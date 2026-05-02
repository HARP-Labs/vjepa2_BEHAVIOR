"""Draft utilities for precomputing DROID encoder tokens.

This module is intentionally lightweight and designed to re-use the existing
DROID dataset/transforms path. It precomputes encoder outputs for full clips
and persists them to shard files, then exposes a fast map-style dataset for
predictor training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from app.vjepa_droid.droid import init_data
from app.vjepa_droid.transforms import make_transforms


@dataclass
class PrecomputeConfig:
    # input/data sampling (kept aligned with droid-256px-8f.yaml)
    data_path: str
    batch_size: int = 8
    frames_per_clip: int = 8
    fps: int = 4
    crop_size: int = 256
    patch_size: int = 16
    camera_views: Tuple[str, ...] = ("left_mp4_path",)
    camera_frame: bool = False
    stereo_view: bool = False
    num_workers: int = 12
    pin_mem: bool = True
    persistent_workers: bool = True

    # writer/output
    output_dir: str = "./precomputed/droid"
    shard_size: int = 4096
    dtype: str = "float16"


def build_raw_droid_loader(cfg: PrecomputeConfig, rank: int = 0, world_size: int = 1):
    """Reuse existing loader path to preserve DROID sampling semantics."""
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[0.75, 1.35],
        random_resize_scale=[1.777, 1.777],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=cfg.crop_size,
    )
    return init_data(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        frames_per_clip=cfg.frames_per_clip,
        tubelet_size=1,
        fps=cfg.fps,
        camera_views=list(cfg.camera_views),
        camera_frame=cfg.camera_frame,
        stereo_view=cfg.stereo_view,
        transform=transform,
        collator=torch.utils.data.default_collate,
        num_workers=cfg.num_workers,
        world_size=world_size,
        pin_mem=cfg.pin_mem,
        persistent_workers=cfg.persistent_workers,
        rank=rank,
    )


def precompute_and_write(
    encoder: torch.nn.Module,
    cfg: PrecomputeConfig,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
):
    """Run encoder once over dataset and save predictor-ready samples.

    Stored sample keys:
      - z: encoder tokens [F, N, D]
      - actions: [F-1, A]
      - states: [F, S]
      - extrinsics: [F, E]
      - indices: video frame indices from source clip
    """
    loader, sampler = build_raw_droid_loader(cfg, rank=rank, world_size=world_size)
    sampler.set_epoch(0)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
    encoder.eval()

    shard: List[Dict[str, torch.Tensor]] = []
    sid = 0
    with torch.inference_mode():
        for clips, actions, states, extrinsics, indices in loader:
            clips = clips.to(device, non_blocking=True)
            z = encoder(clips).to(dtype=dtype).cpu()

            for b in range(z.shape[0]):
                shard.append(
                    {
                        "z": z[b],
                        "actions": actions[b].cpu(),
                        "states": states[b].cpu(),
                        "extrinsics": extrinsics[b].cpu(),
                        "indices": indices[b].cpu(),
                    }
                )
            if len(shard) >= cfg.shard_size:
                torch.save(shard, out / f"rank{rank}_shard{sid:05d}.pt")
                sid += 1
                shard = []

    if shard:
        torch.save(shard, out / f"rank{rank}_shard{sid:05d}.pt")


def predictor_forward_from_precomputed(
    predictor: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    tokens_per_frame: int,
):
    """How states/actions are fed into predictor for precomputed batches.

    Equivalent to the online-train call pattern:
      predictor(z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1])
    """
    z_ctx, actions, states_ctx, extrinsics_ctx = prepare_predictor_inputs_from_precomputed(
        batch, tokens_per_frame=tokens_per_frame
    )
    return predictor(z_ctx, actions, states_ctx, extrinsics_ctx)


class PrecomputedDROIDDataset(Dataset):
    """Fast map-style loader over precomputed shard files."""

    def __init__(self, shard_paths: List[str]):
        self.shard_paths = [str(p) for p in shard_paths]
        self._index: List[Tuple[int, int]] = []
        self._shard_sizes: List[int] = []
        self._cache: Dict[int, List[Dict[str, torch.Tensor]]] = {}

        for sid, p in enumerate(self.shard_paths):
            n = len(torch.load(p, map_location="cpu"))
            self._shard_sizes.append(n)
            for i in range(n):
                self._index.append((sid, i))

    def __len__(self):
        return len(self._index)

    def _load_shard(self, sid: int):
        if sid not in self._cache:
            self._cache[sid] = torch.load(self.shard_paths[sid], map_location="cpu")
        return self._cache[sid]

    def __getitem__(self, idx: int):
        sid, off = self._index[idx]
        return self._load_shard(sid)[off]


def build_precomputed_loader(
    shard_glob: str,
    batch_size: int,
    num_workers: int,
    pin_mem: bool = True,
    persistent_workers: bool = True,
):
    shard_paths = sorted(str(p) for p in Path(os.path.dirname(shard_glob) or ".").glob(Path(shard_glob).name))
    dataset = PrecomputedDROIDDataset(shard_paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
