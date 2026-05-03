# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu

from app.vjepa_droid.droid import DROIDVideoDataset

logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=None,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    state_start_idx=0,
    state_dim=7,
    action_dim=23,
    window_stride=1,
    random_window=True,
):
    dataset = BehaviorVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        state_start_idx=state_start_idx,
        state_dim=state_dim,
        action_dim=action_dim,
        window_stride=window_stride,
        random_window=random_window,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("Behavior video data loader created")
    return data_loader, dist_sampler


class BehaviorVideoDataset(DROIDVideoDataset):
    """BEHAVIOR dataset that reuses DROID preprocessing and action/state generation."""

    def __init__(
        self,
        data_path,
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        state_start_idx=0,
        state_dim=7,
        action_dim=23,
        window_stride=1,
        random_window=True,
    ):
        self.data_path = data_path
        self.dataset_root = os.path.dirname(os.path.abspath(data_path))
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.state_start_idx = state_start_idx
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_stride = max(1, int(window_stride))
        self.random_window = random_window

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        manifest = self._load_manifest(data_path)
        self.samples = self._parse_samples(manifest)
        self.windows = self._build_window_index()

    def _load_manifest(self, manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _resolve_episode_layout(self, task_name, episode_file):
        if task_name is not None and episode_file is not None:
            episode_name = os.path.splitext(os.path.basename(episode_file))[0]
            base = os.path.join(self.dataset_root, task_name)
            return {
                "video": os.path.join(base, "video", f"{episode_name}.mp4"),
                "parquet": os.path.join(base, "data", f"{episode_name}.parquet"),
            }
        return None

    def _parse_samples(self, manifest):
        samples = []
        for ep in manifest.get("episodes", []):
            task_name = ep.get("task_name")
            episode_file = ep.get("episode_file")
            fallback = self._resolve_episode_layout(task_name, episode_file)

            video_rel = ep.get("video_file") or (ep.get("video_files") or [None])[0]
            parquet_rel = ep.get("data_parquet_file")

            video_path = (
                os.path.join(self.dataset_root, video_rel)
                if video_rel is not None
                else (fallback["video"] if fallback is not None else None)
            )
            parquet_path = (
                os.path.join(self.dataset_root, parquet_rel)
                if parquet_rel is not None
                else (fallback["parquet"] if fallback is not None else None)
            )

            if video_path is None or parquet_path is None:
                continue

            samples.append({"video_path": video_path, "parquet_path": parquet_path})

        if not samples:
            raise ValueError(f"No episodes found in manifest: {self.data_path}")
        return samples

    def _episode_sampled_indices(self, sample):
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        vfps = vr.get_avg_fps()
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        vlen = len(vr)
        parquet_len = len(pd.read_parquet(ppath, columns=["action"]))
        max_len = min(vlen, parquet_len)
        if max_len < fstp:
            return None, fstp, max_len
        indices = np.arange(0, max_len, fstp, dtype=np.int64)
        if self.frameskip > 1:
            indices = indices[:: self.frameskip]
        return indices, fstp, max_len

    def _build_window_index(self):
        windows = []
        for sample_idx, sample in enumerate(self.samples):
            try:
                indices, _, _ = self._episode_sampled_indices(sample)
                if indices is None:
                    continue
                n = len(indices)
                if n <= 0:
                    continue
                if n <= self.frames_per_clip:
                    windows.append((sample_idx, 0))
                    continue
                max_start = n - self.frames_per_clip
                for start in range(0, max_start + 1, self.window_stride):
                    windows.append((sample_idx, start))
            except Exception as e:
                logger.info(f"Skipping sample during window indexing sample={sample} {e=}")

        if not windows:
            raise ValueError(f"No valid windows found in manifest: {self.data_path}")
        logger.info(f"Built BEHAVIOR window index with {len(windows)} windows from {len(self.samples)} episodes")
        return windows

    def __len__(self):
        return len(self.windows)

    def loadvideo_decord(self, sample, start_idx=0):
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]

        df = pd.read_parquet(ppath)
        if "observation.state" not in df.columns or "action" not in df.columns:
            raise ValueError(f"Expected `observation.state` and `action` in parquet: {ppath}")

        full_states = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
        full_actions = np.asarray(df["action"].to_list(), dtype=np.float32)

        if full_actions.shape[1] < self.action_dim:
            raise ValueError(
                f"Action dim out of bounds for {ppath}: {full_actions.shape[1]=}, {self.action_dim=}"
            )

        if full_states.shape[1] < self.state_start_idx + self.state_dim:
            raise ValueError(
                f"State slice out of bounds for {ppath}: {full_states.shape[1]=}, "
                f"{self.state_start_idx=}, {self.state_dim=}"
            )

        states = full_states[:, self.state_start_idx : self.state_start_idx + self.state_dim]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))

        vfps = vr.get_avg_fps()
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        vlen = len(vr)

        max_len = min(vlen, states.shape[0], full_actions.shape[0])
        if max_len < fstp:
            raise Exception(f"Episode too short for subsampling {vpath=}, {fstp=}, {max_len=}")

        # Fixed-fps subsampling over episode timeline.
        indices = np.arange(0, max_len, fstp, dtype=np.int64)
        if self.frameskip > 1:
            indices = indices[:: self.frameskip]
        if len(indices) == 0:
            raise Exception(f"No indices after subsampling for {vpath=}, {fstp=}, {max_len=}")

        if self.random_window and len(indices) > self.frames_per_clip:
            max_start = len(indices) - self.frames_per_clip
            start_idx = np.random.randint(0, max_start + 1)

        end_idx = min(start_idx + self.frames_per_clip, len(indices))
        window_indices = indices[start_idx:end_idx]
        if len(window_indices) < self.frames_per_clip:
            pad = np.full((self.frames_per_clip - len(window_indices),), window_indices[-1], dtype=np.int64)
            window_indices = np.concatenate([window_indices, pad])

        states = states[window_indices, :]

        # Aggregate raw per-step actions between sampled points for this window.
        raw_actions = full_actions[:, : self.action_dim]
        actions = []
        for i, start in enumerate(window_indices):
            end = window_indices[i + 1] if i + 1 < len(window_indices) else min(start + fstp, max_len)
            actions.append(raw_actions[start:end].sum(axis=0))
        actions = np.asarray(actions, dtype=np.float32)

        vr.seek(0)
        buffer = vr.get_batch(window_indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        # No extrinsics in BEHAVIOR parquet for now; keep predictor API-compatible.
        extrinsics = np.zeros((states.shape[0], 6), dtype=np.float32)
        return buffer, actions, states, extrinsics, window_indices

    def __getitem__(self, index):
        sample_idx, start_idx = self.windows[index]
        sample = self.samples[sample_idx]

        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(sample, start_idx=start_idx)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading sample={sample} {e=}")
                loaded_video = False
                sample_idx, start_idx = self.windows[np.random.randint(self.__len__())]
                sample = self.samples[sample_idx]

        return buffer, actions, states, extrinsics, indices
