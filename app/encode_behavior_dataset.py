#!/usr/bin/env python3

import argparse

import torch
import yaml

from app.vjepa_droid.behavior import BehaviorEpisodePreencoder, BehaviorVideoDataset
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model, load_pretrained


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    meta_cfg = cfg.get("meta", {})
    out_cfg = cfg["output"]
    aug_cfg = cfg.get("data_aug", {})

    dtype_name = meta_cfg.get("dtype", "float32").lower()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, _ = init_video_model(
        device=device,
        patch_size=data_cfg["patch_size"],
        max_num_frames=max(data_cfg["dataset_fpcs"]),
        tubelet_size=data_cfg["tubelet_size"],
        model_name=model_cfg["model_name"],
        crop_size=data_cfg["crop_size"],
        use_sdpa=meta_cfg.get("use_sdpa", False),
        use_rope=model_cfg.get("use_rope", False),
        use_silu=model_cfg.get("use_silu", False),
        wide_silu=model_cfg.get("wide_silu", False),
    )

    ckpt = meta_cfg.get("pretrain_checkpoint")
    if ckpt:
        encoder, _, _ = load_pretrained(
            ckpt,
            encoder=encoder,
            predictor=None,
            target_encoder=None,
            context_encoder_key=meta_cfg.get("context_encoder_key", "encoder"),
            target_encoder_key=meta_cfg.get("target_encoder_key", "target_encoder"),
            load_predictor=False,
            load_encoder=True,
        )

    transform = make_transforms(
        random_horizontal_flip=aug_cfg.get("horizontal_flip", False),
        random_resize_aspect_ratio=tuple(aug_cfg.get("random_resize_aspect_ratio", [0.75, 1.35])),
        random_resize_scale=tuple(aug_cfg.get("random_resize_scale", [1.777, 1.777])),
        reprob=aug_cfg.get("reprob", 0.0),
        auto_augment=aug_cfg.get("auto_augment", False),
        motion_shift=aug_cfg.get("motion_shift", False),
        crop_size=data_cfg["crop_size"],
    )
    dataset = BehaviorVideoDataset(
        data_path=data_cfg["datasets"][0],
        fpcs=data_cfg["dataset_fpcs"][0],
        fps=data_cfg.get("fps"),
        transform=transform,
        camera_frame=data_cfg.get("camera_frame", False),
        state_start_idx=data_cfg.get("state_start_idx", 0),
        state_dim=data_cfg.get("state_dim", 7),
        action_dim=data_cfg.get("action_dim", 23),
    )

    preencoder = BehaviorEpisodePreencoder(encoder=encoder, device=device, dtype=dtype)
    preencoder.encode_full_episodes(
        dataset,
        output_dir=out_cfg.get("local_output_dir"),
        hf_repo_id=out_cfg.get("hf_repo_id"),
        hf_path_prefix=out_cfg.get("hf_path_prefix", ""),
        episodes_per_shard=out_cfg.get("episodes_per_shard", 1),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_mem", True),
        persistent_workers=data_cfg.get("persistent_workers", True),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="YAML config path")
    args = parser.parse_args()
    main(args.fname)
