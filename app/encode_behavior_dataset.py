#!/usr/bin/env python3
import argparse
import os

import torch
import yaml

from app.vjepa_droid.behavior import BehaviorEpisodePreencoder, BehaviorVideoDataset
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model, load_pretrained


def parse_args():
    parser = argparse.ArgumentParser(description="Encode BEHAVIOR dataset windows with a V-JEPA encoder.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--output", type=str, required=True, help="Directory to write encoded shards.")
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=100,
        help="Number of encoded windows written per shard.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to load encoder weights from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    meta_cfg = cfg.get("meta", {})

    dataset_path = data_cfg["datasets"][0]
    frames_per_clip = max(data_cfg.get("dataset_fpcs", [16]))
    batch_size = int(data_cfg.get("batch_size", 8))
    crop_size = int(data_cfg.get("crop_size", 256))
    patch_size = int(data_cfg.get("patch_size", 16))
    tubelet_size = int(data_cfg.get("tubelet_size", 2))
    fps = data_cfg.get("fps", 5)

    transform = make_transforms(
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
        horizontal_flip=False,
    )

    dataset = BehaviorVideoDataset(
        data_path=dataset_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        camera_frame=bool(data_cfg.get("camera_frame", False)),
        frameskip=tubelet_size,
        state_start_idx=int(data_cfg.get("state_start_idx", 0)),
        state_dim=int(data_cfg.get("state_dim", 7)),
        action_dim=int(data_cfg.get("action_dim", 23)),
        window_stride=data_cfg.get("window_stride", frames_per_clip),
        random_window=bool(data_cfg.get("random_window", False)),
    )

    model_name = model_cfg.get("model_name", "vit_giant2")
    pred_depth = model_cfg.get("pred_depth", 12)
    pred_embed_dim = model_cfg.get("pred_embed_dim", 384)
    pred_num_heads = model_cfg.get("pred_num_heads", 12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = init_video_model(
        uniform_power=model_cfg.get("uniform_power", False),
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=int(data_cfg.get("state_dim", 7)),
        pred_is_frame_causal=model_cfg.get("pred_is_frame_causal", True),
        use_extrinsics=model_cfg.get("use_extrinsics", False),
        use_sdpa=meta_cfg.get("use_sdpa", False),
        use_silu=model_cfg.get("use_silu", False),
        use_pred_silu=model_cfg.get("use_pred_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_rope=model_cfg.get("use_rope", False),
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", False),
    )

    ckpt = args.checkpoint or meta_cfg.get("pretrain_checkpoint")
    if ckpt:
        load_pretrained(
            encoder=encoder,
            predictor=None,
            load_encoder=True,
            load_predictor=False,
            r_path=os.path.expanduser(ckpt),
            context_encoder_key=meta_cfg.get("context_encoder_key", "encoder"),
            target_encoder_key=meta_cfg.get("target_encoder_key", "target_encoder"),
        )

    preencoder = BehaviorEpisodePreencoder(encoder=encoder, device=device, dtype=torch.float32)
    preencoder.encode_full_episodes(
        dataset,
        args.output,
        episodes_per_shard=args.episodes_per_shard,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
