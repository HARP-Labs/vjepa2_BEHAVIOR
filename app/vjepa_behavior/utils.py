# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from functools import partial

import torch
import torch.nn as nn

import src.models.ac_predictor as vit_ac_pred
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_predictor(
    device,
    embed_dim,
    tpf_per_cam,
    n_cameras,
    num_frames,
    patch_size=16,
    pred_depth=24,
    pred_embed_dim=1024,
    pred_num_heads=16,
    action_embed_dim=138,
    state_embed_dim=133,
    pred_is_frame_causal=True,
    use_rope=True,
    use_sdpa=False,
    use_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
    uniform_power=True,
):
    """
    Instantiate VisionTransformerPredictorAC for the BEHAVIOR pre-encoded setting.

    The AC predictor grid is set to (patch_grid, n_cameras * patch_grid) so that
    all camera tokens within a frame share a consistent spatial indexing for RoPE.

    Also returns a learnable camera embedding (nn.Embedding) so the predictor can
    distinguish token origin by camera view.
    """
    patch_grid = int(tpf_per_cam ** 0.5)  # e.g. 16 for 256-token frames
    assert patch_grid * patch_grid == tpf_per_cam, (
        f"tpf_per_cam={tpf_per_cam} must be a perfect square for spatial RoPE "
        f"(got patch_grid={patch_grid}, patch_grid^2={patch_grid*patch_grid})"
    )
    img_h = patch_grid * patch_size                   # e.g. 256
    img_w = n_cameras * patch_grid * patch_size       # e.g. 768 for 3 cams

    predictor = vit_ac_pred.vit_ac_predictor(
        img_size=(img_h, img_w),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads,
        action_embed_dim=action_embed_dim,
        state_embed_dim=state_embed_dim,
        is_frame_causal=pred_is_frame_causal,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_extrinsics=False,
    )

    cam_embed = nn.Embedding(n_cameras, embed_dim)

    predictor.to(device)
    cam_embed.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Predictor parameters: {count_parameters(predictor):,}")
    logger.info(f"Camera embedding parameters: {count_parameters(cam_embed):,}")

    return predictor, cam_embed


def load_checkpoint(r_path, predictor, cam_embed, opt=None, scaler=None):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    pretrained_dict = checkpoint["predictor"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded predictor from epoch {epoch}: {msg}")

    if "cam_embed" in checkpoint:
        pretrained_dict = checkpoint["cam_embed"]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        msg = cam_embed.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded cam_embed from epoch {epoch}: {msg}")

    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"Resumed from epoch {epoch}")
    del checkpoint

    return predictor, cam_embed, opt, scaler, epoch


def init_opt(
    predictor,
    cam_embed,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    """AdamW + WSD LR schedule + cosine WD schedule for predictor + cam_embed."""
    all_modules = [("predictor", predictor), ("cam_embed", cam_embed)]

    param_groups = []
    for _, module in all_modules:
        param_groups += [
            {
                "params": (
                    p for n, p in module.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ),
            },
            {
                "params": (
                    p for n, p in module.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
