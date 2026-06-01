# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_behavior.behavior import make_behavior_dataset
from app.vjepa_behavior.utils import init_opt, init_predictor, load_checkpoint
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    # bfloat16 has float32-range exponents and never needs gradient scaling.
    use_grad_scaler = dtype == torch.float16

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    embed_dim = cfgs_model.get("embed_dim")
    patch_size = cfgs_model.get("patch_size", 16)
    n_cameras = cfgs_model.get("n_cameras")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", 16)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    action_embed_dim = cfgs_model.get("action_embed_dim")
    state_embed_dim = cfgs_model.get("state_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", True)
    use_rope = cfgs_model.get("use_rope", True)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    remote = cfgs_data.get("remote")
    local = cfgs_data.get("local")
    cameras = cfgs_data.get("cameras")
    frames_per_clip = cfgs_data.get("frames_per_clip")
    tpf_per_cam = cfgs_data.get("tpf_per_cam")
    batch_size = cfgs_data.get("batch_size")
    num_workers = cfgs_data.get("num_workers", 8)
    pin_mem = cfgs_data.get("pin_mem", True)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    action_fstp = cfgs_data.get("action_fstp", None)

    assert len(cameras) == n_cameras, (
        f"data.cameras={cameras} (len={len(cameras)}) must match model.n_cameras={n_cameras}"
    )

    if action_fstp is not None:
        assert action_embed_dim == action_fstp * 23, (
            f"model.action_embed_dim={action_embed_dim} != data.action_fstp * 23 = {action_fstp * 23}. "
            f"Update model.action_embed_dim when changing data.action_fstp."
        )

    patch_grid = int(tpf_per_cam ** 0.5)
    assert patch_grid * patch_grid == tpf_per_cam, (
        f"tpf_per_cam={tpf_per_cam} must be a perfect square for spatial RoPE "
        f"(got patch_grid={patch_grid}, patch_grid^2={patch_grid*patch_grid})"
    )
    # Image tokens per physical step (one predictor "frame" packs all cameras)
    tokens_per_phys_step = n_cameras * tpf_per_cam

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    normalize_reps = cfgs_loss.get("normalize_reps", True)
    auto_steps = min(cfgs_loss.get("auto_steps", 2), frames_per_clip - 1)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    if r_file is not None:
        resume_path = os.path.join(folder, r_file)
        if not os.path.exists(resume_path):
            logger.warning(f"Requested resume_checkpoint not found: {resume_path}")
            resume_path = None
    elif os.path.exists(latest_path):
        resume_path = latest_path
    else:
        resume_path = None

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "jloss"),
        ("%.5f", "sloss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- init model: predictor + camera embeddings (no encoder)
    predictor, cam_embed = init_predictor(
        device=device,
        embed_dim=embed_dim,
        tpf_per_cam=tpf_per_cam,
        n_cameras=n_cameras,
        num_frames=frames_per_clip,
        patch_size=patch_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        pred_num_heads=pred_num_heads,
        action_embed_dim=action_embed_dim,
        state_embed_dim=state_embed_dim,
        pred_is_frame_causal=pred_is_frame_causal,
        use_rope=use_rope,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        uniform_power=uniform_power,
    )

    if compile_model:
        logger.info("Compiling predictor.")
        torch._dynamo.config.optimize_ddp = False
        predictor.compile()

    # -- init data-loader
    _, unsupervised_loader, unsupervised_sampler = make_behavior_dataset(
        remote=remote,
        local=local,
        cameras=cameras,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
    )

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"Iterations per epoch / dataset length: {ipe} / {_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        predictor=predictor,
        cam_embed=cam_embed,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        use_grad_scaler=use_grad_scaler,
        betas=betas,
        eps=eps,
    )

    predictor = DistributedDataParallel(predictor, static_graph=False)
    cam_embed = DistributedDataParallel(cam_embed, static_graph=True)

    start_epoch = 0
    if resume_path is not None:
        predictor, cam_embed, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=resume_path,
            predictor=predictor,
            cam_embed=cam_embed,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    # -- camera index tensors (used inside train_step for embedding lookup)
    cam_indices = torch.arange(n_cameras, device=device)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "predictor": predictor.state_dict(),
            "cam_embed": cam_embed.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skipping {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches}")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loader. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Data loading error (retry {iter_retries}): {e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise e

            # -- unpack batch
            # tokens:  [B, T, n_cameras*tpf_per_cam, embed_dim]  float16
            # actions: [B, T, fstp*23]  float32  (raw action chunk per frame)
            # states:  [B, T, 133]      float32  (proprioceptive state per frame)
            def load_batch():
                tokens_cpu = sample[0]   # float16
                actions = sample[1].to(device, dtype=torch.float32, non_blocking=True)
                states = sample[2].to(device, dtype=torch.float32, non_blocking=True)
                # Move tokens to GPU and cast to training dtype
                tokens = tokens_cpu.to(device, dtype=dtype, non_blocking=True)
                return tokens, actions, states

            tokens, actions, states = load_batch()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def build_h(tok):
                    emb = cam_embed(cam_indices)  # [n_cameras, D]

                    # Concatenate cameras contiguously within each physical step so
                    # the predictor sees one frame per physical step containing all
                    # n_cameras views. Per-camera (h, w) RoPE positions are reused
                    # across cameras inside ACRoPEAttention; cam_embed carries identity.
                    cam_segs = []
                    for c in range(n_cameras):
                        seg = tok[:, :, c * tpf_per_cam:(c + 1) * tpf_per_cam, :] + emb[c]
                        cam_segs.append(seg)

                    # [B, T, n_cameras*tpf, D] → [B, T*n_cameras*tpf, D]
                    h = torch.cat(cam_segs, dim=2).flatten(1, 2)
                    if normalize_reps:
                        h = F.layer_norm(h, (h.size(-1),))
                    return h

                def _step_predictor(_z, _a, _s):
                    _z = predictor(_z, _a, _s)
                    if normalize_reps:
                        _z = F.layer_norm(_z, (_z.size(-1),))
                    return _z

                def forward_predictions(z, act, sta):
                    # Teacher forcing: drop the last physical step (all cameras at T-1)
                    _z = z[:, :-tokens_per_phys_step]
                    # One shared action+state token per physical step (no per-camera repeat)
                    z_tf = _step_predictor(_z, act, sta)

                    # Autoregressive rollout
                    _z = torch.cat([z[:, :tokens_per_phys_step], z_tf[:, :tokens_per_phys_step]], dim=1)
                    for n in range(1, auto_steps):
                        _a_n = act[:, : n + 1]
                        _s_n = sta[:, : n + 1]
                        _z_nxt = _step_predictor(_z, _a_n, _s_n)[:, -tokens_per_phys_step:]
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, tokens_per_phys_step:]

                    return z_tf, z_ar

                def loss_fn(z, h):
                    _h = h[:, tokens_per_phys_step : z.size(1) + tokens_per_phys_step]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # Pre-encoded tokens are the targets — no encoder forward needed
                    h = build_h(tokens)

                    # actions/states: context frames only (all but last)
                    z_tf, z_ar = forward_predictions(h, actions[:, :-1], states[:, :-1])
                    jloss = loss_fn(z_tf, h.detach())
                    sloss = loss_fn(z_ar, h.detach())
                    loss = jloss + sloss

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(predictor.parameters()) + list(cam_embed.parameters()),
                    max_norm=1.0,
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return float(loss), float(jloss), float(sloss), _new_lr, _new_wd

            (loss, jloss, sloss, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, jloss, sloss,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [j=%.3f s=%.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_checkpoint(epoch + 1, os.path.join(folder, f"e{epoch}.pt"))
