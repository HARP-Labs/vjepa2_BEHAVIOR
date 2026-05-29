# Plan: BEHAVIOR Pre-encoded Token Predictor Training

## Context

The BEHAVIOR-1K dataset has been pre-encoded: each video frame has already been passed through the V-JEPA2 ViT-G encoder and stored as token arrays in MDS shards (one row = one step). The goal is to train the **action-conditioned predictor** (`ACPredictor`) to predict future token representations from past tokens + robot proprioceptive state/actions — exactly like the DROID training, but skipping the encoder entirely (tokens are already computed). This eliminates the most expensive part of DROID training (encoder forward + EMA target forward) and lets the predictor train much faster on pre-computed representations.

---

## Key Findings

### Dataset schema (DATASET_SCHEMA.md)
- Each MDS row = one frame: `tokens_head (tpf, 1408) float16`, `tokens_left_wrist`, `tokens_right_wrist` (same shape), `actions (fstp*23,) float32`, `states (133,) float32`
- `fstp = ceil(video_fps / target_fps)` = 6 for 30fps @ 5fps; so **action_dim = 6 × 23 = 138** (full chunk, no averaging)
- Episodes are contiguous rows grouped by `episode_idx`; `step_pos` + `episode_len` allow clip construction

### Critical constraint in `src/models/ac_predictor.py`
Lines 54–56 use `action_embed_dim` for **both** action and state linear projections:
```python
self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)  # 138D for BEHAVIOR
self.state_encoder  = nn.Linear(action_embed_dim, predictor_embed_dim)  # needs 133D — different!
```
BEHAVIOR: `action_dim = 6*23 = 138`, `state_dim = 133`. Requires adding a separate `state_embed_dim` parameter. This is a 3-line backward-compatible change.

### DROID training loop (fully understood)
- `forward_target(clips)` → `h = [B, T*tpf, D]` — **replaced by loading pre-encoded tokens**
- `forward_predictions(h)` teacher-forcing + autoregressive rollout — **reused verbatim**
- `loss_fn(z, h)` L1, `loss_exp=1.0` — **reused verbatim**
- `init_opt()` AdamW + WSDSchedule — **reused (predictor-only param groups)**

---

## Architecture Decision: New module vs. extend DROID

The core question: create `app/vjepa_behavior/` as a parallel module, or bolt BEHAVIOR support into `app/vjepa_droid/`?

| | **Option A: New `app/vjepa_behavior/` module** | **Option B: Extend `app/vjepa_droid/`** |
|---|---|---|
| **Pros** | Clean separation; DROID untouched; can evolve independently; no risk of DROID regression | Less total code; shared logic lives in one place; one training loop to maintain |
| **Cons** | ~60 lines of duplicated forward_predictions + loss_fn logic | `droid/train.py` gets encoder/no-encoder branching; dataset routing becomes awkward; risk of breaking existing DROID runs |
| **Risk** | Low — only new files + 3 small edits to shared code | Medium — changes to the shared train.py could silently break DROID |

**Recommendation: Option A — new `app/vjepa_behavior/` module.**

The two datasets are fundamentally different: DROID loads raw video → encodes on the fly → needs encoder + EMA target. BEHAVIOR skips all of that. Merging them into one loop would require deep `if is_preencoded` branching around the most performance-critical paths. The "duplicated" logic (`forward_predictions`, `loss_fn`) is ~60 lines; any future shared refactor can extract it into `src/utils/ac_training.py` without touching either module. The shared infrastructure that matters (`ac_predictor.py`, `src/utils/schedulers.py`, `src/utils/distributed.py`) is already correctly in `src/`.

---

## Files to Touch

### New files
| File | Role |
|---|---|
| `app/vjepa_behavior/__init__.py` | Package marker (empty) |
| `app/vjepa_behavior/behavior.py` | `BehaviorMDSDataset` + `make_behavior_dataset()` |
| `app/vjepa_behavior/train.py` | Training loop (stripped DROID loop, no encoder) |
| `app/vjepa_behavior/utils.py` | `init_predictor()`, `init_opt()`, `load_checkpoint()` |
| `configs/train/behavior/behavior-vitg16-256px-16f.yaml` | Training config |

### Minimal edits to existing files
| File | Change | Size |
|---|---|---|
| `src/models/ac_predictor.py` | Add `state_embed_dim=None` param (defaults to `action_embed_dim`) | 3 lines |
| `app/scaffold.py` | Add `"vjepa_behavior"` routing entry | 1 line |
| `requirements.txt` | Add `mosaicml-streaming` | 1 line |

**Nothing else is touched.**

---

## Step-by-Step Implementation

### Step 1 — Fix `src/models/ac_predictor.py` (backward-compatible)

```python
# __init__ signature — add after action_embed_dim:
state_embed_dim=None,

# __init__ body — replace line 55:
_state_dim = state_embed_dim if state_embed_dim is not None else action_embed_dim
self.state_encoder = nn.Linear(_state_dim, predictor_embed_dim, bias=True)
```

DROID passes no `state_embed_dim` → falls back to `action_embed_dim=7`. No behavior change.

---

### Step 2 — `app/vjepa_behavior/behavior.py` — Dataset

**Multi-camera token concatenation:**

All three cameras are concatenated along the token dimension per frame, with a learned per-camera embedding to distinguish viewpoints:

```
tokens_per_frame = N_active_cams × (patch_grid)²   e.g., 3 × 256 = 768
```

For RoPE positional encoding, the AC predictor is instantiated with a non-square grid:
```
grid_height = patch_grid,  grid_width = N_cams × patch_grid
→ e.g., 16 × 48 for ViT-G/16 at 256px with 3 cameras
```

A learned `camera_embed: nn.Embedding(N_cams, embed_dim)` is added to each camera's tokens before concatenation (added in `train.py` on GPU, trained alongside predictor). This lets the predictor distinguish viewpoints.

**`BehaviorMDSDataset.__init__`:**
1. Open `StreamingDataset(remote=remote, local=local, shuffle=False)`
2. Scan all rows once reading only `episode_idx`, `step_pos`, `episode_len` (integers — fast, no tensor loading)
3. Build `self.clips_index: list[int]` — row index of each valid clip start: `episode_len - step_pos >= T`
4. Cache index to `{local}/_clip_index_T{T}.npy`; reload on subsequent runs

**`BehaviorMDSDataset.__getitem__(idx)`:**
1. `row_start = self.clips_index[idx]`
2. Load T consecutive rows: `ds[row_start + t]` for t in 0..T-1
3. Stack per camera (only active cameras; gracefully skip missing `tokens_*` keys):
   ```
   tokens: (T, N_active_cams * tpf_per_cam, embed_dim) float16
   ```
   Camera tokens concatenated per frame: `[head | left_wrist | right_wrist]`
4. Actions: each row's `actions` is `(fstp*23,)` → keep flat as `(138,)` → stack → `(T, 138)`
5. States: stack → `(T, 133)`

**Collated batch shapes:**
```
tokens:  [B, T, N_cams*tpf, embed_dim]  float16 → cast to bfloat16 on GPU
actions: [B, T, 138]                     float32
states:  [B, T, 133]                     float32
```

**`make_behavior_dataset()` factory:**
- Returns `(dataset, data_loader, dist_sampler)` — same pattern as `make_videodataset()`
- `torch.utils.data.distributed.DistributedSampler` (shuffle=True, seed by epoch)
- `torch.utils.data.default_collate` (no custom collator needed — shapes are uniform)

---

### Step 3 — `app/vjepa_behavior/utils.py`

**`init_predictor()`** — instantiates `vit_ac_predictor` with BEHAVIOR dimensions:

```python
def init_predictor(device, embed_dim, tpf_per_cam, n_cameras, num_frames,
                   patch_size, pred_depth, pred_embed_dim, pred_num_heads,
                   action_embed_dim=138, state_embed_dim=133, ...):
    patch_grid = int(tpf_per_cam**0.5)                    # 16 for 256 tokens
    predictor = vit_ac_predictor(
        img_size=(patch_grid * patch_size,
                  n_cameras * patch_grid * patch_size),   # e.g. (256, 768)
        patch_size=patch_size,
        num_frames=num_frames,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,    # 138
        state_embed_dim=state_embed_dim,      # 133
        depth=pred_depth,
        ...
    )
    cam_embed = nn.Embedding(n_cameras, embed_dim)  # learned camera embeddings
    return predictor.to(device), cam_embed.to(device)
```

**`init_opt(predictor, cam_embed, ...)`** — copy of DROID's `init_opt` without encoder param groups:
- Two param groups covering both `predictor` and `cam_embed`: (weights + WD) and (biases/1D, no WD)
- WSDSchedule + CosineWDSchedule from `src/utils/schedulers.py`

**`load_checkpoint(predictor, cam_embed, opt, scaler)`** — simplified (no encoder/target_encoder):
- Checkpoint keys: `"predictor"`, `"cam_embed"`, `"opt"`, `"scaler"`, `"epoch"`, `"loss"`

---

### Step 4 — `app/vjepa_behavior/train.py` — Training Loop

**Key differences from `app/vjepa_droid/train.py`:**

| DROID | BEHAVIOR |
|---|---|
| `encoder` + `target_encoder` (EMA) initialized | Not needed |
| `forward_target(clips)` encodes T frames | `h = apply_cam_embed(tokens).flatten(1,2).to(dtype)` |
| `action_embed_dim=7`, `state_embed_dim=7` | `action_embed_dim=138`, `state_embed_dim=133` |
| `load_pretrained()` for encoder | Not needed |
| `tokens_per_frame = (crop_size // patch_size)²` | `tokens_per_frame = n_cameras * tpf_per_cam` (from config) |
| checkpoint: encoder + target_encoder + predictor | checkpoint: predictor + cam_embed only |

**`forward_predictions(h)` and `loss_fn(z, h)` — verbatim copy from DROID.**

Core train step:
```python
def train_step():
    # tokens: [B, T, N_cams*tpf, D] float16
    # Add per-camera learned embeddings before flattening
    tpf = tpf_per_cam
    tokens = tokens.to(device, dtype=dtype)
    for cam_idx in range(n_cameras):
        tokens[:, :, cam_idx*tpf:(cam_idx+1)*tpf] += cam_embed(
            torch.tensor(cam_idx, device=device)
        )

    h = tokens.flatten(1, 2)                            # [B, T*N_cams*tpf, D]
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))

    # forward_predictions identical to DROID:
    z_tf, z_ar = forward_predictions(h, actions[:, :-1], states[:, :-1])
    jloss = loss_fn(z_tf, h)
    sloss = loss_fn(z_ar, h)
    loss = jloss + sloss
```

---

### Step 5 — `app/scaffold.py`

```python
elif app == "vjepa_behavior":
    from app.vjepa_behavior.train import main
```

---

### Step 6 — Config `configs/train/behavior/behavior-vitg16-256px-16f.yaml`

```yaml
app: vjepa_behavior

meta:
  resume_checkpoint: null
  dtype: bfloat16

data:
  remote: "hf://datasets/<repo_id>/<path_prefix>"
  local: "/tmp/behavior_mds_cache"
  cameras: [head, left_wrist, right_wrist]
  frames_per_clip: 8
  tpf_per_cam: 256          # tokens per frame per camera (16×16 @ 256px)
  embed_dim: 1408            # ViT-G embed dim
  action_fstp: 6             # fstp used during encoding
  batch_size: 32
  num_workers: 8
  pin_mem: true
  persistent_workers: true

model:
  embed_dim: 1408
  patch_size: 16
  n_cameras: 3
  pred_depth: 24
  pred_embed_dim: 1024
  pred_num_heads: 16
  pred_is_frame_causal: true
  action_embed_dim: 138      # action_fstp * 23 = 6 * 23
  state_embed_dim: 133
  use_rope: true
  use_activation_checkpointing: true

loss:
  loss_exp: 1.0
  auto_steps: 2
  normalize_reps: true

optimization:
  epochs: 100
  warmup: 5
  anneal: 10
  ipe: null
  lr: 0.000425
  start_lr: 0.000075
  final_lr: 0.0
  weight_decay: 0.04
  final_weight_decay: 0.04
  betas: [0.9, 0.999]
  eps: 1.0e-8
```

---

## Verification

1. **Unit test the dataset:**
   ```python
   ds = BehaviorMDSDataset(remote=..., local=..., frames_per_clip=8,
                            cameras=["head", "left_wrist", "right_wrist"])
   tokens, actions, states = ds[0]
   assert tokens.shape == (8, 768, 1408)   # 3 cams × 256 tokens
   assert actions.shape == (8, 138)         # 6 × 23
   assert states.shape  == (8, 133)
   ```

2. **Predictor shape check (no data needed):**
   ```python
   pred, cam_embed = init_predictor(embed_dim=1408, tpf_per_cam=256, n_cameras=3, ...)
   h = torch.randn(2, 8*768, 1408)
   z = pred(h[:, :-768], actions[:, :-1], states[:, :-1])
   assert z.shape == (2, 7*768, 1408)
   ```

3. **Backward compatibility:**
   ```bash
   pytest tests/models/test_predictor.py   # state_embed_dim=None → no change for DROID
   ```

4. **Smoke-test one training step:**
   ```bash
   python -m app.main --fname configs/train/behavior/behavior-vitg16-256px-16f.yaml \
     --devices cuda:0 --debug
   ```
