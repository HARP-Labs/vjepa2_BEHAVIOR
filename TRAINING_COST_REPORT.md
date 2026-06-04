# BEHAVIOR AC Predictor — Training Cost Report

**Hardware:** NVIDIA H100 NVL (1,979 TFLOPS BF16, 94 GB HBM3) on RunPod  
**Model:** AC Predictor — `pred_depth=24`, `pred_embed_dim=1024`, `pred_num_heads=16` (~300 M params)  
**Target steps:** 94,500 (matching DROID post-training scale)  
**Batch size:** B = 64 · `frames_per_clip` = 8 · patch size = 16 px · crop = 256 px  

---

## How FLOPs are counted

Each training step runs **two kinds of predictor calls** (no encoder — tokens are pre-encoded):

| Pass | Input sequence | Purpose |
|---|---|---|
| Teacher-forcing (TF) | `(T−1) × tpps` tokens | One-shot prediction from full context |
| Autoregressive (AR) | grows from `2×tpps` to `auto_steps×tpps` | `auto_steps − 1` sequential calls |

where `tpps = n_cams × 256 + 2` (image tokens + 1 action token + 1 state token per physical step).

**Per-layer forward FLOPs:**

```
F_layer(S) = 24·B·D²·S   (linear: QKV + MLP)
           +  4·B·D·S²   (attention quadratic)
```

**Backward multiplier:**
- Baseline: ×4 (activation checkpointing on — one extra forward recomputation per layer)
- Optimised: ×3 (checkpointing off — fits in 94 GB at B=64)

**Effective MFU (baseline):** ~20% — limited by dense `[S×S]` float `attn_mask` loaded from HBM each layer, no `torch.compile`, sequential AR calls.

---

## Configuration matrix

| Config | n\_cams | auto\_steps | tpps | S\_TF | S\_AR calls |
|---|---|---|---|---|---|
| Multi-view 5-step | 3 | 5 | 770 | 5,390 | 1,540 / 2,310 / 3,080 / 3,850 |
| Multi-view 2-step | 3 | 2 | 770 | 5,390 | 1,540 |
| Single-view 5-step | 1 | 5 | 258 | 1,806 | 516 / 774 / 1,032 / 1,290 |
| Single-view 2-step | 1 | 2 | 258 | 1,806 | 516 |

Note: `auto_steps=N` runs `N−1` AR predictor calls (`for n in range(1, auto_steps)`).

---

## Results — Baseline

> No `torch.compile`, activation checkpointing on (×4), dense float `attn_mask`, single H100 NVL.

| Config | FLOPs / step | Step time | 94,500 steps | RunPod (~$3.99/hr) | Vast.ai spot (~$2.50/hr) |
|---|---|---|---|---|---|
| Multi-view 5-step | 4,037 TF | ~10.2 s | **~11.2 days** | ~$1,070 | ~$670 |
| Multi-view 2-step | 1,862 TF | ~4.7 s | **~5.1 days** | ~$490 | ~$310 |
| Single-view 5-step | 1,010 TF | ~2.6 s | **~2.8 days** | ~$270 | ~$170 |
| Single-view 2-step | 448 TF | ~1.1 s | **~1.2 days** | ~$120 | ~$74 |

---

## Results — Max Optimised

> Three changes applied, **tokens and clip length unchanged**:
>
> 1. `compile_model: true` in config — kernel fusion, ~12% wall-time gain
> 2. `use_activation_checkpointing: false` in config — ×3 instead of ×4, −25% FLOPs
> 3. Replace dense `attn_mask` with `flex_attention` block-causal in `ACRoPEAttention` — eliminates S×S HBM load, attention MFU ~10% → ~45%

| Config | FLOPs / step | Step time | 94,500 steps | RunPod (~$3.99/hr) | Vast.ai spot (~$2.50/hr) | MFU |
|---|---|---|---|---|---|---|
| Multi-view 5-step | 3,028 TF | ~3.2 s | **~3.5 days** | ~$340 | ~$210 | ~47% |
| Multi-view 2-step | 1,397 TF | ~1.5 s | **~1.6 days** | ~$155 | ~$97 | ~48% |
| Single-view 5-step | 758 TF | ~0.8 s | **~0.9 days** | ~$87 | ~$54 | ~46% |
| Single-view 2-step | 336 TF | ~0.4 s | **~10 hours** | ~$38 | ~$24 | ~46% |

---

## Speedup breakdown (multi-view 2-step example)

| Optimisation | Source | Step time | Cumulative gain |
|---|---|---|---|
| Baseline | — | 4.71 s | 1× |
| + no activation checkpointing | fewer FLOPs (×3 not ×4) | 3.53 s | 1.3× |
| + `torch.compile` | kernel fusion | 3.11 s | 1.5× |
| + `flex_attention` block-causal | attention MFU 10%→45% | ~1.48 s | **3.2×** |

The dominant gain comes from `flex_attention`. At S=5,390 the TF attention term is **47% of per-layer FLOPs** and runs memory-bandwidth-bound with the current dense mask. `flex_attention` evaluates the block-causal predicate tile-by-tile in SRAM, eliminating the S×S HBM load entirely.

---

## Why the dense mask is the bottleneck

The predictor registers a precomputed `[S, S]` float buffer as the causal mask
(`src/models/utils/modules.py`, `ACRoPEAttention.forward`).
PyTorch SDPA with a materialized float `attn_mask` cannot dispatch to the efficient
FlashAttention-2 causal kernel — it falls back to the `EFFICIENT_ATTENTION` path,
which must load and apply the full mask from HBM per layer per head.

At S=5,390 and D=1,024 the attention quadratic term equals the linear term
(`S / 6D = 5390 / 6144 ≈ 0.88`), so fixing the attention kernel halves effective
per-layer compute time.

### flex_attention replacement (ACRoPEAttention.forward)

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def _block_causal_fn(tpps):
    def fn(b, h, q_idx, kv_idx):
        return (q_idx // tpps) >= (kv_idx // tpps)
    return fn

# Build once per unique S (cache by S value)
block_mask = create_block_mask(
    _block_causal_fn(tokens_per_phys_step),
    B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
)
out = flex_attention(q, k, v, block_mask=block_mask)
```

Requires PyTorch ≥ 2.5.

---

## Memory check — checkpointing off at B=64

| Component | Memory |
|---|---|
| Model params (BF16) | ~600 MB |
| Optimizer states (AdamW fp32) | ~3.6 GB |
| Activations — TF pass (S=5,390, 24 layers) | ~17 GB |
| Activations — AR pass (S=1,540, 24 layers) | ~5 GB |
| Gradients | ~600 MB |
| **Total** | **~27 GB** |

Comfortably within H100 NVL 94 GB. Checkpointing can be disabled at B=64 for all four configurations.

---

## Constraints

- Token count and clip length are fixed (`tpf_per_cam=256`, `frames_per_clip=8`).
- Estimates assume single H100 NVL. Adding a second GPU halves wall-time linearly (DDP, no model parallelism needed at ~300 M params).
- Data loading is not a bottleneck when the MDS cache is on RunPod NVMe (~3–7 GB/s available vs. ~1.2 GB/s peak requirement for the largest batch).
- `auto_steps` is clamped to `min(auto_steps, frames_per_clip − 1) = min(N, 7)`, so 5-step rollout is valid with `frames_per_clip=8`.
