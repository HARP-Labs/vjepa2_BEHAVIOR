# AC predictor preprocessing/augmentation/encoding pipeline and outsourcing plan

This document maps the current V-JEPA2 DROID action-conditioned (AC) training input path, identifies expensive stages, and proposes a high-level outsourcing/caching design so training can load precomputed tensors and mostly slice/index.

## 1) Current end-to-end input path before AC predictor

1. `app/vjepa_droid/train.py` builds a `VideoTransform`, initializes `DROIDVideoDataset`, and consumes `(clips, actions, states, extrinsics)` from the loader.
2. `DROIDVideoDataset.__getitem__` repeatedly tries loading one sample until successful.
3. `loadvideo_decord` (per sample):
   - reads per-trajectory json metadata (`get_json`),
   - opens `trajectory.h5`,
   - picks one camera view,
   - loads state and camera extrinsics arrays,
   - opens video with decord `VideoReader`, computes sampling stride, samples a random temporal window,
   - computes robot state transforms/differences (`transform_frame`, `poses_to_diffs`),
   - decodes selected frames to numpy,
   - applies visual augmentations/transforms.
4. Training step transfers sample tensors to GPU.
5. `forward_target` encodes clips with frozen target encoder to token features `h`.
6. Predictor input is assembled from token features plus action/state/extrinsics and rolled out autoregressively.

## 2) Time-consuming stages (expected dominant costs)

### A. CPU + storage bound (data pipeline)

- **Frequent small-file IO**: JSON + HDF5 open/read for every sample.
- **Video decode**: decord random-access decode for every sampled clip.
- **CPU math on trajectories**: SciPy Euler/matrix conversions in `poses_to_diffs` and optional `transform_frame`.
- **Frame transforms/augmentation**: random resized crop, horizontal flip, optional auto-augment + random erase.

### B. Host->device + GPU pre-predictor work

- transfer of `(clips, actions, states, extrinsics)` to GPU each iteration,
- frozen target-encoder forward pass that converts clips to latent tokens used as predictor targets/inputs,
- rollout-time token slicing/concats for teacher-forcing + AR loss.

## 2.1) Randomness in the current pipeline (and what full offline caching removes)

If you switch to a **fully offline dataset** (precompute once, then always load fixed tensors), the following stochastic behaviors are no longer sampled per-iteration/per-epoch unless you explicitly preserve them:

1. **DistributedSampler epoch shuffle**
   - Current behavior: sample order is reshuffled each epoch by `DistributedSampler(..., shuffle=True)` and `set_epoch(epoch)`.
   - Full offline risk: if cache reader iterates deterministically without reshuffle, training sees repeated fixed ordering.

2. **Retry path random replacement**
   - Current behavior: if one sample fails to load, `__getitem__` picks a random replacement index with `np.random.randint`.
   - Full offline risk: with no runtime decode failures this disappears (usually good), but it does change effective sampling distribution versus “best-effort” raw loading.

3. **Camera-view random choice**
   - Current behavior: each sample picks a random camera from `camera_views` with `torch.randint`.
   - Full offline risk: precomputing one fixed view per trajectory removes view diversity unless you store multi-view variants and resample at load time.

4. **Temporal window random choice**
   - Current behavior: for each clip, end frame `ef` is randomly sampled (`np.random.randint`), defining random `sf/indices`.
   - Full offline risk: fixed cached window(s) remove temporal diversity and reduce coverage of long trajectories.

5. **Stochastic visual augmentation**
   - Current behavior:
     - random resized crop scale/aspect and optional motion shift,
     - random horizontal flip (p=0.5),
     - optional auto-augment random policy/magnitude path,
     - optional random erasing.
   - Full offline risk: once baked into cache, these become fixed augmentations; epoch-to-epoch augmentation diversity is lost.

6. **Worker/process RNG interplay**
   - Current behavior: DataLoader workers produce slightly different random sequences across epochs/ranks.
   - Full offline risk: deterministic cache lookup can reduce stochastic regularization that previously came “for free” from worker-level randomness.

7. **Potential mixed-source nondeterminism**
   - Current behavior: runtime decode + PIL/tensor conversion/transform order can yield small natural variation across workers/platforms.
   - Full offline risk: cached tensors collapse this to one realized version (better reproducibility, less variation).

### Practical implication

With a fully offline fixed cache, you typically gain throughput and reproducibility, but you also remove several regularizers (multi-view, temporal jitter, augmentation randomness, shuffled ordering if not reintroduced). To retain quality, keep some randomness online (at least sampler shuffle + temporal/view or crop augmentations), or precompute **N variants per trajectory** and randomly draw among variants at train time.

## 3) Can you outsource it?

Yes. The code structure already cleanly separates:
- data loading/augmentation (`app/vjepa_droid/droid.py`, `app/vjepa_droid/transforms.py`),
- training consumption (`app/vjepa_droid/train.py`).

So you can replace the runtime decode+augmentation path with a precompute pipeline and a tensor-backed dataset.

Main caveat: if you fully precompute *after random augmentation*, you lose online augmentation diversity unless you store multiple augmented variants per clip or keep a light online augmentation stage.

## 4) Estimated impact of outsourcing

Use the training CSV metrics (`iter-time`, `gpu-time`, `dataload-time`) to estimate speedup.

Practical estimate:
- If `dataload-time` is ~30-50% of iteration wall time, moving decode/trajectory math/most aug off the training path usually gives **~1.2x-1.7x** iteration speedup.
- If data pipeline dominates (slow storage, high workers contention), speedup can approach **~2x**.
- If GPU compute dominates already, speedup may be only **~1.05x-1.2x**.

Rule-of-thumb upper bound (Amdahl):

`speedup_max = 1 / ((1 - p) + p / s)`

where `p` is fraction of iteration currently spent in preprocessing/dataloading, and `s` is achieved acceleration for that fraction.

Example: `p=0.45`, `s=5` => `1 / (0.55 + 0.09) = 1.56x`.

## 5) Recommended outsource boundaries

### Option 1 (safer for generalization):
- Precompute and store **decoded frame tensors + action/state/extrinsics tensors**,
- keep stochastic crop/flip/erase online.

Benefits: large decode/IO savings, preserve augmentation randomness.

### Option 2 (max throughput):
- Precompute and store **final augmented clips + action/state/extrinsics**,
- train loop only loads tensors and slices.

Benefits: highest throughput; risk: weaker augmentation diversity.

### Option 3 (most aggressive):
- Precompute and store **target-encoder token features `h`** and control tensors,
- predictor training bypasses pixel pipeline and frozen encoder.

Benefits: biggest speedup + lower GPU memory/compute; tradeoff: tightly couples cache to a specific frozen encoder checkpoint + preprocessing config.

## 6) High-level code changes (files and what)

1. **Add offline preprocessor script** (new file, e.g. `app/vjepa_droid/precompute.py`)
   - iterate over trajectory list,
   - run current sampling logic (`loadvideo_decord`-equivalent),
   - write sharded tensor files (e.g. WebDataset tar, zarr, or chunked `.pt`) containing clip/control tensors (and optionally pre-encoded tokens).

2. **Add tensor-backed dataset** (new file, e.g. `app/vjepa_droid/tensor_dataset.py`)
   - `__getitem__` reads precomputed tensors,
   - returns `(clips_or_tokens, actions, states, extrinsics, indices)` with current train-loop-compatible shapes.

3. **Update data init switch** (`app/vjepa_droid/droid.py`)
   - keep current raw-video dataset path,
   - add config-gated path selecting tensor dataset when `data.precomputed_path` exists.

4. **Update train loop input contract** (`app/vjepa_droid/train.py`)
   - branch on `data.input_type`:
     - `pixels`: current behavior,
     - `tokens`: skip `forward_target(clips)` and consume precomputed tokens directly.

5. **Config updates** (`configs/train/vitg16/droid-256px-8f.yaml`)
   - add fields like:
     - `data.precomputed_path`,
     - `data.input_type: pixels|tokens`,
     - `data.cache_format`, `data.cache_num_variants`,
     - `data.cache_checkpoint_id` (for token caches).

6. **Validation/safety checks**
   - add lightweight checksum/schema/version checks so cached tensors match model-critical params (`crop_size`, `fps`, `tubelet_size`, camera view set, encoder checkpoint id).

## 7) Migration strategy

1. Implement Option 1 first (decoded tensors + online stochastic aug).
2. Benchmark 1-2 epochs and compare logged `dataload-time` and `iter-time`.
3. If still bottlenecked and encoder is frozen, move to Option 3 token cache.
4. For any “full offline” mode, explicitly reintroduce randomness knobs (epoch shuffle, variant sampling, optional online light aug) before large-scale training.
5. Keep raw pipeline as fallback for reproducibility and cache regeneration.
