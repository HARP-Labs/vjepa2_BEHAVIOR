# DROID precompute draft (encode once, save before predictor)

## Is this possible with the current codebase?
Yes. The DROID training stack already cleanly separates:
1. dataset/augmentations loading (`app/vjepa_droid/droid.py`, `transforms.py`)
2. model build (`app/vjepa_droid/utils.py` through `train.py`)
3. predictor stage in the train loop (`app/vjepa_droid/train.py`)

So we can add a **separate precompute flow** that reuses the same data and model construction,
and exits after `encoder(clips)` to persist features.

## DORID-oriented config interpretation (from `configs/train/vitg16/droid-256px-8f.yaml`)
These fields should carry over unchanged to precompute because they define the clip distribution:
- `data.datasets`
- `data.dataset_fpcs`
- `data.fps`
- `data.camera_views`
- `data.crop_size`
- `data.patch_size`
- `data.tubelet_size`
- `data_aug.*` (if we want exact train-time view distribution)

For deterministic/reproducible caches, you may intentionally disable stochastic augs in precompute.
The draft uses the current DROID defaults (no hflip/AA/reprob, fixed resize scale).

## Proposed components

### 1) `PrecomputeConfig`
Holds only fields needed for:
- raw DROID loading
- encoder forward
- shard write format

### 2) `build_raw_droid_loader(cfg)`
Calls existing `init_data(...)` and `make_transforms(...)` to avoid duplicating sampling logic.

### 3) `precompute_and_write(encoder, cfg, device)`
Loop:
1. iterate loader
2. move clips to device
3. encode with inference mode
4. store predictor inputs per sample: `z`, `actions`, `states`, `extrinsics`, `indices`
5. flush periodic shard files `rank*_shard*.pt`

### 4) `PrecomputedDROIDDataset`
Map-style dataset that indexes all saved shards and returns single precomputed sample dicts.

### 5) `build_precomputed_loader(...)`
DataLoader tuned for throughput:
- `shuffle=True`
- `pin_memory=True`
- `persistent_workers=True`
- larger batch than online-video path (since decode/augs are gone)

## Efficient data loader notes
- If `.pt` shards become a bottleneck, move to chunked memory-mappable formats
  (e.g. Arrow/Parquet or zarr) while keeping the same dataset interface.
- Shard size should be tuned to avoid too many tiny files.
- In DDP precompute, write one shard stream per rank to avoid write contention.

## Minimal rollout steps
1. Add a dedicated precompute entrypoint script (or mode flag).
2. Reuse DROID model+data config keys from train config.
3. Run one-time precompute job, producing shard directory.
4. Point predictor-training job to `precomputed_data.shard_glob`.
5. Keep online DROID path unchanged for fallback/ablation.
