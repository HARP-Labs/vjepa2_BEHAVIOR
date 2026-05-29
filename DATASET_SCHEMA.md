# BEHAVIOR-1K Pre-encoded MDS Dataset Schema

Each MDS shard is a flat sequence of **steps** (one row = one sampled video frame from one episode). Rows are ordered chronologically within each episode; episodes are contiguous.

---

## Columns

| Column | MDS type | NumPy dtype | Shape | Description |
|---|---|---|---|---|
| `tokens_head` | `ndarray` | `float16` | `(tokens_per_frame, embed_dim)` | V-JEPA2 patch tokens for head camera |
| `tokens_left_wrist` | `ndarray` | `float16` | `(tokens_per_frame, embed_dim)` | V-JEPA2 patch tokens for left wrist camera |
| `tokens_right_wrist` | `ndarray` | `float16` | `(tokens_per_frame, embed_dim)` | V-JEPA2 patch tokens for right wrist camera |
| `actions` | `ndarray` | `float32` | `(fstp * 23,)` | Flattened action chunk: `fstp` raw 23-dim action vectors from frame `t` to `t+fstp` |
| `states` | `ndarray` | `float32` | `(133,)` | Curated 133-dim proprioceptive state at frame `t` (see breakdown below) |
| `cam_rel_poses` | `ndarray` | `float32` | `(21,)` | 3 cameras × `[pos(3) + quat(4)]` — head, left_wrist, right_wrist in order |
| `frame_index` | `int` | — | scalar | Source video frame index (before FPS subsampling) |
| `episode_idx` | `int` | — | scalar | Episode index within the manifest |
| `sample_idx` | `int` | — | scalar | Maps `episode_idx` to the file paths (video/parquet) in the manifest |
| `step_pos` | `int` | — | scalar | This step's 0-indexed position within the episode |
| `episode_len` | `int` | — | scalar | Total number of steps in this episode (for masking/grouping) |

`tokens_*` columns are only present for the views that were encoded. A head-only run omits `tokens_left_wrist` and `tokens_right_wrist`.

---

## Token shape by model variant

| Config | Model | Crop | Patch | `tokens_per_frame` | `embed_dim` |
|---|---|---|---|---|---|
| `behavior-vjepa2-vitg16-256px-16f.yaml` | ViT-G/16 (`facebook/vjepa2-vitg-fpc64-256`) | 256 px | 16 | **256** (16×16) | **1408** |
| `behavior-vjepa21-vitg16-384px-16f.yaml` | ViT-G/16 (`vjepa2_1_vit_giant_384`) | 384 px | 16 | **576** (24×24) | **1408** |

Each frame is encoded **independently**: it is duplicated into a `temporal_patch_size=2` tubelet before the ViT forward pass, so you get one token set per frame (not per tubelet pair).

Tokens are stored as **`float16`** (bfloat16 is cast to float16 before writing to disk).

---

## Actions shape

`actions` has shape `(fstp * 23,)` where:

- `action_dim = 23` — leading 23 dims of the raw action vector (joint velocity commands)
- `fstp = ceil(video_native_fps / target_fps)` — frame stride used during sampling

For 30 fps video at 5 fps target: `fstp = 6`, so `actions.shape = (138,)`.

To reconstruct the chunk: `actions.reshape(fstp, 23)` gives the `fstp` consecutive action vectors starting at this frame.

---

## States breakdown — 133-dim proprioceptive vector

Extracted from the 256-dim `observation.state` vector. Simulator-only global state (base odometry, accumulated robot position, global orientation) is excluded.

| Slice (original index) | Dims | Meaning |
|---|---|---|
| `[6:28]` | 22 | `joint_qpos` — controllable joints (torso + arms + grippers, no base odometry) |
| `[34:56]` | 22 | `joint_qpos_sin` |
| `[62:84]` | 22 | `joint_qpos_cos` |
| `[84:112]` | 28 | `joint_qvel` — all joints (encoder-based velocity) |
| `[152:158]` | 6 | `robot_lin_vel` (3) + `robot_ang_vel` (3) — IMU-observable |
| `[186:197]` | 11 | `eef_left_pos` (3) + `eef_left_quat` (4) + `gripper_left_qpos` (2) + `gripper_left_qvel` (2) |
| `[225:244]` | 19 | `eef_right_pos` (3) + `eef_right_quat` (4) + `gripper_right_qpos` (2) + `gripper_right_qvel` (2) + `trunk_qpos` (4) + `trunk_qvel` (4) |
| `[253:256]` | 3 | `base_qvel` — encoder velocity (not accumulated position) |
| **Total** | **133** | |

---

## cam_rel_poses breakdown

Shape `(21,)` = 3 cameras × 7 dims each, in order: head, left_wrist, right_wrist.

Each camera: `[x, y, z, qw, qx, qy, qz]` (position + quaternion relative to robot base).

---

## Loading example

```python
from streaming import StreamingDataset
import numpy as np

ds = StreamingDataset(remote="hf://datasets/<repo_id>/<path_prefix>", split=None)
sample = ds[0]

tokens_head  = sample["tokens_head"]               # (256, 1408) float16  [or (576, 1408) for 384px]
actions      = sample["actions"].reshape(-1, 23)   # (fstp, 23)  float32
states       = sample["states"]                    # (133,)      float32
cam_poses    = sample["cam_rel_poses"].reshape(3, 7)  # (3, 7)   float32
step_pos     = sample["step_pos"]                  # int — position within episode
episode_len  = sample["episode_len"]               # int — total steps in episode
```

To group rows back into full episodes, collect all rows with the same `episode_idx` and sort by `step_pos`.
