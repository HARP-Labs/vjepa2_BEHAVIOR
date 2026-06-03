# Trajectory Planner for BEHAVIOR AC Predictor

## Context

The action-conditioned predictor trained in [app/vjepa_behavior/train.py](app/vjepa_behavior/train.py) is currently only exercised as a training loss target. To validate it as a world model and produce paper-quality results, we need a runtime planner that:

1. Encodes start + goal observations once via the frozen V-JEPA2 ViT-G encoder
2. Uses the trained AC predictor to roll out batched futures conditioned on candidate action sequences
3. Scores rollouts against subgoal-image embeddings
4. Selects an action, executes it in OmniGibson, and logs everything for paper figures and offline visualization

The existing planning code at [notebooks/utils/mpc_utils.py](notebooks/utils/mpc_utils.py) was built for DROID — 7-DOF pose-delta actions with **analytical state integration** via `compute_new_pose()`. It is unusable for BEHAVIOR's 138-dim action chunks, 133-dim proprioceptive state, 3-camera observation layout, and lack of analytical state integration. This plan implements a clean, BEHAVIOR-native replacement.

Design constraints from prior discussion:
- Single GPU; H100/A100 target
- Short horizons (2–4 plan steps) expected to be most effective — design accordingly
- Encoder is a one-shot cost, never re-run during the inner planning loop
- 3-camera ViT-L predictor at B=400 samples is too expensive; default config will be ~50–64 samples with all standard speedups

---

## Module Layout

```
evals/behavior_planning/
├── __init__.py
├── main.py                     # CLI entry — loads YAML, dispatches to run()
├── run.py                      # episode/MPC loop
├── world_model.py              # encoder + predictor + cam_embed wrapper
├── planner/
│   ├── __init__.py
│   ├── base.py                 # Planner ABC + shared sampling/scoring infra
│   ├── cem.py                  # CEM
│   ├── mppi.py                 # MPPI
│   └── sampling.py             # pre-allocated buffers, action-space adapters
├── env/
│   ├── __init__.py
│   ├── omnigibson_env.py       # OG wrapper
│   └── subgoal_source.py       # held-out trajectory loader + subgoal iterator
├── metrics.py                  # timing / divergence / goal-dist / success trackers
└── logging_utils.py            # CSV + trajectory HDF5 dump

configs/eval/behavior_planning/
├── cem-default.yaml
└── mppi-default.yaml

tests/eval/behavior_planning/
├── test_world_model.py
├── test_planner.py
└── test_subgoal_source.py
```

---

## Existing Code to Reuse (no modification)

| Component | Path | How it's used |
|---|---|---|
| Encoder factory | `src/hub/backbones.py::vjepa2_vit_giant(img_size=256)` | Load pretrained ViT-G from torch.hub |
| Predictor factory | `src/models/ac_predictor.py::vit_ac_predictor` | Build `VisionTransformerPredictorAC` |
| Predictor init wrapper | `app/vjepa_behavior/utils.py::init_predictor` | Reuse to build predictor + cam_embed with matching hyperparams |
| Checkpoint loader | `app/vjepa_behavior/utils.py::load_checkpoint` | Load trained predictor + cam_embed (predictor=DDP stripped at load) |
| Held-out tokens source | `app/vjepa_behavior/behavior.py::BehaviorMDSDataset` | Filter `clips_index` by `episode_idx ∈ holdout_set` to source goal tokens |
| Robust checkpoint | `src/utils/checkpoint_loader.py::robust_checkpoint_loader` | Already used inside `load_checkpoint` |
| Logging | `src/utils/logging.py::CSVLogger`, `gpu_timer` | Per-step CSV + cuda.Event timing |
| Cam-segmenting pattern | training loop `build_h()` in [app/vjepa_behavior/train.py:373-385](app/vjepa_behavior/train.py#L373-L385) | Lifted verbatim into `WorldModel._build_h` |
| Autoregressive rollout | `forward_predictions` in [app/vjepa_behavior/train.py:391-408](app/vjepa_behavior/train.py#L391-L408) | Adapted for variable horizon + batched samples in `WorldModel.rollout` |

---

## Component Specifications

### 1. `world_model.py` — WorldModel

Wraps the frozen encoder + trained predictor + cam_embed into the interface the planner needs.

```python
class WorldModel:
    def __init__(self, encoder, predictor, cam_embed, *, dtype, normalize_reps,
                 cameras, tpf_per_cam, n_cameras, state_strategy):
        # state_strategy: "hold_constant" | "zero" | "env_query"
        ...

    @torch.inference_mode()
    def encode_observation(self, obs_dict) -> Tensor:
        """obs_dict: {cam_name: HxWxC uint8 RGB}.  Returns [1, n_cameras*tpf, D] bf16."""
        # 1. Stack cameras to [n_cameras, C, H, W], apply V-JEPA preprocess.
        # 2. Duplicate frame to tubelet_size=2 per DATASET_SCHEMA (encoder expects video).
        # 3. Single encoder forward, batch=n_cameras.
        # 4. Apply cam_embed[c] per camera segment, concat along token dim.
        # 5. layer_norm if normalize_reps.

    @torch.inference_mode()
    def rollout(self, start_tokens, actions, states0) -> Tensor:
        """
        start_tokens: [1, n_cameras*tpf, D]    # current observation embedding
        actions:      [B, H, action_embed_dim] # candidate action sequences (138-dim native)
        states0:      [1, state_embed_dim]      # observed proprioceptive state at t=0
        Returns:      [B, H, n_cameras*tpf, D]  # predicted tokens at each plan step
        """
        # Expand start to [B, ...], build states tensor according to state_strategy.
        # Autoregressive loop following the training-loop pattern:
        #   _z accumulates tokens, each iter appends -tokens_per_phys_step slice.
        # Pre-allocate _z buffer of max shape [B, H * tokens_per_phys_step, D] once.

    @torch.inference_mode()
    def score(self, rollout_tokens, goal_tokens, metric) -> Tensor:
        """[B, H, N, D], [1, N, D]  ->  [B]   (lower is better; uses final step by default)"""
        # metrics: cosine | l1 | l2 ; applied on layer-normed tokens.
        # Configurable: final-step only vs. discounted sum across horizon.
```

**Hot-loop optimizations:**
- Pre-allocate rollout buffer in `__init__` (sized for max samples × horizon)
- Single `.to(device)` for actions/states on planner side; no per-iter copies
- `torch.compile(mode='reduce-overhead')` on predictor forward, opt-in via config
- AMP autocast (bf16) wraps the rollout
- Encoder weights live in bf16 to halve memory & speed up the one-shot encode

### 2. `planner/base.py` — Planner ABC

```python
class Planner(ABC):
    def __init__(self, *, horizon, samples, iterations, action_dim,
                 action_clip, warm_start, world_model, action_space_adapter,
                 device, dtype):
        # Pre-allocate sample buffer: [samples, horizon, action_dim]
        # Pre-allocate score buffer:  [samples]
        # Distribution state: mean [horizon, action_dim], std [horizon, action_dim]
        ...

    @abstractmethod
    def _update_distribution(self, samples, scores): ...

    def plan(self, start_tokens, goal_tokens, *, observed_state, prev_solution=None):
        # 1. warm_start: shift prev_solution forward by 1, fill last step with prior
        # 2. for iter in range(iterations):
        #      sample()                # in-place into pre-alloc buffer
        #      rollout = world_model.rollout(start_tokens, samples_expanded, observed_state)
        #      scores = world_model.score(rollout, goal_tokens, metric)
        #      _update_distribution(samples, scores)
        #      log iter metrics (best_score, timing breakdown)
        # 3. return mean (the planned trajectory) + diagnostics
```

Action-space adapter (sampling.py) handles the **native ↔ planner-native** switch:
- `NativeAdapter`: samples directly in 138-dim space, passes through unchanged
- `PlannerNativeAdapter`: samples in `[6, 23]` per plan step, flattens to 138-dim for predictor
  - Allows correlated noise within a 6-action chunk
  - Per-dim clipping in 23-dim space, cleaner than clipping the 138-dim flat vector

### 3. `planner/cem.py` — CEM

- Elite ratio (default 10%)
- Mean/std momentum (separate per-dim if needed)
- Optional colored noise (β ∈ {0, 0.5, 1, 2}) via `colorednoise` or hand-rolled FFT-based sampler
- Distribution reset switch (`reset_std_each_call`) for non-warm-start mode

### 4. `planner/mppi.py` — MPPI

- Temperature λ (lower → sharper)
- Reward → weights: `w = softmax(-λ * cost)`
- `new_mean = (w[:, None, None] * samples).sum(0)`
- No elite selection; uses all samples weighted

### 5. `env/omnigibson_env.py` — OmniGibsonEnv

Task-agnostic wrapper (task selection happens via YAML). User has OG installed; no task chosen yet.

```python
class OmniGibsonEnv:
    def __init__(self, *, demo_manifest, task_cfg, scene_cfg, cameras, control_hz):
        # Lazy-import omnigibson; build env from task_cfg + scene_cfg.
        # Index demo_manifest: episode_idx → hdf5_path, sample_idx → frame_offsets.

    def reset(self, demo_id=None) -> dict:
        # if demo_id: read HDF5, set robot qpos / qvel / object poses to demo initial state.
        # else: env.reset() with task defaults.
        # Returns obs_dict {head: RGB, left_wrist: RGB, right_wrist: RGB, state: 133-dim}.

    def step(self, action_23) -> dict: ...

    def step_chunk(self, action_chunk_138, *, dump_intermediate=True) -> list[dict]:
        # Reshape (138,) -> (6, 23), apply sequentially. Returns 6 obs_dicts or just last.

    def get_state(self) -> np.ndarray:
        # Extract 133-dim proprioceptive vector matching DATASET_SCHEMA.md slicing.
        # Lifted from whatever script produced the MDS dataset (need user to confirm path).

    def render_for_dump(self) -> dict: ...
```

**Optimizations in the wrapper:**
- Reuse OG render targets across episodes (no recreate)
- Cache HDF5 file handles per scene
- Optional: in open-loop mode, skip rendering wrist cams every step if `raw_obs_subsample > 1`

### 6. `env/subgoal_source.py` — SubgoalSource

- Constructor: held-out split (config: explicit `holdout_episodes: [...]` OR deterministic `seed + holdout_frac`)
- Reuses `BehaviorMDSDataset` but with `clips_index` filtered to held-out episodes
- For each test episode:
  - `get_demo_id(episode_idx) -> demo_id`  → passed to `env.reset()`
  - `iter_subgoals(episode_idx, spacing) -> Iterator[(goal_tokens, target_env_step)]`
- Goal tokens are loaded directly from MDS (already encoded by the same ViT-G), then `cam_embed + layer_norm` applied to match the predictor's input convention. **No re-encoding needed.**

### 7. `metrics.py`

Each tracker is a small class with `.update()` and `.dump()`:

| Tracker | Captures |
|---|---|
| `TimingTracker` | cuda.Event-based per-phase (sample / rollout / score / update) per planner iteration |
| `SampleEfficiencyTracker` | min cost vs. iteration index — for the paper's convergence figure |
| `ActionDivergenceTracker` | `||a_t - a_{t-1}||_2` between consecutive MPC re-plans (MPC mode only) |
| `GoalDistanceTracker` | encoder-space distance at each subgoal arrival (logged + per-episode summary) |
| `EpisodeTracker` | success flag, length, time-to-subgoal |

### 8. `logging_utils.py`

- Per-step CSV via `CSVLogger`
- Per-episode HDF5 dump (`trajectory_dump_dir/episode_{i:04d}.h5`):
  - `obs/head`, `obs/left_wrist`, `obs/right_wrist` — uint8 RGB stacks
  - `states` — `[T, 133]`
  - `executed_actions` — `[T, 23]`
  - `planner/best_samples_per_iter` — for top-k visualization
  - `subgoal_arrivals` — step indices + goal distance

### 9. `run.py` — episode/MPC loop

```python
def run(cfg):
    wm = WorldModel(...)
    planner = build_planner(cfg.planning, world_model=wm)
    env = OmniGibsonEnv(...)
    subgoals = SubgoalSource(...)
    metrics = MetricsBundle(...)
    dumper = TrajectoryDumper(...)

    for ep_idx in range(cfg.eval.num_episodes):
        demo_id = subgoals.get_demo_id(ep_idx)
        obs = env.reset(demo_id=demo_id)
        start_tokens = wm.encode_observation(obs)
        prev_solution = None
        for goal_tokens, target_step in subgoals.iter_subgoals(ep_idx, cfg.subgoals.spacing):
            while env.step_count < target_step:
                plan, diag = planner.plan(
                    start_tokens, goal_tokens,
                    observed_state=torch.as_tensor(obs["state"]),
                    prev_solution=prev_solution,
                )
                if cfg.planning.mode == "mpc":
                    action_chunk = plan[0]            # execute first plan step (=6 env steps)
                    prev_solution = plan if cfg.planning.warm_start else None
                else:                                  # open_loop
                    action_chunk = plan                # execute all H plan steps
                obs_list = env.step_chunk(action_chunk if open_loop else plan[0])
                dumper.append(obs_list)
                obs = obs_list[-1]
                start_tokens = wm.encode_observation(obs)
                metrics.step_update(diag, plan, prev_solution)
                if cfg.planning.mode == "open_loop": break
            metrics.subgoal_arrival(start_tokens, goal_tokens)
        metrics.episode_end(env.is_success())
        dumper.flush_episode(ep_idx)
    metrics.dump(cfg.folder)
```

---

## Config Schema (`configs/eval/behavior_planning/cem-default.yaml`)

```yaml
app_kind: behavior_planning            # not a training app — own CLI
folder: /eval/behavior_planning/runs/cem_default

model:
  encoder_source: hub                  # hub | path
  encoder_ckpt_path: null              # only when encoder_source=path
  predictor_ckpt: /checkpoints/behavior-vitg16-256px-16f/latest.pt
  embed_dim: 1408
  patch_size: 16
  n_cameras: 3
  tpf_per_cam: 256
  cameras: [head, left_wrist, right_wrist]
  pred_depth: 24
  pred_embed_dim: 1024
  pred_num_heads: 16
  pred_is_frame_causal: true
  action_embed_dim: 138
  state_embed_dim: 133
  use_rope: true
  use_activation_checkpointing: false  # off for inference
  compile_predictor: true              # torch.compile predictor.forward
  dtype: bfloat16
  normalize_reps: true

env:
  demo_manifest: /path/to/behavior_demos_manifest.json   # USER TO FILL
  task: null                            # task-agnostic initially
  scene: null
  control_hz: 30
  predictor_hz: 5                       # = control_hz / fstp
  fstp: 6
  cameras: [head, left_wrist, right_wrist]

planning:
  planner: cem                          # cem | mppi
  mode: mpc                             # mpc | open_loop
  warm_start: true

  horizon: 3                            # plan steps; each = fstp env steps
  samples: 64
  iterations: 5
  action_space: native                  # native (138) | planner_native (6x23)
  state_strategy: hold_constant         # hold_constant | zero | env_query

  goal_metric: cosine                   # cosine | l1 | l2
  goal_aggregation: final_only          # final_only | discounted_sum
  discount: 0.95

  action_clip: { enabled: true, min: -1.0, max: 1.0 }
  noise_std_init: 0.5

  cem:
    elite_frac: 0.1
    momentum_mean: 0.1
    momentum_std: 0.1
    colored_noise_beta: 0.0
  mppi:
    temperature: 1.0

subgoals:
  source: holdout_mds
  remote: "hf://datasets/<org>/<repo>/<split>"
  local: "/tmp/behavior_mds_cache"
  holdout_seed: 0
  holdout_frac: 0.1                     # OR specify holdout_episodes: [...]
  spacing: 6                            # env steps between subgoals
  selection: sequential                 # sequential | random

logging:
  csv_path: log.csv
  trajectory_dump_dir: trajectories
  save_raw_obs: true
  raw_obs_subsample: 1
  timing_breakdown: true

eval:
  num_episodes: 20
  max_steps_per_episode: 600
  success_threshold_goal_dist: 0.1
  seed: 0
```

---

## What NOT to Touch

- `src/models/ac_predictor.py` — predictor loaded as-is; no architecture changes
- `app/vjepa_behavior/train.py`, `app/vjepa_behavior/utils.py` (besides importing `init_predictor` and `load_checkpoint`)
- `app/main.py`, `app/scaffold.py` — eval has its own CLI in `evals/behavior_planning/main.py`
- `notebooks/utils/mpc_utils.py`, `notebooks/utils/world_model_wrapper.py` — leave as DROID reference
- All `configs/train/` configs

## State Prediction at Planning Time — Cost & Risk Analysis

The plan defaults to `state_strategy=hold_constant` because the trained predictor **only outputs vision tokens** ([ac_predictor.py:208-209](src/models/ac_predictor.py#L208-L209)) and the training loss only supervises vision tokens ([train.py:413-414](app/vjepa_behavior/train.py#L413-L414)):

```python
jloss = loss_fn(z_tf, h.detach())   # vision tokens vs vision tokens
sloss = loss_fn(z_ar, h.detach())   # vision tokens vs vision tokens
```

States are input-only — they enter via `state_encoder`, condition the attention, and get stripped from the output. The existing checkpoint has **zero state-prediction signal trained in**. Any path to state prediction therefore requires BOTH:
- An architecture change (state-projection head)
- A NEW loss term (state-prediction loss against next-step ground truth)

Below is the change footprint and risk profile of each path, ranked by effort.

### Option 0 — Keep current plan default (`hold_constant`, no model change)
- **Predictor change:** none
- **Training change:** none
- **Effort:** 0
- **Risk:** Low. Predictor sees stale but plausible state every step. Likely fine for short horizons (2–4 plan steps) where state drift is small.

### Option 1 — Latent-state recurrence (use the hidden state-token output directly)
- **Idea:** Internally the predictor's state-token position *does* receive attention from prior frames. Take that hidden vector (`predictor_embed_dim=1024`) as the state input for the next step instead of re-encoding raw state.
- **Predictor change:** ~5 LoC. Skip the slice at line 208, expose the full sequence or just the state-token output. Add a flag `return_hidden_state=True`. Also need to bypass `state_encoder` for steps that get a latent input (it expects 133-dim raw, not 1024-dim hidden) — either skip the encoder for steps `>0` or add a passthrough mode.
- **Training change:** None initially — at inference you just feed back the hidden vector.
- **Effort:** < 1 day
- **Risk:** **Medium–high.** The state token's hidden output was never *trained* to serve as a state input. It was trained to be a useful context vector for vision-token prediction. Semantic mismatch — may degrade rollouts rather than help. Worth a quick ablation before committing.

### Option 2 — Add state head, fine-tune from existing checkpoint  *(recommended if state pred matters)*
- **Predictor change:** ~10 LoC
  - Add `self.state_proj = nn.Linear(predictor_embed_dim, state_embed_dim)` in `__init__`
  - Extract state-token output before the vision-only slice: `state_tok = x.view(B, T, cond_tokens + img_per_frame, D)[:, :, 1, :]`
  - Apply `predictor_norm` + `state_proj` → `[B, T, 133]`
  - Return `(vision_tokens, state_preds)` or as a dict
  - Update call sites in `train.py` (one site, easy)
- **Training change:** ~20 LoC in `forward_predictions` + `loss_fn`
  - Targets: `states[:, 1:]` (one-step-ahead, matching vision-token target alignment)
  - `state_loss = mse(state_pred, states[:, 1:].detach())`
  - Combined: `loss = jloss + sloss + λ_s * state_loss` (λ_s ≈ 0.1–1.0, tune)
  - Fine-tune existing checkpoint for **1–3 epochs** on the same MDS data
- **Effort:** 2–5 days (most of it is the FT run + tuning λ_s)
- **Risk:** **Low–medium.** Model already knows dynamics from joint training; adding a small head with weak loss weight rarely destabilizes. Backward compat: load old checkpoint with `strict=False`, state_proj initializes randomly, FT pulls it into a useful regime.
- **Compute cost:** Roughly 3–10% of original pre-training cost (a few epochs vs ~100). On the same hardware that did pre-training, expect ~1–3 days wall-clock.

### Option 3 — Add state head, train head only (freeze predictor)
- **Predictor change:** Same as Option 2
- **Training change:** Same head, but `optimizer` sees only `state_proj.parameters()`. All other modules `.requires_grad_(False)`.
- **Effort:** 1–2 days
- **Risk:** Low. Fast to converge, can't degrade existing predictor behavior.
- **Limitation:** The state head can only use the existing frozen state-token representation. If that representation doesn't encode enough about future state, predictions will be weak. **Quality ceiling lower than Option 2.**

### Option 4 — Retrain from scratch with state loss
- **Effort:** 1–3 weeks GPU-bound
- **Risk:** Discards current checkpoint progress. Not recommended unless other training changes are also planned.

### Recommendation
Build the planner with **Option 0** (zero-cost default) and ship the eval framework. **In parallel**, run **Option 3** (head-only fine-tune, 1–2 days) as a low-risk experiment — if the head learns useful predictions, the same architecture change supports Option 2 later. The planner's `state_strategy` enum gets a fourth value `predicted` that hits `world_model.predict_state(...)` when enabled.

### Concrete change list to land Option 2 or 3

| File | Change | LoC |
|---|---|---|
| `src/models/ac_predictor.py` | Add `state_proj`, modify `forward` to return tuple/dict, gate on `predict_states` flag | ~15 |
| `app/vjepa_behavior/utils.py::init_predictor` | Plumb `predict_states` config flag through | ~3 |
| `app/vjepa_behavior/train.py` | Unpack tuple from predictor, add `state_loss` term, log it | ~25 |
| `configs/train/behavior/behavior-vitg16-256px-16f.yaml` | Add `model.predict_states: true`, `loss.state_loss_weight: 0.1` | ~2 |
| `evals/behavior_planning/world_model.py` | Add `state_strategy: predicted` branch in `rollout` | ~10 |

Total: ~55 LoC across 5 files. All additive — old checkpoints still load via `strict=False`.

---

## Open Items (User Action)

These are stubs/configs that need user input before end-to-end runs:
1. **`env.demo_manifest` path + schema** — wrapper needs to know how to map `(episode_idx, sample_idx) → (HDF5 path, initial-state keys)`. User to provide.
2. **State extraction** — the script that built the MDS dataset performed the 256→133 slicing per DATASET_SCHEMA.md. We need to either re-use that slicing function or duplicate it inside `OmniGibsonEnv.get_state`. Find or get pointer to the original.
3. **OG task/scene** — eventual evals need a chosen task; wrapper will work task-agnostic until then.

---

## Open Discussion — State Representation for Training

Training has not started. The choice of state representation is still open and affects both the predictor architecture (`state_embed_dim`) and the analytical integration strategy at planning time.

### Context

DROID's state is 7-dim Cartesian EEF pose + gripper. Its actions are computed as `poses_to_diffs(states)` — pose deltas — so `state_{t+1} = state_t ⊕ action_t` (SE(3) composition). State is always exactly known from the action sequence; no environment queries needed. This is why DROID never had a state prediction problem.

BEHAVIOR's current state is 133-dim proprioceptive (joint angles stored three ways: raw + sin + cos, plus joint velocities, EEF poses, IMU). Actions are 23-dim joint velocity commands, not Cartesian deltas — so DROID's direct trick doesn't apply. However, the MDS dataset already stores `cam_rel_poses` (21-dim: 3 cameras × pos(3)+quat(4) relative to robot base), which is effectively the EEF Cartesian state in camera space.

At planning time, the inner rollout loop (64 samples × 5 steps) must propagate state forward without environment queries. The chosen representation determines whether this is exact or approximate.

### Four Options

| Option | State dim | Training change | Planning integration | Velocity signal |
|---|---|---|---|---|
| A — 133-dim as-is | 133 | none | `q_t = q_0 + Σ a*dt`, recompute all 133 dims via FK + trig | from joint_qvel in state |
| B — cam_rel_poses + gripper | 23 | swap state source | `cam_t = FK(q_t)`, gripper from action cumsum | none |
| C — DROID-style Cartesian delta | 14 (2×7 EEF) | redefine action space | `state_{t+1} = state_t ⊕ action_t` | none |
| D — cam_rel_poses + gripper + EEF velocity | 29 | swap state source | `cam_t = FK(q_t)`, vel = `J(q_t) * a_t` | yes, via Jacobian |

**Option C is ruled out**: BEHAVIOR is bimanual with trunk and base. A single Cartesian pose delta is ambiguous — converting 23 joint-velocity DOFs to Cartesian loses the coupling and doesn't generalise. DROID's trick works because it is a single 7-DOF arm.

**Preferred direction: Option D** (or B as a simpler fallback).

Rationale: the predictor's task is to predict visual tokens. The most relevant conditioning is (1) where the cameras are in 3D space — `cam_rel_poses` — which directly determines the geometric structure of the tokens, and (2) how fast they are moving — EEF velocity via Jacobian. Everything else in the 133-dim vector is either redundant (qpos represented three times) or tangential to vision.

At planning time this reduces to pure math — no simulator, no environment query, no learned state head, exact for free-space motion:

```python
q_t = q_0 + cumsum(actions[:, :t, :22], dim=1) * dt   # [B, 22]
cam_t    = fk(q_t)                                      # [B, 21]
eef_vel  = jacobian(q_t) @ actions[:, t, :22]           # [B, 6]  (Option D only)
gripper  = gripper_0 + cumsum(actions[:, :t, 20:22]) * dt
state_t  = cat([cam_t, gripper, eef_vel], dim=-1)       # [B, 23 or 29]
```

For real-world R1 Pro deployment, `q_0` comes directly from joint encoders — nothing else changes.

### Decision needed before training starts

- Which option to use (A / B / D)?
- If B or D: confirm that `cam_rel_poses` in the MDS is computed from the same FK as will be used at planning time (i.e. same URDF / coordinate convention).
- If D: decide whether to compute the Jacobian offline (during MDS dataset build) or online at training time.

---

## Verification

### Unit tests
- `tests/eval/behavior_planning/test_world_model.py`
  - Pre-encode 3 cameras from a sample image via `vjepa2_vit_giant`; compare against the same image's MDS tokens. Should match within fp16 tolerance (~1e-3 cosine sim ≥ 0.999).
  - `WorldModel.rollout` with `horizon=3, B=8` returns shape `[8, 3, n_cameras*tpf, D]` and gradients are disabled.
- `tests/eval/behavior_planning/test_planner.py`
  - Toy 1D world model: `next = current + 0.5 * action`, goal = `[1.0]`. Both CEM and MPPI converge to action `≈ 2.0` within 10 iterations, samples=32.
  - Warm-start: 2nd `.plan()` call converges in ≤2 iterations from a near-optimum starting distribution.
- `tests/eval/behavior_planning/test_subgoal_source.py`
  - Holdout filter excludes train episodes deterministically given seed.
  - Subgoal iterator yields tokens at correct `step_pos` values.

### Smoke run (no OG required)
```bash
python -m evals.behavior_planning.main \
  --fname configs/eval/behavior_planning/cem-default.yaml \
  --dry-run
```
Replaces `OmniGibsonEnv` with a `FakeEnv` that returns repeated MDS frames; verifies the full plan→execute→encode→re-plan loop, CSV + HDF5 dumps, and metric summaries.

### End-to-end (requires OG + manifest)
```bash
python -m evals.behavior_planning.main \
  --fname configs/eval/behavior_planning/cem-default.yaml \
  --devices cuda:0
```
With `eval.num_episodes=1, planning.samples=8, planning.iterations=2, planning.horizon=2`.
Verify: env resets to demo, raw obs HDF5 contains correct shapes, CSV log records timing breakdown, no errors in 1 episode.

### Performance gate
On a single H100 (or A100), time one `planner.plan()` call with:
- `samples=50, iterations=5, horizon=3, action_space=native`
- predictor compiled, encoder cached

Target: **< 500 ms/step on H100, < 1.5 s/step on A100**. Log full timing breakdown (sample / rollout / score / update). If targets miss, the next levers are: (1) reduce horizon to 2, (2) flash-attn 3 on H100, (3) drop samples to 32 with MPPI (sample-efficient).
