# Position Pipelines

## Overview

Position pipelines in Spyglass turn raw video into time-stamped behavioral
variables — centroid (x, y), head orientation (radians), and velocity (cm/s) —
that can be aligned with neural recordings for downstream decoding or
linearization.

### What a position pipeline does

1. **Pose estimation** — A trained keypoint model (DeepLabCut, SLEAP, or any
    ndx-pose–compatible tool) runs on each video frame and outputs per-bodypart
    (x, y, likelihood) time series.
2. **Likelihood filtering** — Frames where the model's confidence is below a
    threshold are treated as missing data and interpolated over.
3. **Bodypart smoothing** — Each bodypart trajectory is independently
    interpolated across short gaps and smoothed with a moving average, removing
    jitter before orientation or centroid are calculated.
4. **Orientation** — A chosen pair (or triplet) of bodyparts defines the
    animal's head direction at each frame.
5. **Centroid** — One or more bodyparts are combined into a single (x, y)
    position estimate for the animal.
6. **Centroid smoothing** — The centroid is interpolated and smoothed a second
    time, giving a clean positional trajectory.
7. **Velocity** — 2D velocity and scalar speed are computed from the smoothed
    centroid using `np.gradient` (central differences) and optionally smoothed
    with a Gaussian kernel.

### Two pipelines: V1 and V2

|                    | V1                      | V2                          |
| ------------------ | ----------------------- | --------------------------- |
| **Status**         | Stable, production      | Active development          |
| **Tables**         | ~15 separate tables     | 3 core tables               |
| **Tools**          | DLC only                | DLC, SLEAP, ndx-pose import |
| **Backend**        | TensorFlow (DLC 2.x)    | PyTorch (DLC 3.x, SLEAP)    |
| **Storage**        | Custom NWB objects      | ndx-pose extension          |
| **Parameters**     | 3 separate param tables | 1 unified `PoseParams`      |
| **Shared science** | `position/utils/`       | same                        |

Both pipelines share the same underlying mathematical functions
(`position/utils/`). V2 consolidates V1's many intermediate tables into a
cleaner three-step flow: `PoseEstim → PoseV2 → PositionOutput`.

### Why V2 uses the PyTorch backend

DeepLabCut 3.x defaults to the **PyTorch** engine
(`deeplabcut.compat.DEFAULT_ENGINE = Engine.PYTORCH`), and Position V2
standardizes on it. V2's code is **engine-agnostic** — DLC 3.x supports both the
PyTorch and TensorFlow engines, and V2 dispatches to whichever a model was
trained with — so V2 does not strictly *need* PyTorch. But PyTorch is the
recommended and better-supported path, chiefly for **dependency coexistence**.

The TensorFlow backend forces tight, conflicting version pins that collide with
the rest of the modern scientific-Python stack and with the GPU runtimes
Spyglass already loads (the rest of Spyglass pulls in `jax` via
`non_local_detector`):

- **XLA / cuDNN collision.** TensorFlow and `jaxlib` each bundle their own
    XLA/CUDA and both try to register cuDNN/cuFFT/cuBLAS on the GPU
    (`Unable to register cuDNN factory ... already registered`).
- **numpy / jax version wedge.** DeepLabCut 3.x pins `numpy<2`; within that,
    TensorFlow forces an old `jax`, and the newer TensorFlow/`jax` releases that
    would line up each require `numpy 2`. No single set of versions satisfies
    all three.
- **Keras / tf-keras wedge.** TensorFlow 2.16+ ships Keras 3, which DeepLabCut's
    TF compat layer cannot load without `tf-keras` — the loader recurses
    infinitely (a `RecursionError` at import) unless you also set
    `TF_USE_LEGACY_KERAS=1` and install `tf-keras`.

The PyTorch engine has none of these problems: it coexists with modern `numpy`
and `jax`, keeps the whole Spyglass stack in one working environment, and is
DeepLabCut's actively-developed default. It is also the better-supported path
inside Spyglass — continued training resumes cleanly from a parent snapshot on
PyTorch (`train_network(snapshot_path=…)`), whereas the TensorFlow weight-resume
mechanism (`init_weights` in `pose_cfg`) is not exposed as a training kwarg.

If you are locked to a TensorFlow-trained DLC model, run that inference in a
**separate** environment (no `jax` / `non_local_detector`) and ingest the
resulting `.h5`/NWB via `PoseEstimSelection` with `task_mode="load"` or via
`ImportedPose`. See
[Troubleshooting → TensorFlow / jax conflict](./TROUBLESHOOTING.md) for the
migration fix.

### Training parameters: TensorFlow → PyTorch

The two DLC engines **count training length in different units**, and this is
the single most common source of confusion when moving a V1 (TF) workflow to V2
(PyTorch). They are *not* interchangeable:

|                  | TensorFlow (DLC 2.x)                | PyTorch (DLC 3.x)                                    |
| ---------------- | ----------------------------------- | ---------------------------------------------------- |
| Length unit      | **iteration** (one batch)           | **epoch** (one full pass over the train set)         |
| Length knob      | `maxiters`                          | `epochs`                                             |
| Snapshot cadence | `saveiters`                         | `save_epochs`                                        |
| Console cadence  | `displayiters`                      | `display_iters`                                      |
| Default length   | ~1,030,000 iterations               | 200 epochs                                           |
| Snapshot file    | `snapshot-<iter>.{index,meta,data}` | `snapshot-<epoch>.pt` (+ `snapshot-best-<epoch>.pt`) |

One epoch bundles `ceil(n_train_images / batch_size)` gradient steps, so the raw
counters are **not** comparable: on a ~100-frame training set with
`batch_size: 8`, epoch 200 is only ~2,400 gradient steps — nothing like 200 TF
iterations, and nothing like the ~1e6 iterations of a full TF run. PyTorch
generally reaches equal accuracy in **far fewer total updates**, which is why
its default is 200 epochs rather than ~1M iterations.

> **Migration pitfall.** Do not copy a TensorFlow `maxiters` value into the
> PyTorch `epochs` field. A carried-over `epochs: 1030000` (the old TF default)
> means ~1,000,000 passes over the data — a run that never sensibly terminates
> and trains long past its best checkpoint. For Frank Lab–sized training sets,
> ~200 epochs is the right scale.

**How V2 handles this for you.** `Model.train(key, epochs=N)` is
engine-agnostic: `DLCStrategy.apply_epochs` resolves the project's engine and
writes the correct native knob — `epochs` for PyTorch (the DLC 3.x default),
`maxiters` for an explicit `engine: tensorflow` model. The `dlc_default`
`ModelParams` set ships an explicit PyTorch budget (`epochs: 200`,
`save_epochs: 25`), matching DLC's own defaults; pass `epochs=` (or bake a
different value into `ModelParams`) to override. Declaring `epochs` in the tens
of thousands raises a warning — that is the size of a TensorFlow `maxiters`
value, not a PyTorch epoch count.

**How to read training progress.** For a PyTorch model, count epochs: the
highest `snapshot-<epoch>.pt` (and `snapshot-best-<epoch>.pt`) in the model's
`train/` directory, and the `step` column of `learning_stats.csv` (which is the
epoch number). An empty `dlc-models-pytorch/` **and** an empty
`training-datasets/.../UnaugmentedDataSet.../` means training never started (the
training dataset was never created).

______________________________________________________________________

## Table Reference

### V1 → V2 table mapping

| V1 table                         | V2 table              | Notes                                                |
| -------------------------------- | --------------------- | ---------------------------------------------------- |
| `BodyPart`                       | `BodyPart`            | Reclassed `Manual` → `Lookup`                        |
| `DLCProject`                     | `Skeleton`            | Body-part set → explicit skeleton graph              |
| —                                | `Skeleton.BodyPart`   | Part table; no V1 equivalent                         |
| `DLCModelTrainingParams`         | `ModelParams`         |                                                      |
| `DLCModelTrainingSelection`      | `ModelSelection`      |                                                      |
| `DLCModelTraining`               | `Model`               |                                                      |
| `DLCModelInput`                  | `ModelParams`         | Merged into params                                   |
| `DLCModelSource`                 | `ModelParams`         | Merged into params                                   |
| `DLCModelParams`                 | `ModelParams`         |                                                      |
| `DLCModelSelection`              | `ModelSelection`      |                                                      |
| `DLCModel`                       | `Model`               |                                                      |
| `DLCModelEvaluation`             | `Model`               |                                                      |
| `DLCPoseEstimationSelection`     | `PoseEstimSelection`  |                                                      |
| `DLCPoseEstimation`              | `PoseEstim`           |                                                      |
| —                                | `PoseEstimParams`     | New; separates inference params (device, batch size) |
| `DLCSmoothInterpParams`          | `PoseParams`          | Consolidated into `smoothing` sub-dict               |
| `DLCCentroidParams`              | `PoseParams`          | Consolidated into `centroid` sub-dict                |
| `DLCOrientationParams`           | `PoseParams`          | Consolidated into `orient` sub-dict                  |
| `DLCSmoothInterpSelection`       | `PoseSelection`       |                                                      |
| `DLCCentroidSelection`           | `PoseSelection`       |                                                      |
| `DLCOrientationSelection`        | `PoseSelection`       |                                                      |
| `DLCSmoothInterpCohortSelection` | `PoseSelection`       | Cohort concept eliminated                            |
| `DLCSmoothInterpCohort`          | `PoseV2`              | Cohort concept eliminated                            |
| `DLCSmoothInterp`                | `PoseV2`              |                                                      |
| `DLCCentroid`                    | `PoseV2`              |                                                      |
| `DLCOrientation`                 | `PoseV2`              |                                                      |
| `DLCPosSelection`                | `PoseSelection`       |                                                      |
| `DLCPosV1`                       | `PoseV2`              |                                                      |
| `DLCPosVideoParams`              | `VidFileGroup`        |                                                      |
| `DLCPosVideoSelection`           | `VidFileGroup`        |                                                      |
| `DLCPosVideo`                    | `PoseV2.make_video()` | No longer stored as a table                          |
| `TrodesPosParams`                | —                     | No V2 equivalent                                     |
| `TrodesPosSelection`             | —                     | No V2 equivalent                                     |
| `TrodesPosV1`                    | —                     | No V2 equivalent                                     |
| `TrodesPosVideo`                 | —                     | No V2 equivalent                                     |
| `ImportedPose`                   | `ImportedPose`        | Unchanged; ingests external ndx-pose NWB files       |

### Key consolidations in V2

- **`DLCCentroidParams` + `DLCOrientationParams` + `DLCSmoothInterpParams`** →
    single `PoseParams` with three sub-dicts (`centroid`, `orient`,
    `smoothing`).
- **`DLCModelInput` + `DLCModelSource` + `DLCModelParams`** → single
    `ModelParams`.
- **Cohort pattern** (`DLCSmoothInterpCohort*`) eliminated — `PoseV2` handles
    multi-bodypart poses directly. Choosing *which* bodyparts feed
    centroid/orientation carries over to `PoseParams`; the cohort's ability to
    apply *different* smoothing parameters to different bodyparts within one run
    was dropped — a review of production V1 usage found it was never used in
    practice (a single parameter set was always applied uniformly across a
    cohort's bodyparts).
- **Trodes tables** have no V2 equivalent because V2 focuses on video-based pose
    estimation rather than hardware position sensors.

### V2 pipeline diagram

```
VidFileGroup ──► ModelSelection ──► Model
                                      │
                               PoseEstimSelection
                                      │
                               PoseEstim   ◄── PoseEstimParams
                                      │
                               PoseSelection ◄── PoseParams
                                      │
                               PoseV2
                                      │
                               PositionOutput (merge)
```

### PoseParams sub-dicts

`PoseParams` stores three nested dicts in a single JSON blob:

```python
{
    "orient": {
        "method": "two_pt",  # "two_pt" | "bisector" | "none"
        "bodypart1": "greenLED",
        "bodypart2": "redLED",
        "smooth": True,  # Gaussian smooth orientation
        "interpolate": True,
        "smoothing_params": {"std_dev": 0.001},
    },
    "centroid": {
        "method": "2pt",  # "1pt" | "2pt" | "4pt"
        "points": {"point1": "greenLED", "point2": "redLED"},
        "max_LED_separation": 12.0,  # cm; frames exceeding this become NaN
    },
    "smoothing": {
        "likelihood_thresh": 0.95,
        "interpolate": True,
        "interp_params": {"max_pts_to_interp": 10, "max_cm_to_interp": 15.0},
        "smooth": True,
        "smoothing_params": {"method": "moving_avg", "smoothing_duration": 0.05},
        "velocity_smoothing_std_dev": 0.1,  # seconds; None to disable
    },
}
```

Built-in presets:

```python
PoseParams.insert_default()  # 2-LED (greenLED + redLED_C)
PoseParams.insert_4LED_default()  # 4-LED (greenLED + redLED_C/L/R)
```

______________________________________________________________________

## Intentional Differences from V1

The following behaviors differ from V1 by design. They represent scientific
corrections or API improvements rather than missing functionality.

### 1. Velocity smoothed in 2D before speed is derived

**V1 behavior**: `DLCCentroid` computes scalar speed first (`|Δpos| / Δt`), then
Gaussian-smooths that 1D speed signal.

**V2 behavior**: `compute_velocity` (shared by both pipelines via
`position/utils/velocity.py`) computes 2D velocity with `np.gradient`,
optionally Gaussian-smooths the (vx, vy) vector, then derives scalar speed as
`√(vx² + vy²)`.

**Why**: Smoothing a scalar speed signal is asymmetric — it can lower peaks but
cannot raise them back. Smoothing the 2D velocity vector before collapsing to
speed preserves direction information and produces unbiased speed estimates.
This was confirmed in T06 where V1's approach gave a `velocity corr = 0.912` vs.
V2's corrected `corr > 0.999`.

**Migration**: Use `smoothing.velocity_smoothing_std_dev` (seconds) in
`PoseParams` instead of V1's `speed_smoothing_std_dev`.

### 2. Orientation smoothing is opt-in

**V1 behavior**: `DLCOrientation.make()` always Gaussian-smooths the orientation
signal (unwrap → interpolate → Gaussian → wrap) whenever
`orient_method != "none"`, regardless of parameters.

**V2 behavior**: Orientation smoothing only runs when
`orient_params["smooth"] == True`. The default `PoseParams` preset sets
`smooth: True` with `std_dev: 0.001` (1 ms), matching V1 behavior.

**Why**: Making smoothing explicit gives users the ability to skip it for
workflows (e.g., MoSeq) that need raw orientation. Default params preserve
backward compatibility.

### 3. Orientation method names are descriptive

**V1 names**: `"red_green_orientation"`, `"red_led_bisector"`, `"none"`

**V2 names**: `"two_pt"`, `"bisector"`, `"none"`

Both call the same underlying functions from `position/utils/orientation.py`.
The V2 names are tool-agnostic (not LED-specific), supporting use with any
two-keypoint skeleton.

### 4. Three V1 param tables collapsed into one

V1 requires separate `DLCSmoothInterpParams`, `DLCCentroidParams`, and
`DLCOrientationParams` entries, each populated independently. V2 stores all
three as sub-dicts in a single `PoseParams` row, queried by dot notation:

```python
PoseParams & {"orient.method": "two_pt"}
PoseParams & {"smoothing.likelihood_thresh": 0.95}
```

### 5. V1 velocity uses `np.diff`; V2 uses `np.gradient`

**V1 behavior**: `calculate_velocity` in `dlc_utils.py` prepends NaN then uses
`np.diff` (forward differences), producing `n − 1` velocity values that are
padded back to `n` with a leading NaN. This systematically underestimates speed
at the last frame and loses information at boundaries.

**V2 behavior**: `compute_velocity` uses `np.gradient` (central differences at
interior points, one-sided differences at boundaries), returning `n` values with
no boundary artifacts.

### 6. Compute device (GPU/CPU) is a runtime choice, not a parameter

**V1 behavior**: GPU selection (`gputouse`) lived in the DLC parameter tables,
so the device a run happened to use was baked into the stored, hashed
parameters.

**V2 behavior**: V2 does not treat the compute device as a pipeline parameter —
the pose tool selects the best available device at run time (GPU if present,
otherwise CPU), so the tutorials set no `device`/`gputouse` field. Device
selection is a runtime/environment concern, not part of a run's scientific
provenance: the same model and parameters yield the same result on GPU or CPU,
so pinning a device into the content-addressed parameters would only fork
otherwise-identical parameter sets. To force a specific GPU, use the environment
(e.g. `CUDA_VISIBLE_DEVICES`) rather than a pipeline parameter.
