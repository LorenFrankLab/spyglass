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

|                    | V1                      | V2                                    |
| ------------------ | ----------------------- | ------------------------------------- |
| **Status**         | Stable, production      | Active development                    |
| **Tables**         | ~15 separate tables     | 3 core tables                         |
| **Tools**          | DLC only                | DLC, SLEAP (planned), ndx-pose import |
| **Backend**        | TensorFlow (DLC 2.x)    | PyTorch (DLC 3.x, SLEAP)              |
| **Storage**        | Custom NWB objects      | ndx-pose extension                    |
| **Parameters**     | 3 separate param tables | 1 unified `PoseParams`                |
| **Shared science** | `position/utils/`       | same                                  |

Both pipelines share the same underlying mathematical functions
(`position/utils/`). V2 consolidates V1's many intermediate tables into a
cleaner three-step flow: `PoseEstim → PoseV2 → PositionOutput`.

### Why V2 uses the PyTorch backend

V2's code is **engine-agnostic** — DeepLabCut 3.x supports both PyTorch and
TensorFlow engines, and V2 dispatches to whichever a model was trained with. So
V2 does not *need* PyTorch in principle.

In practice PyTorch is required for **dependency coexistence**, not by the pose
code. The rest of Spyglass pulls in `jax` (via `non_local_detector`), and
TensorFlow cannot cleanly share one environment with `jax`:

- **XLA collision.** TensorFlow and `jaxlib` each bundle their own XLA/CUDA and
    both try to register cuDNN/cuFFT/cuBLAS on the GPU
    (`Unable to register cuDNN factory ... already registered`).
- **Version wedge.** DeepLabCut 3.x pins `numpy<2`; within that, TensorFlow
    forces an old `jax`, and the newer TensorFlow/`jax` releases that would line
    up each require `numpy 2`. No single set of versions satisfies all three.

PyTorch has neither problem, so it keeps the whole Spyglass stack in one working
environment. If you are locked to a TensorFlow-trained DLC model, run that
inference in a **separate** environment (no `jax` / `non_local_detector`) and
ingest the resulting `.h5`/NWB via `PoseEstimSelection` with `task_mode="load"`
or via `ImportedPose`. See
[Troubleshooting → TensorFlow / jax conflict](./TROUBLESHOOTING.md) for the
migration fix.

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
    multi-bodypart poses directly.
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
