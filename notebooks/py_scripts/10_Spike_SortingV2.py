# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (spyglass_spikesorting_v2)
#     language: python
#     name: python3
# ---

# # Spike Sorting v2: single-session walkthrough
#
# This notebook runs the modern (`spyglass.spikesorting.v2`) spike-sorting
# pipeline end-to-end on **one already-ingested session**, using the high-level
# `run_v2_pipeline` orchestrator:
#
# > defaults → sort group → **preflight** → sort → auto-label → review (and
# > reopen) → **select units** → analyze
#
# It assumes you have already configured your DataJoint connection (see
# [Setup](./00_Setup.ipynb)) and ingested a session with `insert_sessions` (see
# [Insert Data](./02_Insert_Data.ipynb)).
#
# This is the **first-sort path**. For the deeper how-tos, see:
#
# - [Curation](./10_Spike_SortingV2_Curation.ipynb): browser (FigPack) and
#   step-by-step evaluate → merge → re-evaluate curation.
# - [Presets](./10_Spike_SortingV2_Presets.ipynb): customize a preset
#   (`clone_pipeline_preset` / `register_pipeline_preset`) and sort a whole
#   session at once.
# - [Cross-session](./10_Spike_SortingV2_CrossSession.ipynb): concatenate
#   same-day recordings and track units across days.
# - [Tables](./10_Spike_SortingV2_Tables.ipynb): if you want to see each table,
#   the same stages run one table at a time.

# +
import datajoint as dj
import pandas as pd
from IPython.display import display

from spyglass.common import Electrode, LabTeam
from spyglass.common.common_interval import IntervalList
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.pipeline import (
    describe_pipeline_presets,
    describe_run,
    describe_sort_groups,
    describe_units,
    plot_sort_group_geometry,
    preflight_v2_pipeline,
    run_v2_pipeline,
)
from spyglass.spikesorting.v2.recording import SortGroupV2
from spyglass.spikesorting.v2.sorting import Sorting  # noqa: F401

dj.config["display.limit"] = 12  # cap rows in table reprs
# -

# ## 1. Choose your session
#
# Point the notebook at the session you ingested with `insert_sessions`. A
# full-session sort uses the `"raw data valid times"` interval and the default
# native MountainSort4 hippocampus production preset
# (`franklab_probe_hippocampus_30khz_ms4_2026_06`). MS4 is included in the v2
# extra and runs under the `numpy>=2` baseline. A container execution option is
# available through `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`
# with Singularity/Apptainer.
# Change `pipeline_preset` to any name from `describe_pipeline_presets()` below
# (e.g. a cortex or 20 kHz preset).
#
# List the valid `interval_list_name` values for this session with
# `IntervalList & {"nwb_file_name": nwb_file_name}`. Leave `sort_group_id =
# None` to auto-pick when the session has exactly one sort group; with more than
# one, set it explicitly after reviewing the table and geometry plot in step 2
# (don't just take the first row).

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_probe_hippocampus_30khz_ms4_2026_06"
references = None  # inherit stored references; review before creating groups
# Sort group (shank) to sort. None auto-picks only when the session has exactly
# one sort group; otherwise set it deliberately after reviewing step 2.
sort_group_id = None
# Optional, experimental drift QC and motion correction (section 10).
run_motion_example = False
# -

IntervalList & {"nwb_file_name": nwb_file_name}

# ## 2. One-time setup
#
# Install default parameters and the owning team, review channel quality
# and references, then create and inspect the sort groups.

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "spike sorting"},
    skip_duplicates=True,
)

# ### Review channels and references before creating groups
#
# Inspect the existing bad-channel flags and acquisition reference metadata.
# Finalize any bad-channel edits BEFORE creating groups: grouping omits
# flagged channels, and later flags do not change existing membership.
# If groups already exist, inspect `SortGroupV2.preview_existing_entries(
# nwb_file_name)` before explicitly recreating them; this notebook reuses them.

electrode_config = pd.DataFrame(
    (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
        "electrode_group_name",
        "electrode_id",
        "bad_channel",
        "original_reference_electrode",
        as_dict=True,
    )
)
electrode_config

# Automated detection is optional. Its coherence/PSD thresholds are derived
# from Neuropixels; inspect suggestions for polymer probes and do not rely
# on a clean result for small tetrode groups. To propose changes:
#
# ```python
# from spyglass.spikesorting.v2.bad_channels import suggest_bad_channels
# reviewed_report = suggest_bad_channels(nwb_file_name, persist=False)
# reviewed_report
# ```
#
# After inspecting the report, persist exactly those suggestions separately:
#
# ```python
# suggest_bad_channels(
#     nwb_file_name, persist=True, reviewed_report=reviewed_report
# )
# ```
#
# Choose the sorting reference before grouping. `references=None` inherits
# each group's stored reference (`-1`/None: none, `-2`: global median,
# nonnegative: that electrode). To override it, set `references` to a mapping
# from every included `electrode_group_name` to its reference electrode ID
# or sentinel. The acquisition reference is not automatically a suitable
# sorting reference; inspect the resulting group table and geometry below.

if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, references=references
    )
sort_groups = describe_sort_groups(nwb_file_name)
if sort_groups.empty:
    raise ValueError(f"No SortGroupV2 rows found for {nwb_file_name!r}.")
available_sort_group_ids = [int(g) for g in sort_groups["sort_group_id"]]
plot_sort_group_geometry(nwb_file_name)
sort_groups


# Validate the chosen sort group only after the table and geometry are visible.
if sort_group_id is None:
    if len(available_sort_group_ids) == 1:
        sort_group_id = available_sort_group_ids[0]
    else:
        raise ValueError(
            f"{nwb_file_name!r} has multiple sort groups "
            f"{available_sort_group_ids}; set sort_group_id explicitly after "
            "reviewing the table and geometry plot above — don't default to "
            "the first shank."
        )
elif sort_group_id not in available_sort_group_ids:
    raise ValueError(
        f"sort_group_id={sort_group_id} is not one of "
        f"{available_sort_group_ids} for {nwb_file_name!r}."
    )
sort_group_id


# ## 3. Pick a Pipeline Preset
#
# `describe_pipeline_presets()` returns a table of what each shipping pipeline
# preset does: the sorter, the parameter rows each stage uses, the intended use,
# and the units of the detection threshold.

# Native MountainSort4 is the default hippocampus production recipe.
# For hippocampal polymer probes:
#
# | Choice | Recommendation | Runtime requirement |
# | --- | --- | --- |
# | MountainSort4, local | Lab production recipe; notebook default | Standard v2 environment |
# | MountainSort4, container | Same production recipe on a modern host | Singularity/Apptainer and the catalog's container image |
# | MountainSort5 | Alternative | Standard v2 environment |
#
# Match the catalog's region, sampling rate and probe metadata to your
# recording. `production` means lab-recommended, `alternative` is a supported
# substitute, and `experimental` needs scientific validation for your use.
# Inspect `notes` for the exact execution requirements; selecting a preset
# never silently switches its sorter or backend.
#

describe_pipeline_presets()

# ### What the Frank Lab presets assume
#
# A few scientific choices are intentionally behind dated preset names, so users
# do not have to hand-enter sorter kwargs:
#
# - Sort one shank, tetrode, or sort group at a time. `run_v2_pipeline_session`
#   loops this single-group path across a session.
# - Reference choice lives on `SortGroupV2`, not on the sorter row. The
#   acquisition ground is separate metadata; the spike-sorting reference should
#   be a quiet channel when available. `set_group_by_shank` inherits
#   `Electrode.original_reference_electrode` (`-1`/None -> no reference, `-2`
#   -> global median, `>=0` -> specific electrode), or you can pass
#   `references={...}` / `reference_mode="specific"` explicitly. For
#   hippocampal polymer probes, prefer a quiet anatomical reference when you
#   have one and inspect common-median choices carefully; `plot_sort_group_geometry`
#   marks a specific reference with a star.
# - The Frank Lab MountainSort rows use downward-only `detect_sign=-1`, matching
#   the v0/v1 defaults. If your geometry requires bidirectional detection, use
#   `clone_pipeline_preset` and set the sorter override `detect_sign` to `0`.
# - Hippocampus rows high-pass at 600 Hz; cortex rows at 300 Hz. Recordings are
#   already filtered when MountainSort runs (`filter=False`). MS4 rows whiten,
#   use adjacency radius 100 um, and tune clip/detect intervals to 30 vs 20 kHz.
# - The analyzer keeps two waveform views: unwhitened display waveforms for real
#   shapes/amplitudes, and a whitened metric analyzer for PC/NN cluster metrics.
#   Hippocampal display/metric rows intentionally use 0.5/0.5 ms windows and up
#   to 20000 spikes per unit.

# To **adapt** a preset (tune one knob with `clone_pipeline_preset`, or build a
# custom one with `register_pipeline_preset`) without hand-editing parameter
# rows, see the [Presets how-to](./10_Spike_SortingV2_Presets.ipynb).

# ## 4. Preflight: a fast, fail-early check
#
# `preflight_v2_pipeline` verifies in ~1 s (inserting nothing, never calling
# `populate`) that every prerequisite is in place: the session / interval /
# team / sort-group rows, the pipeline preset's parameter rows, and the sorter binary.
# It returns a `PreflightReport` that is truthy when the configuration is
# runnable; `report.errors` lists each blocking problem with the exact fix, and
# `report.expected_ids` shows the selection PKs the run will produce. (Skip it
# with `run_v2_pipeline(..., preflight=False)`.)
# The summary includes the effective backend/parameters, worker/chunk settings,
# scratch/cache notes, and stages to compute or reuse. Resource notes describe
# known allocations, not a prediction of total memory or runtime.

report = preflight_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)
print(report.summary())

# ## 5. Run the pipeline
#
# `run_v2_pipeline` chains recording → artifact detection → sort → curation into one
# idempotent call and registers the result on the `SpikeSortingOutput` merge
# table. With `preflight=True` (the default) it re-runs the check above before
# any populate. **Re-running with the same inputs is safe**: it finds the
# existing rows and returns the same run summary (same `root_merge_id`) without
# inserting duplicates. After a failure, rerun the same inputs to reuse completed
# stages; a failed sorter restarts its stage, not an internal checkpoint.
# If you want to see each table this call fills, the
# [Tables](./10_Spike_SortingV2_Tables.ipynb) notebook runs them one at a time.

run_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)

# ## 6. Read the run summary
#
# `describe_run(run_summary)` renders the run as a receipt with:
#
# - a summary row: `n_units`, `root_merge_id` (the uncurated root),
#   `auto_labeled_merge_id` (the auto-labeled child, `None` until you curate),
#   and a `"root only"` / `"auto-labeled"` status;
# - one row per stage: `*_status` is `"computed"` (ran this call), `"reused"`
#   (row already existed) or `"skipped"` (the preset has no such stage), and
#   `seconds` is wall-clock time for **this call** (≈0 on a re-run);
# - one row per advisory `warning`;
# - one `config` row per effective sorter setting (the kwargs SpikeInterface
#   received, whiten routing, seed, job kwargs, backend).
#
# A **zero-unit** sort is a legitimate quiet-shank result (it still writes an
# empty curation and merge row) and appears as a warning row. Pass
# `require_units=True` to `run_v2_pipeline` to make zero units an error.
#
# Below the receipt, `describe_units(...)` shows the per-unit, sort-time detail:
# one row per unit with `n_spikes`, `firing_rate_hz` (over the observed
# duration, which excludes masked artifact segments), `peak_amplitude_uv`,
# `peak_electrode_id`, and `brain_region`. It reads only sort-time metadata (no
# waveform recompute); deeper SNR / ISI / nearest-neighbour metrics are computed
# by the curation-evaluation step in section 7 below (`CurationEvaluation`). The raw
# `run_summary` dict carries the same fields programmatically
# (`run_summary["root_merge_id"]`, `run_summary["auto_labeled_merge_id"]`,
# `run_summary["n_units"]`).

display(describe_run(run_summary))
describe_units(run_summary["sorting_id"])

# ## 7. Curate the sort
#
# `run_v2_pipeline` leaves you a root (uncurated) curation.
# `CurationV2.summarize_curation` describes **one** curation and returns a plain
# dict (`n_units`, `labels`, `merge_groups`, `merge_id`, ...); build its key from
# the summary's `root_curation_id` (a run summary has no bare `curation_id`,
# since one run can yield a root and an auto-labeled child).
#
# The quickest path is **one-call auto-curation** below. For browser (FigPack)
# curation and the **step-by-step** evaluate → merge → re-evaluate loop, see the
# [Curation how-to](./10_Spike_SortingV2_Curation.ipynb).

root_key = {
    "sorting_id": run_summary["sorting_id"],
    "curation_id": run_summary["root_curation_id"],
}
CurationV2.summarize_curation(root_key)

# ### 7-auto. Automated labeling in one call
#
# When you trust the rule set for a batch, `auto_curate=True` folds the
# evaluate-and-accept steps into the `run_v2_pipeline` call itself: it scores the
# root curation with the preset's metric + auto-curation rows and commits a child
# curation whose labels ARE the rule set's verdict. The run summary then points
# `auto_labeled_curation_id` / `auto_labeled_merge_id` (and the pinned
# `auto_labeled_curation_uuid`) at that child, alongside the root keys. The
# call is idempotent, so it reuses the sort above and only adds the curation
# step.
#
# Automatic labels are **suggestions written as labels, not approval**, and the
# child holds **every** unit, including noise, reject and artifact. Choosing the
# analysis population is a separate, explicit step (section 9).

auto_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    auto_curate=True,
)
display(describe_run(auto_summary))
# The receipt pins the exact curation generations it produced.
auto_labeled = auto_summary.auto_labeled_curation
auto_labeled

# ## 8. Review in the browser, and reopen it later
#
# `start_review` evaluates the pinned curation with a named review profile and
# builds a seeded FigPack view (a local bundle by default). The review is
# persisted by identity: `FigPackReview.resume(review_id)` reopens it from a
# fresh process with the same parent generation, profile snapshot, evaluation
# and display budget. `review.open()` serves the bundle from this kernel at a
# `http://localhost:<port>/` URL and returns it (`open_browser=False` only
# prints it). Select units, edit labels / merges, then **Preview and commit**.
# Merge commits open a reevaluated child to inspect and record as reviewed.
# **Save draft** keeps unfinished edits. After completing the browser sequence,
# `review.result()` returns the exact reviewed curation for analysis. Hosted
# figures use **Curate Figure**, **Save Annotations**, and the notebook commit
# panel instead. The [Curation](./10_Spike_SortingV2_Curation.ipynb) notebook
# covers commit, merged-unit verification, and the handoff to analysis. The
# review packages are optional
# (`pip install "spyglass-neuro[spikesorting-v2-curation]"`), so this cell is
# skipped when they are absent.

# +
import importlib.util

from spyglass.spikesorting.v2.pipeline import FigPackReview

review = None
if importlib.util.find_spec("figpack") is not None:
    review = auto_summary.start_review(
        "franklab_hippocampus_2026_09_17",
        source="auto_labeled",
        upload=False,
    )
    print("Open in a browser:", review.open(open_browser=False))
    review_id = str(review.review_id)
    print("Save this review ID to resume after a restart:", review_id)
    reopened = FigPackReview.resume(review.review_id)
    assert reopened.parent == review.parent
    print(reopened.preview_import().summary())
else:
    print("FigPack not installed; skipping the browser review cell.")
# -

# To reopen after a kernel restart, connect to the same database and storage
# and replace the placeholder with the printed review ID:
#
# ```python
# from spyglass.spikesorting.v2.pipeline import FigPackReview
# review = FigPackReview.resume("paste-the-printed-review-uuid-here")
# print(review.open(open_browser=False))
# ```
#
# Complete **Preview and commit** and any merged-child verification before
# choosing the browser result for analysis below. `review.result()` refuses
# an unfinished review; finish its commit/verification action and retry.

# ## 9. Select the analysis population, then analyze
#
# `SpikeSortingOutput().get_spike_times({"merge_id": ...})` returns **every**
# unit of a curation, labels ignored. Use it for a quick look, not for
# analysis. The supported handoff is `select_units_for_analysis`: it applies a
# named `UnitSelectionParams` policy to the curation's labels, builds the
# `SortedSpikesGroup` that decoding and firing-rate consumers read, and returns
# a receipt naming the exact curation generation, the policy content, and every
# included / excluded unit with its reason.
#
# | Policy | Include | Deny |
# | --- | --- | --- |
# | `v2_accepted_single_units` (default) | `accept` | `mua`, `noise`, `reject`, `artifact` |
# | `v2_accepted_neural_units` | `accept` or `mua` | `noise`, `reject`, `artifact` |
# | `v2_unflagged_units` | everything not denied (MUA and unlabeled included) | `noise`, `reject`, `artifact` |
# | `all_units` | everything (explicit expert choice) | (none) |
#
# The shipped rule sets only **flag** bad units; they never write `accept`, so
# after auto-labeling alone the two `accepted` policies select nothing. Accept
# units in the browser review (section 8) and hand that child over with the
# default policy, or choose `v2_unflagged_units` to state explicitly that
# rule-passing, never-reviewed units count. Unlabeled units are excluded by the
# `accepted` policies and listed on the receipt either way.
#
# Choose the handoff explicitly. Set `use_browser_result=True` after completing
# section 8 to select `review.result()` with the accepted-single-unit policy.
# The default `False` keeps this notebook executable without a browser and
# selects the automatic, **unreviewed** population with `v2_unflagged_units`.
# That population includes unlabeled units and MUA; browser edits only reach
# this analysis when you choose the completed browser result.

use_browser_result = False  # True only after the browser review is complete

# +
from spyglass.spikesorting.v2.pipeline import select_units_for_analysis

if use_browser_result:
    if globals().get("review") is None:
        raise RuntimeError(
            "No browser review is available. Install "
            "spyglass-neuro[spikesorting-v2-curation] and run section 8, "
            "or resume the saved review ID with FigPackReview.resume(review_id)."
        )
    final_curation = review.result()
    analysis_policy = "v2_accepted_single_units"
    print("Analysis source: completed browser review; accepted single units.")
else:
    final_curation = auto_labeled
    analysis_policy = "v2_unflagged_units"
    print(
        "Analysis source: automatic labels WITHOUT human review; "
        "unflagged units, including unlabeled units and MUA."
    )
receipt = select_units_for_analysis(final_curation, policy=analysis_policy)
print(receipt.summary())  # source, policy content, counts, why if empty
display(receipt.describe())  # per-unit verdict, labels, reason
# -

# The receipt's group is the downstream handle; `fetch_spike_data` returns one
# array of spike times (seconds) per SELECTED unit, through the same
# `SortedSpikesGroup` API decoding uses.

spike_times, unit_ids = receipt.fetch_spike_data(return_unit_ids=True)
print(f"{len(spike_times)} selected unit(s):", unit_ids)
receipt.group_key


# ## 10. Optional: drift QC and motion correction (experimental)
#
# Leave `run_motion_example=False` (parameters cell) unless you want these
# cells to compute; they are slow and the motion stage is experimental.
#
# ### Drift QC
#
# `DriftEstimate` estimates probe motion on a `Recording` to **flag** high-drift
# sessions. Nothing applies it: the `Recording` is unchanged. It always uses
# `dredge_fast` (needs `torch`, in the `spikesorting-v2` extra) and applies
# **no artifact mask**, so its numbers are not comparable to the motion stage's
# `MotionEstimate`.

recording_key = {"recording_id": run_summary["recording_id"]}
if run_motion_example:
    from spyglass.spikesorting.v2.recording import DriftEstimate

    DriftEstimate.populate(recording_key)  # on demand only
    high_drift = DriftEstimate & "max_abs_displacement_um > 20"
    display(high_drift.fetch("recording_id"))
    motion = DriftEstimate().get_motion(recording_key)  # SI Motion object

# ### Motion correction
#
# **No motion recipe is validated for a probe.** Inspect the saved estimate rather
# than trusting a sort's improvement.
#
# - On a simulated 32-contact, single-column, 26 µm-pitch polymer shank,
#   `rigid_fast` was sometimes worse than no correction, and interpolation cost
#   sorting quality even with the true motion.
# - A held-out benchmark (`tests/spikesorting/v2/motion_acceptance_held_out.json`)
#   failed for both shipped recipes.
# - No real lab recording with drift has been tested. The estimator has not been
#   benchmarked on discontinuous inputs (gaps, concatenations) or on sources
#   that are not unit-calibrated float µV.
#
# `motion_mode` on `run_v2_pipeline`, `run_v2_pipeline_session`,
# `run_v2_concat_pipeline`, and their preflights:
#
# - `"off"` (default): no motion stage.
# - `"estimate"`: saves a `MotionEstimate` under the source's mask but sorts the
#   uncorrected traces (same `sorting_id` as `"off"`).
# - `"apply"`: also saves a `MotionCorrectedRecording` and sorts it (a new
#   `sorting_id`).
#
# Rules:
#
# - `motion_correction_params_name` (a `MotionCorrectionParameters` row pairing an
#   estimation and an interpolation recipe) is required for `"estimate"` /
#   `"apply"` and rejected for `"off"`. A contradictory pair raises
#   `PipelineInputError` before any query.
# - A failed motion stage raises `PipelineStageError`. It never falls back to an
#   uncorrected sort.
# - **The source must be filtered.** A recipe without a temporal filter
#   (`bandpass_filter=None`, the `no_filter` row; for a concat, the
#   concatenation's recipe) is refused by
#   `MotionEstimateSelection.insert_selection`, and preflight fails
#   `motion_source_filtered`.
# - **The sorter must not also correct.** `SortingSelection.insert_selection`
#   rejects a corrected recording with a sorter that corrects internally
#   (SpykingCircus2 / Tridesclous2 `apply_motion_correction`, Kilosort
#   `do_correction`, resolved against SI's default when the row omits it), and
#   re-checks at compute. Unknown sorter behavior is refused too. Insert a
#   `SorterParameters` row with that key `False`; the error names the sorter,
#   row, and key.
#
# Estimate, inspect, then apply that exact estimate. For a concatenation, use
# `estimate_concat_motion(owner, name, motion_correction_params_name=...)`, then
# `run_v2_concat_pipeline(owner, name, motion_mode="apply", ...)`.

if run_motion_example:
    from spyglass.spikesorting.v2.motion import MotionEstimate
    from spyglass.spikesorting.v2.pipeline import estimate_motion

    motion_source = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        pipeline_preset=pipeline_preset,
    )
    motion_receipt = estimate_motion(
        **motion_source, motion_correction_params_name="dredge_fast_v1"
    )
    print(motion_receipt["motion_diagnostics"])  # peaks, max displacement, bins
    print(motion_receipt["motion_spans_without_evidence"])  # [] when all kept
    fig, motion_report = MotionEstimate().report(motion_receipt)

    motion_summary = run_v2_pipeline(
        **motion_source,
        motion_mode="apply",
        motion_correction_params_name="dredge_fast_v1",
        motion_estimate_id=motion_receipt["motion_estimate_id"],
    )
    print(motion_summary["motion_estimate_supplied"])  # True; status "reused"

# - `estimate_motion` / `estimate_concat_motion` run preflight, recording, and
#   artifact stages and save only the estimate. They skip sorter-only checks;
#   the `"apply"` run checks them.
# - A supplied estimate must be a populated estimate of this run's source and mask
#   (for concat: same group, recipe, members, recordings, and masks), made with
#   the recipe's estimation row, on unchanged traces. A mismatch errors before
#   anything is corrected. `motion_estimate_id` with `"off"` or `"estimate"` is
#   a `PipelineInputError`.
# - Without an id, runs with the same source, mask, and estimation row share one
#   `motion_estimate_id`, so `"apply"` after `"estimate"` reuses it.
# - An `"apply"` summary reports `motion_corrected_recording_id` and
#   `motion_removed_channel_ids`.
#
# Table-level calls, reusing the `recording_key` and `artifact_detection_key`
# of the [Tables](./10_Spike_SortingV2_Tables.ipynb) notebook:
#
# ```python
# from spyglass.spikesorting.v2.motion import (
#     MotionCorrectedRecording,
#     MotionCorrectedRecordingSelection,
#     MotionEstimate,
#     MotionEstimateSelection,
# )
# from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
#
# # One of recording_id / concat_recording_id. artifact_detection_id is optional
# # and single-recording only (a concat carries its own member masks).
# estimate_key = MotionEstimateSelection.insert_selection(
#     {
#         "recording_id": recording_key["recording_id"],
#         "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
#         "motion_estimation_params_name": "dredge_fast_v1",
#     }
# )
# MotionEstimate.populate(estimate_key)
#
# corrected_key = MotionCorrectedRecordingSelection.insert_selection(
#     {
#         "motion_estimate_id": estimate_key["motion_estimate_id"],
#         "motion_interpolation_params_name": "kriging_force_extrapolate_v1",
#     }
# )
# MotionCorrectedRecording.populate(corrected_key)
#
# # The correction must come from the same source and mask as the sort. A
# # mismatch raises at insert (ValueError) and at compute (SchemaBypassError).
# sorting_key = SortingSelection.insert_selection(
#     {
#         "recording_id": recording_key["recording_id"],
#         "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
#         "motion_corrected_recording_id": corrected_key["motion_corrected_recording_id"],
#         "sorter": "mountainsort5",
#         "sorter_params_name": "franklab_30khz_ms5_2026_06",
#     }
# )
# Sorting.populate(sorting_key)
# ```
#
# Inspecting an estimate (`receipt` is an `estimate_motion` receipt):
#
# ```python
# estimate_key = {"motion_estimate_id": receipt["motion_estimate_id"]}
# t_start = MotionEstimate().get_estimation_clock(estimate_key).source_start_s[0]
# t_start += 60.0  # 60 s after the source's first sample
# fig, report = MotionEstimate().report(
#     receipt,
#     corrected_key=corrected_key,  # optional MotionCorrectedRecording
#     trace_window_s=(t_start, t_start + 0.2),  # optional; needs corrected_key
# )
# report["spans_without_evidence"]  # spans corrected from the prior alone
# report["max_abs_displacement_um"], report["rms_displacement_um"]
# report["masked_fraction"], report["gaps"], report["capped_gap_s"]
# report["border_channel_ids"], report["removed_channel_ids"]
#
# motion = MotionEstimate().get_motion(estimate_key)  # bins on the estimation clock
# mapped = MotionEstimate().get_displacement_on_source_clock(estimate_key)
# MotionEstimate().get_spans_without_evidence(estimate_key)
# row = (MotionEstimate & estimate_key).fetch1()
# row["n_peaks_detected"], row["n_peaks_kept"]  # on the masked recording
# row["peaks_per_temporal_bin"], row["peaks_per_continuity_span"]
# ```
#
# - `report` takes a receipt, a `motion_estimate_id`, or a restriction. Its
#   panels: displacement over source time (a depth heatmap when nonrigid); a
#   timeline of masks, gaps, member joins (hatched where shortened) and spans
#   without evidence; kept peaks per span; border channels moved past the
#   probe's ends (circled when extrapolated, crossed when removed); and, with
#   `trace_window_s`, original vs corrected traces for four central channels
#   (`trace_channel_ids` to choose).
# - **A span with no evidence** kept no peak, so its displacement comes from the
#   estimator's prior alone. It is not refused, but `run_v2_pipeline` warns and
#   lists it in `motion_spans_without_evidence`. Check these before applying.
# - `MotionEstimate` stores the SI `Motion`, its resolved configuration, its
#   spans, and peak counts, never a raw peak array.
# - `get_displacement_on_source_clock` flags a bin inside a capped gap
#   (`in_gap=True`, `source_time_s=NaN`).
#
# Recipes (all DREDge AP registration, SI's
# `estimate_motion(..., method="dredge_ap")`):
#
# | recipe | estimator | peak localization | interpolation border mode |
# | --- | --- | --- | --- |
# | `dredge_v1` (default row) | `dredge_ap`, nonrigid (`rigid=False`) | `monopolar_triangulation` | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
# | `dredge_fast_v1` (default row) | `dredge_ap`, nonrigid (`rigid=False`) | `grid_convolution` | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
# | `rigid_fast` (allowed; insert explicitly, no default row) | `dredge_ap`, **rigid=True, 5 s bins** | `center_of_mass` | `remove_channels` (`kriging_remove_channels_v1`) |
#
# - `MotionEstimationParameters` stores the fully resolved SI configuration, the
#   SI version, and the algorithm version, all in the estimate's identity. An
#   upgrade or a changed override selects a new estimate.
# - `MotionInterpolationParameters` states every `interpolate_motion` argument
#   (`spatial_interpolation_method`, `sigma_um`, `p`, `num_closest`). Only
#   `remove_channels` and `force_extrapolate` border modes are allowed.
# - Estimation and interpolation act on µV. A source that is not float µV with
#   gain 1 and offset 0 is scaled first (float32). Masked samples are exactly 0
#   µV, and the corrected recording is stored with gain 1 and offset 0.
# - **Gap policy.** Each source is estimated **once**, on an estimation clock.
#   Within a continuity span (an uninterrupted stretch, or one concat member)
#   time advances at `1 / fs`. A real gap is kept up to `max_gap_s` (required on
#   every row, 30 s in the shipped rows, part of identity) and shortened beyond
#   it. Concat members must be in acquisition order.
# - **Masks.** A peak is kept only if its localization window lies inside one
#   artifact-free statistics span and its detection window does not cross a
#   join. A `MotionCorrectedRecording` reuses its estimate's spans and
#   re-silences masked samples after interpolation.
# - **`rigid_fast`** has no default row and is not recommended. To compare it,
#   insert a `MotionEstimationParameters` row with
#   `params={"preset": "rigid_fast", "max_gap_s": 30.0}`, pair it with
#   `kriging_remove_channels_v1` in a new `MotionCorrectionParameters` row, and
#   run `motion_mode="apply"` with that name.
# - **Storage.** A `MotionEstimate` is a database row (blob), not an NWB file. A
#   `MotionCorrectedRecording` is a full NWB `ElectricalSeries` about the size
#   of its source; an `"apply"` sort keeps both on disk.
# - **Deletion.** A sort references its corrected recording by foreign key.
#   `delete_quick` on a referenced one fails, and Spyglass's `cautious_delete`
#   cascades through the sorts that read it. The same protects a referenced
#   `MotionEstimate`.

# ## Next steps
#
# - Curate the sort in the browser (FigPack) or step by step with
#   [Curation](./10_Spike_SortingV2_Curation.ipynb).
# - Customize a preset, or sort a whole session at once, with
#   [Presets](./10_Spike_SortingV2_Presets.ipynb).
# - Track units across multiple sessions with
#   [Cross-Session Spike Sorting](./10_Spike_SortingV2_CrossSession.ipynb).
# - Run each stage table yourself with
#   [Tables](./10_Spike_SortingV2_Tables.ipynb).
# - Organize sorts across sessions and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb)
#   (`select_units_for_analysis` builds the `SortedSpikesGroup` for you).
# - Supported workload differences: tetrodes use the
#   `franklab_tetrode_hippocampus_30khz_ms4_2026_06` preset (same rows as the
#   probe preset; `probe_type` is informational); probes with drift can add the
#   optional, experimental motion stage (`motion_mode`; section 10 above), and
#   same-day blocks
#   can be concatenated in the Cross-Session notebook; clusterless decoding
#   features run the `clusterless_thresholder` preset (see Presets) and hand
#   the root curation's `merge_id` to `UnitWaveformFeatures`; there is no
#   unit selection step because the thresholder yields one "unit" per channel
#   group.
# - Stage-by-stage behavior that differs from v1 (ADC phase-shift, bad-channel
#   handling): see the
#   [migration guide](https://lorenfranklab.github.io/spyglass/latest/Features/SpikeSortingV2_Migration/).
