# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (spyglass_spikesorting_v2)
#     language: python
#     name: python3
# ---

# # Spike Sorting v2 — single-session walkthrough
#
# This notebook runs the modern (`spyglass.spikesorting.v2`) spike-sorting
# pipeline end-to-end on **one already-ingested session**, using the
# high-level `run_v2_pipeline` orchestrator:
#
# > defaults → sort group → **preflight** → pipeline → curation summary →
# > downstream fetch
#
# It assumes you have already:
#
# 1. configured your DataJoint connection (see
#    [Setup](./00_Setup.ipynb)), and
# 2. ingested a session with `insert_sessions` (see
#    [Insert Data](./02_Insert_Data.ipynb)).
#
# For a table-by-table tour of the internals (what each `populate` does, how
# identities are content-addressed, how curation is staged), see the
# developer walkthrough notebook `10_Spike_SortingV2_dev_walkthrough.ipynb`.

# +
import datajoint as dj

from spyglass.common import LabTeam
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.pipeline import (
    describe_pipeline_presets,
    describe_sort_groups,
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
# full-session sort uses the `"raw data valid times"` interval and the
# production MountainSort4 pipeline preset; change `pipeline_preset` to one
# of the names from `describe_pipeline_presets()` below if you want a
# different sorter (e.g. the MountainSort5 alternative, or a cortex/20 kHz
# preset).

# + tags=["parameters"]
nwb_file_name = "your_session_.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_tetrode_hippocampus_30khz_ms4_2026_06"
# -

# ## 2. One-time setup
#
# `initialize_v2_defaults()` installs every default parameter row the pipeline
# needs (preprocessing / artifact / sorter), so there is no per-table
# `insert_default()` to remember. The owning `LabTeam` and the per-shank sort
# groups are session-specific user input, so we create them here.
# `describe_sort_groups()` and `plot_sort_group_geometry()` then show the membership,
# metadata, and physical layout you should inspect before deciding which group
# to sort. The walkthrough picks the first group only to keep the example
# reproducible; for real analyses, set `sort_group_id` deliberately after
# reviewing the table and geometry view.
#

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "spike sorting"},
    skip_duplicates=True,
)
if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
sort_groups = describe_sort_groups(nwb_file_name)
if sort_groups.empty:
    raise ValueError(f"No SortGroupV2 rows found for {nwb_file_name!r}.")
sort_group_id = int(sort_groups.iloc[0]["sort_group_id"])
plot_sort_group_geometry(nwb_file_name)
sort_groups


# ## 3. Pick a Pipeline Preset
#
# `describe_pipeline_presets()` returns a table of what each shipping pipeline
# preset does — the sorter, the parameter rows each stage uses, the intended
# use, and (a known footgun) the units of the detection threshold — so you can
# choose one without reading the module source.

describe_pipeline_presets()

# ## 4. Preflight — a fast, fail-early check
#
# `preflight_v2_pipeline` verifies in ~1 s (inserting nothing, never calling
# `populate`) that every prerequisite is in place: the session / interval /
# team / sort-group rows, the pipeline preset's parameter rows, and the sorter binary.
# It returns a `PreflightReport` that is truthy when the configuration is
# runnable; `report.errors` lists each blocking problem with the exact fix, and
# `report.expected_ids` shows the selection PKs the run will produce. (Skip it
# with `run_v2_pipeline(..., preflight=False)`.)

report = preflight_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)
print("preflight ok:", report.ok)
report

# ## 5. Run the pipeline
#
# `run_v2_pipeline` chains recording → artifact detection → sort → curation into one
# idempotent call and registers the result on the `SpikeSortingOutput` merge
# table. With `preflight=True` (the default) it re-runs the check above before
# any populate. **Re-running with the same inputs is safe** — it finds the
# existing rows and returns the same run summary (same `merge_id`) without
# inserting duplicates.

run_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)

# ## 6. Read the run summary
#
# Besides the stable keys (`pipeline_preset` / `recording_id` /
# `artifact_detection_id` / `sorting_id` / `curation_id` / `merge_id` /
# `n_units`), the run summary carries
# per-stage observability: each `*_status` is `"computed"` if the stage did
# work this call or `"reused"` if its row already existed; `stage_seconds` is
# the wall-clock spent **this call** per stage (≈0 on an idempotent re-run, not
# cumulative compute cost); `warnings` collects advisories. **Downstream code
# keys off `merge_id`.**
#
# A sort that finds **zero units** is a legitimate result on a quiet shank: it
# still produces an empty-but-real curation + merge row (with a loud warning),
# so downstream code treats it like any other row. Pass `require_units=True` to
# turn that into a hard error instead.

print("merge_id (downstream key):", run_summary["merge_id"])
print("n_units                  :", run_summary["n_units"])
print(
    "stage status             :",
    {
        stage: run_summary[f"{stage}_status"]
        for stage in ("recording", "artifact_detection", "sorting", "curation")
    },
)
print("stage seconds            :", run_summary["stage_seconds"])
run_summary

# ## 7. Inspect / curate
#
# `summarize_curation` accepts the run summary directly and returns a plain dict
# (`n_units`, `labels`, `merge_groups`, `merges_applied`, `is_merge_preview`,
# `merge_id`, ...). To curate further, reach for the intent-first wrappers on
# `CurationV2` — `create_initial_curation`, `propose_merge_curation`,
# `create_merged_curation` — rather than the expert `insert_curation`.
#
# > Interactive web curation (label/merge in a browser) is coming via FigPack
# > in a later release; it will slot in here.

CurationV2.summarize_curation(run_summary)

# ## 8. Downstream: choose the output accessor
#
# The payoff: the sort is resolvable through the `SpikeSortingOutput` merge
# table, so every existing downstream consumer (decoding, ripple detection,
# `SortedSpikesGroup`) works on the v2 `merge_id` unchanged.
#
# | Goal | Call |
# | --- | --- |
# | Spike times | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})` |
# | Recording | `SpikeSortingOutput().get_recording({"merge_id": merge_id})` |
# | Sorting | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})` |
# | Unit brain regions | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})` |
# | Curation summary | `CurationV2.summarize_curation(run_summary)` |
# | Analyzer/debug internals | `Sorting().get_analyzer({"sorting_id": run_summary["sorting_id"]})` |
#
# Here we fetch spike times: one array of spike times (seconds) per unit.
#

merge_id = run_summary["merge_id"]
spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
print(f"{len(spike_times)} unit(s)")
spike_times


# ## Next steps
#
# - Organize sorts across sessions and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb)
#   (`SortedSpikesGroup`).
# - Drive the stages individually (custom pipeline presets, ADC phase-shift, bad-channel
#   handling, drift QC) — see `docs/src/Features/SpikeSortingV2.md`.
#
