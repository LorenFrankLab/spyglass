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
    preflight_v2_pipeline_session,
    run_v2_pipeline,
    run_v2_pipeline_session,
)
from spyglass.spikesorting.v2.recording import SortGroupV2
from spyglass.spikesorting.v2.sorting import Sorting  # noqa: F401

dj.config["display.limit"] = 12  # cap rows in table reprs

# -

# ## 1. Choose your session
#
# Point the notebook at the session you ingested with `insert_sessions`. A
# full-session sort uses the `"raw data valid times"` interval and the default
# MountainSort5 pipeline preset (it runs under the `numpy>=2` baseline; the MS4
# production recipe needs `numpy<2`). Change `pipeline_preset` to one of the
# names from `describe_pipeline_presets()` below if you want a different sorter
# (e.g. an MS4 production preset on a `numpy<2` install, or a cortex/20 kHz
# preset).
#
# To see the intervals available for this session — the valid
# `interval_list_name` values — list them with
# `IntervalList & {"nwb_file_name": nwb_file_name}`. Leave `sort_group_id =
# None` to auto-pick when the session has exactly one sort group; with more than
# one, set it explicitly after reviewing the table and geometry plot in step 2
# (don't just take the first row).

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
# Sort group (shank) to sort. None auto-picks only when the session has exactly
# one sort group; otherwise set it deliberately after reviewing step 2.
sort_group_id = None
# -

IntervalList & {"nwb_file_name": nwb_file_name}

# ## 2. One-time setup
#
# `initialize_v2_defaults()` installs every default parameter row the pipeline
# needs (preprocessing / artifact / sorter), so there is no per-table
# `insert_default()` to remember. The owning `LabTeam` and the per-shank sort
# groups are session-specific user input, so we create them here.
# `describe_sort_groups()` and `plot_sort_group_geometry()` then show the membership,
# metadata, and physical layout you should inspect before deciding which group
# to sort. With one sort group the cell auto-selects it; with several it makes
# you set `sort_group_id` deliberately (above) rather than defaulting to the
# first shank, and validates your choice against the available groups.
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
available_sort_group_ids = [int(g) for g in sort_groups["sort_group_id"]]
plot_sort_group_geometry(nwb_file_name)
sort_groups


# +
# Validate the chosen sort group only after the table and geometry are visible.
if sort_group_id is None:
    if len(available_sort_group_ids) == 1:
        sort_group_id = available_sort_group_ids[0]
    else:
        raise ValueError(
            f"{nwb_file_name!r} has multiple sort groups "
            f"{available_sort_group_ids}; set sort_group_id explicitly after "
            "reviewing the table and geometry plot below — don't default to "
            "the first shank."
        )
elif sort_group_id not in available_sort_group_ids:
    raise ValueError(
        f"sort_group_id={sort_group_id} is not one of "
        f"{available_sort_group_ids} for {nwb_file_name!r}."
    )
sort_group_id
# -


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
# `describe_run(run_summary)` renders the run as a receipt: a leading summary row
# with `n_units` and the `merge_id` **downstream code keys off**, one row per
# stage (its `*_status` is `"computed"` if the stage ran this call or `"reused"`
# if its row already existed, with the wall-clock `seconds` spent **this call** —
# ≈0 on an idempotent re-run, not cumulative cost), and one row per advisory
# `warning`. Because warnings are their own rows, a **zero-unit** sort — a
# legitimate quiet-shank result that still writes an empty-but-real curation +
# merge row — is impossible to miss. Pass `require_units=True` to
# `run_v2_pipeline` to turn zero units into a hard error instead.
#
# Below the receipt, `describe_units(...)` shows the per-unit, sort-time detail:
# one row per unit with `n_spikes`, `firing_rate_hz` (over the duration the sort
# actually observed — artifact-removed when masking ran, so the rate is not
# inflated by blanked segments), `peak_amplitude_uv`, `peak_electrode_id`, and
# `brain_region`. It reads only sort-time metadata (no waveform recompute);
# deeper SNR / ISI / nearest-neighbour metrics arrive with the analyzer-driven
# curation in a later release. The raw `run_summary` dict carries the same
# fields programmatically (`run_summary["merge_id"]`, `run_summary["n_units"]`).

from IPython.display import display
display(describe_run(run_summary))
describe_units(run_summary["sorting_id"])

# ## 7. Inspect / curate
#
# `summarize_curation` accepts the run summary directly and returns a plain dict
# (`n_units`, `labels`, `merge_groups`, `merges_applied`, `is_merge_preview`,
# `merge_id`, ...). The labels curation *accepts* are the canonical set
# `CurationV2.label_options()` (the `CurationLabel` enum); custom labels are
# possible only via `allow_custom_labels=True`. To curate further,
# reach for the intent-first wrappers on `CurationV2` —
# `create_initial_curation`, `propose_merge_curation`, `create_merged_curation`
# — rather than the expert `insert_curation`.
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


# ## 9. Sort the whole session at once
#
# A real session has one sort group per shank. `run_v2_pipeline_session` loops
# the single-group runner over all of them (run `preflight_v2_pipeline_session`
# first for a read-only whole-session check), returning one entry per group with
# an `outcome` of `"ok"` or `"failed"`. With `continue_on_error=True` a failed
# group is recorded (with `error_type`, `error`, and `partial_run_summary`)
# instead of stopping the batch. Pick `pipeline_preset` explicitly from
# `describe_pipeline_presets()` — the session runner infers no default.
# `describe_run(session_results)` renders the whole batch as one receipt: a
# summary row with the ok / failed / zero-unit / with-warnings counts, then a
# row per group (and per warning), so failed and zero-unit groups don't hide in
# a long list.

# Read-only whole-session check first (run_v2_pipeline_session also preflights
# internally); inspect report.ok / report.errors before committing compute.
session_report = preflight_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)
print("session preflight ok:", session_report.ok)
session_results = run_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    continue_on_error=True,
)
describe_run(session_results)


# ## Next steps
#
# - Organize sorts across sessions and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb)
#   (`SortedSpikesGroup`).
# - Drive the stages individually (custom pipeline presets, ADC phase-shift, bad-channel
#   handling, drift QC) — see `docs/src/Features/SpikeSortingV2.md`.
#
