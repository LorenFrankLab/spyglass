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

# # Spike Sorting v2 — single-session walkthrough
#
# This notebook runs the modern (`spyglass.spikesorting.v2`) spike-sorting
# pipeline end-to-end on **one already-ingested session**, using the high-level
# `run_v2_pipeline` orchestrator:
#
# > defaults → sort group → **preflight** → pipeline → curation summary →
# > downstream fetch
#
# It assumes you have already configured your DataJoint connection (see
# [Setup](./00_Setup.ipynb)) and ingested a session with `insert_sessions` (see
# [Insert Data](./02_Insert_Data.ipynb)).
#
# This is the **first-sort path**. For the deeper how-tos, see:
#
# - [Curation](./10_Spike_SortingV2_Curation.ipynb) — browser (FigPack) and
#   step-by-step evaluate → merge → re-evaluate curation.
# - [Presets](./10_Spike_SortingV2_Presets.ipynb) — customize a preset
#   (`clone_preset` / `register_preset`) and sort a whole session at once.
# - [Cross-session](./10_Spike_SortingV2_CrossSession.ipynb) — concatenate
#   same-day recordings and track units across days.
# - A table-by-table tour of the internals is in
#   `10_Spike_SortingV2_dev_walkthrough.ipynb`.

# +
import datajoint as dj
from IPython.display import display

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
# MountainSort5 pipeline preset (`franklab_probe_hippocampus_30khz_ms5_2026_06`):
# it runs under the `numpy>=2` baseline out of the box. MountainSort4 is the
# scientifically-preferred polymer-probe recipe but needs `numpy<2`, so run it
# via the containerized `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`
# preset on modern (`numpy>=2`) hosts with Docker/Singularity, or the local
# `franklab_probe_hippocampus_30khz_ms4_2026_06` preset on `numpy<2` hosts.
# Change `pipeline_preset` to any name from `describe_pipeline_presets()` below
# (e.g. a cortex or 20 kHz preset).
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
pipeline_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
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


# ## 3. Pick a Pipeline Preset
#
# `describe_pipeline_presets()` returns a table of what each shipping pipeline
# preset does — the sorter, the parameter rows each stage uses, the intended
# use, and (a known footgun) the units of the detection threshold — so you can
# choose one without reading the module source.

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
# - Extracellular spikes are often downward but can flip with geometry, so the
#   Frank Lab MountainSort rows use bidirectional `detect_sign=0`. A downward-only
#   `-1` row is more conservative and usually needs separate validation.
# - Hippocampus rows high-pass at 600 Hz; cortex rows at 300 Hz. Recordings are
#   already filtered when MountainSort runs (`filter=False`). MS4 rows whiten,
#   use adjacency radius 100 um, and tune clip/detect intervals to 30 vs 20 kHz.
# - The analyzer keeps two waveform views: unwhitened display waveforms for real
#   shapes/amplitudes, and a whitened metric analyzer for PC/NN cluster metrics.
#   Hippocampal display/metric rows intentionally use 0.5/0.5 ms windows and up
#   to 20000 spikes per unit.

# To **adapt** a preset (tune one knob with `clone_preset`, or build a custom one
# with `register_preset`) without hand-editing parameter rows, see the
# [Presets how-to](./10_Spike_SortingV2_Presets.ipynb).

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
# stage (its `*_status` is `"computed"` if the stage ran this call, `"reused"`
# if its row already existed, or `"skipped"` if the preset has no such stage
# (e.g. artifact detection for a no-artifact preset), with the wall-clock
# `seconds` spent **this call** —
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
# deeper SNR / ISI / nearest-neighbour metrics are computed by the
# curation-evaluation step in section 7 below (`CurationEvaluation`). The raw
# `run_summary` dict carries the same fields programmatically
# (`run_summary["merge_id"]`, `run_summary["n_units"]`).

display(describe_run(run_summary))
describe_units(run_summary["sorting_id"])

# ## 7. Curate the sort
#
# `run_v2_pipeline` leaves you a root (uncurated) curation;
# `CurationV2.summarize_curation` accepts the run summary directly and returns a
# plain dict (`n_units`, `labels`, `merge_groups`, `merge_id`, ...).
#
# The quickest path is **one-call auto-curation** below. For the other two paths —
# labeling/merging **in a browser** with FigPack, and the hands-on **step-by-step**
# evaluate → merge → re-evaluate loop — see the
# [Curation how-to](./10_Spike_SortingV2_Curation.ipynb).

CurationV2.summarize_curation(run_summary)

# ### 7-auto. Automated labeling in one call
#
# When you trust the rule set for a batch, `auto_curate=True` folds the
# evaluate-and-accept steps into the `run_v2_pipeline` call itself: it scores the
# root curation with the preset's metric + auto-curation rows and commits a child
# curation whose labels ARE the rule set's verdict. The run summary then carries
# `auto_curation_id` / `auto_merge_id` (the committed labeled curation) alongside
# the root keys. It is idempotent like the rest of the pipeline, so this reuses
# the sort already computed above and only adds the curation step.

auto_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    auto_curate=True,
)
display(describe_run(auto_summary))
# Key downstream code off `auto_merge_id` to use the auto-labeled result.
auto_summary["auto_merge_id"]

# ## 8. Downstream: choose the output accessor
#
# The payoff: the sort is resolvable through the `SpikeSortingOutput` merge
# table, so every existing downstream consumer (decoding, ripple detection,
# `SortedSpikesGroup`) works on the v2 `merge_id` unchanged. Key off the
# **auto-curated** result `auto_summary["auto_merge_id"]` (section 7-auto) rather
# than `run_summary["merge_id"]`, which is the uncurated root curation. (If you
# curate by hand instead, see the
# [Curation how-to](./10_Spike_SortingV2_Curation.ipynb) and key off that
# curation's `merge_id`.)
#
# | Goal | Call |
# | --- | --- |
# | Spike times | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})` |
# | Recording | `SpikeSortingOutput().get_recording({"merge_id": merge_id})` |
# | Sorting | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})` |
# | Unit brain regions | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})` |
# | Curation summary | `CurationV2.summarize_curation(auto_summary)` |
# | Analyzer/debug internals | `Sorting().get_analyzer({"sorting_id": run_summary["sorting_id"]})` |
#
# Here we fetch spike times: one array of spike times (seconds) per unit.

merge_id = auto_summary["auto_merge_id"]  # the auto-curated result
spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
print(f"{len(spike_times)} unit(s)")
spike_times


# ## Next steps
#
# - Curate the sort — browser (FigPack) or step-by-step — with
#   [Curation](./10_Spike_SortingV2_Curation.ipynb).
# - Customize a preset, or sort a whole session at once, with
#   [Presets](./10_Spike_SortingV2_Presets.ipynb).
# - Track units across multiple sessions with
#   [Cross-Session Spike Sorting](./10_Spike_SortingV2_CrossSession.ipynb).
# - Organize sorts across sessions and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb)
#   (`SortedSpikesGroup`).
# - Stage-by-stage internals (ADC phase-shift, bad-channel handling, drift QC):
#   see `docs/src/Features/SpikeSortingV2.md`.
