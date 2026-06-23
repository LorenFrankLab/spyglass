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
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.common.common_interval import IntervalList
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.metric_curation import (
    AnalyzerCuration,
    AnalyzerCurationSelection,
)
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
# deeper SNR / ISI / nearest-neighbour metrics are computed by the
# analyzer-driven curation step in section 7 below (`AnalyzerCuration`). The raw
# `run_summary` dict carries the same fields programmatically
# (`run_summary["merge_id"]`, `run_summary["n_units"]`).

display(describe_run(run_summary))
describe_units(run_summary["sorting_id"])

# ## 7. Inspect and curate
#
# `run_v2_pipeline` leaves you a root curation. `summarize_curation` accepts the
# run summary directly and returns a plain dict (`n_units`, `labels`,
# `merge_groups`, `merges_applied`, `is_merge_preview`, `merge_id`, ...). The
# labels curation *accepts* are the canonical set `CurationV2.label_options()`
# (the `CurationLabel` enum); custom labels need `allow_custom_labels=True`.
#
# Curation is a loop, and the second analyzer pass is not redundant:
#
# 1. **Auto-curate** — `AnalyzerCuration` walks the sort's analyzer, computes
#    quality metrics, and proposes labels from a rule set;
#    `materialize_curation` commits them to a child curation.
# 2. **Manually merge** oversplit clusters (MS4/MS5 oversplit and don't track
#    drift) — find burst pairs with `plot_by_sort_group_ids` /
#    `investigate_pair_*`, then merge them with `create_merged_curation`.
# 3. **Re-run auto-curation on the merged curation** — merging changes each
#    unit's template (and so its SNR / ISI-violation fraction / PC-NN
#    separation), so metrics over the *post-merge* templates are the numbers of
#    record.
#
# > Interactive web curation (label/merge in a browser) is coming via FigPack
# > in a later release; it will slot in here.

CurationV2.summarize_curation(run_summary)

# ### 7a. Automatic curation (pass 1)
#
# Pair the root curation with a quality-metric recipe and an auto-curation rule
# set, then populate. `franklab_default` computes `snr` / `isi_violation` /
# `firing_rate` / `num_spikes` / `presence_ratio` / `amplitude_cutoff` /
# `nn_advanced` (PCA); `franklab_default_auto_curation_2026_06` labels
# `nn_noise_overlap > 0.1` units `noise` and `isi_violation > 0.02` (>2%
# refractory violations) `reject`. `plot_units_qc` is the population
# "do these units look reasonable?" view; `get_metrics` returns the per-unit
# table.
#
# `populate` is the heavy step — it grows the analyzer extensions and builds the
# whitened PCA analyzer, so it can take minutes on a real session (it is
# idempotent, so a re-run reuses the result instead of recomputing).

auto_sel = AnalyzerCurationSelection.insert_selection(
    {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["curation_id"],
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
    }
)
AnalyzerCuration.populate(auto_sel)
AnalyzerCuration().plot_units_qc(auto_sel)
AnalyzerCuration.get_metrics(auto_sel)

# #### Reading the labels: proposals to verify, not verdicts
#
# The rule set proposes labels from thresholds — a starting point, not a
# verdict. Before trusting a `noise` / `reject` tag, cross-check the unit:
#
# - **Refractory dip** — `AnalyzerCuration().plot_correlograms(auto_sel)`: a real
#   single unit has a central dip in its autocorrelogram; a symmetric, dip-free
#   autocorrelogram is a noise signature.
# - **The interneuron trap** — a fast-spiking interneuron raises ISI violations
#   and sits just around the `nn_noise_overlap` noise threshold. If its waveform
#   is narrow, its refractory dip clean, and its firing stable, it is a cell, not
#   noise — don't reject it on the metric alone.
# - **Amplitude over time** — `AnalyzerCuration().get_peak_amps(auto_sel)`: a
#   slow, smooth amplitude drift across a place-field traversal is a place cell,
#   not multi-unit activity; sharp amplitude steps or several distinct bands are
#   MUA.
#
# These pattern-recognition cues are calibrated for **hippocampal tetrodes and
# polymer probes**; other regions (cortex, thalamus, striatum) have different
# waveform widths and bursting, so the thresholds and tells above can mislead
# there.

# ### 7b. Find burst pairs to merge, then commit the auto labels
#
# `plot_by_sort_group_ids` scatters waveform similarity vs cross-correlogram
# asymmetry, one point per unit pair — high-similarity, asymmetric pairs are
# merge candidates (drill into a pair with `investigate_pair_xcorrel` /
# `investigate_pair_peaks`). Hippocampal pyramidal cells fire complex-spike
# bursts with an amplitude decrement that MountainSort oversplits into a parent +
# a shorter-waveform daughter — exactly the high-similarity, short-lag-asymmetric
# pair this scatter surfaces, so merging them reassembles one cell.
# `materialize_curation` then commits the proposed auto labels into a child
# curation you merge on top of.

AnalyzerCuration().plot_by_sort_group_ids(auto_sel)
auto_curation = AnalyzerCuration().materialize_curation(auto_sel)
auto_curation  # {"sorting_id", "curation_id"} of the auto-labeled child

# ### 7c. Manual merge, then the final auto-curation pass (pass 2)
#
# List the burst pairs you decided to merge in step 7b in `merge_groups_to_apply`
# (each a list of ≥2 unit ids, e.g. `[[3, 7]]`). It starts EMPTY so a run-all
# never merges arbitrary units — fill it in after inspecting 7b, then re-run.
# When you merge, `create_merged_curation` (intent-first sugar over
# `insert_curation` with `apply_merge=True`) branches off the auto-labeled
# curation, and `AnalyzerCuration` runs once more on the merged curation: the
# metrics over the post-merge templates are the final numbers of record.
# `materialize_curation` commits those final labels so downstream code keys off
# the curated result, not the uncurated root curation. (Leaving the list empty
# keeps the auto-labeled curation from 7b as the result — no merge applied.)

merge_groups_to_apply = []  # e.g. [[3, 7]] after inspecting step 7b

if merge_groups_to_apply:
    merged = CurationV2.create_merged_curation(
        sorting_key={"sorting_id": run_summary["sorting_id"]},
        merge_groups=merge_groups_to_apply,
        parent_curation_id=auto_curation["curation_id"],
        description="manual burst-pair merge",
        reuse_existing=True,
    )
    final_sel = AnalyzerCurationSelection.insert_selection(
        {
            "sorting_id": run_summary["sorting_id"],
            "curation_id": merged["curation_id"],
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
        }
    )
    AnalyzerCuration.populate(final_sel)
    display(AnalyzerCuration.get_metrics(final_sel))  # over merged templates
    final_curation = AnalyzerCuration().materialize_curation(final_sel)
else:
    final_curation = auto_curation  # no manual merge yet; use the 7b result

final_summary = CurationV2.summarize_curation(final_curation)
final_merge_id = final_summary["merge_id"]
final_summary

# ### 7d. Surface waveform shape for cell typing (your thresholds, not the pipeline's)
#
# `get_metrics` returns a waveform-shape column next to the quality metrics:
# `trough_half_width` — the half-amplitude width of the spike trough, in seconds,
# read from the unwhitened display analyzer. Narrow spikes are fast-spiking
# interneurons, wide spikes pyramidal cells. Paired with `firing_rate` (already a
# quality metric) it gives the classic rate × width view for separating putative
# cell types.
#
# **The pipeline ships NO cell-type thresholds.** The boundary below is **your
# own** — region-specific and tuned here for hippocampus (fast + narrow ⇒
# putative interneuron; slow + wide ⇒ putative pyramidal). For cortex, striatum,
# or thalamus the cutoffs differ; pick your own, do not reuse these. (The
# trough-to-peak duration and slope columns are available by adding them to the
# metric row's `template_metric_columns`, but they are not defaults: they need a
# wider post-trough window than the hippocampus display recipe's 0.5 ms and clip
# there — they are reliable on the wider cortex/fallback window.)

shape = AnalyzerCuration.get_metrics(auto_sel)[
    ["firing_rate", "trough_half_width"]
].dropna()

# YOUR thresholds — region-specific, NOT pipeline defaults. Tune per recording.
rate_cut_hz, width_cut_s = 7.0, 0.0003  # 0.3 ms
is_interneuron = (shape["firing_rate"] > rate_cut_hz) & (
    shape["trough_half_width"] < width_cut_s
)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(
    shape.loc[~is_interneuron, "trough_half_width"] * 1e3,
    shape.loc[~is_interneuron, "firing_rate"],
    c="tab:blue",
    label="putative pyramidal",
    alpha=0.8,
)
ax.scatter(
    shape.loc[is_interneuron, "trough_half_width"] * 1e3,
    shape.loc[is_interneuron, "firing_rate"],
    c="tab:red",
    label="putative interneuron",
    alpha=0.8,
)
ax.axvline(width_cut_s * 1e3, ls="--", c="gray")
ax.axhline(rate_cut_hz, ls="--", c="gray")
ax.set_xlabel("trough_half_width (ms)")
ax.set_ylabel("firing_rate (Hz)")
ax.set_title("Putative cell types — YOUR hippocampal thresholds, not the pipeline's")
ax.legend()
print(
    f"{int(is_interneuron.sum())} putative interneuron(s), "
    f"{int((~is_interneuron).sum())} putative pyramidal"
)

# ## 8. Downstream: choose the output accessor
#
# The payoff: the sort is resolvable through the `SpikeSortingOutput` merge
# table, so every existing downstream consumer (decoding, ripple detection,
# `SortedSpikesGroup`) works on the v2 `merge_id` unchanged. Key off
# `final_merge_id` — the merge_id of the curated result from section 7 — rather
# than `run_summary["merge_id"]`, which is the uncurated root curation.
#
# | Goal | Call |
# | --- | --- |
# | Spike times | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})` |
# | Recording | `SpikeSortingOutput().get_recording({"merge_id": merge_id})` |
# | Sorting | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})` |
# | Unit brain regions | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})` |
# | Curation summary | `CurationV2.summarize_curation(final_curation)` |
# | Analyzer/debug internals | `Sorting().get_analyzer({"sorting_id": run_summary["sorting_id"]})` |
#
# Here we fetch spike times: one array of spike times (seconds) per unit.
#

merge_id = final_merge_id  # the curated result (root is run_summary["merge_id"])
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
