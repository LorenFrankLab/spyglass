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

# # Spike Sorting v2 — Curation
#
# Curate a single-session v2 sort, picking up where the
# [first-sort walkthrough](./10_Spike_SortingV2.ipynb) leaves off. It re-runs a
# sort to get a root curation, then shows the two hands-on curation paths:
# browser curation with **FigPack**, and the step-by-step **evaluate → merge →
# re-evaluate** loop built on `CurationEvaluation`.
#
# Assumes a configured DataJoint connection and an ingested session (see
# [Setup](./00_Setup.ipynb) / [Insert Data](./02_Insert_Data.ipynb)).

# +
import datajoint as dj
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.metric_curation import (
    CurationEvaluation,
    CurationEvaluationSelection,
)
from spyglass.spikesorting.v2.pipeline import (
    describe_sort_groups,
    plot_sort_group_geometry,
    preflight_v2_pipeline,
    run_v2_pipeline,
)
from spyglass.spikesorting.v2.recording import SortGroupV2
from spyglass.spikesorting.v2.sorting import Sorting  # noqa: F401

dj.config["display.limit"] = 12

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
# Sort group (shank) to sort. None auto-picks only when the session has exactly
# one sort group; otherwise set it deliberately after reviewing step 2.
sort_group_id = None
# -

# ## 1. One-time setup
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


# ## 2. Run a root sort
#
# Run a sort so we have a root curation to work with (idempotent — reuses an
# existing sort). See the [first-sort walkthrough](./10_Spike_SortingV2.ipynb)
# for the details of the recording → artifact → sort stages.

report = preflight_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)
print("preflight ok:", report.ok)
report

run_summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
)

# ## 3. Inspect and curate
#
# `run_v2_pipeline` leaves you a root curation. `summarize_curation` describes
# **one** curation and returns a plain dict (`n_units`, `labels`, `merge_groups`,
# `merges_applied`, `is_merge_preview`, `merge_id`, ...); build its key from the
# summary's `root_curation_id` (a run summary has no bare `curation_id`). The
# labels curation *accepts* are the canonical set `CurationV2.label_options()`
# (the `CurationLabel` enum); custom labels need `allow_custom_labels=True`.
#
# There are three ways to curate, from most automated to most hands-on:
#
# - **Automated** — `run_v2_pipeline(auto_curate=True)` scores the sort and
#   commits the rule set's labels in the same call (section 3-auto).
# - **In a browser** — `run_v2_pipeline(build_figpack_view=True)` publishes an interactive
#   FigPack view you label and merge in a browser (section 3-browser).
# - **Step by step** — the evaluate → accept → merge → re-evaluate loop below
#   (sections 3a–3e), for full control over each decision.
#
# The step-by-step loop is built on `CurationEvaluation`, which scores a
# **committed** curation in that curation's OWN unit namespace (a merged unit is
# scored over its merged template, not inherited from a contributor):
#
# 1. **Evaluate** the committed curation — `CurationEvaluation` walks its
#    analyzer, computes quality metrics, and proposes labels from a rule set.
# 2. **Accept** the proposals into a committed child — `use_evaluation_labels` writes
#    the evaluation's label verdict (the final-metrics path); `accept_evaluation_outputs`
#    accepts explicit merges too.
# 3. **Manually merge** oversplit clusters (MS4/MS5 oversplit and don't track
#    drift) — find burst pairs with `plot_by_sort_group_ids` /
#    `investigate_pair_*`, then merge them with `create_merged_curation`.
# 4. **Re-evaluate the merged curation** — merging changes each unit's template
#    (and so its SNR / ISI-violation fraction / PC-NN separation), so metrics
#    over the *post-merge* templates are the numbers of record.

root_key = {
    "sorting_id": run_summary["sorting_id"],
    "curation_id": run_summary["root_curation_id"],
}
CurationV2.summarize_curation(root_key)

# ### 3-browser. Curate in a browser with FigPack
#
# To inspect units in a point-and-click view, `build_figpack_view=True` publishes a FigPack
# curation view of the root curation and returns its location in `figpack_uri`.
# This needs the optional FigPack packages (the `spikesorting-v2-curation`
# extra); without them `run_v2_pipeline(build_figpack_view=True)` raises, so the cell below
# runs only when they are installed.
#
# Offline, the view is a self-contained static bundle on disk — open `index.html`
# from `figpack_uri` in a browser to inspect the sort. Edits made to a static
# bundle in the browser are **not** written back automatically. To bring labels
# and merges from a FigPack figure into a Spyglass curation, read them with
# `FigPackCuration.fetch_curation_from_uri(uri)` and commit them with
# `CurationV2.save_manual_curation(...)` (the round-trip cell below). A
# never-edited figure reads back empty.

import importlib.util

figpack_available = (
    importlib.util.find_spec("figpack") is not None
    and importlib.util.find_spec("figpack_spike_sorting") is not None
)
figpack_summary = None
if figpack_available:
    figpack_summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
        pipeline_preset=pipeline_preset,
        build_figpack_view=True,
    )
    print("FigPack view:", figpack_summary["figpack_uri"])
else:
    print(
        "FigPack extra not installed; skipping. Install the "
        "'spikesorting-v2-curation' extra to publish a browser curation view."
    )

# Round-trip: read the figure's annotation state, then commit it as the next
# curation. After labeling/merging in the browser, re-run this to import the
# edits (here, on a fresh sort, the figure reads back empty).

if figpack_summary is not None:
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration

    labels, merge_groups = FigPackCuration.fetch_curation_from_uri(
        figpack_summary["figpack_uri"]
    )
    print("from figure — labels:", labels, "| merge groups:", merge_groups)
    # With real edits, commit them as a CHILD of the root curation (pass the
    # root's root_curation_id as the parent — the default parent_curation_id=-1
    # would try to re-create the root and raise). `merge_action="commit"`
    # applies the browser's merges into the child's unit set; use `"preview"`
    # instead to store them as proposals you review in section 3 first.
    # CurationV2.save_manual_curation(
    #     {"sorting_id": figpack_summary["sorting_id"]},
    #     parent_curation_id=figpack_summary["root_curation_id"],
    #     labels=labels,
    #     merge_groups=merge_groups,
    #     merge_action="commit",
    #     curation_source="figpack",
    #     description="curated in FigPack",
    # )

# ### 3a. Evaluate the root curation (pass 1)
#
# Pair the (committed) root curation with a quality-metric recipe and an
# auto-curation rule set, then populate. `franklab_default` computes `snr` /
# `isi_violation` / `firing_rate` / `num_spikes` / `presence_ratio` /
# `amplitude_cutoff` / `nn_advanced` (PCA); `franklab_default_auto_curation_2026_06`
# labels `nn_noise_overlap > 0.1` units `noise` and `isi_violation > 0.02` (>2%
# refractory violations) `reject`. `plot_units_qc` is the population
# "do these units look reasonable?" view; `get_metrics` returns the per-unit
# table.
#
# `populate` is the heavy step — it grows the analyzer extensions and builds the
# whitened PCA analyzer, so it can take minutes on a real session (it is
# idempotent, so a re-run reuses the result instead of recomputing).

eval_sel = CurationEvaluationSelection.insert_selection(
    {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["root_curation_id"],
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
    }
)
CurationEvaluation.populate(eval_sel)
CurationEvaluation().plot_units_qc(eval_sel)
CurationEvaluation.get_metrics(eval_sel)

# #### Reading the labels: proposals to verify, not verdicts
#
# The rule set proposes labels from thresholds — a starting point, not a
# verdict. Before trusting a `noise` / `reject` tag, cross-check the unit:
#
# - **Refractory dip** — `CurationEvaluation().plot_correlograms(eval_sel)`: a
#   real single unit has a central dip in its autocorrelogram; a symmetric,
#   dip-free autocorrelogram is a noise signature.
# - **The interneuron trap** — a fast-spiking interneuron raises ISI violations
#   and sits just around the `nn_noise_overlap` noise threshold. If its waveform
#   is narrow, its refractory dip clean, and its firing stable, it is a cell, not
#   noise — don't reject it on the metric alone.
# - **Amplitude over time** — `CurationEvaluation().get_peak_amps(eval_sel)`: a
#   slow, smooth amplitude drift across a place-field traversal is a place cell,
#   not multi-unit activity; sharp amplitude steps or several distinct bands are
#   MUA.
#
# These pattern-recognition cues are calibrated for **hippocampal tetrodes and
# polymer probes**; other regions (cortex, thalamus, striatum) have different
# waveform widths and bursting, so the thresholds and tells above can mislead
# there.

# ### 3b. Find burst pairs to merge, then accept the auto labels
#
# `plot_by_sort_group_ids` scatters waveform similarity vs cross-correlogram
# asymmetry, one point per unit pair — high-similarity, asymmetric pairs are
# merge candidates (drill into a pair with `investigate_pair_xcorrel` /
# `investigate_pair_peaks`). Hippocampal pyramidal cells fire complex-spike
# bursts with an amplitude decrement that MountainSort oversplits into a parent +
# a shorter-waveform daughter — exactly the high-similarity, short-lag-asymmetric
# pair this scatter surfaces, so merging them reassembles one cell.
# `use_evaluation_labels` then writes the evaluation's label verdict into a committed
# child you merge on top of (it CLEARS any earlier label the evaluation no
# longer proposes — the authoritative "use the evaluation's labels" path; use
# `overlay_evaluation_labels` instead to KEEP existing labels and only add the proposed
# ones).

CurationEvaluation().plot_by_sort_group_ids(eval_sel)
labeled_curation = CurationEvaluation().use_evaluation_labels(eval_sel)
labeled_curation  # {"sorting_id", "curation_id"} of the auto-labeled child

# ### 3c. Manual merge, then the final evaluation pass (pass 2)
#
# List the burst pairs you decided to merge in step 3b in `merge_groups_to_apply`
# (each a list of ≥2 unit ids, e.g. `[[3, 7]]`). It starts EMPTY so a run-all
# never merges arbitrary units — fill it in after inspecting 3b, then re-run.
# When you merge, `create_merged_curation` (intent-first sugar over
# `insert_curation` with `apply_merge=True`) branches off the auto-labeled
# curation, and `CurationEvaluation` runs once more on the MERGED curation: the
# metrics over the post-merge templates are the final numbers of record. The
# analyzer-backed plots (`plot_units_qc`, `plot_correlograms`, ...) are
# raw-unit-curation only and RAISE on a merged curation, so read the merged
# result with `get_metrics` (it carries the curation's own merged namespace).
# `use_evaluation_labels` commits those final labels so downstream code keys off the
# curated result, not the uncurated root curation. (Leaving the list empty keeps
# the auto-labeled curation from 3b as the result — no merge applied.)

merge_groups_to_apply = []  # e.g. [[3, 7]] after inspecting step 3b

if merge_groups_to_apply:
    merged = CurationV2.create_merged_curation(
        sorting_key={"sorting_id": run_summary["sorting_id"]},
        merge_groups=merge_groups_to_apply,
        parent_curation_id=labeled_curation["curation_id"],
        description="manual burst-pair merge",
        reuse_existing=True,
    )
    final_eval_sel = CurationEvaluationSelection.insert_selection(
        {
            "sorting_id": run_summary["sorting_id"],
            "curation_id": merged["curation_id"],
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
        }
    )
    CurationEvaluation.populate(final_eval_sel)
    display(CurationEvaluation.get_metrics(final_eval_sel))  # merged templates
    final_curation = CurationEvaluation().use_evaluation_labels(final_eval_sel)
else:
    final_curation = labeled_curation  # no manual merge yet; use the 3b result
    final_eval_sel = eval_sel  # the curation-evaluation selection of record

final_summary = CurationV2.summarize_curation(final_curation)
final_merge_id = final_summary["merge_id"]
final_summary

# ### 3d. Surface waveform shape for cell typing (your thresholds, not the pipeline's)
#
# Over the final curated result (`final_eval_sel` from section 3c — post-merge if you
# merged, the pass-1 auto curation otherwise), `get_metrics` returns a
# waveform-shape column next to the quality metrics: `trough_half_width` — the
# half-amplitude width of the spike trough, in seconds, read from the unwhitened
# display analyzer. Narrow spikes are fast-spiking interneurons, wide spikes
# pyramidal cells. Paired with `firing_rate` (already a quality metric) it gives
# the classic rate × width view for separating putative cell types.
#
# **The pipeline ships NO cell-type thresholds.** The boundary below is **your
# own** — region-specific and tuned here for hippocampus (fast + narrow ⇒
# putative interneuron; slow + wide ⇒ putative pyramidal). For cortex, striatum,
# or thalamus the cutoffs differ; pick your own, do not reuse these. (The
# trough-to-peak duration and slope columns are available by adding them to the
# metric row's `template_metric_columns`, but they are not defaults: they need a
# wider post-trough window than the hippocampus display recipe's 0.5 ms and clip
# there — they are reliable on the wider cortex/fallback window.)

shape = CurationEvaluation.get_metrics(final_eval_sel)[
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
ax.set_title(
    "Putative cell types — YOUR hippocampal thresholds, not the pipeline's"
)
ax.legend()
print(
    f"{int(is_interneuron.sum())} putative interneuron(s), "
    f"{int((~is_interneuron).sum())} putative pyramidal"
)

# ### 3e. Inspect with the SpikeInterface bridge (`ssviz`)
#
# The curation views above are lab-specific. For general inspection there is one
# discoverable namespace — `visualization` (import it as `ssviz`) — that wraps
# SpikeInterface's own widgets/exporters behind Spyglass keys, so you
# tab-complete a single module instead of hunting for plot methods across
# `Recording`, `Sorting`, and `CurationEvaluation`. `available_visualizations()`
# catalogs every visualization/export helper with a one-line description, the key
# it takes, what it wraps, and whether it can fill in a missing analyzer
# extension (the `recording_key_for_sorting` convenience used below is a key
# resolver, not a plot, so it is not in the catalog).
#
# Routing is automatic and matters: recording widgets read the saved
# **preprocessed** recording; sorting/waveform/location widgets read the sort's
# **display** (unwhitened) analyzer, so you see real µV waveforms and real probe
# positions — never the whitened metric analyzer. `plot_metrics` plots the
# routed `CurationEvaluation.get_metrics()` table (the same numbers as section 3),
# while the raw SpikeInterface metric widgets are separately named
# (`plot_si_quality_metrics` / `plot_si_template_metrics`) and read analyzer
# extensions directly. `plot_suggested_merges` shows the **persisted**
# `get_suggested_merge_groups()` suggestions and never recomputes candidates at
# plot time.

from spyglass.spikesorting.v2 import visualization as ssviz

ssviz.available_visualizations()

# Plot helpers are read-only by default: a richer widget whose display-safe
# extension has not been computed yet raises a clear error naming the
# `add_extensions(...)` call; pass `compute_missing=True` to compute only that
# display-safe extension first (as `plot_unit_summary` does below for
# `unit_locations`). Most plot helpers default to local `matplotlib`; SI widgets
# without a matplotlib backend expose that explicitly (`plot_sorting_summary`
# requires `backend="spikeinterface_gui"`, `backend="sortingview"`, or
# `backend="figpack"`, while `plot_suggested_merges` defaults to notebook-local
# `ipywidgets`). No step here uploads or publishes anything.
# `plot_recording_probe_map(recording_key)` rounds out the recording view (pass a
# 3D `ax=` for a probe with z-coordinates), and
# `ssviz.export_si_report(sorting_key, folder, force_computation=True)` /
# `ssviz.export_to_phy(sorting_key, folder)` write a local SI report / Phy folder
# off the display analyzer. To label and merge in a browser instead, publish a
# FigPack curation view with `run_v2_pipeline(build_figpack_view=True)` (section 3-browser).

# +
sorting_key = {"sorting_id": run_summary["sorting_id"]}
# The facade resolves the saved recording's key for you (no hunting through
# SortingSelection); the recording widgets take that recording_key.
recording_key = ssviz.recording_key_for_sorting(sorting_key)

ssviz.plot_recording_traces(recording_key, time_range=[0.0, 1.0])

unit_ids = list(Sorting().get_sorting(sorting_key).get_unit_ids())
if unit_ids:
    ssviz.plot_unit_summary(sorting_key, unit_ids[0], compute_missing=True)

ssviz.plot_metrics(final_eval_sel)  # the routed Spyglass metric table, plotted

# ### 3f. Hand-label specific units
#
# When you disagree with a rule-set proposal — or want to tag a unit the rules
# don't (e.g. an oversplit fragment as `mua`) — `save_manual_curation` commits
# your own per-unit labels as a child of the curated result. Keys are unit ids;
# values are labels from `CurationV2.label_options()` (`noise` / `mua` /
# `accept` / ...); pass `allow_custom_labels=True` for a label outside that set.
# `manual_labels` starts EMPTY so a run-all is a no-op — fill it in after
# inspecting the units above, then re-run.

manual_labels = {}  # e.g. {5: "noise", 12: "mua"} after inspecting the units
curated_merge_id = final_merge_id  # default: the section-3c result

if manual_labels:
    hand_labeled = CurationV2.save_manual_curation(
        {"sorting_id": run_summary["sorting_id"]},
        parent_curation_id=final_curation["curation_id"],
        labels=manual_labels,
        description="manual per-unit labels",
    )
    hand_summary = CurationV2.summarize_curation(hand_labeled)
    curated_merge_id = hand_summary["merge_id"]
    display(hand_summary)

# ### 3g. Use the curated result downstream
#
# The curated units are consumed like any Spyglass spike-sorting output: pull
# spike times from `SpikeSortingOutput`, keyed by the curated `merge_id` (from
# section 3c, or your hand-labeled child above) — NOT the uncurated root. Every
# existing consumer (decoding, ripple detection, `SortedSpikesGroup`) works on
# this `merge_id` unchanged.

from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

curated_spike_times = SpikeSortingOutput().get_spike_times(
    {"merge_id": curated_merge_id}
)
print(f"{len(curated_spike_times)} curated unit(s)")
curated_spike_times
