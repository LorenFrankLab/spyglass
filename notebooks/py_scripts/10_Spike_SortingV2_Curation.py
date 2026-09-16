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
# sort to get a root curation, then walks ONE reviewed result through the
# browser: open the **FigPack** review, label and merge, save, preview and
# commit, verify the merged units, and send exactly that curation to
# analysis. A scripted **evaluate → merge → re-evaluate** appendix (opt-in)
# serves automation and debugging without touching the browser result.
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
from spyglass.spikesorting.v2.curation_api import (
    save_manual_curation,
)
from spyglass.spikesorting.v2.pipeline import (
    describe_sort_groups,
    plot_sort_group_geometry,
    preflight_v2_pipeline,
    run_v2_pipeline,
    select_units_for_analysis,
)
from spyglass.spikesorting.v2.recording import SortGroupV2
from spyglass.spikesorting.v2.sorting import Sorting  # noqa: F401

dj.config["display.limit"] = 12

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
review_profile = "franklab_hippocampus_2026_06"
# Keep run-all/headless execution safe. Set True interactively when ready.
open_review_in_browser = False
commit_browser_review = False  # commit the previewed browser edits
commit_merge_verification = False  # commit the post-merge verification review
# Analysis population policy for the final handoff (section 4).
analysis_policy = "v2_accepted_single_units"
# Optional executable example for typed, curation-scoped custom properties.
run_custom_annotation_example = False
# Opt-in scripted curation appendix (separate result names; never overwrites
# the browser result).
run_scripted_curation_example = False
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
# `run_v2_pipeline` leaves you a typed `root_curation` pinned to one immutable
# generation. `summarize_curation` describes **one** curation and returns a
# plain dict (`n_units`, `labels`, `merge_groups`, `merges_applied`,
# `is_merge_preview`, `merge_id`, ...). The labels curation *accepts* are the
# canonical set `CurationV2.label_options()`; custom labels need
# `allow_custom_labels=True`.
#
# The main path is the browser review (3-browser): one review profile fixes the
# evaluation recipes, the metric columns shown, the label palette and the
# import mode; you label and merge in FigPack, save, then preview and commit in
# Python. Two alternatives stay explicit: `run_v2_pipeline(auto_curate=True)`
# commits a rule set's labels without review (see the first-sort notebook), and
# the scripted loop in Appendix A (opt-in) for automation and debugging.
#
# Whatever path you take, the notebook ends with ONE deliberately named result,
# `final_curation`, and section 4 hands exactly that curation to analysis.

root_curation = run_summary.root_curation
CurationV2.summarize_curation(root_curation.as_key())

# ### 3-annotations. Optional typed custom unit properties
#
# Lab-specific computed values live in immutable annotation sets tied to this
# exact curation. They are not labels and do not affect curation identity or the
# final `merge_id`. The common reader never picks a latest evaluation/set: both
# selections are explicit. Set `run_custom_annotation_example=True` to execute
# this small example and include its column in the review's unit table.

root_annotation_sets = []
if run_custom_annotation_example:
    import pandas as pd

    from spyglass.spikesorting.v2.unit_annotation import (
        CurationUnitAnnotationSet,
        UnitAnnotationDefinition,
        read_unit_properties,
    )

    annotation_definition = UnitAnnotationDefinition.insert_definition(
        "custom_score",
        1,
        "float",
        physical_unit="a.u.",
        description="Example lab-specific score",
    )
    unit_rows = (CurationV2.Unit & root_curation.as_key()).fetch(
        "unit_id", "n_spikes", as_dict=True, order_by="unit_id"
    )
    scale = max(1, max((int(row["n_spikes"]) for row in unit_rows), default=1))
    custom_values = pd.DataFrame(
        {"custom_score": [float(row["n_spikes"]) / scale for row in unit_rows]},
        index=pd.Index(
            [int(row["unit_id"]) for row in unit_rows], name="unit_id"
        ),
    )
    custom_annotation = CurationUnitAnnotationSet.from_dataframe(
        root_curation,
        annotation_definition,
        custom_values,
        producer="curation-notebook-example",
        producer_version="1",
        producer_parameters={"normalization": "max_n_spikes"},
    )
    root_annotation_sets = [custom_annotation]
    display(
        read_unit_properties(
            root_curation,
            evaluation=None,
            annotation_sets=root_annotation_sets,
        )
    )

# ### 3-browser. Review in a browser with FigPack
#
# `start_review` resolves the profile, evaluates (or reuses) the exact root,
# seeds its committed labels, and saves a local review bundle
# (`upload=False`, the default). `review.open()` serves that bundle from this
# kernel at a `http://localhost:<port>/` URL -- the exact files the importer
# reads, nothing copied -- and returns the URL (`open_browser=False` only
# prints it). Repeated `open()` calls reuse the server; after a kernel
# restart, `FigPackReview.resume(review_id).open()` serves the same bundle
# again with every saved edit intact.
#
# In the browser:
#
# 1. **Curate Figure** (bottom bar) enables editing.
# 2. Select units in the unit table. Its columns are the profile's official
#    evaluation metrics, any annotation columns, the rule set's
#    `proposed_labels` / `proposed_merge_groups`, and `merged_from` (applied
#    merge provenance) -- all for the **committed curation being reviewed**; a
#    merge you propose here has no merged metrics yet (the continued review
#    after commit shows those).
# 3. In the **Curation** pane, tick/untick labels for the selected units, or
#    select two or more and **Merge Selected**.
# 4. **Save Annotations**. (**Finalize Curation** is only a browser state
#    flag: it neither saves nor commits anything to Spyglass.)
# 5. Back here: preview, then commit.
#
# Remote kernel: forward the port (`ssh -L <port>:localhost:<port> host`) and
# open the same `localhost` URL on your machine -- the frontend enables local
# editing only for a `localhost` origin; a generic Jupyter proxy URL is not
# equivalent. `upload=True` publishes the identical seeded bundle to
# figpack.org instead (needs `FIGPACK_API_KEY` unless `ephemeral=True`). The
# optional `spikesorting-v2-curation` extra is required for this path.

import importlib.util

figpack_available = (
    importlib.util.find_spec("figpack") is not None
    and importlib.util.find_spec("figpack_spike_sorting") is not None
)
review = None
if figpack_available:
    review = run_summary.start_review(
        source="root",
        profile=review_profile,
        upload=False,
        annotation_sets=root_annotation_sets,
    )
    review_url = review.open(open_browser=open_review_in_browser)
    print("Review id:", review.review_id)
    print("Bundle:", review.uri)
    print("Open in a browser:", review_url)
else:
    print(
        "FigPack extra not installed; skipping the browser review. Install "
        "the 'spikesorting-v2-curation' extra to review in a browser."
    )

# After **Save Annotations**, re-run from here. The preview is mutation-free:
# it reads the saved `annotations.json`, re-verifies the review identity, and
# reports the diff -- label additions/removals, proposed merges, the resulting
# unit count, contributor-label conflicts a merge would create, and sibling
# children other reviewers committed meanwhile. Losing the Python variable is
# harmless: resume from the printed review id.

browser_changes = None
if review is not None:
    from spyglass.spikesorting.v2.review_api import FigPackReview

    review = FigPackReview.resume(review.review_id)
    browser_changes = review.preview_import()
    print(
        browser_changes.next_step()
    )  # saved-not-committed / nothing to commit
    print(browser_changes.summary())
    display(browser_changes.changed_units())

# Commit exactly the previewed diff. Leave `commit_browser_review=False` during
# run-all and until you have inspected the summary above. If
# `label_conflicts` is non-empty, map every predicted merged unit id to the
# labels it should carry, e.g. `conflict_resolutions = {12: ("accept",)}`. A
# review with no edits is committed with `confirm_no_changes=True`, which
# records the parent as explicitly verified (a new child with identical
# content).
#
# Every imported merge is re-evaluated with the same profile, and the receipt
# says `needs_merge_verification=True`: the merged units have NOT been looked
# at yet, so the merged child is NOT the result -- `final_curation` becomes
# `None` (pending) until the verification below is committed, and the
# analysis section refuses to run on a pending result. Until any commit
# lands, `final_curation` is the reviewed parent itself.

final_curation = root_curation if review is None else review.parent
browser_receipt = None
pending_verification = None  # a review over merged units awaiting your look
if browser_changes is not None and commit_browser_review:
    conflict_resolutions = {}
    browser_receipt = browser_changes.commit(
        conflict_resolutions=conflict_resolutions,
        confirm_no_changes=not browser_changes.has_changes,
    )
    print(browser_receipt.next_step())  # awaiting verification / available
    print("Merge id:", browser_receipt.curation.merge_id)
    if browser_receipt.needs_merge_verification:
        # Continue into a seeded review over the ACTUAL merged waveforms /
        # correlograms / metrics, not the pre-merge contributors. Opening is
        # non-blocking: the result stays pending until you commit the look.
        pending_verification = browser_receipt.continue_review()
        final_curation = None
        print(
            "Merged units await verification -- inspect:",
            pending_verification.open(open_browser=open_review_in_browser),
        )
    else:
        final_curation = browser_receipt.curation

# Inspect the merged units in that second review (their `merged_from` column
# names the contributors). If they look right, save nothing, THEN set
# `commit_merge_verification=True` and run this cell: it previews and commits
# the verification (`confirm_no_changes` when you saved nothing). If that
# commit imports another merge, the result stays pending and a further review
# opens -- inspect it and run this cell again. Nothing marks a merge verified
# silently.
#
# If a committed merge was WRONG, do not label around it: a committed merge
# is a branch. Go back to the parent's review -- `FigPackReview.find(
# root_curation, profile=review_profile)` returns it with every edit you
# saved (a fresh `start_review` would begin a new review now that a child
# exists) -- select the merged units, **Unmerge Selected**, **Save
# Annotations**, and preview/commit again: the replacement sibling keeps your
# labels and the abandoned merged branch stays as history (the preview lists
# it under `newer_sibling_curations`). Recovery is spelled out in the
# reference ("Where am I, and how do I undo a merge?").

# +
if pending_verification is not None and commit_merge_verification:
    verification = FigPackReview.resume(
        pending_verification.review_id
    ).preview_import()
    print(verification.summary())
    verification_receipt = verification.commit(
        confirm_no_changes=not verification.has_changes
    )
    print(verification_receipt.next_step())
    if verification_receipt.needs_merge_verification:
        pending_verification = verification_receipt.continue_review()
        print(
            "New merges were imported and await verification -- inspect:",
            pending_verification.open(open_browser=open_review_in_browser),
            "then run this cell again.",
        )
    else:
        pending_verification = None
        final_curation = verification_receipt.curation
        print("Verified curation:", final_curation.curation_id)


def require_verified_result():
    """Stop here while a merged curation is still awaiting verification."""
    if final_curation is None:
        raise RuntimeError(
            "final_curation is pending: merged units in review "
            f"{pending_verification.review_id} have not been verified. "
            "Inspect them, set commit_merge_verification=True, and re-run "
            "the verification cell before labeling or selecting units."
        )


# -


# ### 3-hand-label. Override specific units (optional)
#
# When you disagree with a label after the fact -- or want to tag a unit the
# review did not (e.g. an oversplit fragment as `mua`) -- `save_manual_curation`
# commits your per-unit labels as a child of `final_curation`, and that child
# becomes the final result. Keys are unit ids of `final_curation`; values are
# labels from `CurationV2.label_options()` (pass `allow_custom_labels=True` for
# one outside that set). `manual_labels` starts EMPTY so a run-all is a no-op.

manual_labels = {}  # e.g. {5: ["noise"], 12: ["mua"]} after inspecting

if manual_labels:
    require_verified_result()
    final_curation = save_manual_curation(
        parent_curation=final_curation,
        labels=manual_labels,
        description="manual per-unit labels",
    )
    display(CurationV2.summarize_curation(final_curation.as_key()))

# ## 4. Send the reviewed result to analysis
#
# `final_curation` is the one result of this notebook: the reviewed parent
# when nothing was committed, the committed browser child, its verified merged
# child, or your hand-labeled child on top -- never "the latest child".
# `select_units_for_analysis` applies a named `UnitSelectionParams` policy to
# exactly that curation's labels, builds the `SortedSpikesGroup` downstream
# consumers read, and returns a receipt naming the source generation, the
# policy content, and every included / excluded unit with its reason:
#
# | Policy | Include | Deny |
# | --- | --- | --- |
# | `v2_accepted_single_units` (default) | `accept` | `mua`, `noise`, `reject`, `artifact` |
# | `v2_accepted_neural_units` | `accept` or `mua` | `noise`, `reject`, `artifact` |
# | `v2_unflagged_units` | everything not denied (MUA + unlabeled) | `noise`, `reject`, `artifact` |
# | `all_units` | everything (explicit expert choice) | -- |
#
# Rule sets only flag bad units and never write `accept`, so an unreviewed or
# automatic-only curation selects nothing under the `accepted` policies -- the
# receipt says so and names the alternatives. This cell stops while a merged
# curation is still pending verification.

require_verified_result()
final_summary = CurationV2.summarize_curation(final_curation.as_key())
final_merge_id = final_curation.merge_id
print(
    f"final_curation: curation {final_curation.curation_id} of sorting "
    f"{final_curation.sorting_id} (merge_id {final_merge_id})"
)
selection = select_units_for_analysis(final_curation, policy=analysis_policy)
print(selection.summary())  # source curation, policy, counts, why if empty
display(selection.describe())  # per-unit verdict, labels, reason

# The receipt's group is the downstream handle: `fetch_spike_data` returns one
# spike-time array per SELECTED unit through the same `SortedSpikesGroup` API
# decoding uses. `SpikeSortingOutput().get_spike_times({"merge_id": ...})`
# would return every unit, labels ignored.

spike_times, selected_unit_ids = selection.fetch_spike_data(
    return_unit_ids=True
)
print(f"{len(spike_times)} selected unit(s):", selected_unit_ids)
selection.group_key

# ## 5. Inspect with the SpikeInterface bridge (`ssviz`)
#
# One discoverable namespace -- `visualization` (import it as `ssviz`) -- wraps
# SpikeInterface's widgets/exporters behind Spyglass keys.
# `available_visualizations()` catalogs every helper with the key it takes,
# what it wraps, and whether it accepts `compute_missing=True`.
#
# Routing matters: recording widgets read the saved **preprocessed** recording;
# unit-level widgets take an exact **curation** (`CurationRef`) and read its
# **display** (unwhitened) analyzer, so a merged child shows its merged units.
# Plot helpers are read-only by default: a widget whose display-safe extension
# is missing raises a clear error; `compute_missing=True` computes only that
# extension on the analyzer this curation's plots read. `plot_metrics` plots the
# routed `CurationEvaluation.get_metrics()` table (the review's official
# numbers). `plot_suggested_merges` shows the **persisted** suggestions and
# never recomputes candidates. `ssviz.export_si_report(curation, folder,
# compute_missing=True)` / `ssviz.export_to_phy(curation, folder)` write a local
# SI report / Phy folder of exactly that curation's units.

from spyglass.spikesorting.v2 import visualization as ssviz

ssviz.available_visualizations()

# +
sorting_key = {"sorting_id": run_summary["sorting_id"]}
recording_key = ssviz.recording_key_for_sorting(sorting_key)
ssviz.plot_recording_traces(recording_key, time_range=[0.0, 1.0])

# Unit-level plots take the exact final curation, so a merged unit is plotted
# as the merged unit.
require_verified_result()
final_unit_ids = list(
    CurationV2.get_sorting(final_curation.as_key()).get_unit_ids()
)
if final_unit_ids:
    ssviz.plot_unit_summary(
        final_curation, final_unit_ids[0], compute_missing=True
    )
# -

# ## Appendix A. Scripted evaluate → merge → re-evaluate (opt-in)
#
# The same facade drives an automation/debugging loop without a browser. It
# uses its OWN result names (`scripted_*`) and never changes `final_curation`
# above, so running these cells after the browser path cannot replace the
# reviewed result. Set `run_scripted_curation_example=True` to execute it.
#
# `CurationRef.evaluate` scores a **committed** curation in that curation's own
# unit namespace (a merged unit is scored over its merged template):
#
# 1. **Evaluate** the root -- metrics plus a rule set's label proposals.
#    `franklab_default` computes `snr` / `isi_violation` / `firing_rate` /
#    `num_spikes` / `presence_ratio` / `amplitude_cutoff` / `nn_advanced` (PCA);
#    `franklab_default_auto_curation_2026_06` labels `nn_noise_overlap > 0.1`
#    units `noise` and `isi_violation > 0.02` `reject`. `populate` is the heavy
#    step (minutes on a real session; idempotent).
# 2. **Inspect** proposals with the plot accessors. Proposals are thresholds, not
#    verdicts: check the refractory dip (`plots.correlograms()`), the
#    fast-spiking-interneuron trap (narrow waveform + clean dip + stable firing
#    is a cell), and amplitude over time (`plots.peak_over_time(pairs)`; smooth
#    drift is a place cell, bands are MUA). These cues are calibrated for
#    hippocampal tetrodes / polymer probes.
# 3. **Find burst pairs**: `plots.burst_pair_metrics()` scatters waveform
#    similarity vs cross-correlogram asymmetry; MountainSort oversplits a
#    complex-spike burst into a parent + shorter daughter, exactly the
#    high-similarity, short-lag-asymmetric pair to merge.
# 4. **Merge and re-evaluate**: `merge_and_evaluate(groups)` commits the merge
#    child and evaluates its actual merged templates; `accept_labels` commits
#    the rule verdict.

scripted_curation = None
if run_scripted_curation_example:
    scripted_evaluation = root_curation.evaluate(
        metric_params_name="franklab_default",
        auto_curation_rules_name="franklab_default_auto_curation_2026_06",
    )
    scripted_evaluation.plots.units_qc()
    display(scripted_evaluation.metrics)
    scripted_evaluation.plots.burst_pair_metrics()
    display(
        scripted_evaluation.burst_pair_metrics()
        .sort_values("wf_similarity", ascending=False)
        .head()
    )
    display(scripted_evaluation.proposed_labels)

    # Fill in after inspecting (each a list of >=2 unit ids, e.g. [[3, 7]]);
    # EMPTY so a run-all never merges arbitrary units.
    scripted_merge_groups = []
    if scripted_merge_groups:
        scripted_receipt = scripted_evaluation.merge_and_evaluate(
            scripted_merge_groups
        )
        scripted_evaluation = scripted_receipt.evaluation
        display(scripted_evaluation.metrics)  # actual merged templates
    scripted_curation = scripted_evaluation.accept_labels(mode="replace")
    scripted_merge_id = scripted_curation.merge_id
    display(CurationV2.summarize_curation(scripted_curation.as_key()))
    scripted_evaluation.plots.metrics()

# ### Appendix B. Waveform shape for cell typing (your thresholds)
#
# Over a scripted evaluation, `.metrics` carries `trough_half_width` (seconds,
# from the unwhitened display analyzer) next to the quality metrics: narrow
# spikes are fast-spiking interneurons, wide spikes pyramidal cells; with
# `firing_rate` it gives the classic rate x width view. **The pipeline ships NO
# cell-type thresholds** -- the boundary below is yours, tuned for hippocampus;
# other regions need their own. (Trough-to-peak duration and slope columns are
# available via the metric row's `template_metric_columns` but clip on the
# hippocampus display window.)

if run_scripted_curation_example:
    import matplotlib.pyplot as plt

    shape = scripted_evaluation.metrics[
        ["firing_rate", "trough_half_width"]
    ].dropna()
    rate_cut_hz, width_cut_s = 7.0, 0.0003  # YOUR thresholds, 0.3 ms
    is_interneuron = (shape["firing_rate"] > rate_cut_hz) & (
        shape["trough_half_width"] < width_cut_s
    )
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
    ax.set_title("Putative cell types -- YOUR thresholds, not the pipeline's")
    ax.legend()
    print(
        f"{int(is_interneuron.sum())} putative interneuron(s), "
        f"{int((~is_interneuron).sum())} putative pyramidal"
    )
