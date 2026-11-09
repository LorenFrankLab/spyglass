# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (spyglass_spikesorting_v2)
#     language: python
#     name: python3
# ---

# # Spike Sorting v2: Curation
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
import numpy as np
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
pipeline_preset = "franklab_probe_hippocampus_30khz_ms4_2026_06"
review_profile = "franklab_hippocampus_2026_09_17"
# Keep run-all/headless execution safe. Set True interactively when ready.
open_review_in_browser = False
use_connected_browser_result = (
    False  # True after completing the local browser review
)
use_notebook_commit_panel = (
    False  # optional alternative, required for hosted drafts
)
commit_browser_review = False  # commit the previewed browser edits
commit_merge_verification = False  # commit the post-merge verification review
# Analysis population policy for the final handoff (section 4).
analysis_policy = "v2_accepted_single_units"
inspection_unit_ids = []  # set after looking at the review, e.g. [3, 7]
inspection_pair = []  # exactly two final-curation IDs to investigate
snr_threshold = 5.0  # example analysis predicate: choose for your experiment
# Opt-in scripted curation appendix (separate result names; never overwrites
# the browser result).
run_scripted_curation_example = False
commit_recovery = False  # commit the previewed correction in section 3-recover
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
# Run a sort so we have a root curation to work with (idempotent; reuses an
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
# A `CurationRef` handle also describes its place in the lineage:
#
# - `commit_status` is `"preview"` or `"committed"`. `is_root`, `is_leaf`, and
#   `has_committed_children` are independent. `operation_type` reports the
#   producer and change kind. There is no inferred "superseded" state.
# - `created_at` / `created_by` give creation metadata. `visualize_lineage()`
#   draws the tree.
# - Use `preview_curation_delete()` before `delete_subtree()`. `health_report()`
#   audits lineage and analyzer-cache orphans.
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

# ### 3-browser. Review in a browser with FigPack
#
# `start_review` resolves the profile, evaluates (or reuses) the exact root,
# seeds its committed labels, and saves a local review bundle
# (`upload=False`, the default). `review.open()` serves that bundle from this
# kernel at a `http://localhost:<port>/` URL (the exact files the importer
# reads, nothing copied) and returns the URL (`open_browser=False` only prints
# it). Repeated `open()` calls reuse the server; after a kernel
# restart, `FigPackReview.resume(review_id).open()` serves the same bundle
# again with every saved edit intact.
#
# In the browser:
#
# 1. Local reviews are ready to edit; hosted figures use **Curate Figure**.
# 2. Select units in the unit table. Its columns are the profile's official
#    evaluation metrics, the rule set's `proposed_labels` /
#    `proposed_merge_groups`, and `merged_from` (applied merge provenance).
#    All values describe the **committed curation being reviewed**; a proposed
#    merge has no merged metrics until it is committed and the continued review
#    shows them.
# 3. In the **Curation** pane, tick/untick labels for the selected units, or
#    select two or more and **Merge Selected**.
# 4. **Preview and commit**, choose final labels for any conflicting merge,
#    and commit. **Save draft** keeps unfinished edits.
# 5. If merges were committed, inspect the child that opens and record its
#    review in the browser. **Review parent branch** recovers a mistaken merge.
# 6. Set `use_connected_browser_result=True` after completing the review. The
#    handoff cell below reads `review.result()` and refuses an unfinished review.
#
# Select units and use **Inspect selected units / pairs** for all their CCGs.
# Enter a window for an exact raster and, up to 10 seconds, spikes on traces.
# **Time and sampling** maps display seconds to original session seconds;
# red bands show excluded time. Default raster/amplitude budgets match v1's
# `floor(duration_s * 50)` points per unit. Large time plots load on request.
# Hosted figures save with **Save Annotations** and use the notebook alternative.
#
# Remote kernel: forward the port (`ssh -L <port>:localhost:<port> host`) and
# open the same `localhost` URL on your machine. The frontend enables local
# editing only for a `localhost` origin; a generic Jupyter proxy URL is not
# equivalent. `upload=True` publishes the identical seeded bundle to
# figpack.org instead (needs a FigPack API key unless `ephemeral=True`). The
# optional `spikesorting-v2-curation` extra is required for this path.
# Appendix C lists the remaining review details.

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
    )
    review_url = review.open(open_browser=open_review_in_browser)
    print(review.summary())
    print("Review id:", review.review_id)
    print("Bundle:", review.uri)
    print("Open in a browser:", review_url)
else:
    print(
        "FigPack extra not installed; skipping the browser review. Install "
        "the 'spikesorting-v2-curation' extra to review in a browser."
    )

# Optional notebook/script alternative after **Save draft**. The preview is
# mutation-free: it reads the saved `annotations.json`, re-verifies the review
# identity, and reports the diff: label additions/removals, proposed merges,
# the resulting unit count, contributor-label conflicts a merge would create,
# and sibling children other reviewers committed meanwhile. Losing the Python
# variable is harmless: resume from the printed review id.

# The guided route previews the saved draft, provides label choices for each
# conflicting merge, and opens the reevaluated child after committing. A
# no-change review has an explicit **Record reviewed** button. Run this cell
# after saving a draft, then click its action. Nothing commits during run-all.

panel = (
    review.commit_panel(open_browser=open_review_in_browser)
    if review is not None and use_notebook_commit_panel
    else None
)

# After clicking, run the following cells to collect the receipt. The explicit
# preview/commit API remains available below for scripted workflows.

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
# at yet, so the merged child is NOT the result. `final_curation` is
# `None` (pending) until the verification below is committed, and the
# analysis section refuses to run on a pending result. Until any commit
# lands, `final_curation` is the reviewed parent itself.

final_curation = root_curation if review is None else review.parent
browser_receipt = panel.receipt if panel is not None else None
pending_verification = panel.verification_review if panel is not None else None
if browser_receipt is not None:
    final_curation = (
        None
        if browser_receipt.needs_merge_verification
        else browser_receipt.curation
    )
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

# For the guided route, inspect the opened child, save any edits, and run this
# cell to create its verification action. The same panel supports further merges.

verification_panel = (
    pending_verification.commit_panel(open_browser=open_review_in_browser)
    if pending_verification is not None
    else None
)

# After clicking the verification action, run this cell to select its result.

if verification_panel is not None and verification_panel.receipt is not None:
    verification_receipt = verification_panel.receipt
    pending_verification = verification_panel.verification_review
    final_curation = (
        None
        if verification_receipt.needs_merge_verification
        else verification_receipt.curation
    )

# Inspect the merged units in that second review (their `merged_from` column
# names the contributors). If they look right, save nothing, THEN set
# `commit_merge_verification=True` and run this cell: it previews and commits
# the verification (`confirm_no_changes` when you saved nothing). If that
# commit imports another merge, the result stays pending and a further review
# opens; inspect it and run this cell again. Nothing marks a merge verified
# silently.
#
# If a committed merge was WRONG, do not label around it: a committed merge
# is a branch, and the fix is a replacement sibling from THAT merge's parent
# (a merge committed during verification has the earlier merged child as its
# parent; going back to the root would discard the earlier, valid merge).
# Section 3-recover below does this from the receipt the merge came from.

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


# For the connected local workflow, retrieve its explicitly verified result.
# Set the flag only after completing the browser actions; this accessor never
# silently substitutes the latest child or a still-pending merge.

if use_connected_browser_result:
    final_curation = review.result()
    pending_verification = None

# ### 3-recover. Undo a committed merge that was wrong (opt-in)
#
# Replace `None` below with the receipt that committed the mistaken merge:
# `browser_receipt` for the first commit, or the appropriate
# `verification_receipt` for a later merge. This opens that merge's own
# review, whose bundle holds your other saved edits.

# Replace with browser_receipt or verification_receipt.
bad_merge_receipt = None
recovery_review = None
if bad_merge_receipt is not None:
    recovery_review = bad_merge_receipt.changes.review
    bad_merge = bad_merge_receipt.curation
    print(
        "Open the review to correct:",
        recovery_review.open(open_browser=open_review_in_browser),
    )

# **Stop here to edit.** Select only the mistaken merge's units,
# **Undo selected merge**, and **Save draft**. Other valid merges and
# label edits can stay. After saving, run the next cell to preview the
# correction; it does not commit anything.

recovery_changes = None
if recovery_review is not None:
    recovery_changes = recovery_review.preview_import()
    print(recovery_changes.summary())
    display(recovery_changes.changed_units())

# Inspect the preview, resolve any listed label conflicts using its merged
# unit IDs, then set `commit_recovery=True` and run the next cell. A complete
# undo with no label changes confirms a no-change child; remaining merges
# open a verification review. Inspect that review before rerunning the
# verification cell above. The replacement becomes the notebook's result
# only when no verification is pending.
#
# The abandoned branch stays as history (`newer_sibling_curations`); remove
# it later with `bad_merge.delete_subtree()` once nothing downstream refers
# to it.
#
# - The replacement sibling comes from that merge's parent, not necessarily
#   the root; the review it came from is `receipt.changes.review`.
# - Edits made in the merged child's own verification review live on the
#   abandoned branch; redo them on the replacement.
# - Once nothing refers to the abandoned branch,
#   `bad_merge.preview_curation_delete()` lists it leaf-first and
#   `bad_merge.delete_subtree()` removes it. Delete a `SortedSpikesGroup`
#   built from it first.

recovery_conflict_resolutions = {}  # use the IDs from recovery_changes
if recovery_changes is not None and commit_recovery:
    replacement_receipt = recovery_changes.commit(
        conflict_resolutions=recovery_conflict_resolutions,
        confirm_no_changes=not recovery_changes.has_changes,
    )
    print(replacement_receipt.next_step())
    print(
        f"Replacement curation {replacement_receipt.curation.curation_id} "
        f"(parent {bad_merge.parent.curation_id}); abandoned merge "
        f"{bad_merge.curation_id} kept as history."
    )
    if replacement_receipt.needs_merge_verification:
        pending_verification = replacement_receipt.continue_review()
        final_curation = None
        print(
            "Its remaining merges await verification -- inspect:",
            pending_verification.open(open_browser=open_review_in_browser),
        )
    else:
        pending_verification = None
        final_curation = replacement_receipt.curation

# ### 3-hand-label. Override specific units (optional)
#
# To override a label, or tag a unit the review did not (e.g. an oversplit
# fragment as `mua`), `save_manual_curation` commits your per-unit labels as a
# child of `final_curation`, and that child becomes the final result. Keys are
# unit ids of `final_curation`; values are labels from
# `CurationV2.label_options()` (pass `allow_custom_labels=True` for one outside
# that set). `manual_labels` starts EMPTY so a run-all is a no-op.
#
# The full form is
# `save_manual_curation(parent_curation=..., labels={unit_id: [label, ...]}, merge_groups=[[unit_id, ...], ...], merge_action=...)`.
# It takes integer unit ids. `merge_action="preview"` stores unapplied
# proposals; `"commit"` applies them.

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
# child, or your hand-labeled child on top; never "the latest child".
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
# | `all_units` | everything (explicit expert choice) | (none) |
#
# Rule sets only flag bad units and never write `accept`, so an unreviewed or
# automatic-only curation selects nothing under the `accepted` policies; the
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
# The `visualization` namespace (import it as `ssviz`) wraps SpikeInterface's
# widgets/exporters behind Spyglass keys.
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
    final_curation.plots.unit_summary(final_unit_ids[0], compute_missing=True)
# -

# ### Is this unit neural, noise, or MUA? Does it remain plausible late?
#
# Set `inspection_unit_ids` to suspicious units in the FINAL curation. These
# calls use the unwhitened analyzer in microvolts; compare waveform shape,
# refractory behavior and temporal stability rather than equating a narrow
# waveform or high rate with noise. Inspect short windows at beginning, middle,
# and end, and a raster for those units. SI time ranges below are recording-
# relative seconds; fetched analysis spikes use the original session timeline.

final_evaluation = final_curation.evaluate(
    metric_params_name="franklab_default",
    auto_curation_rules_name="franklab_default_auto_curation_2026_09",
)
display(final_evaluation.missing_qc_inputs().to_frame())
if inspection_unit_ids:
    import spikeinterface.widgets as sw

    final_curation.plots.waveforms(inspection_unit_ids)
    with final_curation.open_analyzer() as analyzer:
        duration = (
            analyzer.recording.get_num_samples() / analyzer.sampling_frequency
        )
        width = min(1.0, duration)
        starts = sorted(
            {0.0, max(0.0, (duration - width) / 2), max(0.0, duration - width)}
        )
        for start in starts:
            final_curation.plots.spikes_on_traces(
                unit_ids=inspection_unit_ids,
                time_range=[start, start + width],
                compute_missing=True,
            )
        sw.plot_rasters(
            analyzer.sorting,
            unit_ids=inspection_unit_ids,
            time_range=[0.0, duration],
            backend="matplotlib",
        )

# ### Should this pair merge?
#
# Pick exactly two IDs after inspection; no automatic merge is performed here.
# Compare targeted cross-correlograms, peak distributions and amplitude over
# time. Use the browser commit/continuation above, then rerun these cells with
# the committed merged unit selected. The same evaluation follows that child.
# Pair ISI diagnostics use the evaluation's refractory window and the same
# violating-interval fraction as unit QC. Commit and reevaluate to score the
# actual merged train, including the merge's duplicate-spike handling.

if inspection_pair:
    if len(inspection_pair) != 2:
        raise ValueError("Choose exactly two final-curation unit IDs.")
    pairs = [tuple(inspection_pair)]
    final_evaluation.plots.pair_correlograms(pairs)
    final_evaluation.plots.pair_peaks(pairs)
    final_evaluation.plots.peak_over_time(pairs)
    display(final_evaluation.burst_pair_metrics(pairs))

# The browser's amplitude sampling cap and pair-similarity threshold limit its
# display. An absent point or pair does not establish absent spikes or absent
# correlation; use the targeted views above. Display limits do not bound every
# analyzer computation or the browser's unit-pair similarity payload.
#
# `isi_violation` is the count of violating intervals / (`num_spikes` - 1), using
# this evaluation recipe's refractory window. It is different from SI's
# `isi_violations_ratio` and is not a contamination estimate. `unavailable_qc`
# names missing inputs needed by the enabled rules. With missing-policy pass,
# a blank metric can leave a unit unflagged; it does not establish quality.

# ### Observed time in metrics and analysis
#
# Artifact intervals are stored in NWB `obs_intervals`. The standard review
# displays `observed_firing_rate_hz`, `observed_presence_ratio`, and
# `observed_duration_s`; raw SI duration metrics retain SI's full-timeline
# definitions. `selection.observation` exposes the frozen population's usable
# time. Sorted-spikes decoding intersects encoding/decoding intervals with that
# time; binned counts distinguish excluded bins (NaN) from observed silence.
#
# - Observation intervals are normalized through sample boundaries, so the last
#   sample contributes its full duration. Time is never compressed across
#   exclusions or gaps.
# - Observed firing rate: spikes in usable spans over their total duration.
# - Observed presence: observed duration of occupied bins over total observed
#   duration. Bins are fixed on the original timeline from the first timestamp.
#   One observed spike occupies a bin. Partial bins count only their usable
#   duration. `QualityMetricParameters.observed_presence_bin_duration_s`
#   defaults to 60 s. This is not SI's presence definition.
# - Zero exposure gives unavailable rate and presence. Observed silence gives
#   zero.
# - Evaluations record the observation-definition version, an interval
#   fingerprint, and the presence-bin width.
# - `select_units_for_analysis` freezes each included unit's observed intervals
#   with the membership. `selection.observation` (one session) or
#   `selection.groups[i].observation` exposes `intervals`, `duration_s`, and
#   `unknown_sources`. The common intervals are the **intersection** of the
#   selected units' availability. Empty or unselected groups impose nothing.
#   Concat members are mapped to original session seconds.

# ### Additional SNR selection for this analysis only
#
# Add analysis-specific metric criteria to the chosen label policy. The exact
# evaluation must belong to `final_curation`; missing metric values fail the
# predicate. The resulting membership is persisted, so decoding and fetching
# use the same units. A good unit need not be labeled `reject` merely because
# it fails one analysis's population preference.

metric_selection = select_units_for_analysis(
    final_curation,
    policy=analysis_policy,
    evaluation=final_evaluation,
    unit_criteria={"snr": {">=": snr_threshold}},
)
filtered_spikes, filtered_unit_ids = metric_selection.fetch_spike_data(
    return_unit_ids=True
)
analysis_provenance = dict(metric_selection.selection_provenance)
print(metric_selection.summary())
analysis_provenance

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
# 1. **Evaluate** the root: metrics plus a rule set's label proposals.
#    `franklab_default` computes `snr` / `isi_violation` / `firing_rate` /
#    `num_spikes` / `presence_ratio` / `amplitude_cutoff` / `nn_advanced` (PCA);
#    `franklab_default_auto_curation_2026_09` labels `nn_noise_overlap > 0.1`
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
#
# Evaluation details:
#
# - Preview curations (`apply_merge=False` with an unapplied merge) are
#   rejected; commit the merge first (`curation.commit_merges(groups)` /
#   `insert_curation(..., apply_merge=True)`).
# - It records the exact recipe pair it used. Its proposals become a committed
#   child; the scored row is never changed.
# - Results are written to NWB and returned as an `EvaluationResult` snapshot.
#   `metrics`, `suggested_merges`, and `proposed_labels` are copies.
# - `evaluation.merge_and_evaluate(groups)` is idempotent and resumable. Run it
#   outside a caller-owned transaction: `populate` manages its own.
#   `evaluation.commit_merges(groups)` commits without re-evaluation.
# - `accept_labels(mode="replace")` makes the evaluation the complete verdict;
#   `mode="overlay"` keeps current labels and adds.
# - `evaluation.plots.units_qc()` (table form `CurationEvaluation.plot_units_qc`)
#   draws a histogram per metric plus a unit-depth scatter; pass `axes=` to
#   embed it.
#
# Table methods (the facade calls these):
#
# - `CurationEvaluation.accept_merges` / `accept_all_suggested_merges` commit the
#   merged unit set and **inherit** the curation's labels. They do not apply
#   pre-merge evaluation labels, which cannot attach to a merged unit.
# - Recommended: accept the merge, re-evaluate the merged child, then label with
#   `use_evaluation_labels` (complete verdict: clears labels the evaluation no
#   longer proposes) or `overlay_evaluation_labels` (adds).
# - `accept_evaluation_outputs` applies merges and labels in one call. Its
#   `labels=None` default applies the pre-merge labels; prefer the action
#   methods unless that is intended.
# - `preview_merges` drafts an unapplied merge. Downstream consumers reject that
#   row until it is committed. Every merge needs a group of two or more units.
#
# Custom rule sets:
#
# - Insert `AutoCurationRules` with `insert_rules(master, rule_rows)`, so the
#   master and its ordered rules validate together. Direct `insert1` is blocked.
# - Expected NaN cases (`expected_missing_units`): `nn_isolation` /
#   `nn_noise_overlap` below `nn_advanced`'s `min_spikes` or `min_fr`;
#   `presence_ratio` for a recording shorter than one bin or a unit with no
#   spikes; `amplitude_cutoff` below
#   `num_histogram_bins * amplitudes_bins_min_ratio` spikes (500 by default);
#   `isi_violation` at one spike or fewer; `firing_rate` with no spikes; `snr`
#   and `num_spikes` never.
# - Any other non-finite rule value raises `ValueError`, including any non-finite
#   value on a column with no registered cases (template, custom, or
#   `observed_*` columns). An error SI catches inside a referenced metric aborts
#   the evaluation; one in an unreferenced metric is logged and its columns stay
#   NaN.
# - The shipped rules use `pass`, because `error` would abort a sort over a
#   legitimately short train. `pass` and `fail` warn when a metric is non-finite
#   for every unit. `pass` means unflagged, not good. The review's
#   `unavailable_qc` column names missing inputs for enabled rules.
# - Metric persistence accepts only numeric scalars (NaN included). Arrays and
#   other objects raise `UnsupportedMetricValueError`.

scripted_curation = None
if run_scripted_curation_example:
    scripted_evaluation = root_curation.evaluate(
        metric_params_name="franklab_default",
        auto_curation_rules_name="franklab_default_auto_curation_2026_09",
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
# `observed_firing_rate_hz` it gives the classic rate x width view. **The
# pipeline ships NO cell-type thresholds**. The boundary below is yours and
# tuned for hippocampus; other regions need their own. (Trough-to-peak duration
# and slope columns are available via the metric row's
# `template_metric_columns` but clip on the hippocampus display window.)

if run_scripted_curation_example:
    import matplotlib.pyplot as plt

    shape = scripted_evaluation.metrics[
        ["observed_firing_rate_hz", "trough_half_width"]
    ].dropna()
    rate_cut_hz, width_cut_s = 7.0, 0.0003  # YOUR thresholds, 0.3 ms
    is_interneuron = (shape["observed_firing_rate_hz"] > rate_cut_hz) & (
        shape["trough_half_width"] < width_cut_s
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(
        shape.loc[~is_interneuron, "trough_half_width"] * 1e3,
        shape.loc[~is_interneuron, "observed_firing_rate_hz"],
        c="tab:blue",
        label="putative pyramidal",
        alpha=0.8,
    )
    ax.scatter(
        shape.loc[is_interneuron, "trough_half_width"] * 1e3,
        shape.loc[is_interneuron, "observed_firing_rate_hz"],
        c="tab:red",
        label="putative interneuron",
        alpha=0.8,
    )
    ax.axvline(width_cut_s * 1e3, ls="--", c="gray")
    ax.axhline(rate_cut_hz, ls="--", c="gray")
    ax.set_xlabel("trough_half_width (ms)")
    ax.set_ylabel("observed_firing_rate_hz (Hz)")
    ax.set_title("Putative cell types -- YOUR thresholds, not the pipeline's")
    ax.legend()
    print(
        f"{int(is_interneuron.sum())} putative interneuron(s), "
        f"{int((~is_interneuron).sum())} putative pyramidal"
    )

# ## Appendix C. Browser review reference
#
# Details of the 3-browser review that the walkthrough above does not show.
#
# - **Profile.** One name binds the metric and rule recipes, ordered columns,
#   label palette, and `replace` / `overlay` import mode.
#   `initialize_v2_defaults()` installs `franklab_hippocampus_2026_09_17`
#   (`franklab_default` metrics, `franklab_default_auto_curation_2026_09` rules,
#   `label_import_mode="replace"`, view version 4). Publishing location,
#   credentials, and ephemeral mode are per review.
# - **Labels.** The browser offers only the profile's `label_options` (nonempty,
#   at most 32 characters). Labels already on the parent are kept on import.
# - **Local serving.** `review.open()` starts or reuses one loopback server per
#   Python process and serves `http://localhost:<port>/bundles/<id>/`. The
#   `<id>` is random, so only someone given the URL can read or edit the review.
#   A browser save writes the bundle's `annotations.json`, the file the importer
#   reads; nothing else is writable.
#     - The port is process state. `open(port=...)` on the first open requests
#       the same port again; `open_browser=False` only returns the URL.
#     - Every review, focused inspection, and merged-child review in one process
#       shares the port, so one SSH tunnel covers the sequence.
#     - A missing bundle raises with the recovery step (start the review again).
#     - A reopened bundle gets the installed review controls; its data and saved
#       annotations are unchanged.
# - **Two tabs.** A stale save from an older tab is rejected without discarding
#   its edits. Use **Open latest draft in a new tab** and reapply them.
# - **Operations** (preview, commit, inspect) run one at a time in a worker with
#   its own DataJoint connection.
#     - Progress and committed identities persist in the bundle. Reloading
#       reconnects; ownership survives the launcher exiting while its worker
#       lives.
#     - Progress requests retry automatically. After repeated failures, use
#       **Reconnect review** to read the current operation without resubmitting.
#     - After a restart, resume the review. A live worker keeps running; retry
#       only an interrupted action. A failed re-evaluation keeps the child, and a
#       retry reuses it.
#     - Draft edits and commit stay disabled while an operation runs. Do not edit
#       or discard the bundle meanwhile.
#     - Operation and result journals live beside the bundle; there is no
#       workflow-status table.
# - **Where am I?** `changes.next_step()` and `receipt.next_step()` name one of
#   four states: saved edits differ from the reviewed parent; no saved edits
#   differ; merged result awaiting verification; result available for analysis.
#   The preview compares with the parent only, so an already-committed diff
#   still "differs" and committing it again reuses that child.
# - **Undo a saved but uncommitted merge:** select its units, **Undo selected
#   merge**, **Save draft**. `preview_import()` then shows no merge. A
#   committed wrong merge is a branch; see 3-recover.
# - **Finding a review.** `FigPackReview.resume(review_id)` rebuilds a handle.
#   `FigPackReview.find(parent, profile=...)` lists every built review of a
#   curation, oldest first; each `start_review` after a commit starts a new one.
#   Before any child is committed, `parent.start_review(profile)` returns the
#   existing review. `RunResult.start_review(source="auto_labeled")` raises when
#   no analysis curation exists; it never falls back to the root.
# - **Import safety.** Every import is pinned to the parent's `curation_uuid` and
#   the exact figure and profile. `preview_import()` reports children other
#   reviewers created since the review began. Commit still creates or reuses a
#   sibling from the pinned parent; it never rebases. It re-reads the
#   annotations and refuses a figure changed after preview. A zero-diff review
#   needs `confirm_no_changes=True`. Conflicting contributor labels need
#   `conflict_resolutions`; no contributor wins by precedence.
#   `FigPackCuration.save_curation_from_uri(uri, parent_curation_key)` checks
#   that `FigPackCuration` built the figure from that parent, then decodes its
#   annotations before writing a child. Other figures fail closed.
# - **Delivery.** Local is the default. `upload=True` publishes the identical
#   seeded bundle. A persistent hosted review needs a FigPack API key
#   (`dj.config["custom"]["figpack_api_key"]` or `FIGPACK_API_KEY`);
#   `ephemeral=True` makes a temporary hosted figure. Both publish the same
#   bundle, seeded annotations included. Hosted figures
#   have no compute service: save with **Save Annotations**, then import with
#   `panel = review.commit_panel()` (`panel.receipt`,
#   `panel.verification_review`).
# - **Views.** The unit selector stays visible across **Waveforms**, **Spike
#   amplitudes**, **Autocorrelograms**, **Cross-correlograms**, **Electrode
#   geometry**, and **Raster**.
# - **Display budgets.** Raster and amplitude budgets default to
#   `floor(recording_duration_s * 50)` points per unit (3,000 for a minute,
#   180,000 for an hour). Low-rate trains keep every point; higher-rate samples
#   span the recording reproducibly.
#     - `max_raster_spikes_per_unit` and `max_amplitudes_per_unit` set smaller
#       caps.
#     - Budgets affect display only, not waveform sampling or metrics. Absence
#       from a display is not evidence.
#     - Time plots over `max_initial_points` (default 1,000,000 per view) are
#       **not loaded** in the initial bundle. **Inspect selected units / pairs**
#       loads the selected units with their full budget, every spike in a
#       `[start, stop)` raster window, and every selected CCG pair. A window of
#       at most 10 s adds spikes on traces.
#     - `review.summary()` reports unavailable rule inputs and budgets.
# - **Times.** Displayed times are recording-relative seconds (the synthetic
#   timeline for concatenations). **Time and sampling** maps them to original
#   session seconds, gaps and member boundaries included. Red bands mark
#   excluded time. Inspection opens separately and keeps the active draft and
#   evaluation.
#
# **Hosted or static bundles** inspect from Python:
#
# ```python
# view = review.inspect_units([1, 4], time_range=(100, 110), include_traces=True)
# view.show(title="Selected units", upload=False, ephemeral=False)
# ```
