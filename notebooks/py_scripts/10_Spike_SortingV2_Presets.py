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

# # Spike Sorting v2 — Presets & whole-session sorting
#
# Two how-tos that build on the
# [first-sort walkthrough](./10_Spike_SortingV2.ipynb): **customizing a pipeline
# preset** (tune one knob with `clone_pipeline_preset`, or build a custom one with
# `register_pipeline_preset`) without hand-editing parameter rows, and **sorting an entire
# session** (every sort group) in one call with `run_v2_pipeline_session`.
#
# Assumes a configured DataJoint connection and an ingested session (see
# [Setup](./00_Setup.ipynb) / [Insert Data](./02_Insert_Data.ipynb)).

# +
import datajoint as dj
import pandas as pd
from IPython.display import display

from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

from spyglass.common import Electrode, LabTeam
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    clone_pipeline_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_run,
    describe_sort_groups,
    list_pipeline_presets,
    plot_sort_group_geometry,
    preflight_v2_pipeline_session,
    register_pipeline_preset,
    run_v2_pipeline_session,
    select_units_for_analysis,
)
from spyglass.spikesorting.v2.recording import SortGroupV2

dj.config["display.limit"] = 12

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
references = None  # inherit stored references; review before creating groups
# None sorts every group; set an explicit subset such as [0, 2] if needed.
sort_group_ids = None
analysis_policy = "v2_accepted_single_units"
population_name = "v2_reviewed_session"
# Explicit alternative to human review. Missing QC may pass the rules;
# unflagged is not evidence that a unit is good. Choose the policy deliberately.
use_auto_labels_only = False
review_group_id = None  # set one group after reading the batch report
open_review_in_browser = False
commit_group_review = False
commit_group_verification = False
use_group_browser_result = False  # True after the connected browser review finishes
# Deliberate omissions are reported as a PARTIAL population.
omitted_sort_group_ids = []
# -

# ## 1. One-time setup
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
plot_sort_group_geometry(nwb_file_name)
sort_groups


# ## 2. Pick a pipeline preset
#
# `describe_pipeline_presets()` returns a table of what each shipping pipeline
# preset does — the sorter, the parameter rows each stage uses, the intended
# use, and (a known footgun) the units of the detection threshold — so you can
# choose one without reading the module source.

# The default is the runnable MS5 alternative, not an automatic choice of
# the lab's preferred scientific recipe. For hippocampal polymer probes:
#
# | Choice | Recommendation | Runtime requirement |
# | --- | --- | --- |
# | MountainSort4, local | Lab production recipe | MS4 backend in a compatible `numpy<2` environment |
# | MountainSort4, container | Same production recipe on a modern host | Singularity/Apptainer and the catalog's container image |
# | MountainSort5 | Alternative; notebook default | Standard v2 environment |
#
# Match the catalog's region, sampling rate and probe metadata to your
# recording. `production` means lab-recommended, `alternative` is a supported
# substitute, and `experimental` needs scientific validation for your use.
# Inspect `notes` for the exact execution requirements; selecting a preset
# never silently switches its sorter or backend.
#

describe_pipeline_presets()

# ### Customize a preset
#
# The shipping presets cover the common Frank Lab recipes, but you can adapt them
# without hand-editing parameter rows. `describe_pipeline_preset(name)` expands
# one preset into the exact parameter row each stage uses:

describe_pipeline_preset(pipeline_preset)

# `clone_pipeline_preset` derives a new preset from an existing one by tuning a single
# knob: pass the parameter you want to change as a keyword and it builds only the
# new parameter rows that differ, reusing the base preset's rows for every
# untouched stage. Here we lower the MountainSort5 detection threshold (a common
# tweak for low-amplitude units). The clone is then selectable by name like any
# shipping preset — set `pipeline_preset` to it to use it below. (Registering a
# name that already exists raises, so each cell is guarded to be safe to re-run.)

if "my_lab_ms5_lower_threshold" not in list_pipeline_presets():
    clone_pipeline_preset(
        pipeline_preset,
        "my_lab_ms5_lower_threshold",
        detect_threshold=5.0,
    )
describe_pipeline_preset("my_lab_ms5_lower_threshold")

# For a fully custom pipeline — a different sorter, or a stage combination no
# shipping preset covers — `register_pipeline_preset(name, {...})` adds a preset from the
# parameter-row names you choose (each must already exist; `initialize_v2_defaults`
# seeded the rows below). The registration lives for this session; to ship a
# durable recipe, add it to the preset catalog. The mapping is the stages a run
# touches:

if "my_lab_custom" not in list_pipeline_presets():
    register_pipeline_preset(
        "my_lab_custom",
        {
            "preprocessing_params_name": "franklab_hippocampus_2026_06",
            "artifact_detection_params_name": "default",
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "franklab_default_auto_curation_2026_09",
        },
    )

# ## 3. Sort the whole session at once
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
    sort_group_ids=sort_group_ids,
    auto_curate=True,
)
# Freeze the validated target list so execution uses exactly these groups.
target_sort_group_ids = [
    row["sort_group_id"] for row in session_report.group_reports
]
print(session_report.summary())
sort_groups[sort_groups["sort_group_id"].isin(target_sort_group_ids)]

# Inspect the plan above before running this cell. Groups run sequentially;
# completed stages are reused on a retry, while a failed stage starts again.
session_results = run_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    sort_group_ids=target_sort_group_ids,
    continue_on_error=True,
    auto_curate=True,
)
describe_run(session_results)


# ## 4. Choose the exact final curation for each group
#
# Keep this mapping while reviewing groups. Add only the generation you intend
# to analyze: never choose the greatest curation ID. Run the following cells
# for one group at a time; opening a browser does not wait for your edits.
# Failed, zero-unit, omitted, and still-unreviewed groups remain visible.
#
# For an explicitly automatic workflow, set `use_auto_labels_only=True` and
# choose `analysis_policy="v2_unflagged_units"`. Rules propose bad-unit labels;
# they do not write `accept`. Missing-policy pass means "not flagged", not
# "quality established". Deny labels win: `accept` plus `noise` is excluded.

# +
final_curations = {}  # retain this mapping when rerunning the review cells
successful_runs = {
    row["sort_group_id"]: row
    for row in session_results
    if row["outcome"] == "ok"
}
if use_auto_labels_only:
    final_curations.update(
        {
            group: run.auto_labeled_curation
            for group, run in successful_runs.items()
        }
    )
# -

# Select `review_group_id` above, then run this cell. Optional FigPack extra:
# `pip install -e ".[spikesorting-v2-curation]"`.

group_review = None
if review_group_id is not None:
    review_parent = final_curations.get(
        review_group_id, successful_runs[review_group_id].auto_labeled_curation
    )
    group_review = review_parent.start_review("franklab_hippocampus_2026_09_17")
    print(group_review.summary())
    print(group_review.open(open_browser=open_review_in_browser))

# **Stop to inspect and edit.** In a connected local browser use **Preview and
# commit**, then inspect and record any merged-child verification. After it
# finishes, set `use_group_browser_result=True` and run the handoff below.
# **Save draft** retains unfinished edits. The following cells are the optional
# notebook/script alternative (also used for hosted figures).

group_changes = None
if group_review is not None:
    group_changes = group_review.preview_import()
    print(group_changes.summary())
    display(group_changes.changed_units())

# A label-only commit can be selected immediately. A merge opens another review
# of the merged child's new waveforms and metrics. Pending browser merges never
# change the parent metrics. The separate verification cell prevents an open
# browser from being mistaken for a completed inspection.

group_verification = None
if group_changes is not None and commit_group_review:
    group_receipt = group_changes.commit(
        confirm_no_changes=not group_changes.has_changes,
        conflict_resolutions={},  # resolve any conflicts shown in the preview
    )
    print(group_receipt.next_step())
    if group_receipt.needs_merge_verification:
        final_curations.pop(review_group_id, None)
        group_verification = group_receipt.continue_review()
        print(group_verification.open(open_browser=open_review_in_browser))
    else:
        final_curations[review_group_id] = group_receipt.curation

# Inspect the merged child, save any edits, then set
# `commit_group_verification=True`. Another merge needs another inspection;
# rerun this cell after that inspection. For a mistaken committed merge, follow
# section 3-recover of the [curation notebook](./10_Spike_SortingV2_Curation.ipynb).

if group_verification is not None and commit_group_verification:
    verified_changes = group_verification.preview_import()
    print(verified_changes.summary())
    verified_receipt = verified_changes.commit(
        confirm_no_changes=not verified_changes.has_changes,
        conflict_resolutions={},
    )
    print(verified_receipt.next_step())
    if verified_receipt.needs_merge_verification:
        group_verification = verified_receipt.continue_review()
        print(group_verification.open(open_browser=open_review_in_browser))
    else:
        final_curations[review_group_id] = verified_receipt.curation
        group_verification = None

if use_group_browser_result:
    final_curations[review_group_id] = group_review.result()

# Rerun this report as you finish groups. A zero-unit sort is a completed
# computation, but contributes no units. Explicitly omit a failed/unwanted
# group only after deciding that a partial population suits the analysis.


# +
def population_review_table():
    rows = []
    for run in session_results:
        group = run["sort_group_id"]
        chosen = final_curations.get(group)
        rows.append(
            {
                "sort_group_id": group,
                "outcome": run["outcome"],
                "n_units": run.get("n_units"),
                "error": run.get("error", ""),
                "omitted": group in omitted_sort_group_ids,
                "final_curation_uuid": (
                    None if chosen is None else str(chosen.curation_uuid)
                ),
                "ready": chosen is not None or group in omitted_sort_group_ids,
            }
        )
    return pd.DataFrame(rows)


population_review_table()
# -

# ## 5. Assemble one session population under one label policy
#
# Run this after finishing the mapping (or explicitly omitting groups). Each
# receipt shows selected/excluded, MUA and unlabeled counts. Unit IDs are local:
# use `(spikesorting_merge_id, unit_id)` together across groups. These are
# label policies; additional metric filters are illustrated in the curation
# notebook and do not automatically carry into decoding.
#
# The assembly reuses a named group only if its membership AND policy match.
# Choose a new `population_name` when deliberately changing the population.


# +
def assemble_population():
    targets = set(target_sort_group_ids)
    omitted = set(omitted_sort_group_ids)
    chosen = set(final_curations)
    if (chosen | omitted) - targets or chosen & omitted:
        raise ValueError(
            "Chosen and omitted groups must be disjoint subsets of the run targets."
        )
    pending = targets - chosen - omitted
    if pending:
        print(
            "Population pending; review or explicitly omit groups:",
            sorted(pending),
        )
        return None, {}, [], []
    if not chosen:
        print("No groups chosen; no analysis population created.")
        return None, {}, [], []
    receipts = {}
    for group, ref in final_curations.items():
        if (
            group not in successful_runs
            or ref.sorting_id != successful_runs[group]["sorting_id"]
        ):
            raise ValueError(
                f"Group {group} needs a curation of its successful run."
            )
        receipts[group] = select_units_for_analysis(
            ref, policy=analysis_policy
        )
        print(f"Group {group}:\n{receipts[group].summary()}")
    key = {
        "nwb_file_name": nwb_file_name,
        "sorted_spikes_group_name": population_name,
        "unit_filter_params_name": analysis_policy,
    }
    members = {receipt.curation.merge_id for receipt in receipts.values()}
    # Preserve each receipt's frozen population when combining sort groups.
    snapshots = {}
    for receipt in receipts.values():
        for selected_group in receipt.groups:
            stored = (
                SortedSpikesGroup.UnitSelection
                & dict(selected_group.group_key)
            ).fetch1()
            snapshots[selected_group.merge_id] = {
                "selected_unit_ids": list(stored["selected_unit_ids"]),
                "selection_provenance": stored["selection_provenance"],
            }
    existing = SortedSpikesGroup & {
        "nwb_file_name": nwb_file_name,
        "sorted_spikes_group_name": population_name,
    }
    if existing:
        stored_policies = set(existing.fetch("unit_filter_params_name"))
        stored_members = set(
            (SortedSpikesGroup.Units & key).fetch("spikesorting_merge_id")
        )
        stored_snapshots = {
            row["spikesorting_merge_id"]: {
                "selected_unit_ids": list(row["selected_unit_ids"]),
                "selection_provenance": row["selection_provenance"],
            }
            for row in (SortedSpikesGroup.UnitSelection & key).fetch(
                as_dict=True
            )
        }
        if (
            stored_policies != {analysis_policy}
            or stored_members != members
            or stored_snapshots != snapshots
        ):
            raise ValueError(
                "Population name already has different members or policy; choose a new name."
            )
    else:
        SortedSpikesGroup().create_group(
            population_name,
            nwb_file_name,
            analysis_policy,
            keys=[{"spikesorting_merge_id": merge_id} for merge_id in members],
            unit_selections=snapshots,
        )
    spikes, identities = SortedSpikesGroup.fetch_spike_data(
        key, return_unit_ids=True
    )
    expected = {
        (receipt.curation.merge_id, unit_id)
        for receipt in receipts.values()
        for unit_id in receipt.included_unit_ids
    }
    actual = {
        (row["spikesorting_merge_id"], row["unit_id"]) for row in identities
    }
    assert (
        actual == expected
    ), "Population differs from the per-group receipts."
    print(
        (
            "PARTIAL population; omitted groups:"
            if omitted
            else "All requested groups included:"
        ),
        sorted(omitted or chosen),
    )
    return key, receipts, spikes, identities


population_key, group_selections, population_spikes, population_unit_ids = (
    assemble_population()
)
population_review_table()
# -
