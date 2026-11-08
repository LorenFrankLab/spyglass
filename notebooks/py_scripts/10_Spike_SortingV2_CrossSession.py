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

# # Cross-session spike sorting: concatenate and track units across sessions
#
# Four chronic workflows that build on the single-session pipeline from
# [Spike Sorting v2](./10_Spike_SortingV2.ipynb). They compose: a daily
# concatenation from Part A is a valid matching input for Part B/C. Part C's
# demonstration reuses A/B; Part D can run independently:
#
# 1. **Concatenate same-day recordings and sort them as one** (Part A). When an
#    animal was recorded in several NWB files on the *same day* on the same
#    probe, sorting the concatenation (rather than each file separately) keeps a
#    unit's identity consistent across the files. If the day's epochs (e.g. runs
#    and sleeps) are all in **one** NWB file, you do not need Part A: sort one
#    interval list covering them with `run_v2_pipeline` (notebook 10). Use Part A
#    for one-file epochs only if you want per-epoch artifact detection and
#    per-epoch outputs. Concatenation is intended for
#    same-day recordings (multi-day concatenation is supported but experimental,
#    via `allow_multi_day`) and never corrects motion; to follow units across
#    days, sort each day independently and match them with UnitMatch (Parts B/C).
# 2. **Match units across sessions via a `SessionGroup`** (Part B). Sort each
#    session independently, then link the same biological unit across
#    sessions (typically **across days**) into a *tracked unit*, the basis
#    for following a cell over time.
# 3. **Match independently sorted sorts directly, without a group** (Part C).
#    The group-less counterpart of Part B: name the already-curated
#    `sorting_id`s to match (each one a single-recording sort *or* a same-day
#    concatenation sort from Part A) and match them directly. This is the
#    direct path for two or more **daily concatenations** (sort each day's
#    blocks as one concatenation via Part A, curate each day, then match the
#    days' concatenation sorts here), and it also reads back each tracked
#    unit's spike times and brain regions on each **original recording's own
#    clock**, not the synthetic concatenation timeline.
# 4. **Match separate epochs in one NWB** (Part D, off by default). Discover
#    existing intervals, sort and review each independently, then match the
#    exact reviewed curations. Disjoint original raw samples are required;
#    artifact masks cannot make overlapping epochs independent.
#
# Pick the path by where the day's data lives:
#
# | Your data | Path |
# | --- | --- |
# | Several epochs (e.g. runs and sleeps) in **one** NWB file | One `IntervalList` covering the epochs, then `run_v2_pipeline`. One sort, one `merge_id`; restrict to an epoch's interval at analysis time. |
# | Same-day recordings in **separate** NWB files | `run_v2_concat_pipeline`. One sort, one `member_merge_ids` entry per member on that member's own clock. |
# | Epochs you want sorted and curated **independently** | Sort each epoch's interval, then match them (Part D). |
#
# The first two sort the epochs as one recording, so a unit keeps one
# identity. The single-session path never sorts the gaps between disjoint
# intervals. Concatenation is the default chronic path for one day's files.
#
# Every matching input must be a v2 curation. A sort made and curated in v1
# cannot be matched directly; re-sort and curate it in v2 first (see the
# [migration guide](https://lorenfranklab.github.io/spyglass/latest/Features/SpikeSortingV2_Migration/)).
#
# Parts A and B start from a `SessionGroup`: a named bundle of *members*, where
# each member is a `(session, sort group, interval)` tuple. Part C names sorts
# directly and builds no group. This notebook assumes you have already
# configured a DataJoint connection (see [Setup](./00_Setup.ipynb)), ingested
# the sessions with `insert_sessions` (see [Insert Data](./02_Insert_Data.ipynb)),
# and created per-shank sort groups for each session
# (`SortGroupV2.set_group_by_shank`; see notebook 10).

# +
import datajoint as dj
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    describe_run,
    describe_epochs,
    describe_unit_match_choices,
    plan_v2_unit_match,
    plan_v2_unit_match_from_sorts,
    plan_v2_unit_match_from_curations,
    preflight_v2_unit_match,
    run_v2_concat_pipeline,
    run_v2_pipeline,
    run_v2_unit_match,
    select_units_for_analysis,
)
from spyglass.spikesorting.v2.session_group import SessionGroup
from spyglass.spikesorting.v2.unit_matching import TrackedUnit

dj.config["display.limit"] = 12
# -

# ## 1. One-time setup
#
# `initialize_v2_defaults()` installs the default parameter rows (including the
# `unitmatch_default` matcher), and the owning `LabTeam` namespaces your session
# groups so two teams can each create a group named `"day1"` without collision.

# + tags=["parameters"]
team_name = "my_team"
session_group_owner = team_name

# Part A: same-day recordings to concatenate and sort as ONE. Each member is a
# (session, sort group/shank, interval) tuple, recorded the same day on the same
# probe. Same-day is the default (allow_multi_day stays False).
same_day_members = [
    {
        "nwb_file_name": "day1_block1.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
    },
    {
        "nwb_file_name": "day1_block2.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
    },
]
concat_group_name = "day1_blocks"
# Motion correction is independent of concatenation and off by default: pass
# motion_mode / motion_correction_params_name to run_v2_pipeline or
# run_v2_concat_pipeline to estimate or apply it, the same way for a
# single-session or a concat preset.
concat_preset = "franklab_probe_hippocampus_30khz_ms4_2026_06"

# Part B: sessions to sort INDEPENDENTLY and match, typically across days (same
# animal + probe). The match group allows multiple days.
match_members = [
    {
        "nwb_file_name": "day1.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
    },
    {
        "nwb_file_name": "day2.nwb",
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
    },
]
match_group_name = "day1_to_day2"
single_preset = "franklab_probe_hippocampus_30khz_ms4_2026_06"
# The matcher backend + its parameters. The default uses UnitMatchPy; see
# describe_pipeline_presets()'s sibling MatcherParameters for the shipped rows.
matcher_params_name = "unitmatch_default"

# The two workflows are independent; enable whichever you need.
run_concat = True
run_unit_match = True

# Part D is a separate interactive workflow. Leave False for Run All; after
# selecting existing disjoint intervals, enable it to sort and open reviews.
run_epoch_match = False
epoch_nwb_file_name = "day1.nwb"
epoch_sort_group_id = 0
epoch_interval_names = ["sleep 1", "run 1"]  # choose with describe_epochs
# -

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "cross-session sorting"},
    skip_duplicates=True,
)

# ## Part A: Concatenate same-day recordings and sort
#
# `SessionGroup.create_group` records the members in order and validates them: it
# rejects a member whose session is not ingested or whose sort group / interval
# does not exist, and, because concatenation is intended for same-day blocks,
# leaves `allow_multi_day=False`, so it raises if these members span dates.
# (Recording dates are derived from each session, never supplied.)
#
# `run_v2_concat_pipeline` takes the *group* instead of a single session:
# it preprocesses each member, detects artifacts and masks that member, then
# concatenates the masked traces (motion correction, if requested via
# motion_mode, then runs on the concatenation before sorting). The summary
# is concat-shaped: `member_recording_ids` and `concat_recording_id` in place of
# the single-session `recording_id`, plus `member_artifacts` with exact detection
# IDs, statuses and masked durations. Inspect the scientific setup in the receipt.
# Detection choices are frozen; no mask is inherited from an earlier sort. The
# synthetic concat curation itself stays out of `SpikeSortingOutput`; the
# summary instead returns one wall-clock-aligned `member_merge_ids` entry per
# frozen member, keyed by `member_index`. `auto_curate=True` makes those member
# IDs point to the auto-curated child. Idempotent, like every pipeline run.

if run_concat:
    concat_key = {
        "session_group_owner": session_group_owner,
        "session_group_name": concat_group_name,
    }
    if not (SessionGroup & concat_key):
        SessionGroup.create_group(
            session_group_owner, concat_group_name, same_day_members
        )
    display(SessionGroup.Member & concat_key)

    concat_summary = run_v2_concat_pipeline(
        session_group_owner,
        concat_group_name,
        pipeline_preset=concat_preset,
        auto_curate=True,
    )
    display(describe_run(concat_summary))
    from spyglass.spikesorting.v2.artifact import RecordingArtifactDetection

    for result in concat_summary["member_artifacts"]:
        print(result)
        display(
            RecordingArtifactDetection().get_artifact_removed_intervals(
                {"artifact_detection_id": result["artifact_detection_id"]}
            )
        )
    # Compare these original-session kept intervals with short preprocessed
    # trace windows via visualization.plot_recording_traces. The concat cache
    # itself already contains the mask; motion correction (this run used the
    # default motion_mode="off") is a separate, optional stage on top of it.
    print(
        f"{len(concat_summary['member_recording_ids'])} member recordings -> "
        f"one concatenated sort with {concat_summary['n_units']} unit(s); "
        f"{len(concat_summary['member_merge_ids'])} session-safe outputs"
    )

    # Hand the auto-labeled concat curation to analysis with an EXPLICIT
    # policy. The rule set only flags bad units (it never writes `accept`), so
    # this automatic-only run names `v2_unflagged_units`: everything not
    # flagged, MUA and unlabeled included. The receipt carries one
    # SortedSpikesGroup per member (same curated unit ids; spike times on that
    # member's own NWB clock). `all_units` is the expert choice that ignores
    # labels.
    concat_selection = select_units_for_analysis(
        concat_summary.auto_labeled_curation, policy="v2_unflagged_units"
    )
    print(concat_selection.summary())
    for member_group in concat_selection.groups:
        print(
            f"member {member_group.member_index} ({member_group.nwb_file_name}): "
            f"{member_group.status} group {dict(member_group.group_key)}"
        )
    member_spikes = concat_selection.groups[0].fetch_spike_data()
    print(f"member 0: {len(member_spikes)} selected unit(s)")

# ### Part A details
#
# - **Artifacts are masked per member before concatenation.** Each detection is
#   frozen in the concat's identity, so a changed mask is a new concat and sort.
#   No mask is inherited from an earlier standalone sort.
# - `CurationRef.member_merge_ids` gives a concat-backed curation's member merge
#   ids.
# - If any member keeps a nonzero channel offset (only with `no_filter` and
#   `reference_mode="none"`), every member is converted to float32 µV before
#   masking. Sample counts and boundaries never change.
# - **Observation intervals survive.** The concat NWB stores kept intervals in
#   synthetic seconds; each member export stores them in original session time.
#   `describe_run` reports member detection ids, status, and masked durations.
# - Deleting a detection also deletes the concat selections that use it; the
#   confirmation prompt lists them.
# - **Brain regions anchor to the first member.** `get_unit_brain_regions` on a
#   concat sort raises `ConcatBrainRegionAmbiguousError` unless
#   `allow_anchor_member=True` (labeled `region_resolution="anchor_member"`).
#   For per-session regions, match the sessions and use
#   `TrackedUnit.get_unit_brain_regions`. `get_sort_group_info` and
#   `CurationV2.get_sort_metadata` resolve through the anchor member.
# - Concatenation never corrects motion. Apply the motion stage to the
#   `ConcatenatedRecording` for a corrected sort.
#
# The same steps table by table. Direct callers supply every member index in
# `artifact_detection_ids` (`None` means no mask); the standard runner
# resolves these from the preset.
#
# ```python
# from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
# from spyglass.spikesorting.v2.artifact import (
#     RecordingArtifactDetection,
#     RecordingArtifactSelection,
# )
# from spyglass.spikesorting.v2.concat_member_curation import ConcatMemberCurationV2
# from spyglass.spikesorting.v2.recording import Recording, RecordingSelection
# from spyglass.spikesorting.v2.session_group import (
#     ConcatenatedRecording,
#     ConcatenatedRecordingSelection,
#     SessionGroup,
# )
# from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
#
# # 1. Each member's Recording (one shared recipe) and artifact detection. A
# #    member is (nwb_file_name, sort_group_id, interval_list_name, team_name).
# members = [
#     {
#         "nwb_file_name": f,
#         "sort_group_id": 0,
#         "interval_list_name": "raw data valid times",
#     }
#     for f in (nwb_a, nwb_b)
# ]
# artifact_ids = {}
# for member_index, m in enumerate(members):
#     rec_key = RecordingSelection.insert_selection(
#         {**m, "preprocessing_params_name": "default", "team_name": "my_team"}
#     )
#     Recording.populate(rec_key)
#     detection = RecordingArtifactSelection.insert_selection(
#         {**rec_key, "artifact_detection_params_name": "default"}
#     )
#     RecordingArtifactDetection.populate(detection)
#     artifact_ids[member_index] = detection["artifact_detection_id"]
#
# # 2. Name the group (namespaced by owner).
# SessionGroup.create_group("my_team", "day1", members)
#
# # 3. The masked, unwhitened concat cache.
# concat_key = ConcatenatedRecordingSelection.insert_selection(
#     {
#         "session_group_owner": "my_team",
#         "session_group_name": "day1",
#         "preprocessing_params_name": "default",
#     },
#     artifact_detection_ids=artifact_ids,
# )
# ConcatenatedRecording.populate(concat_key)
#
# # 4. Sort it.
# sort_key = SortingSelection.insert_selection(
#     {
#         "concat_recording_id": concat_key["concat_recording_id"],
#         "sorter": "mountainsort5",
#         "sorter_params_name": "franklab_30khz_ms5_2026_06",
#     }
# )
# Sorting.populate(sort_key)
#
# # 5. After curating, one session-aligned output per member. Every member
# #    carries the same unit ids, with empty trains where a unit did not fire.
# curation_key = {"sorting_id": sort_key["sorting_id"], "curation_id": ...}
# ConcatMemberCurationV2.populate(curation_key)
# member_merge_ids = {
#     row["member_index"]: row["merge_id"]
#     for row in (
#         SpikeSortingOutput.ConcatMemberCurationV2 * ConcatMemberCurationV2 & curation_key
#     ).fetch(as_dict=True)
# }
# ```

# ## Part B: Match units across sessions
#
# The alternative to concatenation is to sort each session **independently** and
# then match units across them. This is the right path when sessions are days
# apart (multi-day concatenation is supported but experimental, and
# concatenation never corrects motion), or when you want each session's units
# kept distinct and simply *linked*. The match group is created with
# `allow_multi_day=True`.

# ### Group the sessions and sort each one
#
# Sort each member with the single-session preset and `auto_curate=True`, so each
# member contributes an **auto-labeled** curation to the match
# rather than the uncurated root (these reuse anything already computed). We drive
# Part B off the group's **persisted** members (the rows `create_group` stored),
# not the parameter list, so editing the group (or a member carrying its own
# `team_name`) sorts exactly the set that will be matched.

# +
import importlib.util

# Check the matcher backend BEFORE the expensive per-member sorts: without the
# optional UnitMatchPy extra there is no point sorting + auto-curating every
# member for a match that then can't run.
unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None
member_summaries = {}
if run_unit_match and not unitmatch_available:
    print(
        "UnitMatchPy not installed (the 'spikesorting-v2-matching' extra); "
        "skipping Part B (member sorts + matching). Install the extra to run it."
    )
elif run_unit_match:
    match_key = {
        "session_group_owner": session_group_owner,
        "session_group_name": match_group_name,
    }
    if not (SessionGroup & match_key):
        SessionGroup.create_group(
            session_group_owner,
            match_group_name,
            match_members,
            allow_multi_day=True,
        )
    group_members = (SessionGroup.Member & match_key).fetch(
        as_dict=True, order_by="member_index"
    )

    for member in group_members:
        # auto_curate=True so each member contributes label suggestions;
        # this automatic-only demonstration is not human-reviewed. To
        # match a hand-curated result instead, curate each member (see the
        # Curation how-to) and pin that (sorting_id, curation_id) below.
        summary = run_v2_pipeline(
            nwb_file_name=member["nwb_file_name"],
            sort_group_id=int(member["sort_group_id"]),
            interval_list_name=member["interval_list_name"],
            team_name=member["team_name"],
            pipeline_preset=single_preset,
            auto_curate=True,
        )
        member_summaries[int(member["member_index"])] = summary
        print(
            f"member {member['member_index']} ({member['nwb_file_name']}): "
            f"sorting_id={summary['sorting_id']} ({summary['n_units']} units)"
        )
# -

# ### Plan the curations to match
#
# `plan_v2_unit_match` pins one curation per member by a named **curation
# strategy** and returns a reviewable **plan**, matching v2's describe → plan →
# run shape. Here the curation strategy is `"auto_curated"`: each member's
# auto-curated child from the sorting step. Pick the curation strategy that
# matches your intent:
#
# - `final_curated`: the member's single terminal curated curation.
# - `auto_curated`: the auto-curated child produced above.
# - `root`: the uncurated root (warns loudly).
# - `manual`: pin `manual_curation_choices={member_index: {...}}` explicitly.
#
# A curation strategy never picks an implicit "latest"; a member it can't
# resolve to exactly one curation is a **blocking error** on the plan
# (`plan.ok` is `False`, listed in `plan.errors`), so a wrong or ambiguous pin
# surfaces here, not silently in the match. `plan.as_dataframe()` shows the
# per-member pins to review before running. `describe_unit_match_choices` shows
# every pinnable curation (a table, one row per member x curation) if you want
# to inspect them or build a `manual` plan by hand.

if run_unit_match and unitmatch_available:
    display(describe_unit_match_choices(session_group_owner, match_group_name))

    # Pin each member's auto-curated child by curation strategy, and review the
    # plan before running. Swap the curation strategy (final_curated / root /
    # manual) to change how curations are pinned.
    plan = plan_v2_unit_match(
        session_group_owner,
        match_group_name,
        curation_strategy="auto_curated",
        matcher_params_name=matcher_params_name,
        unit_policy="v2_unflagged_single_units",  # automatic-only, MUA excluded
    )
    display(plan.as_dataframe())  # one row per member; review before running
    for (
        warning
    ) in plan.warnings:  # e.g. the "root" curation strategy warns loudly
        print("WARNING:", warning)
    if not plan.ok:
        for problem in plan.errors:
            print("UNRESOLVED:", problem)

# ### Match and track
#
# `run_v2_unit_match(plan)` takes the reviewed plan, pins those curations, runs
# the matcher across the members, and derives **tracked units**: one identity per
# biological unit, with the per-session units that compose it. (A not-ok plan
# raises here rather than running a partial match.) The default `unitmatch_default`
# matcher uses [UnitMatchPy](https://github.com/EnnyvanBeest/UnitMatch), an
# **optional** extra (the `spikesorting-v2-matching` extra); without it the run
# raises, so the cell below runs only when it is installed. The summary reports
# `n_pairs` (pairwise matches) and `n_tracked_units` (biological units across the
# group).

# ``unitmatch_available`` was determined once in the sorting step above.
if run_unit_match and unitmatch_available:
    match_summary = run_v2_unit_match(plan)
    display(describe_run(match_summary))
    print(
        f"{match_summary['n_pairs']} cross-session pair(s) -> "
        f"{match_summary['n_tracked_units']} tracked unit(s)"
    )

    # A tracked unit's per-session members (the curated units that compose it)
    # are queryable through TrackedUnit for downstream cross-session analysis.
    tracked_key = {"unitmatch_id": match_summary["unit_match_id"]}
    display(TrackedUnit & tracked_key)
    display(TrackedUnit.Member & tracked_key)
elif run_unit_match:
    print(
        "UnitMatch extra not installed; skipping. Install the "
        "'spikesorting-v2-matching' extra to match units across sessions."
    )

# ## Part C: Match independently sorted daily concatenations directly
#
# `plan_v2_unit_match_from_sorts` is the group-less counterpart of
# `plan_v2_unit_match`: name the already-curated `sorting_id`s to match, in any
# order, with no `SessionGroup` step at all. Each named sort (a
# single-recording sort *or* a same-day concatenation sort) becomes one
# **matching input**; a two-block daily concatenation contributes exactly one
# input, not two. This is the direct path when each day was independently
# concatenated (Part A) and sorted: sort and curate every day first, then match
# the days' `sorting_id`s here. Inputs must not reuse original raw samples:
# a concatenation and its overlapping member sort cannot be matched together.
# Disjoint same-NWB epochs are supported (Part D), and each concatenation
# must lie within one day.
#
# For a runnable example without a second concatenation to configure, this
# matches Part A's same-day concatenation directly against one of Part B's
# independently sorted, independently curated sessions: the first one whose
# session is not among the concatenation's members (if every Part B session
# is, the part is skipped with a message). Substitute another day's
# concatenation `sorting_id` for a concatenation-vs-concatenation match.

run_daily_match = run_concat and run_unit_match and unitmatch_available
if run_daily_match:
    # This demonstration picks a different NWB for clarity. Part D demonstrates
    # the supported alternative: disjoint epochs of one NWB.
    concat_sessions = set(
        (SessionGroup.Member & concat_key).fetch("nwb_file_name")
    )
    other_session_sorting_ids = [
        member_summaries[int(member["member_index"])]["sorting_id"]
        for member in group_members
        if member["nwb_file_name"] not in concat_sessions
    ]
    if not other_session_sorting_ids:
        run_daily_match = False
        print(
            "Skipping Part C: every Part B session is also in Part A's "
            f"concatenation ({sorted(concat_sessions)}), and no two matching "
            "inputs may reuse source samples. Add a Part B session recorded "
            "outside the concatenation for this demo, or use Part D's epochs."
        )

if run_daily_match:
    daily_sorting_ids = [
        concat_summary["sorting_id"],  # Part A's same-day concatenation
        other_session_sorting_ids[0],  # a curated sort of another session
    ]
    daily_plan = plan_v2_unit_match_from_sorts(
        daily_sorting_ids,
        curation_strategy="auto_curated",
        matcher_params_name=matcher_params_name,
        unit_policy="v2_unflagged_single_units",  # unlabeled included, MUA not
    )
    display(daily_plan.as_dataframe())  # one row per matching input
    for warning in daily_plan.warnings:
        print("WARNING:", warning)
    if not daily_plan.ok:
        for problem in daily_plan.errors:
            print("UNRESOLVED:", problem)
        raise RuntimeError(
            "Part C plan is not runnable; fix the UNRESOLVED problems above."
        )

    daily_summary = run_v2_unit_match(daily_plan)
    # One input_<i> row per matching input, in chronological order, with its
    # source, constituent recordings and motion-correction status.
    display(describe_run(daily_summary))
    print(
        f"{daily_summary['n_pairs']} cross-session pair(s) -> "
        f"{daily_summary['n_tracked_units']} tracked unit(s)"
    )

    # Original-member analysis: a concatenation member's spikes and region are
    # read back on that ORIGINAL recording's own clock and sort group, never
    # the synthetic concatenation timeline or a copied anchor region.
    daily_tracked_key = {"unitmatch_id": daily_summary["unit_match_id"]}
    display(TrackedUnit & daily_tracked_key)
    example_tracked_unit_id = int(
        (TrackedUnit & daily_tracked_key).fetch("tracked_unit_id")[0]
    )
    example_tracked_key = {
        **daily_tracked_key,
        "tracked_unit_id": example_tracked_unit_id,
    }
    member_spike_times = TrackedUnit().get_member_spike_times(
        example_tracked_key
    )
    member_regions = TrackedUnit().get_unit_brain_regions(example_tracked_key)
    display(member_spike_times)  # one row per (member unit, original recording)
    display(member_regions)  # per-recording n_spikes / detected / region
elif run_concat and run_unit_match and not unitmatch_available:
    print(
        "UnitMatch extra not installed; skipping Part C. Install the "
        "'spikesorting-v2-matching' extra to match sorts directly."
    )

# ## Part D: Match independently reviewed epochs in one NWB
#
# This part is independent of A/B/C and disabled by default. First inspect
# `describe_epochs(epoch_nwb_file_name, sort_group_id=epoch_sort_group_id)`.
# `kind`, `epoch_label`, `interval_list_name`, `interval_bounds`, and recording
# readiness distinguish task epochs from other interval rows. Bounds use the
# original acquisition clock; a whole-session interval overlaps its epochs.
# Choose existing interval names above. One combined interval list and one
# sort remain the simpler path when you want a shared unit namespace upfront.

epoch_runs = {}
epoch_reviews = {}
if run_epoch_match:
    if len(epoch_interval_names) < 2 or len(set(epoch_interval_names)) != len(
        epoch_interval_names
    ):
        raise RuntimeError(
            "Part D requires at least two distinct epoch intervals."
        )
    if not unitmatch_available or importlib.util.find_spec("figpack") is None:
        raise RuntimeError(
            "Part D requires spyglass-neuro[spikesorting-v2-matching,"
            "spikesorting-v2-curation]. Install the extras before sorting."
        )
    display(
        describe_epochs(epoch_nwb_file_name, sort_group_id=epoch_sort_group_id)
    )
    for interval_name in epoch_interval_names:
        epoch_runs[interval_name] = run_v2_pipeline(
            nwb_file_name=epoch_nwb_file_name,
            sort_group_id=epoch_sort_group_id,
            interval_list_name=interval_name,
            team_name=team_name,
            pipeline_preset=single_preset,
            auto_curate=True,
        )
        epoch_reviews[interval_name] = epoch_runs[interval_name].start_review(
            "franklab_hippocampus_2026_09_17",
            source="auto_labeled",
            upload=False,
        )
        print(
            interval_name, epoch_reviews[interval_name].open(open_browser=False)
        )
        print("Resume review ID:", epoch_reviews[interval_name].review_id)

# Stop here and inspect each browser review. Label intended single units
# `accept`, commit, and complete any merged-child verification. In a later
# cell, replace the empty list below with exact completed results:
#
# ```python
# epoch_final_curations = [epoch_reviews[name].result() for name in epoch_interval_names]
# ```
#
# `final_curated` in strategy-based planners only means a non-root leaf; it
# can be an automatic child. Passing these handles avoids rediscovering a
# sibling branch and does not silently substitute an automatic result.

# +
epoch_final_curations = []  # fill only after completing both reviews
# -

# +
if run_epoch_match:
    if not epoch_final_curations:
        print(
            "Complete both epoch reviews, set epoch_final_curations, and rerun this cell."
        )
    else:
        if len(epoch_final_curations) != len(epoch_interval_names):
            raise RuntimeError(
                "Provide exactly one completed curation for every selected epoch interval."
            )
        expected_sortings = {
            str(epoch_runs[name]["sorting_id"]) for name in epoch_interval_names
        }
        if {
            str(final.sorting_id) for final in epoch_final_curations
        } != expected_sortings:
            raise RuntimeError(
                "Epoch curations must belong to the separate sorts in epoch_runs. "
                "Choose one completed result per interval; any order is accepted."
            )
        epoch_plan = plan_v2_unit_match_from_curations(epoch_final_curations)
        print(epoch_plan.summary())
        display(
            epoch_plan.as_dataframe()
        )  # exact pins and frozen accepted units
        epoch_report = preflight_v2_unit_match(epoch_plan)
        print(epoch_report.summary())  # read-only; never heals missing files
        if not epoch_report.ok:
            raise RuntimeError(
                "Fix epoch preflight findings and rebuild the plan."
            )
        epoch_match = run_v2_unit_match(epoch_plan)
        epoch_key = {"unitmatch_id": epoch_match["unit_match_id"]}
        display(TrackedUnit & epoch_key)
        # A match spanning both epochs has 2 matching inputs, 1 NWB session.
        display(TrackedUnit().get_member_spike_times(epoch_key))
        display(TrackedUnit().get_unit_brain_regions(epoch_key))
        # Matching creates identities, not analysis populations. Use each exact
        # reviewed curation for its own original-clock downstream handoff.
        epoch_selections = [
            select_units_for_analysis(final) for final in epoch_final_curations
        ]
        for selection in epoch_selections:
            print(selection.summary())
# -

# Matching is a computational plan and result readback. There is no
# browser interface for editing match pairs or approving tracked identities.

# ## Matching reference
#
# The recommended path is Part D's **exact curation → plan → preflight →
# run**. The details below apply to every planner.
#
# Inputs:
#
# - A **matching input** is one curated v2 sort: a single-recording `CurationV2`
#   or a same-day concatenation's parent curation. A two-block daily
#   concatenation is one input.
# - **Policies must deny `mua`.** Without supplied snapshots the planners default
#   to `v2_accepted_single_units`, which normally selects nothing after
#   automatic labeling alone. Name `v2_unflagged_single_units` for
#   automatic-only output. `v2_accepted_neural_units`, `v2_unflagged_units`, and
#   `all_units` are refused.
# - To carry an analysis population, pass one `resolve_unit_selection(...)`
#   snapshot per curation as `unit_selections=`, in order. With snapshots, an
#   unset `unit_policy` keeps each snapshot's policy; an explicit one must match
#   all of them. Each evaluation must belong to its curation.
# - `final_curated` (group and named-sort planners) means a unique committed
#   non-root leaf. An auto-labeled child can satisfy it, so it is not proof of
#   review. `manual` takes `manual_curation_choices={sorting_id: curation_id}`.
#   A plan that cannot pin exactly one curation has `plan.ok is False`, and
#   `run_v2_unit_match` raises listing `plan.errors`.
# - Group-based discovery searches each member's standalone sorts. Use named sorts
#   or curation handles for daily concatenation parents.
# - Planning and preflight need local raw NWBs, source recordings, and curated
#   Units files. Multi-input runs also need waveform traces and statistics
#   spans. Neither rebuilds nor downloads anything. Fix blocking findings and
#   rebuild the plan. Execution rechecks the pins, eligibility, and source hash
#   before inserting a selection.
#
# Restrictions (checked by `UnitMatchSelection.insert_inputs` and again in
# `UnitMatch.make`):
#
# - **No source sample may be reused.** Epochs of one NWB are allowed when their
#   raw samples are disjoint. `OverlappingMatchInputError` names the inputs,
#   intervals, and conflicting raw-frame range. Different recipes, interval
#   names, or masks do not make shared samples independent. A concatenation is
#   also checked against inputs holding one of its member epochs.
# - **Selected timestamps must identify raw samples uniquely.** Duplicate
#   timestamps inside the selection or at its boundaries block matching. Repair
#   the acquisition clock and recreate its recordings, or choose unambiguous
#   epochs. Duplicates elsewhere in the acquisition do not block.
# - **A concatenation input lies within one day.**
# - **Electrode space.** The UnitMatch backend rejects a geometry mismatch. The
#   pipeline warns when `(electrode_group_name, electrode_id, brain_region)`
#   differs, since ingestion naming is not stable across labs. Keep
#   electrode-group names stable across an implant's sessions. A real probe
#   mix-up also shows as poor matcher AUC or few pairs. (Concatenation rejects
#   mismatched electrode spaces outright.)
# - **Pins are rechecked.** `insert_selection` checks that each pinned curation
#   belongs to its member. `UnitMatch.make()` rechecks every input's provenance
#   (`UnitMatchSelectionIntegrityError`), so a direct insert, a recreated
#   curation, or changed source content cannot match the wrong units. A concat
#   input whose member `Recording` changed or is gone is refused.
#
# Identity:
#
# - Inputs are numbered by earliest `session_start_time`, same-session epochs by
#   selected source time, so naming order does not matter. Epochs are not sorted
#   by name.
# - Each input pins `(sorting_id, curation_id, curation_uuid)`, its recordings'
#   identity, content hash, session start, frame span, kept intervals, selected
#   unit ids, label policy, verdicts, and source coverage. A later label edit or
#   changed source needs a new plan.
# - Source coverage comes from the unmasked persisted recording's timestamps, not
#   from artifact-free observation intervals.
# - `UnitMatch.make` refuses a run whose frozen `nwb_file_name` or
#   `session_start_time` differs from the live rows. After a
#   `Session.session_start_time` correction, select again.
# - A `SessionGroup` on a selection is provenance only, not a foreign key.
# - A single-input run writes its matchable units and an empty `Pair` table
#   without calling the matcher.
#
# Results:
#
# - **Waveforms come from the traces the sorter read** (masked, corrected when
#   selected). Each unit's sampled spikes (where the full window lies in one
#   statistics span) split in temporal order into two halves. Unlike upstream
#   UnitMatchPy, Spyglass draws a random sample, averages with the mean, and
#   does not smooth. A unit with fewer than two such spikes stays unmatched.
# - **Tracked units are a strict partition.** `TrackedUnit` groups units that
#   match *every* other input in the group (greedy maximal cliques, largest
#   first, ties by median edge probability). If A↔B and B↔C match but A↔C does
#   not, A and C land in different tracked units. An unmatched unit is a
#   singleton (`n_matching_inputs == 1`, NULL `median_match_probability`). More
#   than `max_strict_nodes` (default 2000) units raise
#   `TrackedUnitBudgetExceededError`.
# - `n_matching_inputs` counts inputs. `n_sessions_detected` counts sessions
#   (`nwb_file_name`) where the unit has spikes; two intervals of one session
#   count once.
# - `TrackedUnit.get_unit_brain_regions` resolves regions per constituent
#   recording, read live. `TrackedUnit.get_member_spike_times` returns spikes on
#   each original recording's clock, with interval name and input index. Both
#   first check the inputs against their frozen rows.
# - `summary["inputs"]` holds one `UnitMatchInputSummary` per input with
#   `source_kind` / `source_id`, `nwb_file_names` / `interval_list_names`, and
#   whether its traces were corrected. `describe_run(summary)` renders them;
#   `UnitMatch.get_input_provenance(key, from_nwb=False)` returns them as
#   DataFrames.
# - Matching yields a pair graph and tracked-unit mapping. It does not build an
#   analysis population.
#
# Table-level calls (`initialize_v2_defaults()` installs `unitmatch_default`; a
# days-apart `SessionGroup` needs `allow_multi_day=True`;
# `UnitMatchSelection.insert_inputs(curations, matcher_params_name)` is the
# group-less form):
#
# ```python
# from spyglass.spikesorting.v2.unit_matching import (
#     TrackedUnit,
#     UnitMatch,
#     UnitMatchSelection,
# )
#
# selection_key = UnitMatchSelection.insert_selection(
#     "my_team",
#     "implant_week1",
#     "unitmatch_default",
#     {0: curation_day1, 1: curation_day2},  # exact curation per member_index
# )
# UnitMatch.populate(selection_key)
# pairs = UnitMatch().get_pairs(selection_key)  # UnitMatch.Pair as a DataFrame
# TrackedUnit.populate(selection_key)
# regions = TrackedUnit().get_unit_brain_regions({**selection_key, "tracked_unit_id": 0})
# ```
#
# Limits and evidence:
#
# - **Small unit counts destabilize calibration.** UnitMatch fits its threshold,
#   prior, and score distributions from each run's units. With about 20 or fewer
#   units per session, results vary run to run and can include bursts of false
#   matches. UnitMatchPy 3.2.7 raised an `IndexError` (`get_threshold`) at about
#   12 units in one session in synthetic trials. Match only well-isolated units
#   (UnitMatch issues
#   [#87](https://github.com/EnnyvanBeest/UnitMatch/issues/87),
#   [#146](https://github.com/EnnyvanBeest/UnitMatch/issues/146),
#   [#170](https://github.com/EnnyvanBeest/UnitMatch/issues/170)).
# - **Corrected days must already line up.** Motion correction registers each day
#   to its own mean position; there is no cross-day registration. On a simulated
#   32-contact single-column polymer layout (26 µm pitch), a rigid offset of 3 /
#   6 / 12 µm between corrected days recovered 22 / 12 / 1 of 24 planted
#   neurons. Confirm the days agree within a few micrometers first (e.g. compare
#   their displacement estimates).
# - **Evidence is synthetic only.** No real multi-day lab recording has been
#   evaluated. A ground-truth benchmark
#   (`tests/spikesorting/v2/scripts/unitmatch_daily_concat_benchmark.py`) runs
#   the matching code over simulated two- and three-day concatenations on a
#   16-channel, 2-column probe; thresholds were set on development seeds and met
#   on held-out seeds. Recall for gradually drifting and deliberately
#   conflicting neurons stays low. An end-to-end test with per-day correction on
#   aligned days recovered 19 of 22 cross-day neurons (13 of 22 without
#   correction).
# - UnitMatch's reference probe is the 128-channel LLNL polymer. Custom matchers
#   are described in the
#   [developer guide](https://lorenfranklab.github.io/spyglass/latest/ForDevelopers/SpikeSortingV2Architecture/#custom-unit-matchers).

# ## Next steps
#
# - Curate the concatenated sort (Part A) or any member/named sort (Parts B/C)
#   with [Curation](./10_Spike_SortingV2_Curation.ipynb).
# - Organize sorts and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb).
