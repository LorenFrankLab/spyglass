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
# Three cross-session workflows that build on the single-session pipeline from
# [Spike Sorting v2](./10_Spike_SortingV2.ipynb). They compose — a daily
# concatenation from Part A is a valid matching input for Part B/C — so this
# notebook keeps them in separate, independently runnable parts:
#
# 1. **Concatenate same-day recordings and sort them as one** (Part A). When an
#    animal was recorded in several blocks on the *same day* on the same probe,
#    sorting the concatenation (rather than each block separately) keeps a unit's
#    identity consistent across the blocks. Concatenation is intended for
#    same-day recordings (multi-day concatenation is supported but experimental,
#    via `allow_multi_day`) and never corrects motion; to follow units across
#    days, sort each day independently and match them with UnitMatch (Parts B/C).
# 2. **Match units across sessions via a `SessionGroup`** (Part B). Sort each
#    session independently, then link the same biological unit across
#    sessions — typically **across days** — into a *tracked unit*, the basis
#    for following a cell over time.
# 3. **Match independently sorted sorts directly, without a group** (Part C).
#    The group-less counterpart of Part B: name the already-curated
#    `sorting_id`s to match — each one a single-recording sort *or* a same-day
#    concatenation sort from Part A — and match them directly. This is the
#    direct path for two or more **daily concatenations** (sort each day's
#    blocks as one concatenation via Part A, curate each day, then match the
#    days' concatenation sorts here), and it also reads back each tracked
#    unit's spike times and brain regions on each **original recording's own
#    clock**, not the synthetic concatenation timeline.
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
    describe_unit_match_choices,
    plan_v2_unit_match,
    plan_v2_unit_match_from_sorts,
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

# Part A — same-day blocks to concatenate and sort as ONE. Each member is a
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
# motion_mode / motion_correction_params_name to run_v2_pipeline to estimate or
# apply it, the same way for a single-session or a concat preset.
concat_preset = "franklab_concat_hippocampus_30khz_ms5_2026_09"

# Part B — sessions to sort INDEPENDENTLY and match, typically across days (same
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
single_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
# The matcher backend + its parameters. The default uses UnitMatchPy; see
# describe_pipeline_presets()'s sibling MatcherParameters for the shipped rows.
matcher_params_name = "unitmatch_default"

# The two workflows are independent; enable whichever you need.
run_concat = True
run_unit_match = True
# -

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "cross-session sorting"},
    skip_duplicates=True,
)

# ## Part A — Concatenate same-day recordings and sort
#
# `SessionGroup.create_group` records the members in order and validates them: it
# rejects a member whose session is not ingested or whose sort group / interval
# does not exist, and — because concatenation is intended for same-day blocks —
# leaves `allow_multi_day=False`, so it raises if these members span dates.
# (Recording dates are derived from each session, never supplied.)
#
# In concat mode, `run_v2_pipeline` takes the *group* instead of a single session:
# it preprocesses each member, detects artifacts and masks that member, then
# concatenates the masked traces (motion correction, if requested via
# motion_mode, then runs on the concatenation before sorting). The summary
# is concat-shaped — `member_recording_ids` and `concat_recording_id` in place of
# the single-session `recording_id`, plus `member_artifacts` with exact detection
# IDs, statuses and masked durations. Inspect the scientific setup in the receipt.
# Detection choices are frozen; no mask is inherited from an earlier sort. The synthetic concat curation itself stays out of `SpikeSortingOutput`;
# the summary instead returns one wall-clock-aligned `member_merge_ids` entry per
# frozen member, keyed by `member_index`. `auto_curate=True` makes those member
# IDs point to the auto-curated child. Idempotent, like every `run_v2_pipeline`
# call.

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

    concat_summary = run_v2_pipeline(
        concat_session_group_owner=session_group_owner,
        concat_session_group_name=concat_group_name,
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
    # member's own NWB clock) -- `all_units` remains the explicit expert
    # choice that ignores labels entirely.
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

# ## Part B — Match units across sessions
#
# The alternative to concatenation is to sort each session **independently** and
# then match units across them — the right path when sessions are days apart
# (multi-day concatenation is supported but experimental, and concatenation
# never corrects motion), or when you want each session's units kept distinct
# and simply *linked*. The match group is created with `allow_multi_day=True`.

# ### Group the sessions and sort each one
#
# Sort each member with the single-session preset and `auto_curate=True`, so each
# member contributes an **auto-labeled** curation to the match
# rather than the uncurated root (these reuse anything already computed). We drive
# Part B off the group's **persisted** members — the rows `create_group` stored —
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
        # auto_curate=True so each member contributes an ANALYSIS-ready
        # (auto-labeled) curation to the match, not the uncurated root. To
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
# `plan_v2_unit_match` pins one curation per member by a named **curation strategy** and
# returns a reviewable **plan** — the plan-then-run shape mirroring the rest of
# v2 (describe → plan → run). Here the curation strategy is `"auto_curated"`: each member's
# auto-curated child from the sorting step. Pick the curation strategy that matches your intent:
#
# - `final_curated` — the member's single terminal curated curation.
# - `auto_curated` — the auto-curated child produced above.
# - `root` — the uncurated root (warns loudly).
# - `manual` — pin `manual_curation_choices={member_index: {...}}` explicitly.
#
# A curation strategy never picks an implicit "latest" — a member it can't resolve to
# exactly one curation is a **blocking error** on the plan (`plan.ok` is `False`,
# listed in `plan.errors`), so a wrong or ambiguous pin surfaces here, not
# silently in the match. `plan.as_dataframe()` shows the per-member pins to review
# before running. `describe_unit_match_choices` still shows every pinnable
# curation (a table, one row per member x curation) if you want to inspect them
# or build a `manual` plan by hand.

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
    )
    display(plan.as_dataframe())  # one row per member -- review before running
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
# the matcher across the members, and derives **tracked units** — one identity per
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

# ## Part C — Match independently sorted daily concatenations directly
#
# `plan_v2_unit_match_from_sorts` is the group-less counterpart of
# `plan_v2_unit_match`: name the already-curated `sorting_id`s to match, in any
# order, with no `SessionGroup` step at all. Each named sort — a
# single-recording sort *or* a same-day concatenation sort — becomes one
# **matching input**; a two-block daily concatenation contributes exactly one
# input, not two. This is the direct path when each day was independently
# concatenated (Part A) and sorted: sort and curate every day first, then match
# the days' `sorting_id`s here. No two matching inputs may share a session
# (`nwb_file_name`), so a concatenation and one of its own member sessions
# cannot be matched together, and a concatenation must lie within one day.
#
# For a runnable example without a second concatenation to configure, this
# matches Part A's same-day concatenation directly against one of Part B's
# independently sorted, independently curated sessions — the first one whose
# session is not among the concatenation's members (if every Part B session
# is, the part is skipped with a message). Substitute another day's
# concatenation `sorting_id` for a concatenation-vs-concatenation match.

run_daily_match = run_concat and run_unit_match and unitmatch_available
if run_daily_match:
    # No two matching inputs may share a session: pick a Part B sort of a
    # session outside Part A's concatenation.
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
            "inputs may share a session. Add a Part B session recorded "
            "outside the concatenation to run it."
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

# ## Next steps
#
# - Curate the concatenated sort (Part A) or any member/named sort (Parts B/C)
#   with the inspect-and-curate tools in
#   [Spike Sorting v2](./10_Spike_SortingV2.ipynb).
# - Organize sorts and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb).
# - For the matcher's parameters, internals, overlap restrictions, and the
#   held-out matching-recovery evidence (including the per-day
#   motion-correction alignment limitation), see
#   `docs/src/Features/SpikeSortingV2.md`.
