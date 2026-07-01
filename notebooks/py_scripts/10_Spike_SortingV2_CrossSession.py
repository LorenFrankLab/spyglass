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
# Two cross-session workflows that build on the single-session pipeline from
# [Spike Sorting v2](./10_Spike_SortingV2.ipynb). They apply to **different**
# session sets, so this notebook keeps them separate — run either or both:
#
# 1. **Concatenate same-day recordings and sort them as one** (Part A). When an
#    animal was recorded in several blocks on the *same day* on the same probe,
#    sorting the concatenation (rather than each block separately) keeps a unit's
#    identity consistent across the blocks. Concatenation only makes sense within
#    a day — there is no shared drift to align across days.
# 2. **Match units across sessions** (Part B). Sort each session independently,
#    then link the same biological unit across sessions — typically **across
#    days** — into a *tracked unit*, the basis for following a cell over time.
#
# Both start from a `SessionGroup`: a named bundle of *members*, where each member
# is a `(session, sort group, interval)` tuple. The same-day concat group and the
# (possibly cross-day) match group are distinct, so they get separate parameters
# below. This notebook assumes you have already configured a DataJoint connection
# (see [Setup](./00_Setup.ipynb)), ingested the sessions with `insert_sessions`
# (see [Insert Data](./02_Insert_Data.ipynb)), and created per-shank sort groups
# for each session (`SortGroupV2.set_group_by_shank`; see notebook 10).

# +
import datajoint as dj
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    describe_run,
    describe_unit_match_choices,
    plan_v2_unit_match,
    run_v2_pipeline,
    run_v2_unit_match,
)
from spyglass.spikesorting.v2.session_group import SessionGroup

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
# probe. Concatenation requires a single day (allow_multi_day stays False).
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
# A concat preset pins a motion-correction recipe (motion is estimated across the
# joined blocks); a single-session preset does not.
concat_preset = "franklab_concat_hippocampus_30khz_ms5_2026_06"

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
# does not exist, and — because concatenation only makes sense within a day —
# leaves `allow_multi_day=False`, so it raises if these members span dates.
# (Recording dates are derived from each session, never supplied.)
#
# In concat mode, `run_v2_pipeline` takes the *group* instead of a single session:
# it populates each member's recording, concatenates them into one
# motion-corrected recording, and sorts the result as a single piece. The summary
# is concat-shaped — `member_recording_ids` and `concat_recording_id` in place of
# the single-session `recording_id`, and no artifact stage (a concat preset runs
# none). The sort is then a normal `SpikeSortingOutput` `merge_id`, curated with
# the section-7 tools from notebook 10. Idempotent, like every `run_v2_pipeline`
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
    )
    display(describe_run(concat_summary))
    print(
        f"{len(concat_summary['member_recording_ids'])} member recordings -> "
        f"one concatenated sort with {concat_summary['n_units']} unit(s); "
        f"root_merge_id={concat_summary['root_merge_id']}"
    )

# ## Part B — Match units across sessions
#
# The alternative to concatenation is to sort each session **independently** and
# then match units across them — the right path when sessions are days apart (no
# shared drift to correct), or when you want each session's units kept distinct
# and simply *linked*. The match group is created with `allow_multi_day=True`.

# ### B1. Group the sessions and sort each one
#
# Sort each member with the single-session preset and `auto_curate=True`, so each
# member contributes an **analysis-ready** (auto-labeled) curation to the match
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

# ### B2. Plan the curations to match
#
# `plan_v2_unit_match` pins one curation per member by a named **strategy** and
# returns a reviewable **plan** — the plan-then-run shape mirroring the rest of
# v2 (describe → plan → run). Here the strategy is `"auto_curated"`: each member's
# auto-curated child from B1. Pick the strategy that matches your intent:
#
# - `single_leaf_curated` — the member's single terminal curated curation.
# - `auto_curated` — the auto-curated child (what B1 produced here).
# - `root` — the uncurated root (warns loudly).
# - `manual` — pin `curation_choices={member_index: {...}}` explicitly.
#
# A strategy never picks an implicit "latest" — a member it can't resolve to
# exactly one curation is a **blocking error** on the plan (`plan.ok` is `False`,
# listed in `plan.errors`), so a wrong or ambiguous pin surfaces here, not
# silently in the match. `plan.as_dataframe()` shows the per-member pins to review
# before running. `describe_unit_match_choices` still lists every pinnable
# curation if you want to inspect them or build a `manual` plan by hand.

if run_unit_match and unitmatch_available:
    display(describe_unit_match_choices(session_group_owner, match_group_name))

    # Pin each member's auto-curated child (from B1) by strategy, and review the
    # plan before running. Swap the strategy (single_leaf_curated / root /
    # manual) to change how curations are pinned.
    plan = plan_v2_unit_match(
        session_group_owner,
        match_group_name,
        strategy="auto_curated",
        matcher_params_name=matcher_params_name,
    )
    display(plan.as_dataframe())  # one row per member -- review before running
    for warning in plan.warnings:  # e.g. the "root" strategy warns loudly
        print("WARNING:", warning)
    if not plan.ok:
        for problem in plan.errors:
            print("UNRESOLVED:", problem)

# ### B3. Match and track
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

import importlib.util

unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None
if run_unit_match and unitmatch_available:
    match_summary = run_v2_unit_match(plan)
    display(describe_run(match_summary))
    print(
        f"{match_summary['n_pairs']} cross-session pair(s) -> "
        f"{match_summary['n_tracked_units']} tracked unit(s)"
    )

    # A tracked unit's per-session members (the curated units that compose it)
    # are queryable through TrackedUnit for downstream cross-session analysis.
    from spyglass.spikesorting.v2.unit_matching import TrackedUnit

    tracked_key = {"unitmatch_id": match_summary["unitmatch_id"]}
    display(TrackedUnit & tracked_key)
    display(TrackedUnit.Member & tracked_key)
elif run_unit_match:
    print(
        "UnitMatch extra not installed; skipping. Install the "
        "'spikesorting-v2-matching' extra to match units across sessions."
    )

# ## Next steps
#
# - Curate the concatenated sort (Part A) or any member sort (Part B) with the
#   inspect-and-curate tools in [Spike Sorting v2](./10_Spike_SortingV2.ipynb).
# - Organize sorts and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb).
# - For the matcher's parameters and internals, see
#   `docs/src/Features/SpikeSortingV2.md`.
