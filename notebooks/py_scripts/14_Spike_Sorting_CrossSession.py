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
# [Spike Sorting v2](./10_Spike_SortingV2.ipynb):
#
# 1. **Concatenate same-day recordings and sort them as one.** When an animal was
#    recorded in several blocks on the same day on the same probe, sorting the
#    concatenation (rather than each block separately) keeps a unit's identity
#    consistent across the blocks.
# 2. **Match units across sessions.** Sort each session independently, then link
#    the same biological unit across sessions (e.g. across days) into a *tracked
#    unit* — the basis for following a cell over time.
#
# Both start from a `SessionGroup`: a named bundle of *members*, where each member
# is a `(session, sort group, interval)` tuple. This notebook assumes you have
# already:
#
# 1. configured your DataJoint connection (see [Setup](./00_Setup.ipynb)),
# 2. ingested the sessions with `insert_sessions` (see
#    [Insert Data](./02_Insert_Data.ipynb)), and
# 3. created per-shank sort groups for each session
#    (`SortGroupV2.set_group_by_shank`; see notebook 10).

# +
import datajoint as dj
from IPython.display import display

from spyglass.common import LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    describe_run,
    describe_unit_match_choices,
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
# Same-day sessions recorded on the same probe, to group together. Each member is
# a (session, sort group/shank, interval) tuple — the same fields you sort a
# single session with. Point these at your ingested sessions.
members = [
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
group_name = "day1"
# A concat preset pins a motion-correction recipe (concatenation runs motion
# correction across the joined blocks); a single-session preset does not.
concat_preset = "franklab_concat_hippocampus_30khz_ms5_2026_06"
single_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
# -

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "cross-session sorting"},
    skip_duplicates=True,
)

# ## 2. Group the sessions
#
# `SessionGroup.create_group` records the members in order and validates them: it
# rejects a member whose session is not ingested or whose sort group / interval
# does not exist, and (because concatenation only makes sense within a day)
# requires `allow_multi_day=True` to span dates. Recording dates are derived from
# each session — never supplied. Re-running is safe: skip creation if the group
# already exists.

group_key = {
    "session_group_owner": session_group_owner,
    "session_group_name": group_name,
}
if not (SessionGroup & group_key):
    SessionGroup.create_group(session_group_owner, group_name, members)
display(SessionGroup.Member & group_key)

# ## Part A — Concatenate same-day recordings and sort
#
# In concat mode, `run_v2_pipeline` takes the *group* instead of a single
# session: it populates each member's recording, concatenates them into one
# motion-corrected recording, and sorts the result as a single piece. The summary
# is concat-shaped — `concat_recording_id` and `member_recording_ids` in place of
# the single-session `recording_id`, and no artifact stage (a concat preset runs
# none). The sort is then a normal `SpikeSortingOutput` `merge_id`, curated with
# the section-7 tools from notebook 10. Idempotent, like every `run_v2_pipeline`
# call.

concat_summary = run_v2_pipeline(
    concat_session_group_owner=session_group_owner,
    concat_session_group_name=group_name,
    pipeline_preset=concat_preset,
)
display(describe_run(concat_summary))
print(
    f"{len(concat_summary['member_recording_ids'])} member recordings -> "
    f"one concatenated sort with {concat_summary['n_units']} unit(s); "
    f"merge_id={concat_summary['merge_id']}"
)

# ## Part B — Match units across sessions
#
# The alternative to concatenation is to sort each session **independently** and
# then match units across them — the right path when sessions are days apart (no
# shared drift to correct), or when you want each session's units kept distinct
# and simply *linked*.

# ### B1. Sort each member session
#
# Matching pins one curation per member, so each member needs its own
# single-session sort + curation. Sort each with the single-session preset (these
# reuse anything already computed above). We use the freshly-created root curation
# of each; curate it first (notebook 10) when you want labeled/merged inputs.

member_summaries = []
for member in members:
    summary = run_v2_pipeline(
        nwb_file_name=member["nwb_file_name"],
        sort_group_id=member["sort_group_id"],
        interval_list_name=member["interval_list_name"],
        team_name=team_name,
        pipeline_preset=single_preset,
    )
    member_summaries.append(summary)
    print(
        f"{member['nwb_file_name']}: sorting_id={summary['sorting_id']} "
        f"({summary['n_units']} units)"
    )

# ### B2. List each member's curation choices
#
# `describe_unit_match_choices` walks each member's recordings → sortings →
# committed curations and lists, per member, every `(sorting_id, curation_id)` you
# may pin — so you assemble the match input without hand-querying. Pick one entry
# per member (here the root curation each sort just produced). Choosing the
# curation explicitly is required by design: an implicit "latest" would make a
# match silently change when a source session gains a new curation.

choices = describe_unit_match_choices(session_group_owner, group_name)
for member in choices:
    print(
        f"member {member['member_index']} ({member['nwb_file_name']}): "
        f"{len(member['choices'])} curation(s) available"
    )

# Build the per-member input: pin the root curation (parent_curation_id == -1)
# of each member. Swap in a labeled/merged curation_id to match curated units.
curation_choices = {}
for member in choices:
    root = next(
        (c for c in member["choices"] if c["parent_curation_id"] == -1),
        member["choices"][0] if member["choices"] else None,
    )
    if root is None:
        raise ValueError(
            f"member {member['member_index']} has no curation to pin; sort it "
            "first (Part B1)."
        )
    curation_choices[member["member_index"]] = {
        "sorting_id": root["sorting_id"],
        "curation_id": root["curation_id"],
    }
curation_choices

# ### B3. Match and track
#
# `run_v2_unit_match` pins those curations, runs the matcher across the members,
# and derives **tracked units** — one identity per biological unit, with the
# per-session units that compose it. The default `unitmatch_default` matcher uses
# [UnitMatchPy](https://github.com/EnnyvanBeest/UnitMatch), an **optional** extra
# (the `spikesorting-v2-matching` extra); without it the run raises, so the cell
# below runs only when it is installed. The summary reports `n_pairs` (pairwise
# matches) and `n_tracked_units` (biological units across the group).

import importlib.util

unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None
if unitmatch_available:
    match_summary = run_v2_unit_match(
        session_group_owner,
        group_name,
        matcher_params_name="unitmatch_default",
        curation_choices=curation_choices,
    )
    display(describe_run(match_summary))
    print(
        f"{match_summary['n_pairs']} cross-session pair(s) -> "
        f"{match_summary['n_tracked_units']} tracked unit(s)"
    )
else:
    print(
        "UnitMatch extra not installed; skipping. Install the "
        "'spikesorting-v2-matching' extra to match units across sessions."
    )

# A tracked unit's per-session members (the curated units that compose it) are
# queryable through `TrackedUnit` for downstream cross-session analysis:

if unitmatch_available:
    from spyglass.spikesorting.v2.unit_matching import TrackedUnit

    match_key = {"unitmatch_id": match_summary["unitmatch_id"]}
    display(TrackedUnit & match_key)
    display(TrackedUnit.Member & match_key)

# ## Next steps
#
# - Curate the concatenated sort (Part A) or any member sort (Part B) with the
#   inspect-and-curate tools in [Spike Sorting v2](./10_Spike_SortingV2.ipynb).
# - Organize sorts and filter units with
#   [Spike Sorting Analysis](./11_Spike_Sorting_Analysis.ipynb).
# - For the matcher's parameters and internals, see
#   `docs/src/Features/SpikeSortingV2.md`.
