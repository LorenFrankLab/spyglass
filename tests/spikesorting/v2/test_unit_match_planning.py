"""DB-free tests for cross-session unit-match curation-choice planning.

``build_unit_match_plan`` turns the per-member curation lists (from
``describe_unit_match_choices``) into a pinned ``curation_choices`` dict via a
named STRATEGY, producing a reviewable ``UnitMatchPlan`` (with warnings /
errors) BEFORE the expensive match. No database -- exercised directly.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._unit_match_planning import (
    UnitMatchPlan,
    build_unit_match_plan,
)

pytestmark = pytest.mark.unit

_S0, _S1 = "sort-0", "sort-1"


def _cur(sorting_id, curation_id, parent, source="manual", desc=""):
    return {
        "sorting_id": sorting_id,
        "curation_id": curation_id,
        "parent_curation_id": parent,
        "curation_source": source,
        "description": desc,
    }


def _member(idx, nwb, choices):
    return {"member_index": idx, "nwb_file_name": nwb, "choices": choices}


def _plan(members, strategy, **kw):
    return build_unit_match_plan(
        session_group_owner="owner",
        session_group_name="grp",
        matcher_params_name="unitmatch_default",
        strategy=strategy,
        members=members,
        **kw,
    )


def test_root_strategy_picks_root_and_warns_loudly():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(members, "root")
    assert plan.ok
    assert plan.curation_choices == {
        0: {"sorting_id": _S0, "curation_id": 0},
        1: {"sorting_id": _S1, "curation_id": 0},
    }
    # Pinning an uncurated root is legal but loud.
    assert plan.warnings
    assert any("root" in w.lower() for w in plan.warnings)


def test_single_leaf_curated_picks_the_terminal_child():
    # root(0) -> child(1); the leaf is the curated child.
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)])]
    plan = _plan(members, "single_leaf_curated")
    assert plan.ok
    assert plan.curation_choices == {0: {"sorting_id": _S0, "curation_id": 1}}


def test_single_leaf_curated_follows_a_chain_to_the_terminal_leaf():
    # root(0) -> child(1) -> grandchild(2): only 2 is a leaf.
    members = [
        _member(
            0,
            "day1.nwb",
            [_cur(_S0, 0, -1), _cur(_S0, 1, 0), _cur(_S0, 2, 1)],
        )
    ]
    plan = _plan(members, "single_leaf_curated")
    assert plan.curation_choices == {0: {"sorting_id": _S0, "curation_id": 2}}


def test_single_leaf_curated_ambiguous_two_leaves_errors():
    # root(0) -> child(1) AND child(2): two curated leaves -> not resolvable.
    members = [
        _member(
            0,
            "day1.nwb",
            [_cur(_S0, 0, -1), _cur(_S0, 1, 0), _cur(_S0, 2, 0)],
        )
    ]
    plan = _plan(members, "single_leaf_curated")
    assert not plan.ok
    assert any("member 0" in e for e in plan.errors)
    assert any("manual" in e for e in plan.errors)  # points to the escape hatch


def test_single_leaf_curated_root_only_errors():
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    plan = _plan(members, "single_leaf_curated")
    assert not plan.ok
    assert any("member 0" in e for e in plan.errors)


def test_auto_curated_picks_the_evaluation_child():
    members = [
        _member(
            0,
            "day1.nwb",
            [
                _cur(_S0, 0, -1),
                _cur(_S0, 1, 0, source="manual"),
                _cur(_S0, 2, 0, source="curation_evaluation"),
            ],
        )
    ]
    plan = _plan(members, "auto_curated")
    assert plan.curation_choices == {0: {"sorting_id": _S0, "curation_id": 2}}


def test_auto_curated_none_errors_with_auto_curate_hint():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0, "manual")])
    ]
    plan = _plan(members, "auto_curated")
    assert not plan.ok
    assert any("auto_curate" in e for e in plan.errors)


def test_manual_canonicalizes_and_validates_membership():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    # A valid explicit pin per member.
    plan = _plan(
        members,
        "manual",
        manual_choices={
            0: {"sorting_id": _S0, "curation_id": 1},
            1: {"sorting_id": _S1, "curation_id": 0},
        },
    )
    assert plan.ok
    assert plan.curation_choices == {
        0: {"sorting_id": _S0, "curation_id": 1},
        1: {"sorting_id": _S1, "curation_id": 0},
    }


def test_manual_rejects_a_pin_not_among_the_members_choices():
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    plan = _plan(
        members,
        "manual",
        manual_choices={0: {"sorting_id": _S0, "curation_id": 99}},
    )
    assert not plan.ok
    assert any("member 0" in e for e in plan.errors)


def test_manual_requires_a_choice_for_every_member():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(
        members,
        "manual",
        manual_choices={0: {"sorting_id": _S0, "curation_id": 0}},
    )
    assert not plan.ok
    assert any("member 1" in e for e in plan.errors)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        _plan([_member(0, "day1.nwb", [_cur(_S0, 0, -1)])], "latest")


def test_single_leaf_curated_ambiguous_across_two_sortings_errors():
    # A re-sorted member: TWO separate sortings, each with its own curated leaf.
    # "single leaf" spans all the member's curations, so this is ambiguous
    # across sortings -> error (not a silent pick of one sorting).
    members = [
        _member(
            0,
            "day1.nwb",
            [
                _cur(_S0, 0, -1),
                _cur(_S0, 1, 0),  # sort-0: root -> leaf 1
                _cur(_S1, 0, -1),
                _cur(_S1, 1, 0),  # sort-1: root -> leaf 1
            ],
        )
    ]
    plan = _plan(members, "single_leaf_curated")
    assert not plan.ok
    assert any("member 0" in e and "manual" in e for e in plan.errors)


def test_run_v2_unit_match_rejects_plan_plus_explicit_args():
    # Passing a plan AND an explicit arg would silently override one -> reject.
    # DB-free: the guard fires before any table access.
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    plan = UnitMatchPlan(
        session_group_owner="owner",
        session_group_name="grp",
        matcher_params_name="unitmatch_default",
        strategy="root",
        curation_choices={0: {"sorting_id": _S0, "curation_id": 0}},
        rows=[],
        warnings=[],
        errors=[],
    )
    assert plan.ok
    with pytest.raises(PipelineInputError, match="not both"):
        run_v2_unit_match(plan, "grp")  # explicit session_group_name too


def test_run_v2_unit_match_rejects_a_not_ok_plan():
    # The plan overload short-circuits a not-ok plan with the collected errors
    # BEFORE any database access -- so this is DB-free.
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    plan = UnitMatchPlan(
        session_group_owner="owner",
        session_group_name="grp",
        matcher_params_name="unitmatch_default",
        strategy="single_leaf_curated",
        curation_choices={},
        rows=[],
        warnings=[],
        errors=["member 0 (day1.nwb): strategy='single_leaf_curated' ..."],
    )
    assert not plan.ok
    with pytest.raises(PipelineInputError, match="could not pin"):
        run_v2_unit_match(plan)


def test_plan_dataframe_and_truthiness():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(members, "single_leaf_curated")
    # A member with no curated leaf makes the whole plan not-ok, but the frame
    # still lists every member for review.
    df = plan.as_dataframe()
    assert list(df["member_index"]) == [0, 1]
    assert "day2.nwb" in set(df["nwb_file_name"])
    # bool(plan) mirrors plan.ok.
    assert bool(plan) is plan.ok
    assert isinstance(plan, UnitMatchPlan)
