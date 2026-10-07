"""DB-free tests for cross-session unit-match curation-choice planning.

``build_unit_match_plan`` turns the per-member curation lists (from
``describe_unit_match_choices``) into a pinned ``curation_choices`` dict via a
named curation strategy, producing a reviewable ``UnitMatchPlan`` (with
warnings / errors) BEFORE the expensive match. No database -- exercised
directly.
"""

from __future__ import annotations

import uuid

import pytest

from spyglass.spikesorting.v2._unit_match_planning import (
    UnitMatchPlan,
    build_unit_match_plan,
)

pytestmark = pytest.mark.unit

_S0, _S1 = "sort-0", "sort-1"
_GENERATIONS = {
    (_S0, 0): "ae0bb75e-7519-472b-a7ed-9e58fce09d6d",
    (_S0, 1): "7ed44ce2-b8a5-4d42-a9bb-e8df4c699955",
    (_S0, 2): "f06a3d4a-ad1f-4c65-bc7e-5f7324fd2a70",
    (_S1, 0): "763b6c78-eb12-4115-90be-33026bcfa5ed",
    (_S1, 1): "d7c7c4f0-d709-4f5b-96e4-39c8b46d5341",
    (_S1, 2): "1500c877-90fa-40fd-b8a7-269cb402c15e",
}


def _pin(sorting_id, curation_id):
    return {
        "sorting_id": sorting_id,
        "curation_id": curation_id,
        "curation_uuid": _GENERATIONS[sorting_id, curation_id],
    }


def _cur(sorting_id, curation_id, parent, source="manual", desc=""):
    return {
        **_pin(sorting_id, curation_id),
        "parent_curation_id": parent,
        "curation_source": source,
        "description": desc,
    }


def _member(idx, nwb, choices):
    return {"member_index": idx, "nwb_file_name": nwb, "choices": choices}


def _plan(members, curation_strategy, **kw):
    return build_unit_match_plan(
        session_group_owner="owner",
        session_group_name="grp",
        matcher_params_name="unitmatch_default",
        curation_strategy=curation_strategy,
        members=members,
        **kw,
    )


def _full_member(idx, nwb, choices):
    """A structured member with all identity columns (as the DB builder emits)."""
    return {
        "member_index": idx,
        "nwb_file_name": nwb,
        "sort_group_id": 0,
        "interval_list_name": "raw data valid times",
        "team_name": "t",
        "choices": choices,
    }


def test_member_choices_to_dataframe_one_row_per_choice():
    """The ``describe_*`` view flattens to one row per (member, choice)."""
    from spyglass.spikesorting.v2._unit_match_planning import (
        member_choices_to_dataframe,
    )

    members = [
        _full_member(
            0,
            "a.nwb",
            [_cur(_S0, 0, -1), _cur(_S0, 1, 0, source="curation_evaluation")],
        ),
        _full_member(1, "b.nwb", []),  # no sort yet
    ]
    df = member_choices_to_dataframe(members)

    assert list(df.columns) == [
        "member_index",
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        "sorting_id",
        "curation_id",
        "curation_uuid",
        "parent_curation_id",
        "curation_source",
        "description",
    ]
    # two choice rows for member 0 + one placeholder row for the empty member 1
    assert len(df) == 3
    member0 = df[df["member_index"] == 0]
    assert set(zip(member0["sorting_id"], member0["curation_id"])) == {
        (_S0, 0),
        (_S0, 1),
    }
    # A member with no sort stays visible with null curation columns.
    member1 = df[df["member_index"] == 1]
    assert len(member1) == 1
    assert member1["sorting_id"].isna().all()

    # Integer id columns stay nullable-integer (Int64), NOT float64: the None
    # placeholder row would otherwise upcast valid ids to e.g. 0.0, reintroducing
    # the copy-paste footgun this table exists to prevent.
    for col in (
        "member_index",
        "sort_group_id",
        "curation_id",
        "parent_curation_id",
    ):
        assert str(df[col].dtype) == "Int64", (col, df[col].dtype)


def test_root_curation_strategy_picks_root_and_warns_loudly():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(members, "root")
    assert plan.ok
    assert plan.curation_choices == {
        0: _pin(_S0, 0),
        1: _pin(_S1, 0),
    }
    # Pinning an uncurated root is legal but loud.
    assert plan.warnings
    assert any("root" in w.lower() for w in plan.warnings)


def test_final_curated_picks_the_terminal_child():
    # root(0) -> child(1); the leaf is the curated child.
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)])]
    plan = _plan(members, "final_curated")
    assert plan.ok
    assert plan.curation_choices == {0: _pin(_S0, 1)}


def test_final_curated_follows_a_chain_to_the_terminal_leaf():
    # root(0) -> child(1) -> grandchild(2): only 2 is a leaf.
    members = [
        _member(
            0,
            "day1.nwb",
            [_cur(_S0, 0, -1), _cur(_S0, 1, 0), _cur(_S0, 2, 1)],
        )
    ]
    plan = _plan(members, "final_curated")
    assert plan.curation_choices == {0: _pin(_S0, 2)}


def test_final_curated_ambiguous_two_leaves_errors():
    # root(0) -> child(1) AND child(2): two curated leaves -> not resolvable.
    members = [
        _member(
            0,
            "day1.nwb",
            [_cur(_S0, 0, -1), _cur(_S0, 1, 0), _cur(_S0, 2, 0)],
        )
    ]
    plan = _plan(members, "final_curated")
    assert not plan.ok
    assert any("member 0" in e for e in plan.errors)
    assert any("manual" in e for e in plan.errors)  # points to the escape hatch


def test_final_curated_root_only_errors():
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    plan = _plan(members, "final_curated")
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
    assert plan.curation_choices == {0: _pin(_S0, 2)}


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
        manual_curation_choices={
            0: {"sorting_id": _S0, "curation_id": 1},
            1: {"sorting_id": _S1, "curation_id": 0},
        },
    )
    assert plan.ok
    assert plan.curation_choices == {
        0: _pin(_S0, 1),
        1: _pin(_S1, 0),
    }


def test_manual_rejects_a_pin_not_among_the_members_choices():
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    plan = _plan(
        members,
        "manual",
        manual_curation_choices={0: {"sorting_id": _S0, "curation_id": 99}},
    )
    assert not plan.ok
    assert any("member 0" in e for e in plan.errors)


def test_manual_pin_ids_are_normalized_losslessly():
    """A fractional / boolean manual curation_id is rejected rather than
    truncated into a (possibly valid) different curation; Python and NumPy
    integers pin as themselves."""
    import numpy as np

    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)])]
    for good in (1, np.int64(1)):
        plan = _plan(
            members,
            "manual",
            manual_curation_choices={
                0: {"sorting_id": _S0, "curation_id": good}
            },
        )
        assert plan.ok and plan.curation_choices[0]["curation_id"] == 1
    for bad in (1.9, True):
        with pytest.raises(ValueError, match="member 0 curation_id"):
            _plan(
                members,
                "manual",
                manual_curation_choices={
                    0: {"sorting_id": _S0, "curation_id": bad}
                },
            )
    # Mapping KEYS obey the same rule: True / 1.0 hash equal to 1 and would
    # otherwise silently pin member 1.
    two = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    for bad_key in (True, 1.0):
        with pytest.raises(ValueError, match="member index must be an integer"):
            _plan(
                two,
                "manual",
                manual_curation_choices={
                    0: {"sorting_id": _S0, "curation_id": 0},
                    bad_key: {"sorting_id": _S1, "curation_id": 0},
                },
            )


def test_manual_rejects_extra_member_indices():
    # A manual_curation_choices entry for a member index that is not in the group
    # (stale / mistyped) is a blocking error -- exact coverage, not silently
    # dropped.
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    plan = _plan(
        members,
        "manual",
        manual_curation_choices={
            0: {"sorting_id": _S0, "curation_id": 0},
            9: {"sorting_id": _S1, "curation_id": 0},  # not a member
        },
    )
    assert not plan.ok
    assert any("[9]" in e for e in plan.errors)


def test_manual_curation_choices_rejected_with_a_non_manual_strategy():
    # Explicit pins with an automatic curation strategy would be silently ignored
    # -> the planner rejects the combination up front.
    members = [_member(0, "day1.nwb", [_cur(_S0, 0, -1)])]
    with pytest.raises(ValueError, match="curation_strategy='manual'"):
        _plan(
            members,
            "auto_curated",
            manual_curation_choices={0: {"sorting_id": _S0, "curation_id": 0}},
        )


def test_manual_requires_a_choice_for_every_member():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(
        members,
        "manual",
        manual_curation_choices={0: {"sorting_id": _S0, "curation_id": 0}},
    )
    assert not plan.ok
    assert any("member 1" in e for e in plan.errors)


def test_unknown_curation_strategy_raises():
    with pytest.raises(ValueError, match="curation_strategy"):
        _plan([_member(0, "day1.nwb", [_cur(_S0, 0, -1)])], "latest")


def test_final_curated_ambiguous_across_two_sortings_errors():
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
    plan = _plan(members, "final_curated")
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
        curation_strategy="root",
        curation_choices={0: {"sorting_id": _S0, "curation_id": 0}},
        rows=[],
        warnings=[],
        errors=[],
    )
    assert plan.ok
    with pytest.raises(PipelineInputError, match="not both"):
        run_v2_unit_match(plan, session_group_name="grp")


def test_run_v2_unit_match_rejects_a_not_ok_plan():
    # The plan overload short-circuits a not-ok plan with the collected errors
    # BEFORE any database access -- so this is DB-free.
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    plan = UnitMatchPlan(
        session_group_owner="owner",
        session_group_name="grp",
        matcher_params_name="unitmatch_default",
        curation_strategy="final_curated",
        curation_choices={},
        rows=[],
        warnings=[],
        errors=["member 0 (day1.nwb): curation_strategy='final_curated' ..."],
    )
    assert not plan.ok
    with pytest.raises(PipelineInputError, match="could not pin"):
        run_v2_unit_match(plan)


def test_plan_dataframe_and_truthiness():
    members = [
        _member(0, "day1.nwb", [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _member(1, "day2.nwb", [_cur(_S1, 0, -1)]),
    ]
    plan = _plan(members, "final_curated")
    # A member with no curated leaf makes the whole plan not-ok, but the frame
    # still lists every member for review.
    df = plan.as_dataframe()
    assert list(df["member_index"]) == [0, 1]
    assert "day2.nwb" in set(df["nwb_file_name"])
    # bool(plan) mirrors plan.ok.
    assert bool(plan) is plan.ok
    assert isinstance(plan, UnitMatchPlan)


# ---- named-sort plans (one matching input per sort) -------------------------


def _sort(sorting_id, choices, kind="recording", nwbs=("day1.nwb",)):
    return {
        "sorting_id": sorting_id,
        "source_kind": kind,
        "source_id": f"src-{sorting_id}",
        "nwb_file_names": nwbs,
        "interval_list_names": tuple(f"interval {i}" for i in range(len(nwbs))),
        "choices": choices,
    }


def _input_plan(sorts, curation_strategy, **kw):
    from spyglass.spikesorting.v2._unit_match_planning import (
        build_unit_match_input_plan,
    )

    return build_unit_match_input_plan(
        matcher_params_name="unitmatch_default",
        curation_strategy=curation_strategy,
        sorts=sorts,
        **kw,
    )


def test_input_plan_pins_one_curation_per_sort_in_named_order():
    """Each automatic strategy resolves within each sort's own curations,
    and the plan has one row (and one pin) per named sort, in named order."""
    day2 = _sort(
        _S1,
        [
            _cur(_S1, 0, -1),
            _cur(_S1, 1, 0),
            _cur(_S1, 2, 0, source="curation_evaluation"),
        ],
        kind="concatenated_recording",
        nwbs=("day2.nwb", "day2.nwb"),
    )
    day1 = _sort(
        _S0, [_cur(_S0, 0, -1), _cur(_S0, 1, 0, "curation_evaluation")]
    )
    expected = {
        "root": [(_S1, 0), (_S0, 0)],
        "auto_curated": [(_S1, 2), (_S0, 1)],
    }
    for strategy, pins in expected.items():
        plan = _input_plan([day2, day1], strategy)
        assert plan.ok, plan.errors
        assert [
            (c["sorting_id"], c["curation_id"]) for c in plan.curations
        ] == (pins)
        frame = plan.as_dataframe()
        assert frame.to_dict(orient="records") == [
            {
                "sorting_id": _S1,
                "source_kind": "concatenated_recording",
                "source_id": f"src-{_S1}",
                "nwb_file_names": ("day2.nwb", "day2.nwb"),
                "interval_list_names": ("interval 0", "interval 1"),
                "curation_id": pins[0][1],
                "curation_uuid": _pin(*pins[0])["curation_uuid"],
                "status": "pinned",
            },
            {
                "sorting_id": _S0,
                "source_kind": "recording",
                "source_id": f"src-{_S0}",
                "nwb_file_names": ("day1.nwb",),
                "interval_list_names": ("interval 0",),
                "curation_id": pins[1][1],
                "curation_uuid": _pin(*pins[1])["curation_uuid"],
                "status": "pinned",
            },
        ]
    # root pins uncurated units: advisory, one per sort, naming the sort.
    root_plan = _input_plan([day2, day1], "root")
    assert len(root_plan.warnings) == 2
    assert f"sort {_S1} (concatenated_recording: day2.nwb, day2.nwb)" in (
        root_plan.warnings[0]
    )


def test_input_plan_final_curated_blocks_an_ambiguous_sort_only():
    """final_curated with two curated leaves in one sort blocks that sort
    (pointing at manual) while the other sort still resolves."""
    ambiguous = _sort(_S0, [_cur(_S0, 0, -1), _cur(_S0, 1, 0), _cur(_S0, 2, 0)])
    clean = _sort(_S1, [_cur(_S1, 0, -1), _cur(_S1, 1, 0)])
    plan = _input_plan([ambiguous, clean], "final_curated")
    assert not plan.ok
    assert len(plan.errors) == 1
    assert f"sort {_S0}" in plan.errors[0] and "manual" in plan.errors[0]
    assert plan.curations == [_pin(_S1, 1)]
    assert plan.as_dataframe()["status"].tolist() == ["UNRESOLVED", "pinned"]
    no_curation = _input_plan([_sort(_S0, [])], "root")
    assert no_curation.errors == [
        f"sort {_S0} (recording: day1.nwb): curation_strategy='root' has no "
        "curation yet; curate the sort first."
    ]


def test_input_plan_manual_pins_by_sorting_id():
    """manual takes one curation_id per named sort, checked against that
    sort's curations; a missing, foreign, fractional or extra pin is caught."""
    sorts = [
        _sort(_S0, [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]),
        _sort(_S1, [_cur(_S1, 0, -1)]),
    ]
    plan = _input_plan(
        sorts, "manual", manual_curation_choices={_S0: 1, _S1: 0}
    )
    assert plan.ok
    assert plan.curations == [
        _pin(_S0, 1),
        _pin(_S1, 0),
    ]

    missing = _input_plan(sorts, "manual", manual_curation_choices={_S0: 1})
    assert [e for e in missing.errors if f"sort {_S1}" in e]
    assert "manual_curation_choices entry for this sort" in missing.errors[0]

    foreign = _input_plan(
        sorts, "manual", manual_curation_choices={_S0: 9, _S1: 0}
    )
    assert "not among this sort's committed curations" in foreign.errors[0]

    extra = _input_plan(
        sorts,
        "manual",
        manual_curation_choices={_S0: 1, _S1: 0, "sort-9": 0},
    )
    assert not extra.ok
    assert extra.errors == [
        "manual_curation_choices has entries for sorting_id(s) ['sort-9'] "
        "that are not among the named sorts."
    ]
    with pytest.raises(ValueError, match="sorting sort-0 curation_id"):
        _input_plan(sorts, "manual", manual_curation_choices={_S0: 1.5})


def test_input_plan_rejects_bad_arguments():
    """No sorts, a sort named twice, an unknown strategy, and manual pins
    with an automatic strategy are argument errors, not plan errors."""
    one = _sort(_S0, [_cur(_S0, 0, -1)])
    with pytest.raises(ValueError, match="at least one sort"):
        _input_plan([], "root")
    with pytest.raises(ValueError, match=r"more than once: \['sort-0'\]"):
        _input_plan([one, one], "root")
    with pytest.raises(ValueError, match="unknown curation_strategy"):
        _input_plan([one], "latest")
    with pytest.raises(ValueError, match="only used by"):
        _input_plan([one], "root", manual_curation_choices={_S0: 0})


@pytest.mark.parametrize("form", ["members", "sorts"])
@pytest.mark.parametrize(
    "strategy", ["root", "manual", "auto_curated", "final_curated"]
)
def test_plans_require_the_reviewed_curation_generation(form, strategy):
    curation_id = 0 if strategy == "root" else 1
    choice = _cur(
        _S0,
        curation_id,
        -1 if strategy == "root" else 0,
        source="curation_evaluation",
    )
    original = choice["curation_uuid"]

    def build():
        if form == "members":
            manual = (
                {
                    "manual_curation_choices": {
                        0: {
                            "sorting_id": _S0,
                            "curation_id": curation_id,
                        }
                    }
                }
                if strategy == "manual"
                else {}
            )
            return _plan([_member(0, "day1.nwb", [choice])], strategy, **manual)
        manual = (
            {"manual_curation_choices": {_S0: curation_id}}
            if strategy == "manual"
            else {}
        )
        return _input_plan([_sort(_S0, [choice])], strategy, **manual)

    def pin(plan):
        return (
            plan.curation_choices[0] if form == "members" else plan.curations[0]
        )

    plan = build()
    assert plan.ok, plan.errors
    assert pin(plan)["curation_uuid"] == original
    # Generations are opaque: they cannot be reconstructed from the numeric
    # curation key. Mutating a candidate must also leave the reviewed plan fixed.
    replacement = uuid.UUID("c7b7a213-757d-4903-9a62-506614b9858b")
    choice["curation_uuid"] = replacement
    assert pin(plan)["curation_uuid"] == original
    assert pin(build())["curation_uuid"] == str(replacement)
    choice.pop("curation_uuid")
    missing = build()
    assert not missing.ok
    assert any("curation_uuid" in error for error in missing.errors)


def test_run_v2_unit_match_checks_an_input_plan_before_the_database():
    """An input plan with explicit args, or a not-ok input plan, raises
    before any table access."""
    from spyglass.spikesorting.v2._unit_match_planning import (
        UnitMatchInputPlan,
    )
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    def _plan_with(errors):
        return UnitMatchInputPlan(
            matcher_params_name="unitmatch_default",
            curation_strategy="root",
            curations=[{"sorting_id": _S0, "curation_id": 0}],
            rows=[],
            errors=errors,
        )

    with pytest.raises(PipelineInputError, match="not both"):
        run_v2_unit_match(_plan_with([]), curation_choices={})
    with pytest.raises(PipelineInputError, match="for every sort"):
        run_v2_unit_match(_plan_with(["sort sort-0 ..."]))
    with pytest.raises(PipelineInputError, match="lacks a curation_uuid"):
        run_v2_unit_match(_plan_with([]))


@pytest.mark.parametrize("form", ["members", "sorts"])
def test_runner_refuses_unpinned_plans_before_loading_tables(monkeypatch, form):
    import sys

    from spyglass.spikesorting.v2._unit_match_planning import UnitMatchInputPlan
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

    # Make the database boundary unavailable so querying it before validating
    # the plan fails independently of any local database configuration.
    monkeypatch.setitem(
        sys.modules, "spyglass.spikesorting.v2.unit_matching", None
    )
    if form == "members":
        plan = UnitMatchPlan(
            session_group_owner="owner",
            session_group_name="group",
            matcher_params_name="unitmatch_default",
            curation_strategy="manual",
            curation_choices={0: {"sorting_id": _S0, "curation_id": 1}},
            rows=[],
        )
    else:
        plan = UnitMatchInputPlan(
            matcher_params_name="unitmatch_default",
            curation_strategy="manual",
            curations=[{"sorting_id": _S0, "curation_id": 1}],
            rows=[],
        )
    with pytest.raises(PipelineInputError, match="lacks a curation_uuid"):
        run_v2_unit_match(plan)
