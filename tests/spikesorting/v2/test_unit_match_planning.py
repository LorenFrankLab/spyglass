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

from spyglass.spikesorting.v2._matching.planning import (
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
    from spyglass.spikesorting.v2._matching.planning import (
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
    from spyglass.spikesorting.v2._matching.planning import (
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


@pytest.mark.parametrize("form", ["sorts", "group"])
def test_plan_display_keeps_nullable_integer_ids_with_unresolved_inputs(form):
    import pandas as pd

    from spyglass.spikesorting.v2._matching.preflight import (
        preflight_v2_unit_match,
    )

    choices = [_cur(_S0, 0, -1), _cur(_S0, 1, 0)]
    if form == "group":
        plan = _plan(
            [_member(0, "day1.nwb", choices), _member(1, "day2.nwb", [])],
            "final_curated",
        )
    else:
        plan = _input_plan(
            [_sort(_S0, choices), _sort(_S1, [])], "final_curated"
        )
    frame = plan.as_dataframe()
    assert str(frame["curation_id"].dtype) == "Int64"
    assert frame.iloc[0]["curation_id"] == 1
    assert pd.isna(frame.iloc[1]["curation_id"])
    if form == "group":
        assert str(frame["member_index"].dtype) == "Int64"
        assert frame["member_index"].tolist() == [0, 1]

    # An incomplete input has no canonical input index. A displayed preflight
    # table must preserve the resolved integer beside its missing placeholder.
    plan.rows[0]["input_index"] = 0
    plan.preflight = preflight_v2_unit_match(plan)
    frame = plan.as_dataframe()
    assert str(frame["input_index"].dtype) == "Int64"
    assert frame.iloc[0]["input_index"] == 0
    assert pd.isna(frame.iloc[1]["input_index"])


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
    from spyglass.spikesorting.v2._matching.planning import UnitMatchInputPlan
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

    from spyglass.spikesorting.v2._matching.planning import UnitMatchInputPlan
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


@pytest.fixture
def explicit_matching_plan(monkeypatch):
    """Real immutable snapshots with isolated selection and source boundaries."""
    import json
    from types import SimpleNamespace

    from spyglass.spikesorting.v2 import analysis_selection
    from spyglass.spikesorting.v2._curation.unit_selection import (
        UnitSelectionSnapshot,
        unit_selection_hash,
    )
    from spyglass.spikesorting.v2._matching import inputs
    from spyglass.spikesorting.v2.curation_api import CurationRef

    refs = [
        CurationRef(uuid.UUID(int=index + 10), 3, uuid.UUID(int=index + 100))
        for index in range(2)
    ]
    state = SimpleNamespace(
        refs=refs,
        resolved=[],
        prepared=[],
        source_hash="checked-sources",
        preparation_error=None,
        eligibility_error=None,
        changed_labels=False,
        resolver_kwargs=[],
    )

    def resolve(pin, *, policy, **kwargs):
        state.resolved.append((pin, policy))
        state.resolver_kwargs.append(kwargs)
        if state.eligibility_error is not None:
            raise state.eligibility_error
        sorting_id = (
            pin.sorting_id
            if isinstance(pin, CurationRef)
            else pin["sorting_id"]
        )
        ref = next(
            value for value in refs if str(value.sorting_id) == str(sorting_id)
        )
        labels = {"1": ["accept"], "2": ["mua"], "3": ["noise"], "4": []}
        if state.changed_labels:
            labels["1"] = ["accept", "changed"]
        label_policy = {
            key: list(value)
            for key, value in analysis_selection.V2_UNIT_SELECTION_POLICIES[
                policy
            ].items()
        }
        included = (
            [1]
            if policy == "v2_accepted_single_units"
            else [1, 2] if policy == "v2_accepted_neural_units" else [1, 2, 4]
        )
        criteria = kwargs.get("unit_criteria") or {}
        excluded = {
            key: "excluded by reviewed policy"
            for key in labels
            if int(key) not in included
        }
        data = {
            "sorting_id": str(ref.sorting_id),
            "curation_id": ref.curation_id,
            "curation_uuid": str(ref.curation_uuid),
            "policy_name": policy,
            "policy": label_policy,
            "unit_criteria": criteria,
            "included_unit_ids": included,
            "excluded_units": excluded,
            "labels_by_unit": labels,
            "selection_provenance": {
                "curation_uuid": str(ref.curation_uuid),
                "label_policy": label_policy,
                "unit_criteria": criteria,
                "evaluation_id": None,
                "evaluation_recipes": None,
                "annotation_sets": [],
            },
        }
        data["selection_hash"] = unit_selection_hash(data)
        return UnitSelectionSnapshot(
            ref,
            json.dumps(data),
            evaluation=kwargs.get("evaluation"),
            annotation_sets=tuple(kwargs.get("annotation_sets", ())),
        )

    def prepare(curations, matcher_params_name, **kwargs):
        assert kwargs["allow_rebuild"] is False
        assert kwargs["validate_backend"] is True
        state.prepared.append((curations, matcher_params_name, kwargs))
        if state.preparation_error is not None:
            raise state.preparation_error
        # Canonical ordering is determined by physical acquisition coverage,
        # independently of the order in which the user supplied curations.
        ordered = [
            {
                **pin,
                "input_index": index,
                "source_kind": "recording",
                "source_id": uuid.UUID(int=index + 200),
                "recordings": [
                    {
                        "nwb_file_name": f"epoch-{index}.nwb",
                        "interval_list_name": "selected epoch",
                        "source_coverage": {
                            "version": 1,
                            "source_object_id": f"raw-source-{index}",
                            "sample_ranges": [[index * 100, index * 100 + 50]],
                            "time_ranges": [[index * 10, index * 10 + 4.9]],
                        },
                    }
                ],
            }
            for index, pin in enumerate(reversed(curations))
        ]
        return {
            "inputs": ordered,
            "input_rows": [],
            "recording_rows": [],
            "unit_selections": kwargs["unit_selections"],
            "input_set_hash": state.source_hash,
            "unitmatch_id": uuid.UUID(int=900),
        }

    monkeypatch.setattr(analysis_selection, "resolve_unit_selection", resolve)
    monkeypatch.setattr(inputs, "prepare_match_inputs", prepare)
    return state


def test_explicit_curation_plan_freezes_population_and_reviews_source_order(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert plan.ok, plan.errors
    assert plan.unit_policy == "v2_accepted_single_units"
    assert [
        selection.included_unit_ids for selection in plan.unit_selections
    ] == [(1,), (1,)]
    assert plan.curations[0]["curation_uuid"] == str(
        state.refs[0].curation_uuid
    )
    frame = plan.as_dataframe()
    assert str(frame["curation_id"].dtype) == "Int64"
    assert str(frame["input_index"].dtype) == "Int64"
    assert frame["input_index"].tolist() == [0, 1]
    assert frame["sorting_id"].tolist() == [
        str(ref.sorting_id) for ref in reversed(state.refs)
    ]
    assert frame["n_included_units"].tolist() == [1, 1]
    assert frame["n_excluded_units"].tolist() == [3, 3]
    assert frame["preflight_status"].tolist() == ["ready", "ready"]
    assert plan.input_set_hash == "checked-sources"
    assert "1 included, 3 excluded" in plan.summary()
    assert "revalidated before writes" in plan.summary()
    assert set(plan.preflight.as_dataframe()["status"]) == {"passed"}
    assert frame.iloc[0]["source_coverage"][0]["sample_ranges"] == [[0, 50]]
    frame.iloc[0]["source_coverage"][0]["sample_ranges"][0][0] = 999
    assert plan.as_dataframe().iloc[0]["source_coverage"][0][
        "sample_ranges"
    ] == [[0, 50]]
    # Copying a report never offers a mutable handle to the frozen population.
    changed = plan.unit_selections[0].snapshot()
    changed["included_unit_ids"].append(4)
    assert plan.unit_selections[0].included_unit_ids == (1,)


def test_explicit_policy_choice_includes_unreviewed_units_visibly(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    plan = plan_v2_unit_match_from_curations(
        explicit_matching_plan.refs, unit_policy="v2_unflagged_units"
    )
    assert plan.ok
    assert plan.unit_selections[0].included_unit_ids == (1, 2, 4)
    assert any(
        "included units are unlabeled" in value for value in plan.warnings
    )
    assert "v2_unflagged_units" in plan.summary()


@pytest.mark.parametrize(
    ("snapshot_policy", "explicit_policy"),
    [
        ("v2_unflagged_units", "v2_accepted_single_units"),
        ("v2_accepted_neural_units", "v2_accepted_single_units"),
        ("v2_accepted_single_units", "v2_unflagged_units"),
        ("v2_accepted_single_units", "v2_accepted_neural_units"),
        ("v2_unflagged_units", "v2_accepted_neural_units"),
        ("v2_accepted_neural_units", "v2_unflagged_units"),
    ],
)
def test_explicit_policy_conflicts_reject_supplied_populations(
    explicit_matching_plan, snapshot_policy, explicit_policy
):
    from spyglass.spikesorting.v2.analysis_selection import (
        resolve_unit_selection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    populations = [
        resolve_unit_selection(ref, policy=snapshot_policy)
        for ref in state.refs
    ]
    with pytest.raises(ValueError, match="unit_policy conflicts"):
        plan_v2_unit_match_from_curations(
            state.refs,
            unit_policy=explicit_policy,
            unit_selections=populations,
        )
    assert not state.prepared


@pytest.mark.parametrize(
    "policy",
    [
        "v2_accepted_single_units",
        "v2_accepted_neural_units",
        "v2_unflagged_units",
    ],
)
def test_explicit_policy_agrees_with_supplied_populations(
    explicit_matching_plan, policy
):
    from spyglass.spikesorting.v2.analysis_selection import (
        resolve_unit_selection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    populations = [
        resolve_unit_selection(ref, policy=policy) for ref in state.refs
    ]
    plan = plan_v2_unit_match_from_curations(
        state.refs, unit_policy=policy, unit_selections=populations
    )
    assert plan.ok, plan.errors
    assert plan.unit_policy == policy


def test_omitted_policy_preserves_each_supplied_population_policy(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.analysis_selection import (
        resolve_unit_selection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    policies = ["v2_accepted_single_units", "v2_unflagged_units"]
    populations = [
        resolve_unit_selection(ref, policy=policy)
        for ref, policy in zip(state.refs, policies, strict=True)
    ]
    plan = plan_v2_unit_match_from_curations(
        state.refs, unit_policy=None, unit_selections=populations
    )
    assert plan.ok, plan.errors
    assert plan.unit_policy == "provided selections"
    assert plan.as_dataframe()["unit_policy"].tolist() == list(
        reversed(policies)
    )
    with pytest.raises(ValueError, match="unit_policy conflicts"):
        plan_v2_unit_match_from_curations(
            state.refs,
            unit_policy="v2_accepted_single_units",
            unit_selections=populations,
        )


def test_explicit_plan_retains_typed_evidence_populations(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.analysis_selection import (
        resolve_unit_selection,
    )
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )

    state = explicit_matching_plan
    evaluation = object()
    annotations = (object(),)
    criteria = {"observed_firing_rate_hz": {"ge": 0}}
    populations = [
        resolve_unit_selection(
            ref,
            policy="v2_accepted_neural_units",
            evaluation=evaluation,
            unit_criteria=criteria,
            annotation_sets=annotations,
        )
        for ref in state.refs
    ]
    plan = plan_v2_unit_match_from_curations(
        state.refs, unit_selections=populations
    )
    assert plan.ok, plan.errors
    assert plan.unit_policy == "v2_accepted_neural_units"
    assert plan.unit_selections[0] is populations[0]
    assert (
        plan.as_dataframe()["unit_policy"].tolist()
        == ["v2_accepted_neural_units"] * 2
    )
    assert preflight_v2_unit_match(plan).ok
    for forwarded in state.resolver_kwargs[-2:]:
        assert forwarded["evaluation"] is evaluation
        assert forwarded["annotation_sets"] == annotations
        assert forwarded["unit_criteria"] == criteria
    with pytest.raises(ValueError, match="exactly one unit selection"):
        plan_v2_unit_match_from_curations(
            state.refs, unit_selections=populations[:1]
        )
    with pytest.raises(ValueError, match="different curation generation"):
        plan_v2_unit_match_from_curations(
            state.refs, unit_selections=list(reversed(populations))
        )
    with pytest.raises(TypeError, match="UnitSelectionSnapshot objects"):
        plan_v2_unit_match_from_curations(
            state.refs,
            unit_selections=[value.snapshot() for value in populations],
        )


def test_explicit_plan_reports_missing_assets_without_healing(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    state.preparation_error = FileNotFoundError(
        "Recording cache is missing; explicitly rebuild the recording before matching."
    )
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert not plan.ok
    assert "Recording cache is missing" in plan.errors[0]
    checks = plan.preflight.as_dataframe().set_index("check")
    assert checks.loc["matching_inputs", "status"] == "blocked"
    for name in ("source_frames_and_coverage", "geometry", "backend"):
        assert checks.loc[name, "status"] == "not_run"
    assert "remaining checks not run" in plan.preflight.summary()


def test_explicit_plan_collects_input_errors_before_shared_validation(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    state.eligibility_error = ValueError(
        "Curation generation changed; rebuild the plan."
    )
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert not plan.ok
    assert len(plan.errors) == 2
    assert "Input 0" in plan.errors[0] and "Input 1" in plan.errors[1]
    assert not state.prepared


def test_stale_ref_finding_retains_generation_recovery_contract(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.exceptions import CurationNotFoundError
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    state = explicit_matching_plan
    state.eligibility_error = CurationNotFoundError(
        "The numeric curation key was replaced."
    )
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert not plan.ok
    assert all(
        "changed curation_uuid since planning" in value for value in plan.errors
    )
    assert not state.prepared


@pytest.mark.parametrize("drift", ["eligibility", "source"])
def test_matching_preflight_rejects_drift_from_reviewed_plan(
    explicit_matching_plan, drift
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )

    state = explicit_matching_plan
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert plan.ok
    if drift == "eligibility":
        state.changed_labels = True
        expected = "Unit selection changed since planning"
    else:
        state.source_hash = "different-source-frames"
        expected = "changed after planning"
    checked = preflight_v2_unit_match(plan)
    assert not checked.ok
    assert any(expected in value for value in checked.errors)


def test_explicit_plan_rejects_empty_duplicate_and_untyped_inputs():
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
    )

    ref = CurationRef(uuid.UUID(int=10), 3, uuid.UUID(int=100))
    with pytest.raises(ValueError, match="at least one curation"):
        plan_v2_unit_match_from_curations([])
    with pytest.raises(ValueError, match="one curation generation"):
        plan_v2_unit_match_from_curations([ref, ref])
    with pytest.raises(TypeError, match="CurationRef objects"):
        plan_v2_unit_match_from_curations([{"sorting_id": str(ref.sorting_id)}])


def test_public_preflight_refuses_primitive_population_bypass(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )

    state = explicit_matching_plan
    plan = plan_v2_unit_match_from_curations(state.refs)
    plan.unit_selections = tuple(
        value.snapshot() for value in plan.unit_selections
    )
    state.prepared.clear()
    report = preflight_v2_unit_match(plan)
    assert not report.ok
    assert all(
        "typed UnitSelectionSnapshot objects" in value
        for value in report.errors
    )
    assert not state.prepared


@pytest.mark.parametrize("form", ["sorts", "group"])
def test_public_preflight_blocks_partially_resolved_requested_inputs(
    explicit_matching_plan,
    monkeypatch,
    form,
):
    from spyglass.spikesorting.v2._matching import preflight
    from spyglass.spikesorting.v2._matching.planning import (
        build_unit_match_input_plan,
    )

    state = explicit_matching_plan
    ref = state.refs[0]
    choice = {
        "sorting_id": str(ref.sorting_id),
        "curation_id": ref.curation_id,
        "curation_uuid": str(ref.curation_uuid),
        "parent_curation_id": 0,
        "curation_source": "figpack",
        "description": "reviewed",
    }
    if form == "group":
        plan = _plan(
            [
                _member(0, "epoch-0.nwb", [choice]),
                _member(1, "epoch-1.nwb", []),
            ],
            "final_curated",
        )
        monkeypatch.setattr(
            preflight,
            "_validate_group_plan",
            lambda plan: pytest.fail(
                "An unresolved plan reached DB validation."
            ),
        )
    else:
        sorts = [
            {
                "sorting_id": str(value.sorting_id),
                "source_kind": "recording",
                "source_id": str(uuid.UUID(int=200 + index)),
                "nwb_file_names": (f"epoch-{index}.nwb",),
                "interval_list_names": ("selected epoch",),
                "choices": [choice] if index == 0 else [],
            }
            for index, value in enumerate(state.refs)
        ]
        plan = build_unit_match_input_plan(
            sorts=sorts,
            curation_strategy="final_curated",
            matcher_params_name="unitmatch_default",
        )
    assert not plan.ok
    assert "Pinned inputs: 1 of 2 requested" in plan.summary()
    report = preflight.preflight_v2_unit_match(plan)
    assert not report.ok
    assert "has no" in report.errors[0]
    assert "remaining checks not run" in report.summary()
    assert not state.prepared


@pytest.mark.parametrize("mutation", ["pins", "rows", "populations"])
def test_public_preflight_rejects_mutated_plan_review_integrity(
    explicit_matching_plan,
    mutation,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )

    state = explicit_matching_plan
    plan = plan_v2_unit_match_from_curations(state.refs)
    if mutation == "pins":
        plan.curations = plan.curations[:1]
        plan.unit_selections = plan.unit_selections[:1]
    elif mutation == "rows":
        plan.rows[0]["curation_uuid"] = str(uuid.UUID(int=9999))
    else:
        plan.unit_selections = tuple(reversed(plan.unit_selections))
    state.prepared.clear()
    report = preflight_v2_unit_match(plan)
    assert not report.ok
    assert any("rebuild" in value.lower() for value in report.errors)
    assert not state.prepared


def test_public_preflight_rechecks_historical_asset_findings(
    explicit_matching_plan,
):
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_curations,
        preflight_v2_unit_match,
    )

    state = explicit_matching_plan
    state.preparation_error = FileNotFoundError(
        "Restore a missing source cache."
    )
    plan = plan_v2_unit_match_from_curations(state.refs)
    assert not plan.ok
    assert not plan.curation_resolution_errors
    state.preparation_error = None
    assert preflight_v2_unit_match(plan).ok


def test_match_runner_passes_frozen_population_and_verified_source_hash(
    explicit_matching_plan,
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._orchestration import matching

    state = explicit_matching_plan
    plan = matching.plan_v2_unit_match_from_curations(state.refs)
    inserted = []
    populated = []

    class Relation:
        def __and__(self, key):
            return self

        def __bool__(self):
            return True

        def __len__(self):
            return 2

        def fetch1(self, field):
            assert field == "n_pairs"
            return 3

    def insert(curations, matcher, **kwargs):
        inserted.append((curations, matcher, kwargs))
        return {"unitmatch_id": uuid.UUID(int=900)}

    selections = SimpleNamespace(
        insert_inputs=insert,
        pinned_curations=lambda key: {},
        _divergent_electrode_space_members=lambda pins: [],
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.unit_matching",
        SimpleNamespace(
            MatcherParameters=Relation(),
            UnitMatch=Relation(),
            TrackedUnit=Relation(),
            UnitMatchSelection=selections,
        ),
    )
    monkeypatch.setattr(
        matching, "_populate_once", lambda table, key: populated.append(key)
    )
    monkeypatch.setattr(matching, "_unit_match_input_summaries", lambda key: ())
    result = matching.run_v2_unit_match(plan)
    assert len(inserted) == 1
    assert inserted[0][0] == plan.curations
    assert inserted[0][2]["expected_input_set_hash"] == plan.input_set_hash
    assert inserted[0][2]["unit_selections"] == [
        value.snapshot() for value in plan.unit_selections
    ]
    assert result["n_pairs"] == 3 and result["n_tracked_units"] == 2
    assert len(populated) == 2
