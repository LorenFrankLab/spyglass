"""SortGroupV2 construction and reference-mode tests for the v2 single-session pipeline."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from tests.spikesorting.v2._ingest_helpers import (
    _clean_session_v2,
)

# ---------- SortGroupV2 by shank -------------------------------------------


@pytest.mark.slow
def test_set_group_by_shank_creates_one_group_per_shank(polymer_smoke_session):
    """The 128-channel 4-shank polymer fixture yields 4 sort groups of 32."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    # Clean any rows a previous module run may have left.
    _clean_session_v2(polymer_smoke_session)

    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    masters = SortGroupV2 & polymer_smoke_session
    assert len(masters) == 4
    parts = SortGroupV2.SortGroupElectrode & polymer_smoke_session
    assert len(parts) == 128
    counts = {
        sg_id: len(
            SortGroupV2.SortGroupElectrode
            & polymer_smoke_session
            & {"sort_group_id": sg_id}
        )
        for sg_id in masters.fetch("sort_group_id")
    }
    assert all(c == 32 for c in counts.values()), counts


@pytest.mark.slow
def test_set_group_by_shank_can_include_bad_channels(polymer_smoke_session):
    """``omit_bad_channels=False`` includes channels the default omits."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    rows = (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
        "electrode_group_name",
        "electrode_id",
        "bad_channel",
        as_dict=True,
        order_by="electrode_id",
    )
    first = rows[0]

    def _set_bad_channel(value):
        Electrode.update1(
            {
                "nwb_file_name": nwb_file_name,
                "electrode_group_name": first["electrode_group_name"],
                "electrode_id": int(first["electrode_id"]),
                "bad_channel": value,
            }
        )

    try:
        _set_bad_channel("True")
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
        parts = SortGroupV2.SortGroupElectrode & polymer_smoke_session
        assert len(parts) == 127

        _clean_session_v2(polymer_smoke_session)
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, omit_bad_channels=False
        )
        parts = SortGroupV2.SortGroupElectrode & polymer_smoke_session
        assert len(parts) == 128
    finally:
        _clean_session_v2(polymer_smoke_session)
        _set_bad_channel(first["bad_channel"])


@pytest.mark.slow
def test_set_group_by_shank_refuses_overlapping_rerun(polymer_smoke_session):
    """Rerun without an override raises (fixes v1 silent-overwrite bug)."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    with pytest.raises(ValueError, match="silently extend"):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)


@pytest.mark.slow
def test_set_group_by_shank_overwrite_requires_confirm(polymer_smoke_session):
    """``delete_existing_entries=True, confirm=False`` returns a preview
    and raises; ``confirm=True`` performs the cautious delete + reinsert."""
    from spyglass.spikesorting.v2.recording import (
        DeletionPreview,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    preview = SortGroupV2.preview_existing_entries(nwb_file_name)
    assert isinstance(preview, DeletionPreview)
    assert preview.sort_group_rows == 4
    assert preview.electrode_rows == 128

    with pytest.raises(ValueError, match="requires confirm=True"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            delete_existing_entries=True,
        )

    # Confirmed overwrite succeeds and produces a fresh set.
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name,
        delete_existing_entries=True,
        confirm=True,
    )
    assert len(SortGroupV2 & polymer_smoke_session) == 4


# ---------- SortGroupV2 by electrode-table column --------------------------


@pytest.mark.slow
def test_set_group_by_column_matches_by_shank(polymer_smoke_session):
    """Grouping by the ``probe_shank`` column reproduces ``by_shank``."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    shanks = sorted(
        set(
            int(s)
            for s in (Electrode & polymer_smoke_session).fetch("probe_shank")
        )
    )
    groups = [[shank] for shank in shanks]
    SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        electrode_column="probe_shank",
        value_groups=groups,
    )
    assert len(SortGroupV2 & polymer_smoke_session) == 4
    assert len(SortGroupV2.SortGroupElectrode & polymer_smoke_session) == 128


@pytest.mark.slow
def test_set_group_by_column_lists_valid_columns_on_typo(
    polymer_smoke_session,
):
    """A bogus column name fails with the valid-column list in the error."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    with pytest.raises(ValueError) as excinfo:
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=polymer_smoke_session["nwb_file_name"],
            electrode_column="probe_shanke",  # typo
            value_groups=[[0]],
        )
    msg = str(excinfo.value)
    assert "probe_shanke" in msg
    assert "probe_shank" in msg  # the suggestion is implicit -- it's listed


def test_electrode_group_sort_key_tolerates_non_numeric(dj_conn):
    """``set_group_by_shank``'s group ordering must not assume numeric
    ``electrode_group_name`` values.

    A plain ``int(name)`` raised ``ValueError`` on free-form NWB group
    names (e.g. ``"probeA"``). The sort key now sorts numeric names
    numerically (and ahead of non-numeric) and non-numeric names
    lexically, so valid datasets no longer crash while purely numeric
    names keep their natural order. (``dj_conn`` only because importing
    the module declares its schema.)
    """
    from spyglass.spikesorting.v2.recording import _electrode_group_sort_key

    # Numeric names keep NUMERIC (not lexical) order: "10" after "2".
    assert sorted(["10", "2", "1"], key=_electrode_group_sort_key) == [
        "1",
        "2",
        "10",
    ]
    # Non-numeric names do not raise and sort after the numeric ones.
    assert sorted(
        ["probeB", "2", "probeA", "1"], key=_electrode_group_sort_key
    ) == ["1", "2", "probeA", "probeB"]


def test_handle_existing_rejects_duplicate_sort_group_ids(dj_conn):
    """Explicit ``sort_group_ids`` with intra-list duplicates raises up
    front instead of failing late on a DataJoint duplicate-key error
    when the master rows are inserted. Auto-allocated ranges
    (``explicit_sort_group_ids=False``) are always unique, so the check
    is gated on the explicit path. (Found by generalizing from the
    merge-curation ``len(list)`` vs ``len(set)`` bug -- the same pattern
    lived here.)
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    # Duplicates surface on a fresh session (no existing rows): the
    # check runs BEFORE the ``len(existing) == 0`` early return so it
    # fires either way.
    with pytest.raises(ValueError, match="duplicate"):
        SortGroupV2._handle_existing(
            nwb_file_name="nonexistent.nwb",
            new_sort_group_ids=[3, 3, 4],
            explicit_sort_group_ids=True,
            delete_existing_entries=False,
            confirm=False,
        )
    # The auto-allocated path is exempt (range() is unique by
    # construction); a same-shape list with no duplicate must NOT raise.
    SortGroupV2._handle_existing(
        nwb_file_name="nonexistent.nwb",
        new_sort_group_ids=[3, 4, 5],
        explicit_sort_group_ids=True,
        delete_existing_entries=False,
        confirm=False,
    )


@pytest.mark.slow
def test_set_group_by_column_surfaces_unitrode_skip(polymer_smoke_session):
    """``set_group_by_electrode_table_column`` returns a skip summary for
    a single-channel (unitrode) group instead of silently succeeding.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    eids = sorted(
        int(e)
        for e in (
            Electrode & {"nwb_file_name": nwb_file_name, "bad_channel": "False"}
        ).fetch("electrode_id")
    )
    assert len(eids) >= 3

    skipped = SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        electrode_column="electrode_id",
        value_groups=[[eids[0]], [eids[1], eids[2]]],
        omit_unitrode=True,
    )
    assert isinstance(skipped, list)
    assert any(s["reason"] == "unitrode" for s in skipped), skipped
    # The kept 2-channel group was still created.
    assert len(SortGroupV2 & polymer_smoke_session) == 1


# ---------- SortGroupV2 reference-mode split --------------------------------


@pytest.mark.slow
def test_sort_group_reference_mode_enforced(polymer_smoke_session):
    """SortGroupV2.insert1 enforces the reference_mode / electrode invariants.

    ``reference_mode`` is validated against the ``ReferenceMode`` Literal
    at insert time (a typo raises), ``"specific"`` requires a
    ``reference_electrode_id``, and the non-specific modes reject one. A
    valid ``"specific"`` row stores both fields; a default (mode omitted)
    row stores ``"none"`` with a NULL electrode.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    base = {"nwb_file_name": nwb_file_name, "sort_group_id": 900}

    # Typo'd mode rejected by the Literal guard (varchar, not enum).
    with pytest.raises(ValueError, match="reference_mode"):
        SortGroupV2.insert1({**base, "reference_mode": "globalmedian"})

    # specific requires an electrode id.
    with pytest.raises(ValueError, match="requires a non-null"):
        SortGroupV2.insert1({**base, "reference_mode": "specific"})

    # non-specific must not carry an electrode id.
    with pytest.raises(ValueError, match="must leave"):
        SortGroupV2.insert1(
            {
                **base,
                "reference_mode": "global_median",
                "reference_electrode_id": 3,
            }
        )

    # A valid specific row inserts and stores both fields.
    SortGroupV2.insert1(
        {**base, "reference_mode": "specific", "reference_electrode_id": 0}
    )
    stored = (SortGroupV2 & base).fetch1(
        "reference_mode", "reference_electrode_id"
    )
    assert stored == ("specific", 0)

    # A default (mode omitted) row stores "none" with NULL electrode.
    default_base = {"nwb_file_name": nwb_file_name, "sort_group_id": 901}
    SortGroupV2.insert1(default_base)
    mode, eid = (SortGroupV2 & default_base).fetch1(
        "reference_mode", "reference_electrode_id"
    )
    assert mode == "none"
    assert eid is None


# ===========================================================================
# Edge-case coverage for SortGroupV2 reference handling and group construction.
#
# reference_mode validation (the integer sentinel was replaced by a
# ``reference_mode`` varchar validated against the ReferenceMode Literal),
# the all-shanks-filtered guard, the omit_ref no-op under the default mode,
# additive inserts, length-mismatch, and the empty-match guard on the
# electrode-table-column constructor. The RecordingSelection duplicate guard
# (test_recording_selection_raises_on_duplicate_logical_identity, len==2) and
# _electrode_group_sort_key (test_electrode_group_sort_key_tolerates_non_numeric)
# are already covered, so they are not re-added here.
# ===========================================================================


@pytest.mark.usefixtures("dj_conn")
def test_sort_group_rejects_invalid_reference_mode():
    """``SortGroupV2.insert1`` rejects an unknown ``reference_mode``.

    The integer ``sort_reference_electrode_id`` sentinel was replaced by
    a ``reference_mode`` varchar validated against the ``ReferenceMode``
    Literal at the insert boundary. An invalid mode (here ``"banana"``)
    raises before any DB write -- the varchar's typo guard standing in for a
    MySQL enum. The message names the valid modes.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    with pytest.raises(ValueError, match="ReferenceMode"):
        SortGroupV2().insert1(
            {
                "nwb_file_name": "a27_irrelevant_.nwb",
                "sort_group_id": 0,
                "reference_mode": "banana",
            }
        )


@pytest.mark.usefixtures("dj_conn")
def test_sort_group_reference_electrode_id_consistency():
    """``reference_electrode_id`` is non-null iff mode=='specific'.

    The split's second invariant: a 'specific' reference needs a channel to
    subtract, and a non-specific mode must not carry a stray channel id the
    runtime would silently ignore. Both directions raise at insert.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    base = {"nwb_file_name": "a27_irrelevant_.nwb", "sort_group_id": 0}
    # 'specific' without a channel id.
    with pytest.raises(ValueError, match="requires a non-null"):
        SortGroupV2().insert1(
            {
                **base,
                "reference_mode": "specific",
                "reference_electrode_id": None,
            }
        )
    # non-specific WITH a stray channel id.
    with pytest.raises(ValueError, match="must leave"):
        SortGroupV2().insert1(
            {
                **base,
                "reference_mode": "global_median",
                "reference_electrode_id": 3,
            }
        )


@pytest.mark.slow
def test_set_group_by_shank_all_shanks_filtered_raises(polymer_smoke_session):
    """When every shank is filtered out, ``set_group_by_shank`` raises.

    The smoke fixture's 4 shanks all live in one electrode group ("0"), so
    ``omit_ref_electrode_group=True`` with ``reference_mode='specific'`` and a
    reference electrode in that group skips ALL shanks (the skip is keyed on
    the electrode group, and the whole probe is one group). No sort group
    survives -> ValueError naming "no sort groups".
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="no sort groups produced"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            omit_ref_electrode_group=True,
            reference_mode="specific",
            reference_electrode_id=0,
        )


@pytest.mark.slow
def test_set_group_by_shank_omit_ref_noop_under_default_mode(
    polymer_smoke_session,
):
    """``omit_ref_electrode_group=True`` is a no-op when no group resolves
    to a ``"specific"`` reference.

    The omit-reference-group skip is gated on a group's *resolved* reference
    being ``"specific"`` (a ``"none"`` / ``"global_median"`` reference names
    no electrode to exclude). The smoke fixture's electrodes all carry
    ``original_reference_electrode = -1``, so the default auto-inheritance
    resolves every group to ``"none"`` and the flag drops nothing: all 4
    shanks become sort groups and the returned skip list has no
    reference-group entries.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    skipped = SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name,
        omit_ref_electrode_group=True,  # no-op: config resolves to "none"
    )

    assert len(SortGroupV2 & polymer_smoke_session) == 4, (
        "omit_ref_electrode_group must not drop groups when every group "
        "resolves to reference_mode='none'"
    )
    assert not any(
        s.get("reason") == "reference_electrode_group" for s in skipped
    ), "no group should be skipped for the reference-electrode reason"


# ---------- SortGroupV2 reference inheritance ------------------------------
#
# The default referencing behavior now inherits each group's configured
# reference (Electrode.original_reference_electrode) instead of defaulting to
# no reference. The smoke fixture ships every electrode at -1 (no reference),
# so these tests temporarily set a real configured reference to exercise the
# inheritance paths, restoring it afterward via the context manager below so
# the session-scoped fixture is unchanged for other tests.


@contextmanager
def _temp_original_reference(nwb_file_name, updates):
    """Temporarily override ``Electrode.original_reference_electrode`` values.

    ``updates`` maps ``electrode_id`` -> reference value (an electrode id or a
    ``-1`` / ``-2`` sentinel). The prior values are restored on exit.
    """
    from spyglass.common.common_ephys import Electrode

    rows = (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
        "electrode_group_name",
        "electrode_id",
        "original_reference_electrode",
        as_dict=True,
    )
    prior = {int(r["electrode_id"]): r for r in rows}

    def _set(eid, ref):
        r = prior[int(eid)]
        Electrode.update1(
            {
                "nwb_file_name": nwb_file_name,
                "electrode_group_name": r["electrode_group_name"],
                "electrode_id": int(eid),
                "original_reference_electrode": int(ref),
            }
        )

    try:
        for eid, ref in updates.items():
            _set(eid, ref)
        yield
    finally:
        for eid in updates:
            _set(eid, prior[int(eid)]["original_reference_electrode"])


def _electrodes_by_shank(nwb_file_name):
    """Return ``{probe_shank: [good electrode ids]}`` for a session."""
    from spyglass.common.common_ephys import Electrode

    rows = (
        Electrode & {"nwb_file_name": nwb_file_name, "bad_channel": "False"}
    ).fetch("electrode_id", "probe_shank", as_dict=True)
    by_shank: dict[int, list[int]] = {}
    for row in rows:
        by_shank.setdefault(int(row["probe_shank"]), []).append(
            int(row["electrode_id"])
        )
    return by_shank


@pytest.mark.slow
def test_set_group_by_shank_inherits_specific_reference_from_config(
    polymer_smoke_session,
):
    """Default referencing inherits each group's configured reference.

    With no ``reference_mode`` / ``references`` argument, each sort group's
    reference is read from its members' ``original_reference_electrode``. Here
    one shank's electrodes are configured to reference an electrode on a
    DIFFERENT shank, so that group resolves to ``("specific", <that id>)``
    while a shank still configured ``-1`` resolves to ``("none", None)``.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    by_shank = _electrodes_by_shank(nwb_file_name)
    shanks = sorted(by_shank)
    assert len(shanks) >= 2
    cfg_shank, ref_shank = shanks[0], shanks[1]
    ref_id = sorted(by_shank[ref_shank])[0]
    # Reference cfg_shank to ref_id (on a different shank, so it is not a
    # member of cfg_shank's sort group and passes the membership guard).
    updates = {eid: ref_id for eid in by_shank[cfg_shank]}

    with _temp_original_reference(nwb_file_name, updates):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
        # Sort groups are created in (electrode_group, shank) sorted order on
        # a fresh session, so the i-th shank -> sort_group_id i.
        cfg_sg, none_sg = shanks.index(cfg_shank), shanks.index(ref_shank)
        cfg_members = {
            int(e)
            for e in (
                SortGroupV2.SortGroupElectrode
                & {**polymer_smoke_session, "sort_group_id": cfg_sg}
            ).fetch("electrode_id")
        }
        assert cfg_members == set(by_shank[cfg_shank])
        mode, eid = (
            SortGroupV2 & {**polymer_smoke_session, "sort_group_id": cfg_sg}
        ).fetch1("reference_mode", "reference_electrode_id")
        assert (mode, int(eid)) == ("specific", ref_id)
        mode, eid = (
            SortGroupV2 & {**polymer_smoke_session, "sort_group_id": none_sg}
        ).fetch1("reference_mode", "reference_electrode_id")
        assert mode == "none" and eid is None

    _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_set_group_by_shank_references_dict_and_missing_key(
    polymer_smoke_session,
):
    """A ``references`` mapping overrides config per electrode group.

    ``references={"0": -2}`` forces global-median on every sort group in
    electrode group ``"0"``; a mapping that omits a produced group raises; and
    combining ``references`` with ``reference_mode`` raises.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    e_group = str(
        (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
            "electrode_group_name"
        )[0]
    )

    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, references={e_group: -2}
    )
    modes = set((SortGroupV2 & polymer_smoke_session).fetch("reference_mode"))
    assert modes == {"global_median"}, modes

    _clean_session_v2(polymer_smoke_session)
    with pytest.raises(ValueError, match="missing from"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, references={"not-a-real-group": -2}
        )
    with pytest.raises(ValueError, match="not both"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            references={e_group: -2},
            reference_mode="none",
        )


@pytest.mark.slow
def test_set_group_by_shank_explicit_none_overrides_config(
    polymer_smoke_session,
):
    """An explicit ``reference_mode="none"`` forces ``"none"`` on every group,
    overriding a configured reference."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    by_shank = _electrodes_by_shank(nwb_file_name)
    all_eids = [eid for ids in by_shank.values() for eid in ids]
    # Config says global-median everywhere; the explicit override wins.
    with _temp_original_reference(nwb_file_name, {eid: -2 for eid in all_eids}):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, reference_mode="none"
        )
        modes = set(
            (SortGroupV2 & polymer_smoke_session).fetch("reference_mode")
        )
        assert modes == {"none"}, modes

    _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_set_group_by_shank_omit_ref_electrode_group(polymer_smoke_session):
    """``omit_ref_electrode_group`` skips the electrode group owning a group's
    *resolved* specific reference, and a missing reference raises.

    The skip now keys off each group's resolved reference (not a call-wide
    scalar). The fixture's shanks all share electrode group ``"0"``, so a
    specific reference inside ``"0"`` skips every sort group -> "no sort
    groups produced" (the skip runs BEFORE the in-group membership check,
    which would otherwise raise a different error). A specific reference id
    that is not an Electrode at all raises a clear error.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    by_shank = _electrodes_by_shank(nwb_file_name)
    all_eids = sorted(eid for ids in by_shank.values() for eid in ids)
    ref_in_group = all_eids[0]

    # AUTO-resolved specific: every electrode references one of its own
    # group's electrodes. With omit, the owning group ("0") is skipped before
    # the membership check, so no group survives.
    with _temp_original_reference(
        nwb_file_name, {eid: ref_in_group for eid in all_eids}
    ):
        with pytest.raises(ValueError, match="no sort groups produced"):
            SortGroupV2.set_group_by_shank(
                nwb_file_name=nwb_file_name, omit_ref_electrode_group=True
            )

    # A specific reference id absent from the Electrode table raises rather
    # than silently failing to find the group to omit.
    with pytest.raises(ValueError, match="not in the Electrode table"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            omit_ref_electrode_group=True,
            reference_mode="specific",
            reference_electrode_id=10_000_000,
        )


@pytest.mark.slow
def test_set_group_by_electrode_table_column_auto_and_global_override(
    polymer_smoke_session,
):
    """``set_group_by_electrode_table_column`` inherits per group by default
    and honors a call-wide ``reference_mode`` override.

    Grouping by ``probe_shank`` with no override inherits each group's config:
    one shank configured global-median (-2) resolves to ``"global_median"``
    while the others (still ``-1``) resolve to ``"none"``. A call-wide
    ``reference_mode="global_median"`` then forces CMR on every group.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    by_shank = _electrodes_by_shank(nwb_file_name)
    shanks = sorted(by_shank)
    assert len(shanks) >= 2
    cmr_shank = shanks[0]
    groups = [[s] for s in shanks]

    with _temp_original_reference(
        nwb_file_name, {eid: -2 for eid in by_shank[cmr_shank]}
    ):
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb_file_name,
            electrode_column="probe_shank",
            value_groups=groups,
        )
        # groups[i] == [shanks[i]] -> sort_group_id i on a fresh session.
        cmr_sg = shanks.index(cmr_shank)
        assert (
            SortGroupV2 & {**polymer_smoke_session, "sort_group_id": cmr_sg}
        ).fetch1("reference_mode") == "global_median"
        others = set(
            (
                SortGroupV2
                & polymer_smoke_session
                & f"sort_group_id != {cmr_sg}"
            ).fetch("reference_mode")
        )
        assert others == {"none"}, others

    _clean_session_v2(polymer_smoke_session)
    SortGroupV2.set_group_by_electrode_table_column(
        nwb_file_name=nwb_file_name,
        electrode_column="probe_shank",
        value_groups=groups,
        reference_mode="global_median",
    )
    modes = set((SortGroupV2 & polymer_smoke_session).fetch("reference_mode"))
    assert modes == {"global_median"}, modes
    _clean_session_v2(polymer_smoke_session)


@pytest.mark.usefixtures("dj_conn")
def test_set_group_reference_electrode_id_without_mode_raises():
    """``reference_electrode_id`` without ``reference_mode`` raises on both
    grouping helpers, before any electrode fetch (a likely user error: an id
    with the mode forgotten)."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    with pytest.raises(ValueError, match="only meaningful"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name="ref_guard_irrelevant_.nwb",
            reference_electrode_id=5,
        )
    with pytest.raises(ValueError, match="only meaningful"):
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name="ref_guard_irrelevant_.nwb",
            electrode_column="electrode_id",
            value_groups=[[0]],
            reference_electrode_id=5,
        )


@pytest.mark.slow
def test_set_group_by_shank_in_group_specific_reference_raises(
    polymer_smoke_session,
):
    """A resolved ``"specific"`` reference that is a member of its own sort
    group fails early at group creation (not deep in ``make``), via both
    auto-inheritance and an explicit ``references`` mapping.

    The fixture's shanks all share electrode group ``"0"``; configuring every
    electrode to reference one of group ``"0"``'s own electrodes makes the
    reference a member of whichever shank owns it, so the membership guard
    must raise rather than silently sort that shank with a dropped channel.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    by_shank = _electrodes_by_shank(nwb_file_name)
    all_eids = sorted(eid for ids in by_shank.values() for eid in ids)
    in_group_ref = all_eids[0]

    # Auto-inherited specific reference that is its own member -> raises.
    with _temp_original_reference(
        nwb_file_name, {eid: in_group_ref for eid in all_eids}
    ):
        with pytest.raises(ValueError, match="also a sort-group channel"):
            SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)

    # Same via an explicit `references` mapping carrying a specific id: the
    # id flows through the resolver to the membership guard and raises.
    e_group = str(
        (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
            "electrode_group_name"
        )[0]
    )
    with pytest.raises(ValueError, match="also a sort-group channel"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            references={e_group: in_group_ref},
        )
    _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_set_group_by_nonexistent_specific_reference_raises(
    polymer_smoke_session,
):
    """A ``"specific"`` reference id absent from the session raises at group
    creation even without ``omit_ref_electrode_group`` -- not later at
    ``Recording.populate``. Covers both grouping helpers."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="not in the Electrode table"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name,
            reference_mode="specific",
            reference_electrode_id=10_000_000,  # no such electrode
        )
    with pytest.raises(ValueError, match="not in the Electrode table"):
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb_file_name,
            electrode_column="probe_shank",
            value_groups=[[0]],
            reference_mode="specific",
            reference_electrode_id=10_000_000,
        )


@pytest.mark.usefixtures("dj_conn")
def test_reference_electrode_group_existence_and_uniqueness():
    """``_reference_electrode_group`` validates a specific reference: it must
    exist and belong to exactly one electrode group.

    Operates on a synthetic electrode recarray (the ``Electrode`` primary key
    allows the same ``electrode_id`` under different ``electrode_group_name``
    values, so an ambiguous owner must be rejected, not silently resolved to
    the first match).
    """
    import numpy as np

    from spyglass.spikesorting.v2.recording import _reference_electrode_group

    electrodes = np.array(
        [(0, "0"), (1, "0"), (1, "1"), (2, "1")],
        dtype=[("electrode_id", "i8"), ("electrode_group_name", "U8")],
    )
    # Unique owner -> returned.
    assert _reference_electrode_group(electrodes, 0) == "0"
    assert _reference_electrode_group(electrodes, 2) == "1"
    # Absent id -> raises.
    with pytest.raises(ValueError, match="not in the Electrode table"):
        _reference_electrode_group(electrodes, 99)
    # Same electrode_id in two groups -> ambiguous owner raises.
    with pytest.raises(ValueError, match="ambiguous"):
        _reference_electrode_group(electrodes, 1)


@pytest.mark.slow
def test_set_group_by_shank_additive_insert_with_explicit_ids(
    polymer_smoke_session,
):
    """Explicit non-overlapping ``sort_group_ids`` opt into an additive
    insert that coexists with prior rows.

    The default rerun refuses to silently extend (covered elsewhere); passing
    explicit non-overlapping ids is the opt-in. After grouping once (ids
    0-3), a second call with ids 10-13 leaves all eight groups present.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    first_ids = set(
        int(i)
        for i in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert first_ids == {0, 1, 2, 3}

    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, sort_group_ids=[10, 11, 12, 13]
    )
    all_ids = set(
        int(i)
        for i in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert all_ids == {
        0,
        1,
        2,
        3,
        10,
        11,
        12,
        13,
    }, "additive insert with explicit ids must coexist with prior rows"


@pytest.mark.slow
def test_set_group_by_shank_length_mismatch_raises(polymer_smoke_session):
    """``sort_group_ids`` whose length differs from the derived group
    count raises.

    The fixture derives 4 groups from shank metadata; passing only two ids
    is a mismatch the helper rejects before any insert.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="Lengths must match"):
        SortGroupV2.set_group_by_shank(
            nwb_file_name=nwb_file_name, sort_group_ids=[0, 1]
        )


@pytest.mark.slow
def test_set_group_by_electrode_table_column_empty_match_raises(
    polymer_smoke_session,
):
    """A ``groups`` sublist matching zero electrodes raises.

    ``set_group_by_electrode_table_column`` with a value that no electrode
    carries produces an empty group, which is a user error (typo'd id) the
    helper rejects with a message naming the offending sort group.
    """
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)

    with pytest.raises(ValueError, match="matched no electrodes"):
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb_file_name,
            electrode_column="electrode_id",
            value_groups=[[10_000_000]],  # no electrode has this id
        )


@pytest.mark.slow
def test_sortgroup_overwrite_preview_lists_cross_team_downstream(
    polymer_smoke_session,
):
    """The overwrite preview enumerates downstream rows owned by EACH team.

    Sort groups in one session can belong to different teams, so an overwrite
    can cascade-delete another team's downstream rows. The preview surfaces
    that cross-team blast radius (visibility only; it does not block).
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        RecordingSelection,
        SortGroupV2,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_ids = sorted(
        int(s)
        for s in (SortGroupV2 & polymer_smoke_session).fetch("sort_group_id")
    )
    assert len(sort_group_ids) >= 2, "need >=2 sort groups for a two-team test"

    for team in ("team_alpha", "team_beta"):
        LabTeam.insert1(
            {"team_name": team, "team_description": "cross-team preview test"},
            skip_duplicates=True,
        )

    try:
        # Two teams own DIFFERENT sort groups of the SAME session.
        RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_ids[0],
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "team_alpha",
            }
        )
        RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_ids[1],
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "team_beta",
            }
        )

        preview = SortGroupV2.preview_existing_entries(nwb_file_name)
        by_team = {
            row["team_name"]: row for row in preview.cross_team_downstream
        }
        assert {"team_alpha", "team_beta"} <= set(by_team), (
            "overwrite preview must enumerate both teams' downstream rows; "
            f"got {preview.cross_team_downstream}"
        )
        assert by_team["team_alpha"]["recording_selection_rows"] >= 1
        assert by_team["team_beta"]["recording_selection_rows"] >= 1
    finally:
        _clean_session_v2(polymer_smoke_session)
