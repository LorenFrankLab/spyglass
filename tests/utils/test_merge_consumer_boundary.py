"""Regression tests for two consumer-boundary bugs in ``_Merge``.

Both bugs live in the generic merge-table layer
(``spyglass.utils.dj_merge_tables``) and are therefore exercised here, on a
hermetic two-source merge table, rather than against any one pipeline's
merge master:

A2 -- ``Merge.fetch_nwb`` advertised a ``multi_source`` argument that the
implementation never used: a restriction spanning >=2 source part types
raised a cryptic "Found N potential parents" from ``merge_get_parent``, and
``return_merge_ids`` resolved each file's merge_id against the *cumulative*
``nwb_list`` instead of the current source's files. The fix makes the
advertised API true:

  - ``multi_source=False`` (default) + a restriction spanning >1 source
    warns and still fetches across all sources.
  - ``multi_source=True`` iterates source-by-source, scoping the parent
    resolution to one source per loop, and returns ``len(merge_ids) ==
    len(nwb_list)`` with each id the owner of its paired file.
  - the single-source path (used by every merge master today) is unchanged.

A6 -- the ``delete_downstream_merge`` shim called
``ActivityLog.deprecate_log(..., alternate=...)`` but the signature kwarg is
``alt=``, so the shim raised ``TypeError`` before doing anything.
"""

from __future__ import annotations

import datajoint as dj
import pytest
from datajoint.errors import DataJointError


@pytest.fixture(scope="module")
def two_source_merge(dj_conn, mini_dict):
    """A hermetic ``_Merge`` table with two distinct Nwbfile-backed sources.

    ``LeafA`` and ``LeafB`` are two different part-table source types, each
    pointing at the same already-ingested sample raw NWB file (so
    ``fetch_nwb`` finds a real file on disk without building analysis
    files). One row is inserted into each source, giving the merge master
    the >=2-source-type condition the A2 bug requires.

    Yields a dict with the activated classes and the two merge_ids
    (``merge_id_a`` owns the ``LeafA`` file, ``merge_id_b`` the ``LeafB``).
    """
    from spyglass.common import Nwbfile
    from spyglass.utils import SpyglassMixin, _Merge

    nwb_file_name = mini_dict["nwb_file_name"]

    class LeafA(SpyglassMixin, dj.Manual):
        definition = """
        leaf_a_id: int
        ---
        -> Nwbfile
        """
        _nwb_table = Nwbfile

    class LeafB(SpyglassMixin, dj.Manual):
        definition = """
        leaf_b_id: int
        ---
        -> Nwbfile
        """
        _nwb_table = Nwbfile

    class TwoSourceMerge(_Merge, SpyglassMixin):
        definition = """
        merge_id: uuid
        ---
        source: varchar(32)
        """

        class LeafA(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> LeafA
            """

        class LeafB(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> LeafB
            """

    context = {
        "Nwbfile": Nwbfile,
        "LeafA": LeafA,
        "LeafB": LeafB,
        "TwoSourceMerge": TwoSourceMerge,
    }
    # NB: the schema name must NOT contain the merge master's table_name
    # ("two_source_merge") as a substring. ``_merge_restrict_parents`` filters
    # part-parents with ``cls().table_name not in parent.full_table_name``; a
    # schema named e.g. "test_two_source_merge" embeds it and would wrongly
    # exclude the leaf parents (real merge masters never collide because the
    # schema name differs from the table name).
    schema = dj.Schema(
        "test_merge_boundary", context=context, connection=dj_conn
    )
    for table in (LeafA, LeafB, TwoSourceMerge):
        schema(table)

    # ``Merge.source_class_dict`` resolves each part's parent class via
    # ``getattr(getmodule(self), part_name)``. The leaf classes are declared
    # inside this fixture (per the conftest "declare within fixture" pattern),
    # so expose them on this module for that lookup to succeed.
    import sys

    _mod = sys.modules[__name__]
    _mod.LeafA = LeafA
    _mod.LeafB = LeafB

    LeafA().insert1({"leaf_a_id": 0, "nwb_file_name": nwb_file_name})
    LeafB().insert1({"leaf_b_id": 0, "nwb_file_name": nwb_file_name})
    TwoSourceMerge().insert(
        [{"leaf_a_id": 0, "nwb_file_name": nwb_file_name}], part_name="LeafA"
    )
    TwoSourceMerge().insert(
        [{"leaf_b_id": 0, "nwb_file_name": nwb_file_name}], part_name="LeafB"
    )

    merge_id_a = (TwoSourceMerge.LeafA & {"leaf_a_id": 0}).fetch1("merge_id")
    merge_id_b = (TwoSourceMerge.LeafB & {"leaf_b_id": 0}).fetch1("merge_id")

    yield {
        "Merge": TwoSourceMerge,
        "merge_id_a": merge_id_a,
        "merge_id_b": merge_id_b,
        "nwb_file_name": nwb_file_name,
    }

    prev = dj.logger.level
    dj.logger.setLevel("ERROR")
    schema.drop(force=True)
    dj.logger.setLevel(prev)
    for attr in ("LeafA", "LeafB"):
        if hasattr(_mod, attr):
            delattr(_mod, attr)


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_multi_source_warns_without_opt_in(two_source_merge, caplog):
    """A restriction spanning >=2 source types now WARNS (not raises) and
    fetches across both; ``multi_source=True`` silences the warning."""
    import logging as _logging

    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    merge_id_b = two_source_merge["merge_id_b"]

    restriction = [{"merge_id": merge_id_a}, {"merge_id": merge_id_b}]
    with caplog.at_level(_logging.WARNING):
        nwb_list, merge_ids = (Merge & restriction).fetch_nwb(
            return_merge_ids=True
        )
    assert "multi_source=True" in caplog.text
    assert len(nwb_list) == 2
    assert set(merge_ids) == {merge_id_a, merge_id_b}


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_multi_source_aligned(two_source_merge):
    """With ``multi_source=True`` a >=2-source restriction returns one
    merge_id per file, each the owner of its paired file."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    merge_id_b = two_source_merge["merge_id_b"]

    restriction = [{"merge_id": merge_id_a}, {"merge_id": merge_id_b}]
    nwb_list, merge_ids = (Merge & restriction).fetch_nwb(
        return_merge_ids=True, multi_source=True
    )

    assert len(nwb_list) == 2
    assert len(merge_ids) == len(nwb_list)
    # Each merge_id owns its paired file: the LeafA file pairs with
    # merge_id_a, the LeafB file with merge_id_b (the two sources are
    # distinguishable by which leaf-id key the fetched row carries).
    for file, merge_id in zip(nwb_list, merge_ids):
        if "leaf_a_id" in file:
            assert merge_id == merge_id_a
        elif "leaf_b_id" in file:
            assert merge_id == merge_id_b
        else:  # pragma: no cover - defensive
            raise AssertionError(f"fetched row from neither source: {file}")
    assert set(merge_ids) == {merge_id_a, merge_id_b}


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_parent_key_restriction_resolves_one_source(
    two_source_merge,
):
    """A parent-key restriction that uniquely identifies one source resolves
    to just that source -- it neither (default) raises "multiple sources" nor
    (multi_source=True) leaks files from the other source.

    Source discovery must apply the FULL restriction through the parts;
    ``extract_merge_id`` alone collapses a parent-key restriction to a
    universal set and would surface every source.
    """
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    # The restriction is passed as the fetch_nwb argument: a parent-key (here
    # leaf_a_id) is not in the merge master's heading, so ``Merge & {...}``
    # would silently drop it -- consumers pass it through ``fetch_nwb`` /
    # ``get_restricted_merge_ids`` instead.

    # Default (no multi_source): leaf_a_id picks LeafA only -> no raise.
    nwb_list, merge_ids = Merge().fetch_nwb(
        {"leaf_a_id": 0}, return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert "leaf_a_id" in nwb_list[0]
    assert merge_ids == [merge_id_a]

    # multi_source=True must NOT leak the other source's file.
    nwb_list2, merge_ids2 = Merge().fetch_nwb(
        {"leaf_a_id": 0}, return_merge_ids=True, multi_source=True
    )
    assert len(nwb_list2) == 1
    assert "leaf_a_id" in nwb_list2[0]
    assert merge_ids2 == [merge_id_a]


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_string_parent_key_restriction(two_source_merge):
    """A STRING restriction naming a parent key resolves like the dict form.

    ``fetch_nwb`` documents accepting parent-attribute restrictions, but a
    string (e.g. ``"leaf_a_id = 0"``) is un-introspectable, so it must NOT be
    applied raw to the master (which has no ``leaf_a_id`` column) during source
    discovery. It resolves to just the owning source, not a crash and not the
    other source's file.
    """
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    nwb_list, merge_ids = Merge().fetch_nwb(
        "leaf_a_id = 0", return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert "leaf_a_id" in nwb_list[0]
    assert merge_ids == [merge_id_a]


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_string_unknown_column_raises(two_source_merge):
    """A string restriction naming a column NO source's parent has re-raises the
    original DataJointError -- a typo must not silently read as a no-match.

    The string-restriction fallback probes each source; if the restriction is
    evaluable against neither the master nor any source's parent, it re-raises
    rather than returning an empty result (which would hide the typo).
    """
    Merge = two_source_merge["Merge"]
    with pytest.raises(DataJointError):
        Merge().fetch_nwb("not_a_real_column = 0")


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_string_valid_column_no_match_is_empty(two_source_merge):
    """A string restriction on a REAL parent column that matches no rows returns
    an empty result (not an error) -- the genuine-no-match case must stay empty
    even though a typo'd column (above) raises."""
    Merge = two_source_merge["Merge"]
    nwb_list, merge_ids = Merge().fetch_nwb(
        "leaf_a_id = 999999", return_merge_ids=True
    )
    assert nwb_list == []
    assert merge_ids == []


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_mixed_master_and_parent_restriction(two_source_merge):
    """A restriction mixing a master column (merge_id) and a parent key
    (leaf_a_id) resolves correctly -- the source whose parent lacks merge_id
    must NOT be skipped (master and parent attributes are evaluated on the
    master+part+parent relation, not all required on the parent)."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    nwb_list, merge_ids = Merge().fetch_nwb(
        {"merge_id": merge_id_a, "leaf_a_id": 0}, return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert "leaf_a_id" in nwb_list[0]
    assert merge_ids == [merge_id_a]


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_parent_secondary_restriction(two_source_merge):
    """A restriction on a parent SECONDARY attribute (nwb_file_name -- not
    propagated to the part heading) is applied through the part*parent join
    for both discovery and the fetch, rather than being silently dropped.

    Both leaves point at the sample NWB, so it matches both sources; this
    exercises the secondary-attribute path end-to-end (discovery, fetch, and
    aligned merge_id resolution) under multi_source=True.
    """
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    merge_id_b = two_source_merge["merge_id_b"]
    nwb_file_name = two_source_merge["nwb_file_name"]

    nwb_list, merge_ids = Merge().fetch_nwb(
        {"nwb_file_name": nwb_file_name},
        return_merge_ids=True,
        multi_source=True,
    )
    assert len(nwb_list) == 2
    assert len(merge_ids) == len(nwb_list)
    for file, merge_id in zip(nwb_list, merge_ids):
        if "leaf_a_id" in file:
            assert merge_id == merge_id_a
        elif "leaf_b_id" in file:
            assert merge_id == merge_id_b
    assert set(merge_ids) == {merge_id_a, merge_id_b}


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_source_plus_parent_restriction(two_source_merge):
    """A restriction mixing the master ``source`` column with a parent
    attribute resolves to that one source. Both leaves share the sample NWB,
    so only the ``source`` clause distinguishes them -- it must be applied
    (the discovery join includes the merge master)."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    nwb_file_name = two_source_merge["nwb_file_name"]

    nwb_list, merge_ids = Merge().fetch_nwb(
        {"source": "LeafA", "nwb_file_name": nwb_file_name},
        return_merge_ids=True,
        multi_source=True,
    )
    assert len(nwb_list) == 1
    assert "leaf_a_id" in nwb_list[0]
    assert merge_ids == [merge_id_a]


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_or_restriction_across_sources(two_source_merge):
    """An OR-list of parent keys for DIFFERENT sources matches each branch
    against its own source, rather than requiring every parent to carry every
    attribute (which would return nothing)."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    merge_id_b = two_source_merge["merge_id_b"]

    nwb_list, merge_ids = Merge().fetch_nwb(
        [{"leaf_a_id": 0}, {"leaf_b_id": 0}],
        return_merge_ids=True,
        multi_source=True,
    )
    assert len(nwb_list) == 2
    assert set(merge_ids) == {merge_id_a, merge_id_b}


def test_fetch_nwb_return_merge_ids_single_source_unchanged(two_source_merge):
    """A single-source restriction returns aligned ``(nwb_list, merge_ids)``.

    Guards the shared-method change: scoping merge_id resolution to the
    current source must not perturb the single-source path (which is what
    every real merge master uses today). Reverting the A2 fix leaves this
    green -- it asserts the behavior the fix preserves, not the bug it
    removes.
    """
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    nwb_list, merge_ids = (Merge & {"merge_id": merge_id_a}).fetch_nwb(
        return_merge_ids=True
    )

    assert len(nwb_list) == 1
    assert merge_ids == [merge_id_a]
    assert "leaf_a_id" in nwb_list[0]


def test_delete_downstream_merge_shim_logs_not_raises(dj_conn):
    """The deprecated ``delete_downstream_merge`` shim logs its deprecation
    and does NOT raise ``TypeError`` (the A6 ``alt=`` kwarg fix).

    With the bug (``alternate=``) ``deprecate_log`` raised ``TypeError``
    before running, so no ``ActivityLog`` row was written and the expected
    ``ValueError`` (non-Spyglass input) was never reached.
    """
    from spyglass.common.common_usage import ActivityLog
    from spyglass.utils.dj_merge_tables import delete_downstream_merge

    log_restr = {"function": "delete_downstream_merge"}
    before = len(ActivityLog & log_restr)

    # A non-Spyglass input reaches the explicit ValueError only AFTER
    # ``deprecate_log`` runs; with the bug a TypeError fires first instead.
    with pytest.raises(ValueError, match="Spyglass Table"):
        delete_downstream_merge(object())

    assert len(ActivityLog & log_restr) == before + 1


@pytest.fixture(scope="module")
def same_file_two_row_source(dj_conn, mini_dict):
    """A merge master whose single source has TWO rows pointing at the SAME NWB
    file, so ``return_merge_ids`` alignment must key off the parent PK, not the
    (shared) file or a widened source restriction.
    """
    from spyglass.common import Nwbfile
    from spyglass.utils import SpyglassMixin, _Merge

    nwb_file_name = mini_dict["nwb_file_name"]

    class LeafC(SpyglassMixin, dj.Manual):
        definition = """
        leaf_c_id: int
        ---
        -> Nwbfile
        """
        _nwb_table = Nwbfile

    class PkAlignMerge(_Merge, SpyglassMixin):
        definition = """
        merge_id: uuid
        ---
        source: varchar(32)
        """

        class LeafC(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> LeafC
            """

    context = {
        "Nwbfile": Nwbfile,
        "LeafC": LeafC,
        "PkAlignMerge": PkAlignMerge,
    }
    # Schema name must NOT embed the master's table_name ("pk_align_merge").
    schema = dj.Schema(
        "test_merge_pkalign", context=context, connection=dj_conn
    )
    for table in (LeafC, PkAlignMerge):
        schema(table)

    import sys

    _mod = sys.modules[__name__]
    _mod.LeafC = LeafC

    # Two source rows, SAME nwb_file_name -> a file-keyed id lookup is ambiguous.
    LeafC().insert1({"leaf_c_id": 0, "nwb_file_name": nwb_file_name})
    LeafC().insert1({"leaf_c_id": 1, "nwb_file_name": nwb_file_name})
    PkAlignMerge().insert(
        [{"leaf_c_id": 0, "nwb_file_name": nwb_file_name}], part_name="LeafC"
    )
    PkAlignMerge().insert(
        [{"leaf_c_id": 1, "nwb_file_name": nwb_file_name}], part_name="LeafC"
    )
    merge_id_0 = (PkAlignMerge.LeafC & {"leaf_c_id": 0}).fetch1("merge_id")

    yield {
        "Merge": PkAlignMerge,
        "merge_id_0": merge_id_0,
        "nwb_file_name": nwb_file_name,
    }

    prev = dj.logger.level
    dj.logger.setLevel("ERROR")
    schema.drop(force=True)
    dj.logger.setLevel(prev)
    if hasattr(_mod, "LeafC"):
        delattr(_mod, "LeafC")


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_string_return_merge_ids_aligns_by_parent_pk(
    same_file_two_row_source,
):
    """A string restriction + return_merge_ids resolves the CORRECT merge_id even
    when two rows in the source share an NWB file.

    The string fallback must align each file's merge_id by the parent PK (the
    parent-attribute branch's method), not by the shared file against a widened
    source restriction (``extract_merge_id(str) == True``) -- which would match
    both rows and make fetch1 raise or mis-pick.
    """
    Merge = same_file_two_row_source["Merge"]
    # Precondition: the ambiguity is real -- two merge rows, one shared file.
    assert len(Merge()) == 2

    nwb_list, merge_ids = Merge().fetch_nwb(
        "leaf_c_id = 0", return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert merge_ids == [same_file_two_row_source["merge_id_0"]]
