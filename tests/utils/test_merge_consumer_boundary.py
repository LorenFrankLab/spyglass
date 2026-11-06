"""Tests for ``_Merge.fetch_nwb`` across multiple merge source types.

Uses a hermetic two-source merge table. The contract:

- ``multi_source=False`` (default) raises on a restriction spanning >1 source.
- ``multi_source=True`` iterates per source and returns one merge_id per file,
  each the owner of its paired file.
- The single-source path returns aligned ``(nwb_list, merge_ids)``.
"""

from __future__ import annotations

import inspect
import logging

import datajoint as dj
import pytest
from datajoint.errors import DataJointError


@pytest.fixture(scope="module")
def two_source_merge(dj_conn, mini_dict):
    """A hermetic ``_Merge`` table with two Nwbfile-backed sources.

    ``LeafA`` and ``LeafB`` each hold one row pointing at the sample raw NWB.
    Yields a dict with the classes, ``merge_id_a`` and ``merge_id_b``.
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
    # Schema name must not contain the master's table_name:
    # ``_merge_restrict_parents`` excludes parents by substring match.
    schema = dj.Schema(
        "test_merge_boundary", context=context, connection=dj_conn
    )
    for table in (LeafA, LeafB, TwoSourceMerge):
        schema(table)

    # ``source_class_dict`` looks up parents on the module, so expose them.
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
def test_fetch_nwb_multi_source_raises_without_opt_in(two_source_merge):
    """A restriction spanning multiple sources requires explicit opt-in."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]
    merge_id_b = two_source_merge["merge_id_b"]

    restriction = [{"merge_id": merge_id_a}, {"merge_id": merge_id_b}]
    with pytest.raises(ValueError, match="spans 2 sources"):
        (Merge & restriction).fetch_nwb(return_merge_ids=True)


def test_fetch_nwb_has_no_disable_warning_kwarg():
    """``fetch_nwb`` takes no ``disable_warning`` argument."""
    from spyglass.utils.dj_merge_tables import _Merge

    params = inspect.signature(_Merge.fetch_nwb).parameters
    assert "disable_warning" not in params


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
    # The leaf-id key in each fetched row identifies its source.
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
    """A parent-key restriction naming one source resolves to that source.

    ``extract_merge_id`` alone would collapse it and surface every source.
    """
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    # Passed to fetch_nwb: ``Merge & {...}`` would drop a parent key.

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
    """A string restriction naming a parent key resolves like the dict form.

    It must not be applied raw to the master, which lacks the column.
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
    """A string restriction on a column no parent has raises DataJointError.

    A typo must not read as an empty result.
    """
    Merge = two_source_merge["Merge"]
    with pytest.raises(DataJointError):
        Merge().fetch_nwb("not_a_real_column = 0")


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_string_valid_column_no_match_is_empty(two_source_merge):
    """A string restriction on a real parent column with no match is empty."""
    Merge = two_source_merge["Merge"]
    nwb_list, merge_ids = Merge().fetch_nwb(
        "leaf_a_id = 999999", return_merge_ids=True
    )
    assert nwb_list == []
    assert merge_ids == []


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_mixed_master_and_parent_restriction(two_source_merge):
    """A restriction mixing merge_id and a parent key resolves correctly.

    Attributes are evaluated on master * part * parent, not on the parent.
    """
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
    """A parent secondary-attribute restriction applies through part * parent.

    ``nwb_file_name`` is not in the part heading; here it matches both sources.
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
    """Master ``source`` plus a shared parent attribute selects one source."""
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
    """An OR-list of parent keys matches each branch against its own source."""
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
    """A single-source restriction returns aligned ``(nwb_list, merge_ids)``."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    nwb_list, merge_ids = (Merge & {"merge_id": merge_id_a}).fetch_nwb(
        return_merge_ids=True
    )

    assert len(nwb_list) == 1
    assert merge_ids == [merge_id_a]
    assert "leaf_a_id" in nwb_list[0]


@pytest.fixture(scope="module")
def same_file_two_row_source(dj_conn, mini_dict):
    """A merge master whose single source has two rows on the same NWB file."""
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
    """A string restriction aligns merge_ids by parent PK, not by shared file."""
    Merge = same_file_two_row_source["Merge"]
    # Precondition: two merge rows share one file.
    assert len(Merge()) == 2

    nwb_list, merge_ids = Merge().fetch_nwb(
        "leaf_c_id = 0", return_merge_ids=True
    )
    assert len(nwb_list) == 1
    assert merge_ids == [same_file_two_row_source["merge_id_0"]]


@pytest.mark.slow
@pytest.mark.integration
def test_fetch_nwb_warns_when_restriction_skips_a_source(
    two_source_merge, caplog
):
    """A clause naming an attribute one parent lacks warns about the skip."""
    Merge = two_source_merge["Merge"]
    merge_id_a = two_source_merge["merge_id_a"]

    with caplog.at_level(logging.WARNING, logger="spyglass"):
        nwb_list, merge_ids = Merge().fetch_nwb(
            {"leaf_a_id": 0}, return_merge_ids=True
        )

    # Warns, not raises: the matching file is still returned.
    assert len(nwb_list) == 1
    assert "leaf_a_id" in nwb_list[0]
    assert merge_ids == [merge_id_a]

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert any(
        "LeafB" in message and "not searched" in message for message in messages
    ), f"no warning naming the skipped source; got {messages}"
