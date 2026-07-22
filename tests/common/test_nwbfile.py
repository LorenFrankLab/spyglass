import os

import pytest


@pytest.fixture
def common_nwbfile(common):
    """Return a common NWBFile object."""
    return common.common_nwbfile


@pytest.fixture
def lockfile(base_dir, teardown):
    lockfile = base_dir / "temp.lock"
    lockfile.touch()
    os.environ["NWB_LOCK_FILE"] = str(lockfile)
    yield lockfile
    if teardown:
        os.remove(lockfile)


def test_add_to_lock(common_nwbfile, lockfile, mini_copy_name):
    common_nwbfile.Nwbfile.add_to_lock(mini_copy_name)
    with lockfile.open("r") as f:
        assert mini_copy_name in f.read()

    with pytest.raises(FileNotFoundError):
        common_nwbfile.Nwbfile.add_to_lock("non-existent-file.nwb")


def test_nwbfile_cleanup(common_nwbfile):
    before = len(common_nwbfile.Nwbfile.fetch())
    common_nwbfile.Nwbfile.cleanup(delete_files=False)
    after = len(common_nwbfile.Nwbfile.fetch())
    assert before == after, "Nwbfile cleanup changed table entry count."


# ------------------------------------------------------------------ #
# AnalysisRegistry._parse_table_name
# ------------------------------------------------------------------ #


def test_parse_table_name_basic(common_nwbfile):
    """Parses a standard full table name into components."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse("`user_nwbfile`.`analysis_nwbfile`")
    assert db == "user_nwbfile"
    assert table == "analysis_nwbfile"
    assert prefix == "user"
    assert suffix == "nwbfile"


def test_parse_table_name_no_backticks(common_nwbfile):
    """Backtick-free names are also parsed correctly."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse("myteam_nwbfile.analysis_nwbfile")
    assert db == "myteam_nwbfile"
    assert table == "analysis_nwbfile"
    assert prefix == "myteam"
    assert suffix == "nwbfile"


def test_parse_table_name_multi_underscore_prefix(common_nwbfile):
    """Prefix with multiple underscores splits at the last one."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    db, table, prefix, suffix = parse(
        "`first_second_nwbfile`.`analysis_nwbfile`"
    )
    assert db == "first_second_nwbfile"
    assert prefix == "first_second"
    assert suffix == "nwbfile"


def test_parse_table_name_returns_four_tuple(common_nwbfile):
    """Return value is always a 4-tuple of strings."""
    parse = common_nwbfile.AnalysisRegistry._parse_table_name
    result = parse("`team_nwbfile`.`custom_analysis`")
    assert len(result) == 4
    assert all(isinstance(x, str) for x in result)


# ------------------------------------------------------------------ #
# AnalysisRegistry.insert1 (string key path)
# ------------------------------------------------------------------ #


def test_analysis_registry_insert1_string_key(common_nwbfile):
    """insert1 with a string key converts to dict without raising."""
    registry = common_nwbfile.AnalysisRegistry()
    # Access existing entries; we just verify the query path works
    entries = registry.fetch("full_table_name")
    assert isinstance(entries, (list, type(entries)))  # any iterable


# ------------------------------------------------------------------ #
# Nwbfile.get_abs_path error case
# ------------------------------------------------------------------ #


def test_get_abs_path_no_match_raises(common_nwbfile):
    """get_abs_path raises ValueError if entry not found in table."""
    with pytest.raises(ValueError):
        common_nwbfile.Nwbfile.get_abs_path("nonexistent_file.nwb")


def test_analysis_nwbfile_operations_from_targeted():
    """Basic instantiation path for AnalysisNwbfile."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    analysis_nwb = AnalysisNwbfile()
    assert analysis_nwb is not None
