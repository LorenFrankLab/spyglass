from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def export_tbls(common):
    from spyglass.common.common_usage import Export, ExportSelection

    return ExportSelection(), Export()


@pytest.fixture(scope="session")
def custom_analysis_file(mini_copy_name, dj_conn, common, teardown):
    """Create a custom AnalysisNwbfile table and file for testing."""
    import datajoint as dj

    from spyglass.utils.dj_mixin import SpyglassAnalysis, SpyglassMixin

    prefix = "testexport"
    original_prefix = dj.config.get("custom", {}).get("database.prefix")

    if "custom" not in dj.config:
        dj.config["custom"] = {}
    dj.config["custom"]["database.prefix"] = prefix

    try:
        schema = dj.schema(f"{prefix}_nwbfile")
        Nwbfile = common.common_nwbfile.Nwbfile  # noqa F401

        @schema
        class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
            definition = """This definition is managed by SpyglassAnalysis"""

        @schema
        class CustomDownstream(SpyglassMixin, dj.Manual):
            definition = """
            foreign_id: int auto_increment
            -> AnalysisNwbfile
            """

            @property
            def _nwb_table_tuple(self):
                """Return tuple of (table, attr) for NWB file access."""
                return (AnalysisNwbfile, "analysis_file_name")

            def insert_by_name(self, fname):
                super().insert1(dict(analysis_file_name=fname))

        table = AnalysisNwbfile()
        downstream = CustomDownstream()
        table.delete(safemode=False)  # Clean slate
        analysis_file_name = table.create(mini_copy_name)
        table.add(mini_copy_name, analysis_file_name)
        downstream.insert_by_name(analysis_file_name)

        yield table, downstream, analysis_file_name

        if teardown:
            (table & {"analysis_file_name": analysis_file_name}).delete(
                safemode=False
            )

            file_path = Path(table.get_abs_path(analysis_file_name))
            if file_path.exists():
                file_path.unlink()
    finally:
        if original_prefix:
            dj.config["custom"]["database.prefix"] = original_prefix
        elif "database.prefix" in dj.config.get("custom", {}):
            del dj.config["custom"]["database.prefix"]


@pytest.fixture(scope="session")
def gen_export_selection(
    lfp,
    trodes_pos_v1,
    track_graph,
    export_tbls,
    populate_lfp,
    pos_merge_tables,
    pop_common_electrode_group,
    common,
    upsample_position,
    custom_analysis_file,
):
    ExportSelection, _ = export_tbls
    pos_merge, lin_merge = pos_merge_tables
    _ = populate_lfp
    custom_table, custom_downstream, custom_file = custom_analysis_file

    ExportSelection.delete(safemode=False)

    ExportSelection.start_export(paper_id=1, analysis_id=1)
    lfp.v1.LFPV1().fetch_nwb()
    trodes_pos_v1.fetch()
    ExportSelection.start_export(paper_id=1, analysis_id=2)
    track_graph.fetch()
    ExportSelection.start_export(paper_id=1, analysis_id=3)

    _ = pop_common_electrode_group & "electrode_group_name = 1"
    _ = common.IntervalPositionInfoSelection * (
        common.IntervalList & "interval_list_name = 'pos 1 valid times'"
    )

    ExportSelection.start_export(paper_id=1, analysis_id=4)

    merge_key = (
        pos_merge.TrodesPosV1 & "trodes_pos_params_name LIKE '%ups%'"
    ).fetch1("KEY")
    (pos_merge & merge_key).fetch_nwb()

    ExportSelection.start_export(paper_id=1, analysis_id=5)

    # NEW: Test fetch_nwb on downstream table with custom AnalysisNwbfile parent
    # This should trigger _copy_to_master and insert into ExportSelection.File
    custom_downstream.fetch_nwb()

    ExportSelection.stop_export()

    yield dict(
        paper_id=1,
        custom_table=custom_table,
        custom_downstream=custom_downstream,
        custom_file=custom_file,
    )

    ExportSelection.stop_export()
    ExportSelection.super_delete(warn=False, safemode=False)


def test_export_selection_files(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    paper_key = {"paper_id": gen_export_selection["paper_id"]}

    len_fi = len(ExportSelection * ExportSelection.File & paper_key)
    assert (
        len_fi == 3
    ), "Selection files not captured correctly (expected 2 standard + 1 custom)"


def test_export_selection_tables(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    paper_key = {"paper_id": gen_export_selection["paper_id"]}

    paper = ExportSelection * ExportSelection.Table & paper_key
    len_tbl_1 = len(paper & dict(analysis_id=1))
    len_tbl_2 = len(paper & dict(analysis_id=2))
    assert len_tbl_1 == 3, "Selection tables not captured correctly"
    assert len_tbl_2 == 1, "Selection tables not captured correctly"


def test_export_selection_joins(gen_export_selection, export_tbls, common):
    ExportSelection, _ = export_tbls
    paper_key = {"paper_id": gen_export_selection["paper_id"]}

    restr = (
        ExportSelection * ExportSelection.Table
        & paper_key
        & dict(analysis_id=3)
    )

    elect_grp_tbl = common.ElectrodeGroup.full_table_name
    restr_elect_grp = restr & dict(table_name=elect_grp_tbl)
    assert "electrode_group_name = 1" in restr_elect_grp.fetch1(
        "restriction"
    ), "Export restriction not captured correctly"

    int_pos_tbl = common.IntervalPositionInfoSelection.full_table_name
    restr_int_pos = restr & dict(table_name=int_pos_tbl)
    assert "pos 1 valid times" in restr_int_pos.fetch1(
        "restriction"
    ), "Export join not captured correctly"


def test_export_selection_merge_fetch(
    gen_export_selection, export_tbls, trodes_pos_v1
):
    ExportSelection, _ = export_tbls
    paper_key = {"paper_id": gen_export_selection["paper_id"]}

    paper = ExportSelection * ExportSelection.Table & paper_key
    restr = paper & dict(analysis_id=4)

    assert trodes_pos_v1.full_table_name in restr.fetch(
        "table_name"
    ), "Export merge not captured correctly"


def tests_export_selection_max_id(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    _ = gen_export_selection

    exp_id = max(ExportSelection.fetch("export_id"))
    got_id = ExportSelection._max_export_id(1)
    assert exp_id == got_id, "Max export id not captured correctly"


@pytest.fixture(scope="session")
def populate_export(export_tbls, gen_export_selection):
    _, Export = export_tbls
    paper_key = {"paper_id": gen_export_selection["paper_id"]}
    Export.populate_paper(**paper_key)
    key = (Export & paper_key).fetch("export_id", as_dict=True)

    yield (Export.Table & key), (Export.File & key)

    Export.super_delete(warn=False, safemode=False)


def test_export_populate(populate_export, custom_analysis_file):
    table, file = populate_export

    assert len(file) == 5, "Export files not captured correctly"
    assert len(table) == 41, "Export tables not captured correctly"


def test_invalid_export_id(export_tbls):
    ExportSelection, _ = export_tbls
    ExportSelection.start_export(paper_id=2, analysis_id=1)
    with pytest.raises(RuntimeError):
        ExportSelection.export_id = 99
    ExportSelection.stop_export()


def test_del_export_id(export_tbls):
    ExportSelection, _ = export_tbls
    ExportSelection.start_export(paper_id=2, analysis_id=1)
    del ExportSelection.export_id
    assert ExportSelection.export_id == 0, "Export id not reset correctly"


# ==================== CUSTOM ANALYSISNWBFILE EXPORT TESTS ====================


def test_custom_analysis_copy_to_master(gen_export_selection, common):
    """Test that custom AnalysisNwbfile entries are copied to master table.

    Verifies the copy-to-master approach documented in TASK4_EXPORT.md:
    - Custom entries exist in both custom and master AnalysisNwbfile tables
    - Entry in master table has matching data
    - Export includes the custom file
    """
    custom_table = gen_export_selection["custom_table"]
    custom_file = gen_export_selection["custom_file"]
    custom_dict = {"analysis_file_name": custom_file}
    master_table = common.common_nwbfile.AnalysisNwbfile()

    # Verify custom entry exists
    custom_entry = custom_table & custom_dict
    assert len(custom_entry) == 1, "Custom table entry not found"

    # Verify entry was copied to master table
    master_entry = master_table & custom_dict
    assert len(master_entry) == 1, "Entry not copied to master AnalysisNwbfile"

    # Verify the copied entry has correct data
    custom_data = custom_entry.fetch1()
    master_data = master_entry.fetch1()

    assert custom_data == master_data, "Data mismatch in copied AnalysisNwbfile"


def test_custom_analysis_in_export(gen_export_selection, populate_export):
    """Test that custom AnalysisNwbfile is included in final Export.

    Verifies:
    - Custom file appears in Export.File
    - File path is correct
    """
    custom_table = gen_export_selection["custom_table"]
    custom_file = gen_export_selection["custom_file"]
    _, export_files = populate_export

    # Get all file paths from export
    file_paths = export_files.fetch("file_path")

    # Check that our custom file's path appears in export
    custom_abs_path = custom_table.get_abs_path(custom_file)
    assert any(
        custom_abs_path in fp for fp in file_paths
    ), f"Custom file {custom_abs_path} not in export files"
