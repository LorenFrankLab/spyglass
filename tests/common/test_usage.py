import pytest


@pytest.fixture(scope="session")
def export_tbls(common):
    from spyglass.common.common_usage import Export, ExportSelection

    return ExportSelection(), Export()


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
):
    ExportSelection, _ = export_tbls
    pos_merge, lin_merge = pos_merge_tables
    _ = populate_lfp

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

    ExportSelection.stop_export()

    yield dict(paper_id=1)

    ExportSelection.stop_export()
    ExportSelection.super_delete(warn=False, safemode=False)


def test_export_selection_files(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

    len_fi = len(ExportSelection * ExportSelection.File & paper_key)
    assert len_fi == 2, "Selection files not captured correctly"


def test_export_selection_tables(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

    paper = ExportSelection * ExportSelection.Table & paper_key
    len_tbl_1 = len(paper & dict(analysis_id=1))
    len_tbl_2 = len(paper & dict(analysis_id=2))
    assert len_tbl_1 == 3, "Selection tables not captured correctly"
    assert len_tbl_2 == 1, "Selection tables not captured correctly"


def test_export_selection_joins(gen_export_selection, export_tbls, common):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

    restr = (
        ExportSelection * ExportSelection.Table
        & paper_key
        & dict(analysis_id=3)
    )

    assert "electrode_group_name = 1" in (
        restr & {"table_name": common.ElectrodeGroup.full_table_name}
    ).fetch1("restriction"), "Export restriction not captured correctly"

    assert "pos 1 valid times" in (
        restr
        & {"table_name": common.IntervalPositionInfoSelection.full_table_name}
    ).fetch1("restriction"), "Export join not captured correctly"


def test_export_selection_merge_fetch(
    gen_export_selection, export_tbls, trodes_pos_v1
):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

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
    Export.populate_paper(**gen_export_selection)
    key = (Export & gen_export_selection).fetch("export_id", as_dict=True)

    yield (Export.Table & key), (Export.File & key)

    Export.super_delete(warn=False, safemode=False)


def test_export_populate(populate_export):
    table, file = populate_export

    assert len(file) == 4, "Export tables not captured correctly"
    assert len(table) == 37, "Export files not captured correctly"


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
