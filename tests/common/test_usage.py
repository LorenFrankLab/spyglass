import pytest


@pytest.fixture(scope="session")
def export_tbls(common):
    from spyglass.common.common_usage import Export, ExportSelection

    return ExportSelection(), Export()


@pytest.fixture(scope="session")
def intersect_export_selection(trodes_pos_v1, export_tbls, teardown):
    ExportSelection, _ = export_tbls
    trodes_pos_v1.fetch_nwb()

    ExportSelection.start_export(paper_id="intersect_selection", analysis_id=1)
    _ = trodes_pos_v1 & "interval_list_name = 'pos 0 valid times'"
    ExportSelection.stop_export()

    yield dict(paper_id="intersect_selection")

    if teardown:
        ExportSelection.stop_export()
        ExportSelection.super_delete(warn=False, safemode=False)


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
    teardown,
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
    _ = trodes_pos_v1 * (
        common.IntervalList & "interval_list_name = 'pos 0 valid times'"
    )  # Note for PR: table and restriction change because join no longer logs empty results

    ExportSelection.start_export(paper_id=1, analysis_id=4)
    merge_key = (
        pos_merge.TrodesPosV1 & "trodes_pos_params_name LIKE '%ups%'"
    ).fetch1("KEY")
    (pos_merge & merge_key).fetch_nwb()

    ExportSelection.start_export(paper_id=1, analysis_id=5)
    trodes_pos_v1._export_cache.clear()  # Clear cache to ensure proj table is captured
    projected_table = trodes_pos_v1.proj(
        proj_interval_list_name="interval_list_name"
    )
    proj_restr = (
        projected_table & "proj_interval_list_name = 'pos 0 valid times'"
    )
    assert len(proj_restr) > 0, "No entries found for projected table"

    ExportSelection.start_export(paper_id=1, analysis_id=6)
    trodes_pos_v1._export_cache.clear()  # Clear cache to ensure proj table is captured
    _ = trodes_pos_v1 & (
        common.IntervalList & "interval_list_name = 'pos 0 valid times'"
    )

    ExportSelection.stop_export()

    yield dict(paper_id=1)

    if teardown:
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


def test_export_selection_joins(
    gen_export_selection, export_tbls, common, trodes_pos_v1
):
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

    assert "pos 0 valid times" in (
        restr & {"table_name": trodes_pos_v1.full_table_name}
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


def test_export_selection_proj(
    gen_export_selection, export_tbls, trodes_pos_v1
):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

    paper = ExportSelection * ExportSelection.Table & paper_key
    restr = paper & dict(analysis_id=5)

    assert trodes_pos_v1.full_table_name in restr.fetch(
        "table_name"
    ), "Export projection not captured correctly"

    assert "proj_interval_list_name" not in restr.fetch1(
        "restriction"
    ), "Export projection restriction not remapped correctly"


def test_export_selection_compound(
    gen_export_selection, export_tbls, trodes_pos_v1, common
):
    ExportSelection, _ = export_tbls
    paper_key = gen_export_selection

    paper = ExportSelection * ExportSelection.Table & paper_key
    restr = paper & dict(analysis_id=6)

    assert trodes_pos_v1.full_table_name in restr.fetch(
        "table_name"
    ), "Export compound did not capture outer table correctly"

    assert common.IntervalList.full_table_name in restr.fetch(
        "table_name"
    ), "Export compound did not capture inner table correctly"

    trodes_restriction = (
        restr & dict(table_name=trodes_pos_v1.full_table_name)
    ).fetch1("restriction")
    assert (
        len(trodes_pos_v1 & trodes_restriction) == 2
    ), "Export compound did not capture outer restriction correctly"


def tests_export_selection_max_id(gen_export_selection, export_tbls):
    ExportSelection, _ = export_tbls
    _ = gen_export_selection

    exp_id = max(ExportSelection.fetch("export_id"))
    got_id = ExportSelection._max_export_id(1)
    assert exp_id == got_id, "Max export id not captured correctly"


@pytest.fixture(scope="session")
def populate_export(export_tbls, gen_export_selection, teardown):
    _, Export = export_tbls
    Export.populate_paper(**gen_export_selection)
    key = (Export & gen_export_selection).fetch("export_id", as_dict=True)

    yield (Export.Table & key), (Export.File & key)

    if teardown:
        Export.super_delete(warn=False, safemode=False)


@pytest.fixture(scope="session")
def populate_intersect_export(
    intersect_export_selection, export_tbls, teardown
):
    _, Export = export_tbls
    included_nwb_files = ["null_.nwb"]
    Export.populate_paper(
        **intersect_export_selection,
        included_nwb_files=included_nwb_files,
        n_processes=4,
    )
    key = (Export & intersect_export_selection).fetch("export_id", as_dict=True)

    yield (Export.Table & key), (Export.File & key)

    if teardown:
        Export.super_delete(warn=False, safemode=False)


def test_export_populate(populate_export):
    table, file = populate_export

    assert len(file) == 4, "Export files not captured correctly"
    assert (
        len(table) == 37
    ), "Export tables not captured correctly"  # Note for PR: Update because not using common.IntervalPositionInfoSelection (and param table)


def test_intersect_export_populate(populate_intersect_export, common):
    table, file = populate_intersect_export

    assert len(file) == 0, "Intersection failed to censor files"
    assert (
        len(table & {"table_name": common.Nwbfile.full_table_name}) == 0
    ), "Intersection failed to censor entries"


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
