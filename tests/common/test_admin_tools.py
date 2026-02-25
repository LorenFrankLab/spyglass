from spyglass.utils.dj_helper_fn import _resolve_external_table


def test_update_external_raw(mini_insert, mini_copy_name, common):
    path = common.Nwbfile().get_abs_path(mini_copy_name)
    _resolve_external_table(path, mini_copy_name, "raw")


def test_update_external_analysis(trodes_pos_v1, common):
    analysis_file = trodes_pos_v1.fetch("analysis_file_name")
    path = common.AnalysisNwbfile().get_abs_path(analysis_file)
    _resolve_external_table(path, analysis_file, "analysis")
