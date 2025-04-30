import pytest


@pytest.fixture(scope="session")
def spike_v0(common):
    import spyglass.spikesorting.v0 as spike_v0

    yield spike_v0


@pytest.fixture(scope="session")
def pop_sort_group(spike_v0, mini_copy_name, mini_insert):
    _ = mini_insert
    spike_v0.SortGroup().set_group_by_shank(mini_copy_name)

    yield spike_v0.SortGroup()


@pytest.fixture(scope="session")
def pop_sort_interval(spike_v0, mini_copy_name, lfp_constants, add_interval):
    # LFP constants do the same 'first n' interval as v0 notebook
    spike_v0.SortInterval.insert1(
        {
            "nwb_file_name": mini_copy_name,
            "sort_interval_name": add_interval,
            "sort_interval": lfp_constants["interval_key"]["valid_times"],
        },
        skip_duplicates=True,
    )

    yield spike_v0.SortInterval()


@pytest.fixture(scope="session")
def pop_rec_params(spike_v0):
    params_tbl = spike_v0.SpikeSortingPreprocessingParameters()
    params_tbl.insert_default()
    preproc_params = params_tbl.fetch("preproc_params")[0]
    preproc_params["frequency_min"] = 600
    preproc_params_name = "default_hippocampus"
    params_tbl.insert1(
        {
            "preproc_params_name": preproc_params_name,
            "preproc_params": preproc_params,
        },
        skip_duplicates=True,
    )
    yield params_tbl, preproc_params_name, preproc_params


@pytest.fixture(scope="session")
def pop_rec(
    spike_v0,
    pop_sort_group,
    pop_sort_interval,
    pop_rec_params,
    mini_copy_name,
    add_interval,
    team_name,
):
    _, params_name, _ = pop_rec_params
    ssr_key = dict(
        nwb_file_name=mini_copy_name,
        sort_group_id=0,  # See SortGroup
        sort_interval_name=add_interval,
        preproc_params_name=params_name,
        interval_list_name="01_s1",
        team_name=team_name,
    )
    _ = pop_sort_interval
    spike_v0.SpikeSortingRecordingSelection.insert1(
        ssr_key, skip_duplicates=True
    )
    spike_v0.SpikeSortingRecordingSelection() & ssr_key
    ssr_pk = (spike_v0.SpikeSortingRecordingSelection & ssr_key).proj()
    spike_v0.SpikeSortingRecording.populate([ssr_pk])

    yield spike_v0.SpikeSortingRecording() & ssr_pk


# Next Artifact Detection
