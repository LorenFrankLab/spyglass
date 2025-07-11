import datajoint as dj
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
def pop_rec_v0(
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


@pytest.fixture(scope="session")
def pop_art(spike_v0, pop_rec_v0):
    spike_v0.ArtifactDetectionParameters().insert_default()
    art_key = dict(pop_rec_v0.fetch1("KEY"), artifact_params_name="none")
    spike_v0.ArtifactDetectionSelection.insert1(art_key, skip_duplicates=True)
    spike_v0.ArtifactDetection.populate(art_key)

    yield spike_v0.ArtifactDetection() & art_key


@pytest.fixture(scope="session")
def sorter_params(spike_v0):
    # Could speed up by changing defaults?
    sorter_key = {"sorter": "mountainsort4"}
    spike_v0.SpikeSorterParameters().insert_default()
    yield spike_v0.SpikeSorterParameters() & sorter_key & dj.Top(limit=1)


@pytest.fixture(scope="session")
def pop_sort(spike_v0, pop_art, sorter_params):
    param = sorter_params.fetch("sorter", "sorter_params_name", as_dict=True)[0]
    query = pop_art.proj() * spike_v0.ArtifactRemovedIntervalList().proj()
    ss_key = dict(**query.fetch1(), **param)
    ss_key.pop("artifact_params_name")
    spike_v0.SpikeSortingSelection.insert1(ss_key, skip_duplicates=True)
    spike_v0.SpikeSorting.populate(ss_key)

    yield spike_v0.SpikeSorting() & ss_key


@pytest.fixture(scope="session")
def pop_curation(spike_v0, pop_sort):
    cur_fields = spike_v0.Curation.heading.names
    for sorting_key in pop_sort:
        cur_key = {k: v for k, v in sorting_key.items() if k in cur_fields}
        spike_v0.Curation.insert_curation(cur_key)

    yield spike_v0.Curation()


@pytest.fixture(scope="session")
def wave_params_key(spike_v0):
    params_key = dict(waveform_params_name="default_whitened")
    spike_v0.WaveformParameters().insert_default()
    yield params_key


@pytest.fixture(scope="session")
def pop_waves(spike_v0, pop_curation, wave_params_key):
    curation_keys = [{**k, **wave_params_key} for k in pop_curation.proj()]
    spike_v0.WaveformSelection.insert(curation_keys, skip_duplicates=True)
    spike_v0.Waveforms.populate()
    yield spike_v0.Waveforms() & curation_keys


@pytest.fixture(scope="session")
def metric_params_key(spike_v0):
    params_key = dict(metric_params_name="franklab_default3")
    spike_v0.MetricParameters().insert_default()
    yield params_key


@pytest.fixture(scope="session")
def pop_metrics(spike_v0, pop_waves, metric_params_key):
    waveform_keys = [{**k, **metric_params_key} for k in pop_waves.proj()]
    spike_v0.MetricSelection.insert(waveform_keys, skip_duplicates=True)
    spike_v0.QualityMetrics().populate(waveform_keys)
    yield spike_v0.QualityMetrics() & waveform_keys


@pytest.fixture(scope="session")
def curation_params_key(spike_v0, pop_metrics):
    params_key = dict(auto_curation_params_name="default")
    spike_v0.AutomaticCurationParameters().insert_default()
    yield params_key


@pytest.fixture(scope="session")
def pop_auto_curation(spike_v0, pop_metrics, curation_params_key):
    metric_keys = [{**k, **curation_params_key} for k in pop_metrics.proj()]
    spike_v0.AutomaticCurationSelection.insert(
        metric_keys, skip_duplicates=True
    )
    spike_v0.AutomaticCuration().populate(metric_keys)
    restr_auto = spike_v0.AutomaticCuration() & metric_keys
    restr_curation = spike_v0.Curation() & metric_keys
    yield restr_auto, restr_curation


@pytest.fixture(scope="session")
def pop_curated(spike_v0, pop_auto_curation):
    _, curation = pop_auto_curation
    for cur_key in curation.proj():
        spike_v0.CuratedSpikeSortingSelection.insert1(
            cur_key, skip_duplicates=True
        )
    spike_v0.CuratedSpikeSorting.populate()
    yield spike_v0.CuratedSpikeSorting()


@pytest.fixture(scope="session")
def burst_params_key(spike_v0):
    yield dict(burst_params_name="default")


@pytest.fixture(scope="session")
def pop_burst(spike_v0, pop_curated, burst_params_key):
    burst_keys = [{**k, **burst_params_key} for k in pop_curated.proj()]
    for key in burst_keys:
        spike_v0.BurstPairSelection().insert_by_sort_group_ids(
            nwb_file_name=key["nwb_file_name"],
            session_name=key["sort_interval_name"],
            curation_id=1,
        )
    spike_v0.BurstPairSelection.insert(burst_keys, skip_duplicates=True)
    spike_v0.BurstPair().populate(burst_keys)
    yield spike_v0.BurstPair() & burst_keys


@pytest.fixture(scope="session")
def pop_merge(spike_v0, pop_curated):  # Not yet used
    spike_v0.SpikeSortingOutput.insert(
        pop_curated.fetch("KEY"),
        skip_duplicates=True,
        part_name="CuratedSpikeSorting",
    )


@pytest.fixture(scope="session")
def pop_figurl(spike_v0, pop_curated, pop_auto_curation):
    yield None  # not yet implemented

    # username = "username"
    # fig_url_repo = f"gh://LorenFrankLab/sorting-curations/main/{username}/"
    # gh_url = (
    #     fig_url_repo
    #     + str(nwb_file_name + "_pytest")  # session id
    #     + "/{}"  # tetrode using auto_id['sort_group_id']
    #     + "/curation.json"
    # )
    # auto_curation, curation = pop_auto_curation
    # inserts = [
    #     dict(
    #         **(auto_curation & auto_id).fetch1("KEY"),
    #         curation_figurl=gh_url.format(str(auto_id["sort_group_id"])),
    #     )
    #     for auto_id in auto_curation.fetch("auto_curation_key")
    # ]
    #
    # spike_v0.CurationFigurlSelection().insert1(inserts, skip_duplicates=True)
    # fig_tbl = spike_v0.CurationFigurl()
    # fig_tbl.populate()
    # yield fig_tbl
