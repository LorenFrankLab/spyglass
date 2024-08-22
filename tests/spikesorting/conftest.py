import pytest
from datajoint.hash import key_hash


@pytest.fixture(scope="session")
def spike_v1(common):
    from spyglass.spikesorting import v1

    yield v1


@pytest.fixture(scope="session")
def pop_rec(spike_v1, mini_dict, team_name):
    spike_v1.SortGroup.set_group_by_shank(**mini_dict)
    key = {
        **mini_dict,
        "sort_group_id": 0,
        "preproc_param_name": "default",
        "interval_list_name": "01_s1",
        "team_name": team_name,
    }
    spike_v1.SpikeSortingRecordingSelection.insert_selection(key)
    ssr_pk = (
        (spike_v1.SpikeSortingRecordingSelection & key).proj().fetch1("KEY")
    )
    spike_v1.SpikeSortingRecording.populate(ssr_pk)

    yield ssr_pk


@pytest.fixture(scope="session")
def pop_art(spike_v1, mini_dict, pop_rec):
    key = {
        "recording_id": pop_rec["recording_id"],
        "artifact_param_name": "default",
    }
    spike_v1.ArtifactDetectionSelection.insert_selection(key)
    spike_v1.ArtifactDetection.populate()

    yield spike_v1.ArtifactDetection().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def sorter_dict():
    return {"sorter": "mountainsort4"}


@pytest.fixture(scope="session")
def pop_sort(spike_v1, pop_rec, pop_art, mini_dict, sorter_dict):
    key = {
        **mini_dict,
        **sorter_dict,
        "recording_id": pop_rec["recording_id"],
        "interval_list_name": str(pop_art["artifact_id"]),
        "sorter_param_name": "franklab_tetrode_hippocampus_30KHz",
    }
    spike_v1.SpikeSortingSelection.insert_selection(key)
    spike_v1.SpikeSorting.populate()

    yield spike_v1.SpikeSorting().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def sorting_objs(spike_v1, pop_sort):
    sort_nwb = (spike_v1.SpikeSorting & pop_sort).fetch_nwb()
    sort_si = spike_v1.SpikeSorting.get_sorting(pop_sort)
    yield sort_nwb, sort_si


@pytest.fixture(scope="session")
def pop_curation(spike_v1, pop_sort):
    spike_v1.CurationV1.insert_curation(
        sorting_id=pop_sort["sorting_id"],
        description="testing sort",
    )

    yield spike_v1.CurationV1().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def pop_metric(spike_v1, pop_sort):
    key = {
        "sorting_id": pop_sort["sorting_id"],
        "curation_id": 0,
        "waveform_param_name": "default_not_whitened",
        "metric_param_name": "franklab_default",
        "metric_curation_param_name": "default",
    }

    spike_v1.MetricCurationSelection.insert_selection(key)
    spike_v1.MetricCuration.populate(key)

    yield spike_v1.MetricCuration().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def metric_objs(spike_v1, pop_metric):
    key = {"metric_curation_id": pop_metric["metric_curation_id"]}
    labels = spike_v1.MetricCuration.get_labels(key)
    merge_groups = spike_v1.MetricCuration.get_merge_groups(key)
    metrics = spike_v1.MetricCuration.get_metrics(key)
    yield labels, merge_groups, metrics


@pytest.fixture(scope="session")
def pop_curation_metric(spike_v1, pop_metric, metric_objs):
    labels, merge_groups, metrics = metric_objs
    parent_dict = {"parent_curation_id": 0}
    spike_v1.CurationV1.insert_curation(
        sorting_id=(
            spike_v1.MetricCurationSelection
            & {"metric_curation_id": pop_metric["metric_curation_id"]}
        ).fetch1("sorting_id"),
        **parent_dict,
        labels=labels,
        merge_groups=merge_groups,
        metrics=metrics,
        description="after metric curation",
    )

    yield (spike_v1.CurationV1 & parent_dict).fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def pop_figurl(spike_v1, pop_sort, metric_objs):
    # WON'T WORK UNTIL CI/CD KACHERY_CLOUD INIT
    sort_dict = {"sorting_id": pop_sort["sorting_id"], "curation_id": 1}
    curation_uri = spike_v1.FigURLCurationSelection.generate_curation_uri(
        sort_dict
    )
    _, _, metrics = metric_objs
    key = {
        **sort_dict,
        "curation_uri": curation_uri,
        "metrics_figurl": list(metrics.keys()),
    }
    spike_v1.FigURLCurationSelection.insert_selection(key)
    spike_v1.FigURLCuration.populate()

    yield spike_v1.FigURLCuration().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def pop_figurl_json(spike_v1, pop_metric):
    # WON'T WORK UNTIL CI/CD KACHERY_CLOUD INIT
    gh_curation_uri = (
        "gh://LorenFrankLab/sorting-curations/main/khl02007/test/curation.json"
    )
    key = {
        "sorting_id": pop_metric["sorting_id"],
        "curation_id": 1,
        "curation_uri": gh_curation_uri,
        "metrics_figurl": [],
    }
    spike_v1.FigURLCurationSelection.insert_selection(key)
    spike_v1.FigURLCuration.populate()

    labels = spike_v1.FigURLCuration.get_labels(gh_curation_uri)
    merge_groups = spike_v1.FigURLCuration.get_merge_groups(gh_curation_uri)
    _, _, metrics = metric_objs
    spike_v1.CurationV1.insert_curation(
        sorting_id=pop_sort["sorting_id"],
        parent_curation_id=1,
        labels=labels,
        merge_groups=merge_groups,
        metrics=metrics,
        description="after figurl curation",
    )
    yield spike_v1.CurationV1().fetch("KEY", as_dict=True)  # list of dicts


@pytest.fixture(scope="session")
def spike_merge(spike_v1):
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    yield SpikeSortingOutput()


@pytest.fixture(scope="session")
def pop_merge(
    spike_v1, pop_curation_metric, spike_merge, mini_dict, sorter_dict
):
    # TODO: add figurl fixtures when kachery_cloud is initialized

    spike_merge.insert([pop_curation_metric], part_name="CurationV1")
    yield spike_merge.fetch("KEY", as_dict=True)[0]


def hash_sort_info(sort_info):
    """Hashes attributes of a dj.Table object that are not uuids."""
    non_uuid = [  # uuid randomly assigned, do not hash
        k for k, v in sort_info.heading.attributes.items() if v.type != "uuid"
    ]
    info_dict = sort_info.fetch(*non_uuid, as_dict=True)[0]
    return key_hash(info_dict)
