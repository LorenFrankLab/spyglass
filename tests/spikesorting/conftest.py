import re

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

    yield (spike_v1.CurationV1() & {"parent_curation_id": -1}).fetch(
        "KEY", as_dict=True
    )[0]


@pytest.fixture(scope="session")
def pop_metric(spike_v1, pop_sort, pop_curation):
    _ = pop_curation  # make sure this happens first
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


def is_uuid(text):
    uuid_pattern = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    )
    return uuid_pattern.fullmatch(str(text)) is not None


def hash_sort_info(sort_info):
    """Hashes attributes of a dj.Table object that are not randomly assigned."""
    no_str_uuid = {
        k: v
        for k, v in sort_info.fetch(as_dict=True)[0].items()
        if not is_uuid(v) and k != "analysis_file_name"
    }
    return key_hash(no_str_uuid)


@pytest.fixture(scope="session")
def spike_v1_group():
    from spyglass.spikesorting.analysis.v1 import group

    yield group


@pytest.fixture(scope="session")
def pop_group(spike_v1_group, spike_merge, mini_dict, pop_merge):

    _ = pop_merge  # make sure this happens first

    spike_v1_group.UnitSelectionParams().insert_default()
    spike_v1_group.SortedSpikesGroup().create_group(
        **mini_dict,
        group_name="demo_group",
        keys=spike_merge.proj(spikesorting_merge_id="merge_id").fetch("KEY"),
        unit_filter_params_name="default_exclusion",
    )
    yield spike_v1_group.SortedSpikesGroup().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def spike_v1_ua():
    from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation

    yield UnitAnnotation()


@pytest.fixture(scope="session")
def pop_annotations(spike_v1_group, spike_v1_ua, pop_group):
    spike_times, unit_ids = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_group, return_unit_ids=True
    )
    for spikes, unit_key in zip(spike_times, unit_ids):
        quant_key = {
            **unit_key,
            "annotation": "spike_count",
            "quantification": len(spikes),
        }
        label_key = {
            **unit_key,
            "annotation": "cell_type",
            "label": "pyridimal" if len(spikes) < 1000 else "interneuron",
        }

        spike_v1_ua.add_annotation(quant_key, skip_duplicates=True)
        spike_v1_ua.add_annotation(label_key, skip_duplicates=True)

    yield (
        spike_v1_ua.Annotation
        # * (spike_v1_group.SortedSpikesGroup.Units & pop_group)
        & {"annotation": "spike_count"}
    )
