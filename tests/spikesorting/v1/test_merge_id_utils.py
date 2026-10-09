"""Regression tests for `get_spiking_sorting_v1_merge_ids` (issue #1159).

Three defects are covered here:

1. A single ``restriction`` dict was applied to both
   ``SpikeSortingRecordingSelection`` and ``ArtifactDetectionSelection``.
   The two tables do not share a heading, so a restriction that is
   meaningful for one is silently dropped by (or invalid for) the other.
2. The artifact lookup used ``fetch1``, which assumes a 1:1
   recording-to-artifact mapping that the schema does not enforce.
3. The fallback for sortings run without artifact detection ran
   unconditionally, so a recording with no artifact entries contributed merge
   ids even when the caller named an artifact parameter.
"""

from uuid import UUID

import pytest


@pytest.fixture(scope="module")
def merge_id_util():
    """The function under test."""
    from spyglass.spikesorting.v1.utils import get_spiking_sorting_v1_merge_ids

    yield get_spiking_sorting_v1_merge_ids


@pytest.fixture(scope="module")
def art_sel(spike_v1):
    """ArtifactDetectionSelection table."""
    yield spike_v1.ArtifactDetectionSelection


@pytest.fixture(scope="module")
def merged_curation_id(spike_merge, pop_spike_merge):
    """Curation id behind the merge entry, rather than a hard-coded 1."""
    yield (spike_merge.CurationV1 & pop_spike_merge).fetch1("curation_id")


@pytest.fixture(scope="module")
def second_artifact(art_sel, pop_rec, pop_spike_merge):
    """Add a second ArtifactDetectionSelection row for one recording.

    Nothing in the schema forbids this, so the util must tolerate it.
    Depends on ``pop_spike_merge`` so the full pipeline is populated from
    the original 1:1 state before the extra row appears.
    """
    _ = pop_spike_merge  # ensure pipeline populated before extra row

    key = {
        "recording_id": pop_rec["recording_id"],
        "artifact_param_name": "none",
    }
    art_sel.insert_selection(key)

    yield (art_sel & key).fetch1("artifact_id")

    (art_sel & key).delete(safemode=False)


@pytest.fixture(scope="module")
def raw_sorting(spike_v1, spike_merge, mini_dict, team_name, pop_spike_merge):
    """A curated, merged sorting on a recording with no artifact entries.

    Yields
    ------
    dict
        ``sort_group_id`` isolating this recording, and its ``merge_id``.
    """
    # pop_sort runs a blanket SpikeSorting.populate(), so the artifact-bearing
    # pipeline must finish before this selection exists.
    _ = pop_spike_merge

    sort_group_id = 1
    rec_key = {
        **mini_dict,
        "sort_group_id": sort_group_id,
        "preproc_param_name": "default",
        "interval_list_name": "01_s1",
        "team_name": team_name,
    }
    spike_v1.SpikeSortingRecordingSelection.insert_selection(rec_key)
    rec_pk = (
        (spike_v1.SpikeSortingRecordingSelection & rec_key).proj().fetch1("KEY")
    )
    spike_v1.SpikeSortingRecording.populate(rec_pk)

    param_name = "no_artifact_raw"
    params = dict(
        (
            spike_v1.SpikeSorterParameters
            & {
                "sorter": "clusterless_thresholder",
                "sorter_param_name": "default_clusterless",
            }
        ).fetch1("sorter_params")
    )
    params["detect_threshold"] = 1e9  # uV -- nothing crosses this
    spike_v1.SpikeSorterParameters.insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_param_name": param_name,
            "sorter_params": params,
        },
        skip_duplicates=True,
    )

    sort_key = {
        **mini_dict,
        **rec_pk,
        "sorter": "clusterless_thresholder",
        "sorter_param_name": param_name,
        "interval_list_name": "01_s1",  # raw interval, not an artifact id
    }
    spike_v1.SpikeSortingSelection.insert_selection(sort_key)
    sel_pk = (spike_v1.SpikeSortingSelection & sort_key).fetch1("KEY")
    spike_v1.SpikeSorting.populate(sel_pk)

    if not (spike_v1.SpikeSorting & sel_pk):
        pytest.fail("Could not build a sorting without artifact detection.")

    curation_key = spike_v1.CurationV1.insert_curation(
        sorting_id=str(sel_pk["sorting_id"]),
        description="sorting without artifact detection",
    )
    merge_src = {
        "sorting_id": curation_key["sorting_id"],
        "curation_id": curation_key["curation_id"],
        "analysis_file_name": curation_key["analysis_file_name"],
    }
    spike_merge.insert([merge_src], part_name="CurationV1")

    yield {
        "sort_group_id": sort_group_id,
        "merge_id": (spike_merge << merge_src).fetch1("merge_id"),
    }

    # Restore the one-recording state the rest of the suite asserts on, e.g.
    # test_recording.py::test_spike_sorting. The cascade takes the sorting,
    # curation and merge entry with it.
    (spike_v1.SpikeSortingRecordingSelection & rec_key).delete(safemode=False)


def test_backcompat_single_positional_restriction(
    merge_id_util, mini_dict, pop_spike_merge, merged_curation_id
):
    """Existing single-arg calls keep working and keep returning UUIDs."""
    ret = merge_id_util(dict(mini_dict, curation_id=merged_curation_id))

    assert len(ret) > 0, "No merge_ids returned for a valid restriction"
    assert isinstance(ret[0], UUID), "Unexpected type from util"
    assert pop_spike_merge["merge_id"] in list(
        ret
    ), "Expected merge_id missing from util return"


def test_multiple_artifacts_does_not_raise(
    merge_id_util,
    mini_dict,
    second_artifact,
    pop_spike_merge,
    merged_curation_id,
    art_sel,
):
    """A recording with two artifact entries must not raise `fetch1`."""
    assert (
        len(art_sel & {"artifact_id": second_artifact}) == 1
    ), "Fixture failed to add a second artifact entry"

    ret = merge_id_util(dict(mini_dict, curation_id=merged_curation_id))

    assert pop_spike_merge["merge_id"] in list(
        ret
    ), "Expected merge_id missing when a recording has 2 artifact entries"
    assert len(set(ret)) == len(ret), "Duplicate merge_ids returned"


def test_recording_only_attr_in_shared_restriction(
    merge_id_util,
    mini_dict,
    team_name,
    second_artifact,
    pop_spike_merge,
    merged_curation_id,
):
    """Recording-only keys in the shared restriction stay off the artifacts."""
    restriction = dict(
        mini_dict,
        team_name=team_name,
        preproc_param_name="default",
        curation_id=merged_curation_id,
    )

    ret = merge_id_util(restriction)

    assert pop_spike_merge["merge_id"] in list(
        ret
    ), "Recording-only attributes broke the artifact query"


def test_recording_restr_kwarg(
    merge_id_util,
    mini_dict,
    team_name,
    second_artifact,
    pop_spike_merge,
    merged_curation_id,
):
    """`recording_restr` is never applied to ArtifactDetectionSelection."""
    ret = merge_id_util(
        mini_dict,
        recording_restr={
            "team_name": team_name,
            "preproc_param_name": "default",
        },
        curation_id=merged_curation_id,
    )

    assert pop_spike_merge["merge_id"] in list(
        ret
    ), "recording_restr should not filter out the expected merge_id"


def test_artifact_restr_kwarg(
    merge_id_util,
    mini_dict,
    second_artifact,
    pop_spike_merge,
    merged_curation_id,
):
    """`artifact_restr` narrows the artifact table, not the recordings."""
    ret = merge_id_util(
        mini_dict,
        artifact_restr={"artifact_param_name": "default"},
        curation_id=merged_curation_id,
    )

    assert pop_spike_merge["merge_id"] in list(
        ret
    ), "artifact_restr dropped the expected merge_id"

    none_only = merge_id_util(
        mini_dict,
        artifact_restr={"artifact_param_name": "none"},
        curation_id=merged_curation_id,
    )

    assert pop_spike_merge["merge_id"] not in list(
        none_only
    ), "artifact_restr had no effect on the artifact table"


def test_raw_sorting_kept_without_artifact_restr(
    merge_id_util, mini_dict, raw_sorting
):
    """An artifact-free sorting is still reachable by an unrestricted lookup."""
    ret = merge_id_util(
        dict(mini_dict, sort_group_id=raw_sorting["sort_group_id"])
    )

    assert raw_sorting["merge_id"] in list(
        ret
    ), "Artifact-free sorting missing from an unrestricted lookup"


def test_raw_sorting_excluded_by_artifact_restr(
    merge_id_util, mini_dict, raw_sorting
):
    """An artifact filter the recording cannot satisfy returns nothing."""
    ret = merge_id_util(
        dict(mini_dict, sort_group_id=raw_sorting["sort_group_id"]),
        artifact_restr={"artifact_param_name": "default"},
    )

    assert (
        ret == []
    ), f"Artifact-free sorting bypassed an explicit artifact_restr: {ret}"


def test_latest_curation_default(
    merge_id_util, mini_dict, second_artifact, pop_spike_merge
):
    """With no curation_id, the util still resolves without raising."""
    ret = merge_id_util(mini_dict)

    assert isinstance(ret, list), "Expected a list of merge_ids"
    for merge_id in ret:
        assert isinstance(merge_id, UUID), "Unexpected type from util"


def test_no_match_returns_empty(merge_id_util):
    """A restriction matching no recording returns an empty list."""
    assert (
        merge_id_util({"nwb_file_name": "not_a_real_file_.nwb"}) == []
    ), "Expected an empty list for a non-matching restriction"
