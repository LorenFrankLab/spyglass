"""Sorting accessors, source dispatch, and selection diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._core.recipe_catalog import (
    CORTEX_DISPLAY_WAVEFORMS,
)
from tests.spikesorting.v2._ingest_helpers import (
    _plant_concat_sorting_selection,
)

# populated_sorting uses the 'default' preprocessing recipe -> cortex display.
_DISPLAY = CORTEX_DISPLAY_WAVEFORMS


def _plant_fake_recording(recording_id, nwb_file_name, sampling_frequency):
    """Insert a minimal ``Recording`` + ``RecordingSelection`` pair via the
    FK-checks-off bypass (the right scalar fields for the guards under test,
    no real populate). Returns the ``recording_id``.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingSelection.insert1(
            {
                "recording_id": recording_id,
                "nwb_file_name": nwb_file_name,
                "sort_group_id": 0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_a25_team",
                "recording_input_hash": "0" * 64,
            },
            allow_direct_insert=True,
        )
        Recording.insert1(
            {
                "recording_id": recording_id,
                "analysis_file_name": "a25_fake.nwb",
                "electrical_series_path": "/fake/es",
                "object_id": "a25-fake-object-id",
                "n_channels": 4,
                "sampling_frequency": sampling_frequency,
                "duration_s": 60.0,
                "content_hash": "0" * 64,
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    return recording_id


def _drop_fake_recording(recording_id):
    """Tear down a ``_plant_fake_recording`` pair (parts-first)."""
    import datajoint as dj

    from spyglass.spikesorting.v2.recording import Recording, RecordingSelection

    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        (Recording & {"recording_id": recording_id}).delete_quick()
        (RecordingSelection & {"recording_id": recording_id}).delete_quick()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


@pytest.mark.usefixtures("dj_conn")
def test_sorting_key_source_includes_concat_rows():
    """Concat-source selections are part of ``Sorting.key_source``."""
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    try:
        assert len(Sorting.key_source & {"sorting_id": sid}) == 1, (
            "concat-source selection should be in Sorting.key_source now "
            "that the antijoin is removed"
        )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_get_unit_brain_regions_concat_raises_without_anchor():
    """A concat-backed sort raises ``ConcatBrainRegionAmbiguousError``
    unless ``allow_anchor_member=True``.

    A concat unit's peak channel maps to one Electrode row per member
    session, so a sort-level region is ambiguous. The default refuses and
    points to ``TrackedUnit.get_unit_brain_regions`` over a matched curation,
    which resolves each member recording's region; the opt-in returns
    anchor-member regions.
    """
    import uuid

    from spyglass.spikesorting.v2.exceptions import (
        ConcatBrainRegionAmbiguousError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    try:
        with pytest.raises(
            ConcatBrainRegionAmbiguousError,
            match=(
                "match a curation of this sort with UnitMatch and use "
                "TrackedUnit.get_unit_brain_regions"
            ),
        ):
            Sorting().get_unit_brain_regions(
                {"sorting_id": sid}, allow_anchor_member=False
            )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_get_unit_brain_regions_concat_anchor_member_df(
    populated_sorting,
):
    """``allow_anchor_member=True`` returns a DataFrame whose rows carry
    ``region_resolution == 'anchor_member'``.

    One ``Sorting.Unit`` row, copied from the populated fixture, is planted
    so the frame is non-empty.
    """
    import datetime as dt
    import uuid

    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._recording.source import SourceResolution

    template_unit = (Sorting.Unit & populated_sorting).fetch(as_dict=True)[0]

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a26_concat_fake.nwb",
                "object_id": "a26-concat-object-id",
                "n_units": 1,
                "time_of_sort": dt.datetime(2020, 1, 1),
                "display_waveform_params_name": _DISPLAY,
                # Synthetic provenance for this bypassed row; the column is
                # NOT NULL (set from si.__version__ on a real sort).
                "spikeinterface_version": "0.0.0",
            },
            allow_direct_insert=True,
        )
        unit_row = {**template_unit, "sorting_id": sid, "unit_id": 0}
        Sorting.Unit.insert1(unit_row, allow_direct_insert=True)
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        result = Sorting().get_unit_brain_regions(
            {"sorting_id": sid}, allow_anchor_member=True
        )
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, SourceResolution)
        assert "region_resolution" in result.columns
        assert len(result) == 1, "anchor-member df should carry the one unit"
        assert (result["region_resolution"] == "anchor_member").all()
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (Sorting.Unit & {"sorting_id": sid}).delete_quick()
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_selection_missing_sorter_params_diagnostic():
    """A missing ``SorterParameters`` row raises a diagnostic ValueError.

    ``_ensure_lookup_row_exists`` translates the would-be FK IntegrityError
    into a message naming ``SorterParameters`` and ``insert_default()``. The
    recording source has no matching master, so control reaches the
    lookup pre-check.
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import SortingSelection

    with pytest.raises(ValueError) as excinfo:
        SortingSelection.insert_selection(
            {
                "recording_id": uuid.uuid4(),
                "sorter": "mountainsort5",
                "sorter_params_name": "a26_no_such_sorter_row",
            }
        )
    message = str(excinfo.value)
    assert "SorterParameters" in message
    assert "insert_default" in message


@pytest.mark.usefixtures("dj_conn")
def test_sorting_selection_rejects_cross_recording_artifact_detection_source():
    """``insert_selection`` rejects an artifact detected on another recording
    of the same session, before writing any source row.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    rid_sort = _plant_fake_recording(
        uuid.uuid4(), "session_cross_artifact_.nwb", 30000.0
    )
    rid_artifact = _plant_fake_recording(
        uuid.uuid4(), "session_cross_artifact_.nwb", 30000.0
    )
    artifact_detection_id = uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        RecordingArtifactSelection.insert1(
            {
                "artifact_detection_id": artifact_detection_id,
                "artifact_detection_params_name": "v2_a26_cross_artifact_params",
                "recording_id": rid_artifact,
            },
            allow_direct_insert=True,
        )
        RecordingArtifactDetection.insert1(
            {"artifact_detection_id": artifact_detection_id},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        SorterParameters.insert_default()
        with pytest.raises(ValueError, match="belongs to recording_id"):
            SortingSelection.insert_selection(
                {
                    "recording_id": rid_sort,
                    "sorter": "clusterless_thresholder",
                    "sorter_params_name": "default",
                    "artifact_detection_id": artifact_detection_id,
                }
            )
        # Rejected before registering the artifact in ArtifactDetectionOutput.
        from spyglass.spikesorting.v2.artifact_output import (
            ArtifactDetectionOutput,
        )

        assert (
            len(
                ArtifactDetectionOutput.RecordingSource
                & {"artifact_detection_id": artifact_detection_id}
            )
            == 0
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            sort_keys = (
                SortingSelection.RecordingSource & {"recording_id": rid_sort}
            ).fetch("KEY", as_dict=True)
            if sort_keys:
                (
                    SortingSelection.ArtifactDetectionSource & sort_keys
                ).delete_quick()
                (
                    SortingSelection.RecordingSource
                    & {"recording_id": rid_sort}
                ).delete_quick()
                (SortingSelection & sort_keys).delete_quick()
            (
                RecordingArtifactDetection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
            (
                RecordingArtifactSelection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rid_sort)
        _drop_fake_recording(rid_artifact)


@pytest.mark.usefixtures("dj_conn")
def test_get_sorting_zero_unit_returns_empty_numpysorting(populated_sorting):
    """A zero-unit sort returns an empty single-segment ``NumpySorting`` at
    the recording's sampling frequency, without reading the units NWB.
    """
    import datetime as dt
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        SortingSelection.RecordingSource.insert1(
            {"sorting_id": sid, "recording_id": recording_id},
            allow_direct_insert=True,
        )
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a26_zero_fake.nwb",
                "object_id": "a26-zero-object-id",
                "n_units": 0,
                "time_of_sort": dt.datetime(2020, 1, 1),
                "display_waveform_params_name": _DISPLAY,
                # Synthetic provenance for this bypassed row; the column is
                # NOT NULL (set from si.__version__ on a real sort).
                "spikeinterface_version": "0.0.0",
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        sorting = Sorting().get_sorting({"sorting_id": sid})
        assert len(sorting.unit_ids) == 0
        assert sorting.get_num_segments() == 1
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (Sorting & {"sorting_id": sid}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_get_sorting_dataframe_casts_unit_ids_to_python_int(populated_sorting):
    """``get_sorting(as_dataframe=True)`` indexes by Python ``int``, matching
    v1's ``nwb.units.to_dataframe()``.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    raw = Sorting().get_sorting(populated_sorting)
    assert len(raw.unit_ids) >= 1, "fixture sort must have units"
    # SI unit_ids are numpy integers, so the cast is not a no-op.
    assert all(
        isinstance(uid, np.integer) for uid in raw.unit_ids
    ), "precondition: NumpySorting yields numpy unit_ids"

    df = Sorting().get_sorting(populated_sorting, as_dataframe=True)
    assert all(
        type(uid) is int for uid in df.index
    ), "DataFrame index unit_ids must be Python int, not numpy scalars"


@pytest.mark.usefixtures("dj_conn")
def test_populate_unit_part_peak_channel_not_in_sort_group(
    populated_sorting, monkeypatch
):
    """A peak channel absent from the sort group raises RuntimeError in
    ``build_unit_rows_from_analyzer``.
    """
    from spikeinterface.core import template_tools

    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._sorting.fetch import (
        fetch_unit_electrode_metadata,
    )
    from spyglass.spikesorting.v2._sorting.units import (
        build_unit_rows_from_analyzer,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb_file_name, recording_sort_group_id = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name", "sort_group_id")
    # The analyzer folder built by populated_sorting's populate.
    analyzer_folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
    sorting = Sorting().get_sorting(populated_sorting)

    bad_channel = 10_000_000
    original_extremum_channel = template_tools.get_template_extremum_channel

    def _bad_peak_channels(analyzer, **kwargs):
        # Only the ``outputs="id"`` (peak-channel) call returns the planted
        # channel; ``get_template_extremum_amplitude`` calls this without it.
        if kwargs.get("outputs") != "id":
            return original_extremum_channel(analyzer, **kwargs)
        return {uid: bad_channel for uid in sorting.unit_ids}

    monkeypatch.setattr(
        template_tools, "get_template_extremum_channel", _bad_peak_channels
    )

    sort_group_id, electrode_by_id, _region = fetch_unit_electrode_metadata(
        recording_sort_group_id, nwb_file_name
    )
    sorter_row = (
        SortingSelection * SorterParameters & populated_sorting
    ).fetch1()
    with pytest.raises(RuntimeError, match=str(bad_channel)):
        build_unit_rows_from_analyzer(
            sorting=sorting,
            analyzer_folder=analyzer_folder,
            sorter_row=sorter_row,
            electrode_by_id=electrode_by_id,
            sort_group_id=sort_group_id,
            nwb_file_name=nwb_file_name,
            key=dict(populated_sorting),
        )
