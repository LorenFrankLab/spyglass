"""Sorting accessors, source dispatch, and selection diagnostics.

Covers concat-source rows being included in ``Sorting.key_source`` (the
antijoin is gone now that the concat path is wired), concat-backed
``get_unit_brain_regions`` (raise vs anchor-member DataFrame), the
missing-SorterParameters and cross-recording-artifact diagnostics on
``insert_selection``, zero-unit / DataFrame ``get_sorting``, and the
peak-channel-not-in-sort-group guard in ``_populate_unit_part``.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._recipe_catalog import CORTEX_DISPLAY_WAVEFORMS
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
    """Concat-source selections ARE part of ``Sorting.key_source``.

    The concat populate path is wired (``ConcatenatedRecording.make`` +
    ``Sorting.make`` concat dispatch), so the antijoin that previously dropped
    ``ConcatenatedRecordingSource`` rows from ``key_source`` is gone -- a concat
    selection is handed to ``populate()`` like any other. A planted concat
    selection suffices to pin that it is NOT antijoined out; the full concat
    sort populate is covered by the chronic smoke in
    ``tests/spikesorting/v2/test_session_group_concat.py``.
    """
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
    session, so per-session regions are ambiguous without cross-session
    matching (not in this build). The default refuses; the opt-in returns
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
        with pytest.raises(ConcatBrainRegionAmbiguousError):
            Sorting().get_unit_brain_regions(
                {"sorting_id": sid}, allow_anchor_member=False
            )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_get_unit_brain_regions_concat_anchor_member_df(
    populated_sorting,
):
    """``allow_anchor_member=True`` returns the
    ``unit_brain_region_df`` DataFrame labeled ``anchor_member``.

    The return is the DataFrame (NOT a ``SourceResolution`` dataclass --
    that type lives inside ``make_fetch`` dispatch, not on this accessor).
    Non-vacuous: one ``Sorting.Unit`` row is planted (copied from the
    populated fixture so its Electrode FK resolves through BrainRegion), so
    the frame has a real row carrying ``region_resolution == 'anchor_member'``.
    """
    import datetime as dt
    import uuid

    import datajoint as dj
    import pandas as pd

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import SourceResolution

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
    """A sort cannot link an artifact detected on another recording.

    Both recordings are in the same session, so the old interval lookup by
    ``nwb_file_name`` + artifact interval name could succeed and apply the
    wrong artifact mask. ``insert_selection`` must reject the mismatch before
    writing ``SortingSelection.ArtifactDetectionSource``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
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
        ArtifactDetectionSelection.insert1(
            {
                "artifact_detection_id": artifact_detection_id,
                "artifact_detection_params_name": "v2_a26_cross_artifact_params",
            },
            allow_direct_insert=True,
        )
        ArtifactDetectionSelection.RecordingSource.insert1(
            {
                "artifact_detection_id": artifact_detection_id,
                "recording_id": rid_artifact,
            },
            allow_direct_insert=True,
        )
        ArtifactDetection.insert1(
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
        assert (
            len(
                SortingSelection.ArtifactDetectionSource
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
                ArtifactDetection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
            (
                ArtifactDetectionSelection.RecordingSource
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
            (
                ArtifactDetectionSelection
                & {"artifact_detection_id": artifact_detection_id}
            ).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")
        _drop_fake_recording(rid_sort)
        _drop_fake_recording(rid_artifact)


@pytest.mark.usefixtures("dj_conn")
def test_get_sorting_zero_unit_returns_empty_numpysorting(populated_sorting):
    """A zero-unit sort returns an empty single-segment ``NumpySorting``.

    The zero-unit branch short-circuits before any NWB read and returns an
    empty ``NumpySorting`` at the recording's sampling frequency (reading an
    empty units table would otherwise fail). Built on the fixture's real
    recording (the branch still reads ``sampling_frequency`` from it) via a
    planted zero-unit Sorting row.
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
    """``get_sorting(as_dataframe=True)`` indexes by Python ``int``.

    v2 writes integer unit_ids; the DataFrame path casts ``int(uid)`` so the
    index matches v1's ``nwb.units.to_dataframe()`` shape. Non-vacuous: the
    SI sorting object (a ``NumpySorting``) yields numpy integers, so the cast
    genuinely changes the type.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    raw = Sorting().get_sorting(populated_sorting)
    assert len(raw.unit_ids) >= 1, "fixture sort must have units"
    # The SI sorting object's unit_ids are numpy integers -- the cast is not
    # a no-op.
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
    """A peak channel absent from the sort group raises RuntimeError.

    ``_populate_unit_part`` resolves each unit's peak channel to a
    ``SortGroupV2.SortGroupElectrode`` row; a channel id outside the group
    is a recording/sort-group mismatch and must fail loudly. We monkeypatch
    the extremum-channel lookup to return an out-of-group id and call the
    helper with the fixture's real analyzer.
    """
    from spikeinterface.core import template_tools

    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb_file_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")
    # The transient analyzer folder _populate_unit_part loads (built by the
    # populate that created populated_sorting; resolved from sorting_id).
    analyzer_folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
    sorting = Sorting().get_sorting(populated_sorting)

    bad_channel = 10_000_000
    original_extremum_channel = template_tools.get_template_extremum_channel

    def _bad_peak_channels(analyzer, **kwargs):
        # ``_populate_unit_part`` resolves peak channels with ``outputs="id"``
        # (and a configured ``peak_sign``); ``get_template_extremum_amplitude``
        # calls this internally with ``peak_sign``/``mode`` but NOT
        # ``outputs="id"``. Both carry ``peak_sign``, so discriminate on
        # ``outputs`` -- only the unit-attribution call returns the planted
        # out-of-group channel; delegate the amplitude path to the real fn.
        if kwargs.get("outputs") != "id":
            return original_extremum_channel(analyzer, **kwargs)
        return {uid: bad_channel for uid in sorting.unit_ids}

    monkeypatch.setattr(
        template_tools, "get_template_extremum_channel", _bad_peak_channels
    )

    # The per-unit row construction (peak attribution + channel-mismatch guard)
    # now lives in ``_build_unit_rows_from_analyzer`` (run once in
    # make_compute); the Electrode FK / sort group are resolved at fetch time.
    sort_group_id, electrode_by_id, _region = (
        Sorting._fetch_unit_electrode_metadata(recording_id, nwb_file_name)
    )
    sorter_row = (
        SortingSelection * SorterParameters & populated_sorting
    ).fetch1()
    with pytest.raises(RuntimeError, match=str(bad_channel)):
        Sorting._build_unit_rows_from_analyzer(
            sorting=sorting,
            analyzer_folder=analyzer_folder,
            sorter_row=sorter_row,
            electrode_by_id=electrode_by_id,
            sort_group_id=sort_group_id,
            nwb_file_name=nwb_file_name,
            key=dict(populated_sorting),
        )
