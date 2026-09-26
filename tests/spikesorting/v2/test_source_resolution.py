"""Tests for the DB-free sort source-resolution service.

Covers the lineage / effective-traces split built by
``effective_source_from_base``, the artifact-mask gate of
``read_effective_recording``, and DataJoint's tri-part ``DeepHash`` over the
carriers. No DB.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

_FS = 30_000.0
_N_FRAMES = 3_000
_T0 = 7.0
#: Frames masked by the fixture's valid times: [1000, 2000).
_MASK_START, _MASK_STOP = 1_000, 2_000


@pytest.fixture
def written_traces(tmp_path):
    """A small NWB whose every sample is nonzero, plus its valid times.

    Returns ``(abs_path, row, raw_traces, valid_times)``. ``valid_times`` keep
    frames ``[0, 1000)`` and ``[2000, 3000)``, so a correct mask zeros exactly
    frames ``[1000, 2000)`` and a missing mask leaves them nonzero.
    """
    from tests.spikesorting.v2._ingest_helpers import (
        write_processed_recording_nwb,
    )

    rng = np.random.default_rng(0)
    raw = rng.integers(1, 100, size=(_N_FRAMES, 4)).astype(np.int16)
    raw *= rng.choice([-1, 1], size=raw.shape).astype(np.int16)
    timestamps = _T0 + np.arange(_N_FRAMES) / _FS
    path, series_path = write_processed_recording_nwb(
        tmp_path / "effective.nwb",
        traces=raw,
        timestamps=timestamps,
        rel_positions=[[0, 0], [0, 20], [0, 40], [0, 60]],
    )
    # The mask treats a valid interval's end as exclusive in frames.
    valid_times = np.array(
        [
            [timestamps[0], timestamps[_MASK_START]],
            [timestamps[_MASK_STOP], timestamps[-1]],
        ]
    )
    row = {
        "analysis_file_name": "effective.nwb",
        "electrical_series_path": series_path,
    }
    return str(path), row, raw, valid_times


def _lineage(kind="recording", artifact_detection_id=None):
    from spyglass.spikesorting.v2._source_resolution import SourceLineage

    key_name = "recording_id" if kind == "recording" else "concat_recording_id"
    return SourceLineage(
        kind=kind,
        key={key_name: uuid.UUID(int=1)},
        artifact_detection_id=artifact_detection_id,
    )


@pytest.mark.parametrize(
    ("kind", "artifact_detection_id", "expected_mask"),
    [
        ("recording", uuid.UUID(int=2), True),
        ("recording", None, False),
        # A concat artifact carries its member masks; a stray detection id on
        # its lineage must not add a second mask at load.
        ("concatenated_recording", uuid.UUID(int=2), False),
        ("concatenated_recording", None, False),
    ],
)
def test_base_effective_source_reads_lineage_source(
    kind, artifact_detection_id, expected_mask
):
    from spyglass.spikesorting.v2._source_resolution import (
        effective_source_from_base,
    )

    lineage = _lineage(kind, artifact_detection_id)
    row = {"analysis_file_name": "x.nwb", "electrical_series_path": "a/b"}
    effective = effective_source_from_base(lineage, row)

    assert effective.lineage is lineage
    assert effective.traces.kind == kind
    assert effective.traces.key == lineage.key
    assert effective.traces.row is row
    assert effective.traces.apply_artifact_mask is expected_mask


def test_masks_exactly_when_traces_require_it(written_traces):
    from spyglass.spikesorting.v2._recording_nwb import read_recording_nwb
    from spyglass.spikesorting.v2._source_resolution import (
        effective_source_from_base,
        read_effective_recording,
    )

    abs_path, row, _raw, valid_times = written_traces
    detection_id = uuid.UUID(int=2)
    masked_traces = effective_source_from_base(
        _lineage("recording", detection_id), row
    ).traces
    unmasked_traces = effective_source_from_base(
        _lineage("recording", None), row
    ).traces
    expected_raw = read_recording_nwb(
        abs_path, electrical_series_path=row["electrical_series_path"]
    ).get_traces()
    assert np.all(expected_raw != 0)

    unmasked = read_effective_recording(abs_path, unmasked_traces)
    masked = read_effective_recording(
        abs_path,
        masked_traces,
        artifact_valid_times=valid_times,
        artifact_detection_id=detection_id,
    )

    for recording in (unmasked, masked):
        assert recording.get_annotation("is_filtered") is True
        assert recording.get_num_samples() == _N_FRAMES
        np.testing.assert_array_equal(
            recording.get_times(), _T0 + np.arange(_N_FRAMES) / _FS
        )
    np.testing.assert_array_equal(unmasked.get_traces(), expected_raw)
    masked_data = masked.get_traces()
    np.testing.assert_array_equal(masked_data[_MASK_START:_MASK_STOP], 0)
    np.testing.assert_array_equal(
        masked_data[:_MASK_START], expected_raw[:_MASK_START]
    )
    np.testing.assert_array_equal(
        masked_data[_MASK_STOP:], expected_raw[_MASK_STOP:]
    )


def test_concat_traces_are_never_masked_at_load(written_traces):
    from spyglass.spikesorting.v2._source_resolution import (
        effective_source_from_base,
        read_effective_recording,
    )

    abs_path, row, raw, valid_times = written_traces
    traces = effective_source_from_base(
        _lineage("concatenated_recording", uuid.UUID(int=2)), row
    ).traces

    loaded = read_effective_recording(abs_path, traces)
    assert np.all(loaded.get_traces()[_MASK_START:_MASK_STOP] != 0)
    with pytest.raises(ValueError, match="not artifact-masked at load"):
        read_effective_recording(
            abs_path, traces, artifact_valid_times=valid_times
        )


def test_mask_without_valid_times_raises(written_traces):
    from spyglass.spikesorting.v2._source_resolution import (
        effective_source_from_base,
        read_effective_recording,
    )

    abs_path, row, _raw, _valid_times = written_traces
    traces = effective_source_from_base(
        _lineage("recording", uuid.UUID(int=2)), row
    ).traces
    with pytest.raises(ValueError, match="must be artifact-masked at load"):
        read_effective_recording(abs_path, traces)


def test_effective_source_deep_hash_tracks_content():
    """DataJoint's tri-part check ``DeepHash``es every fetched carrier."""
    from deepdiff import DeepHash

    from spyglass.spikesorting.v2._source_resolution import (
        effective_source_from_base,
    )

    def build(detection_id):
        # Fetched rows carry UUIDs and blob arrays.
        row = {
            "recording_id": uuid.UUID(int=1),
            "analysis_file_name": "x.nwb",
            "electrical_series_path": "acquisition/ProcessedElectricalSeries",
            "obs_intervals": np.array([[0.0, 1.5], [2.0, 3.0]]),
        }
        return effective_source_from_base(
            _lineage("recording", detection_id), row
        )

    def digest(value):
        return DeepHash(value, ignore_iterable_order=False)[value]

    first = build(uuid.UUID(int=2))
    assert digest(first) == digest(build(uuid.UUID(int=2)))
    assert digest(first) != digest(build(None))
    assert digest(first) != digest(build(uuid.UUID(int=3)))


def test_motion_corrected_traces_are_never_masked_again(written_traces):
    """A motion-corrected artifact is persisted masked: it loads as stored,
    and asking to mask it again is refused."""
    from spyglass.spikesorting.v2._source_resolution import (
        EffectiveTraces,
        read_effective_recording,
    )

    abs_path, row, raw, _valid_times = written_traces
    key = {"motion_corrected_recording_id": uuid.UUID(int=3)}
    traces = EffectiveTraces(
        kind="motion_corrected_recording",
        key=key,
        row=row,
        apply_artifact_mask=False,
    )
    loaded = read_effective_recording(abs_path, traces)
    assert loaded.get_annotation("is_filtered")
    np.testing.assert_array_equal(loaded.get_traces(return_in_uV=False), raw)
    with pytest.raises(ValueError, match="must not be artifact-masked"):
        read_effective_recording(
            abs_path, traces._replace(apply_artifact_mask=True)
        )
