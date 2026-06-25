"""Tests for the sorting-stage service helpers.

Covers the sorter dispatch noise-level precedence (``_sorting_dispatch``),
the artifact-mask complement walker guards (``_sorting_artifact_mask``),
the ``Sorting.Unit`` row builder (``_sorting_units``), and the analyzer
zero-unit access guard (``_sorting_analyzer``). All DB-free: pure numpy /
SpikeInterface objects and in-test relation stubs.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec(traces, fs=1000.0, gain=1.0, times=None):
    import spikeinterface as si

    n_ch = traces.shape[1]
    rec = si.NumpyRecording(
        traces_list=[traces.astype("float32")], sampling_frequency=fs
    )
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    if times is not None:
        rec.set_times(np.asarray(times, dtype=float))
    return rec


def test_clusterless_noise_levels_precedence():
    """Explicit noise_levels win; else uv->[1.0], mad->None."""
    from spyglass.spikesorting.v2._sorting_dispatch import (
        _clusterless_noise_levels,
    )

    assert _clusterless_noise_levels([2.0, 3.0], "uv") == [2.0, 3.0]
    assert _clusterless_noise_levels([2.0, 3.0], "mad") == [2.0, 3.0]
    assert _clusterless_noise_levels(None, "uv") == [1.0]
    assert _clusterless_noise_levels(None, "mad") is None


def test_apply_artifact_mask_rejects_malformed_valid_times():
    """The complement walker rejects inputs that would silently under-mask."""
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )
    from spyglass.spikesorting.v2.exceptions import (
        EmptyArtifactValidTimesError,
    )

    rec = _rec(np.zeros((100, 2), dtype="float32"))
    with pytest.raises(EmptyArtifactValidTimesError):
        apply_artifact_mask(rec, np.empty((0, 2)))
    with pytest.raises(ValueError):
        apply_artifact_mask(rec, np.array([0.0, 1.0, 2.0]))  # not (n, 2)
    with pytest.raises(ValueError):
        apply_artifact_mask(rec, np.array([[1.0, 0.0]]))  # end < start
    with pytest.raises(ValueError):
        apply_artifact_mask(
            rec, np.array([[0.0, 0.05], [0.02, 0.08]])
        )  # overlapping / unsorted


def test_apply_artifact_mask_rejects_nonmonotonic_recording_times():
    """The mask walks the recording timeline with ``searchsorted``, so a
    backward-stepping ``get_times()`` would silently mis-mask. Pins that the
    monotonic guard is actually wired into this boundary (not just the pure
    ``_signal_math`` helpers)."""
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    times = np.arange(100, dtype=float) / 1000.0
    times[50] = times[49] - 0.05  # inject a backward step
    rec = _rec(np.zeros((100, 2), dtype="float32"), times=times)
    with pytest.raises(ValueError, match="monotonic"):
        apply_artifact_mask(rec, np.array([[0.0, 0.05]]))


def test_build_sorting_unit_rows_constructs_rows_from_peak_metadata():
    """One ``Sorting.Unit`` row per unit, carrying the peak channel's Electrode
    FK fields, the peak amplitude (as float), and the precomputed spike count,
    merged onto the base key."""
    from spyglass.spikesorting.v2._sorting_units import (
        build_sorting_unit_rows,
    )

    electrode_by_id = {
        10: {
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": 10,
            "ignored_extra_column": "dropped",
        },
        20: {
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "1",
            "electrode_id": 20,
        },
    }
    rows = build_sorting_unit_rows(
        unit_ids=[0, 1],
        peak_channels={0: 10, 1: 20},
        peak_amplitudes={0: 50.0, 1: 30.5},
        n_spikes_by_unit={0: 100, 1: 40},
        electrode_by_id=electrode_by_id,
        key={"sorting_id": "s"},
        sort_group_id=0,
        nwb_file_name="x.nwb",
    )

    assert rows == [
        {
            "sorting_id": "s",
            "unit_id": 0,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "0",
            "electrode_id": 10,
            "peak_amplitude_uv": 50.0,
            "n_spikes": 100,
        },
        {
            "sorting_id": "s",
            "unit_id": 1,
            "nwb_file_name": "x.nwb",
            "electrode_group_name": "1",
            "electrode_id": 20,
            "peak_amplitude_uv": 30.5,
            "n_spikes": 40,
        },
    ]


def test_build_sorting_unit_rows_rejects_peak_channel_outside_sort_group():
    """A unit whose peak channel is not in the sort group's electrode map is a
    channel-id mismatch -- raise rather than build an invalid Electrode FK."""
    from spyglass.spikesorting.v2._sorting_units import (
        build_sorting_unit_rows,
    )

    with pytest.raises(RuntimeError, match="not in sort group"):
        build_sorting_unit_rows(
            unit_ids=[0],
            peak_channels={0: 99},  # 99 not in electrode_by_id
            peak_amplitudes={0: 50.0},
            n_spikes_by_unit={0: 100},
            electrode_by_id={10: {"nwb_file_name": "x.nwb"}},
            key={"sorting_id": "s"},
            sort_group_id=0,
            nwb_file_name="x.nwb",
        )


def test_build_sorting_unit_rows_raises_typed_error_on_non_integer_unit_id():
    """A sorter unit id that does not convert to int raises the typed
    NonIntegerUnitIDError (not a bare ValueError)."""
    from spyglass.spikesorting.v2._sorting_units import (
        build_sorting_unit_rows,
    )
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError

    with pytest.raises(NonIntegerUnitIDError):
        build_sorting_unit_rows(
            unit_ids=["noise_3"],
            peak_channels={"noise_3": 10},
            peak_amplitudes={"noise_3": 50.0},
            n_spikes_by_unit={"noise_3": 100},
            electrode_by_id={10: {"nwb_file_name": "x.nwb"}},
            key={"sorting_id": "s"},
            sort_group_id=0,
            nwb_file_name="x.nwb",
        )


def test_load_or_rebuild_analyzer_raises_zero_unit_without_path_io():
    """Zero-unit analyzer access fails before touching SI or cache paths."""
    from spyglass.spikesorting.v2._sorting_analyzer import (
        load_or_rebuild_analyzer,
    )
    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError

    class _ZeroUnitSortingRelation:
        def __and__(self, key):
            assert key == {"sorting_id": "zero"}
            return self

        def fetch1(self, *attrs):
            assert attrs == ("sorting_id", "n_units")
            return "zero", 0

    with pytest.raises(ZeroUnitAnalyzerError, match="zero units"):
        load_or_rebuild_analyzer(
            _ZeroUnitSortingRelation(), {"sorting_id": "zero"}
        )


def test_load_or_rebuild_analyzer_no_rebuild_raises_invalid_for_bad_folder(
    monkeypatch, tmp_path
):
    """An existing but unloadable analyzer folder is not treated as healthy."""
    import sys
    import types

    from spyglass.spikesorting.v2 import _analyzer_cache
    from spyglass.spikesorting.v2._sorting_analyzer import (
        load_or_rebuild_analyzer,
    )
    from spyglass.spikesorting.v2.exceptions import AnalyzerFolderInvalidError

    folder = tmp_path / "bad.zarr"
    folder.mkdir()

    class _OneUnitSortingRelation:
        def __and__(self, key):
            assert key == {"sorting_id": "s1"}
            return self

        def fetch1(self, *attrs):
            assert attrs == ("sorting_id", "n_units")
            return "s1", 1

    def _raise_load(_folder):
        raise RuntimeError("not a valid zarr store")

    monkeypatch.setattr(
        _analyzer_cache, "analyzer_path", lambda _sid, _recipe: folder
    )
    monkeypatch.setitem(
        sys.modules,
        "spikeinterface",
        types.SimpleNamespace(load_sorting_analyzer=_raise_load),
    )

    with pytest.raises(AnalyzerFolderInvalidError, match="could not be loaded"):
        load_or_rebuild_analyzer(
            _OneUnitSortingRelation(),
            {"sorting_id": "s1"},
            waveform_params_name="display",
            rebuild=False,
        )
    assert folder.exists(), "no-rebuild audit path must not mutate the folder"


def test_load_or_rebuild_analyzer_rebuilds_invalid_folder(monkeypatch, tmp_path):
    """Default analyzer access removes an invalid cache folder and rebuilds it."""
    import sys
    import types

    from spyglass.spikesorting.v2 import _analyzer_cache, _sorting_analyzer

    folder = tmp_path / "bad.zarr"
    folder.mkdir()
    rebuilt_marker = folder / "rebuilt"
    loaded = object()

    class _OneUnitSortingRelation:
        def __and__(self, key):
            assert key == {"sorting_id": "s1"}
            return self

        def fetch1(self, *attrs):
            assert attrs == ("sorting_id", "n_units")
            return "s1", 1

    def _load(_folder):
        if rebuilt_marker.exists():
            return loaded
        raise RuntimeError("not a valid zarr store")

    def _rebuild(_sorting_table, key, waveform_params_name=None):
        assert key == {"sorting_id": "s1"}
        assert waveform_params_name == "display"
        assert not folder.exists(), "invalid folder must be removed before rebuild"
        folder.mkdir()
        rebuilt_marker.write_text("ok")

    monkeypatch.setattr(
        _analyzer_cache, "analyzer_path", lambda _sid, _recipe: folder
    )
    monkeypatch.setattr(
        _sorting_analyzer, "rebuild_analyzer_folder", _rebuild
    )
    monkeypatch.setitem(
        sys.modules,
        "spikeinterface",
        types.SimpleNamespace(load_sorting_analyzer=_load),
    )

    analyzer = _sorting_analyzer.load_or_rebuild_analyzer(
        _OneUnitSortingRelation(),
        {"sorting_id": "s1"},
        waveform_params_name="display",
    )

    assert analyzer is loaded
    assert rebuilt_marker.exists()
