"""Interrupted canonical analyzers self-heal without loading array payloads."""

import hashlib
import json

import numpy as np
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2 import _analyzer_cache as cache
from spyglass.spikesorting.v2 import _sorting_analyzer as analyzer_service
from spyglass.spikesorting.v2._sorting_analyzer import (
    BASE_ANALYZER_EXTENSIONS,
    _load_analyzer_folder_or_rebuild,
    build_analyzer,
    ensure_extensions,
    load_or_rebuild_analyzer_from_resolved,
)
from spyglass.spikesorting.v2.exceptions import AnalyzerFolderInvalidError

pytestmark = [pytest.mark.integration, pytest.mark.io_heavy]

_JOB_KWARGS = {"n_jobs": 1, "progress_bar": False, "random_seed": 0}
_SPIKE_FRAMES = {1: (500, 1500, 2500), 2: (1000, 2000, 3000)}


def _build_from_sources(folder, recording, sorting, params):
    """Reopen the original trace artifact, independently of any analyzer cache."""
    source_recording = si.load(recording._kwargs["folder_path"])
    source_sorting = si.NumpySorting.from_unit_dict(
        {
            unit_id: sorting.get_unit_spike_train(unit_id).copy()
            for unit_id in sorting.unit_ids
        },
        sampling_frequency=sorting.get_sampling_frequency(),
    )
    return analyzer_service.build_analyzer(
        source_sorting,
        source_recording,
        {"sorting_id": "test"},
        sorter_row={"job_kwargs": {}},
        job_kwargs=_JOB_KWARGS,
        analyzer_folder=folder,
        waveform_params=params,
        statistics_spans=[(0, source_recording.get_num_samples())],
    )


def _folder_digest(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _base_snapshot(analyzer):
    """Small immutable reference values captured before introducing damage."""
    return {
        name: {
            key: value.copy()
            for key, value in analyzer.get_extension(name).data.items()
        }
        for name in BASE_ANALYZER_EXTENSIONS
    }


def _assert_recovered_values(analyzer, source, expected):
    np.testing.assert_array_equal(analyzer.unit_ids, [1, 2])
    for unit_id, frames in _SPIKE_FRAMES.items():
        np.testing.assert_array_equal(
            analyzer.sorting.get_unit_spike_train(unit_id),
            frames,
        )
    assert analyzer.return_in_uV == source.return_in_uV
    assert analyzer.is_sparse() == source.is_sparse()
    if source.is_sparse():
        # Injected units peak on different channels; one channel per unit.
        np.testing.assert_array_equal(
            analyzer.sparsity.mask, np.eye(2, dtype=bool)
        )
    for name, arrays in expected.items():
        actual = analyzer.get_extension(name).data
        assert set(actual) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(actual[key], value)

    # Independent waveform and template oracle: slice the effective source
    # traces at the known spike frames, without using SI's waveform/template
    # readers to derive the expected values. Metric traces already include
    # whitening; display traces include the non-uniform electrode gains.
    traces = source.recording.get_traces(return_in_uV=source.return_in_uV)
    for unit_index, (unit_id, frames) in enumerate(_SPIKE_FRAMES.items()):
        channels = (
            np.array([unit_index])
            if source.is_sparse()
            else np.arange(traces.shape[1])
        )
        # SI stores extracted waveforms as float32, even after whitening.
        snippets = np.stack(
            [traces[frame - 1 : frame + 2, channels] for frame in frames]
        ).astype(np.float32)
        # Whitening matrix multiplication may round one float32 ULP
        # differently for a whole-recording read versus extraction chunks.
        np.testing.assert_allclose(
            analyzer.get_extension("waveforms").get_waveforms_one_unit(unit_id),
            snippets,
            rtol=1e-6,
            atol=1e-7,
        )
        for operator, values in (
            ("average", snippets.mean(axis=0)),
            ("std", snippets.std(axis=0)),
        ):
            dense = np.zeros((3, traces.shape[1]), dtype=values.dtype)
            dense[:, channels] = values
            np.testing.assert_allclose(
                analyzer.get_extension("templates").get_data(operator)[
                    unit_index
                ],
                dense,
                rtol=1e-6,
                atol=1e-7,
            )


@pytest.fixture(params=["dense_display", "sparse_display", "dense_metric"])
def complete_analyzer(request, tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache, "analyzer_cache_root", lambda: tmp_path / "cache_root"
    )
    spike_frames = {
        unit_id: np.asarray(frames) for unit_id, frames in _SPIKE_FRAMES.items()
    }
    traces = np.random.default_rng(0).normal(size=(10_000, 2)).astype("float32")
    for unit_index, frames in enumerate(spike_frames.values()):
        traces[frames, unit_index] -= 10 * (unit_index + 1)
    recording = si.NumpyRecording(
        traces,
        sampling_frequency=1000.0,
    )
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    recording.set_channel_gains([2.0, 5.0])
    recording.set_channel_offsets(0.0)
    recording = recording.save(
        folder=tmp_path / "recording", n_jobs=1, progress_bar=False
    )
    sorting = si.NumpySorting.from_unit_dict(
        spike_frames,
        sampling_frequency=1000.0,
    )
    params = {
        "ms_before": 1.0,
        "ms_after": 2.0,
        "max_spikes_per_unit": 10,
        "whiten": request.param == "dense_metric",
        "purpose": "metric" if request.param == "dense_metric" else "display",
        "sparsity": (
            {"method": "best_channels", "num_channels": 1}
            if request.param == "sparse_display"
            else {"method": "dense"}
        ),
    }
    folder = tmp_path / "complete.analyzer"
    _build_from_sources(folder, recording, sorting, params)
    original = cache.load_analyzer_folder(folder)
    assert original.return_in_uV == (not params["whiten"])
    assert original.is_sparse() == (request.param == "sparse_display")
    return original, recording, sorting, params


@pytest.mark.parametrize("load_extensions", [False, True])
@pytest.mark.parametrize(
    "interruption",
    [
        "missing_extensions",
        "unfinished_run",
        "missing_run_info",
        "missing_data",
        "malformed_header",
        "truncated_waveforms",
    ],
)
def test_incomplete_cache_is_rejected_or_rebuilt(
    complete_analyzer, tmp_path, interruption, load_extensions
):
    original, recording, sorting, params = complete_analyzer
    expected = _base_snapshot(original)
    folder = tmp_path / "interrupted.analyzer"
    if interruption == "missing_extensions":
        # A real SI build can be killed after only its first extension finishes.
        partial = si.create_sorting_analyzer(
            sorting,
            recording,
            format="binary_folder",
            folder=folder,
            sparse=False,
        )
        partial.compute("random_spikes", seed=0)
    else:
        cache.copy_analyzer_folder(original, folder)
        if interruption == "unfinished_run":
            run_file = folder / "extensions/templates/run_info.json"
            run_info = json.loads(run_file.read_text())
            run_info["run_completed"] = False
            run_file.write_text(json.dumps(run_info))
        elif interruption == "missing_run_info":
            # A waveform mmap already exists while extraction is still running.
            (folder / "extensions/waveforms/run_info.json").unlink()
        elif interruption == "missing_data":
            # SI saves completed run metadata before writing extension data.
            (folder / "extensions/noise_levels/noise_levels.npy").unlink()
        elif interruption == "malformed_header":
            (folder / "extensions/templates/average.npy").write_bytes(b"broken")
        else:
            data_file = folder / "extensions/waveforms/waveforms.npy"
            data_file.write_bytes(data_file.read_bytes()[:-8])

    rebuilds = []

    def rebuild():
        rebuilds.append(True)
        _build_from_sources(folder, recording, sorting, params)

    options = {
        "folder": folder,
        "recipe_label": params["purpose"],
        "sorting_id": "test",
        "rebuild_fn": rebuild,
        "load_extensions": load_extensions,
    }
    damaged_digest = _folder_digest(folder)
    with pytest.raises(AnalyzerFolderInvalidError):
        _load_analyzer_folder_or_rebuild(**options, rebuild=False)
    assert rebuilds == []
    assert _folder_digest(folder) == damaged_digest
    repaired = _load_analyzer_folder_or_rebuild(**options, rebuild=True)
    assert rebuilds == [True]
    _assert_recovered_values(repaired, original, expected)
    # A consumer previously failed here because templates were absent.
    ensure_extensions(repaired, ["template_similarity"])
    assert repaired.get_extension("template_similarity") is not None
    repaired_digest = _folder_digest(folder)
    reused = _load_analyzer_folder_or_rebuild(**options, rebuild=True)
    assert rebuilds == [True]
    assert _folder_digest(folder) == repaired_digest
    _assert_recovered_values(reused, original, expected)


def test_complete_cache_validation_keeps_extensions_lazy_and_reuses_bytes(
    complete_analyzer,
):
    original, _recording, _sorting, _params = complete_analyzer
    original_digest = _folder_digest(original.folder)

    def unexpected_rebuild():
        raise AssertionError("A complete cache must be reused.")

    for rebuild in (False, True, True):
        analyzer = _load_analyzer_folder_or_rebuild(
            original.folder,
            recipe_label="display",
            sorting_id="test",
            rebuild=rebuild,
            rebuild_fn=unexpected_rebuild,
            load_extensions=False,
        )
        assert _folder_digest(original.folder) == original_digest
        assert set(analyzer.extensions) == {"waveforms"}
        assert isinstance(
            analyzer.extensions["waveforms"].data["waveforms"], np.memmap
        )


def test_resolved_evaluation_loader_repairs_partial_cache(
    complete_analyzer, monkeypatch
):
    original, recording, sorting, params = complete_analyzer
    expected = _base_snapshot(original)
    (original.folder / "extensions/noise_levels/noise_levels.npy").unlink()
    builder = analyzer_service.build_analyzer
    rebuilds = []

    def observed_builder(*args, **kwargs):
        rebuilds.append(kwargs["analyzer_folder"])
        return builder(*args, **kwargs)

    monkeypatch.setattr(analyzer_service, "build_analyzer", observed_builder)
    options = {
        "sorting_id": "test",
        "n_units": 2,
        "analyzer_folder": original.folder,
        "waveform_params": params,
        "recording": si.load(recording._kwargs["folder_path"]),
        "sorting": sorting,
        "sorter_row": {"job_kwargs": {}},
        "job_kwargs": _JOB_KWARGS,
        "statistics_spans": [(0, recording.get_num_samples())],
    }
    with pytest.raises(AnalyzerFolderInvalidError):
        load_or_rebuild_analyzer_from_resolved(**options, rebuild=False)
    assert rebuilds == []
    repaired = load_or_rebuild_analyzer_from_resolved(**options)
    assert rebuilds == [original.folder]
    _assert_recovered_values(repaired, original, expected)
    digest = _folder_digest(original.folder)
    load_or_rebuild_analyzer_from_resolved(**options)
    assert rebuilds == [original.folder]
    assert _folder_digest(original.folder) == digest


def test_low_level_loader_accepts_intentional_extension_subset(
    complete_analyzer, tmp_path
):
    _original, recording, sorting, params = complete_analyzer
    folder = tmp_path / "subset.analyzer"
    build_analyzer(
        sorting,
        recording,
        {"sorting_id": "test"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"n_jobs": 1, "progress_bar": False, "random_seed": 0},
        analyzer_folder=folder,
        waveform_params=params,
        extensions=("noise_levels",),
    )
    analyzer = cache.load_analyzer_folder(folder)
    assert analyzer.get_saved_extension_names() == ["noise_levels"]
    assert analyzer.get_extension("noise_levels") is not None
