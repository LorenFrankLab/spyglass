"""Old sparse masks follow the same polarity contract as fresh analyzers."""

import json

import numpy as np
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2._params.analyzer_waveform import (
    AnalyzerWaveformParamsSchema,
    SparsityParams,
)
from spyglass.spikesorting.v2._sorting import analyzer as analyzer_service
from spyglass.spikesorting.v2._storage import analyzer_cache

pytestmark = pytest.mark.unit


def _legacy_cache(tmp_path, *, sign, sparsity):
    """Build an unmarked cache with the pre-fix recipe-only SI sparsity."""
    frames = np.array([1000, 3000, 5000, 7000, 9000])
    traces = np.zeros((10_000, 2), dtype=np.float32)
    traces[frames, 0] = sign * 100
    traces[frames, 1] = -sign * 20
    traces[frames + 1, 1] = sign * 5
    recording = si.NumpyRecording(traces, 30_000, channel_ids=[12, 24])
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 500]]))
    recording.set_channel_gains(1.0)
    recording.set_channel_offsets(0.0)
    recording = recording.save(
        folder=tmp_path / "recording", n_jobs=1, progress_bar=False
    )
    sorting = si.NumpySorting.from_unit_dict({17: frames}, 30_000)
    params = AnalyzerWaveformParamsSchema(sparsity=sparsity).model_dump()
    folder = tmp_path / "sparsity_cache__legacy.analyzer"
    legacy = si.create_sorting_analyzer(
        sorting,
        recording,
        format="binary_folder",
        folder=folder,
        **SparsityParams.model_validate(params["sparsity"]).si_create_kwargs(),
    )
    legacy.compute(
        list(analyzer_service.BASE_ANALYZER_EXTENSIONS),
        extension_params={
            "random_spikes": {"seed": 0},
            "noise_levels": {"random_slices_kwargs": {"seed": 0}},
            "waveforms": {
                "ms_before": params["ms_before"],
                "ms_after": params["ms_after"],
            },
        },
        n_jobs=1,
        progress_bar=False,
    )
    return {
        "sorting_id": "sparsity_cache",
        "n_units": 1,
        "analyzer_folder": folder,
        "waveform_params": params,
        "recording": recording,
        "sorting": sorting,
        "job_kwargs": {"n_jobs": 1, "progress_bar": False, "random_seed": 0},
        "statistics_spans": [[0, 10_000]],
    }


@pytest.mark.parametrize(
    "sorter_params, sign, sparsity",
    [
        ({"peak_sign": "pos"}, 1, {}),
        ({"detect_sign": 0}, 1, {}),
        ({"detect_sign": -1}, -1, {"peak_sign": "pos"}),
        ({"peak_sign": "both"}, -1, {"peak_sign": "pos"}),
    ],
)
def test_incompatible_legacy_sparse_cache_is_rejected_then_rebuilt(
    tmp_path, monkeypatch, sorter_params, sign, sparsity
):
    from spyglass.spikesorting.v2.exceptions import AnalyzerFolderInvalidError

    inputs = _legacy_cache(tmp_path, sign=sign, sparsity=sparsity)
    inputs["sorter_row"] = {"params": sorter_params}
    folder = inputs["analyzer_folder"]
    with pytest.raises(AnalyzerFolderInvalidError):
        analyzer_service.load_or_rebuild_analyzer_from_resolved(
            **inputs, rebuild=False
        )
    assert folder.exists(), "a no-rebuild read must leave the old cache intact"

    builds = []
    original_build = analyzer_service.build_analyzer

    def build(*args, **kwargs):
        builds.append(kwargs["analyzer_folder"])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(analyzer_service, "build_analyzer", build)
    rebuilt = analyzer_service.load_or_rebuild_analyzer_from_resolved(**inputs)
    template = rebuilt.get_extension("templates").get_data()[0]
    assert np.abs(template[:, 0]).max() == 100
    assert builds == [folder]
    analyzer_service.load_or_rebuild_analyzer_from_resolved(**inputs)
    assert builds == [folder], "the corrected cache must be reusable"

    # Curation derivatives and expert working copies preserve the receipt;
    # otherwise a positive cache copy would look like an incompatible legacy
    # neg cache when reused.
    copied = analyzer_cache.copy_analyzer_folder(
        rebuilt, tmp_path / "copy.analyzer"
    )
    analyzer_cache.assert_analyzer_sparsity_polarity(
        copied,
        waveform_params=inputs["waveform_params"],
        sorter_params=sorter_params,
    )
    assert (copied.folder / analyzer_cache.ANALYZER_SPARSITY_RECEIPT).is_file()


@pytest.mark.parametrize(
    "sorter_params, sign, sparsity",
    [
        ({}, -1, {}),
        ({"detect_sign": -1}, -1, {}),
        ({"peak_sign": "pos"}, 1, {"peak_sign": "pos"}),
        ({"peak_sign": "both"}, 1, {"peak_sign": "both"}),
        ({"detect_sign": 1}, 1, {"method": "dense"}),
    ],
)
def test_compatible_legacy_and_dense_caches_are_reused(
    tmp_path, monkeypatch, sorter_params, sign, sparsity
):
    inputs = _legacy_cache(tmp_path, sign=sign, sparsity=sparsity)
    inputs["sorter_row"] = {"params": sorter_params}

    def must_not_build(*args, **kwargs):
        raise AssertionError("a compatible cache must be reused")

    monkeypatch.setattr(analyzer_service, "build_analyzer", must_not_build)
    cached = analyzer_service.load_or_rebuild_analyzer_from_resolved(**inputs)
    assert (
        np.abs(cached.get_extension("templates").get_data()[0, :, 0]).max()
        == 100
    )


def test_public_sort_analyzer_rebuilds_incompatible_legacy_cache(
    tmp_path, monkeypatch
):
    from spyglass.spikesorting.v2._sorting import fetch as sorting_fetch

    inputs = _legacy_cache(tmp_path, sign=1, sparsity={})
    sorter_row = {"params": {"detect_sign": 1}}
    builds = []

    class SortingRelation:
        def __and__(self, key):
            assert key == {"sorting_id": inputs["sorting_id"]}
            return self

        def fetch1(self, *attributes):
            assert attributes == ("sorting_id", "n_units")
            return inputs["sorting_id"], 1

    def rebuild(table, key, waveform_params_name):
        builds.append(key)
        analyzer_service.build_analyzer(
            inputs["sorting"],
            inputs["recording"],
            key,
            sorter_row=sorter_row,
            job_kwargs=inputs["job_kwargs"],
            analyzer_folder=inputs["analyzer_folder"],
            waveform_params=inputs["waveform_params"],
        )

    monkeypatch.setattr(
        analyzer_cache, "analyzer_path", lambda *args: inputs["analyzer_folder"]
    )
    monkeypatch.setattr(
        analyzer_service,
        "fetch_waveform_params",
        lambda name: inputs["waveform_params"],
    )
    monkeypatch.setattr(
        sorting_fetch,
        "fetch_sorter_analyzer_inputs",
        lambda key: (sorter_row, {}),
    )
    monkeypatch.setattr(analyzer_service, "rebuild_analyzer_folder", rebuild)
    cached = analyzer_service.load_or_rebuild_analyzer(
        SortingRelation(),
        {"sorting_id": inputs["sorting_id"]},
        waveform_params_name="legacy",
    )
    assert builds == [{"sorting_id": inputs["sorting_id"]}]
    assert cached.get_extension("templates").get_data()[0, :, 0].max() == 100


def test_curation_manifest_cache_rebuilds_incompatible_legacy_mask(tmp_path):
    from spyglass.spikesorting.v2._curation import analyzer as curation_service

    inputs = _legacy_cache(tmp_path, sign=1, sparsity={})
    folder = inputs["analyzer_folder"]
    expected = {
        "curated_unit_ids": (17,),
        "merged_spike_content_hash": curation_service.hash_sorting_spike_content(
            inputs["sorting"]
        ),
        "extension_request": {},
    }

    def write_manifest(analyzer, target):
        manifest = {
            **expected,
            "curated_unit_ids": [17],
            "extension_inventory": curation_service._extension_inventory(
                analyzer, "metric"
            ),
            "storage_fingerprint": curation_service._folder_storage_fingerprint(
                target
            ),
        }
        (target / curation_service.CURATION_ANALYZER_MANIFEST).write_text(
            json.dumps(manifest)
        )

    write_manifest(analyzer_cache.load_analyzer_folder(folder), folder)
    validate = analyzer_service.analyzer_sparsity_validator(
        inputs["waveform_params"], lambda: {"peak_sign": "pos"}
    )
    # Its old manifest is intact, so the polarity guard must discriminate.
    assert (
        curation_service._load_valid_cached_analyzer(folder, expected, "metric")
        is not None
    )
    assert (
        curation_service._load_valid_cached_analyzer(
            folder, expected, "metric", validate_analyzer=validate
        )
        is None
    )
    builds = []

    def build(target):
        builds.append(target)
        analyzer_service.build_analyzer(
            inputs["sorting"],
            inputs["recording"],
            {"sorting_id": inputs["sorting_id"]},
            sorter_row={"params": {"peak_sign": "pos"}},
            job_kwargs=inputs["job_kwargs"],
            analyzer_folder=target,
            waveform_params=inputs["waveform_params"],
        )
        write_manifest(analyzer_cache.load_analyzer_folder(target), target)

    for _ in range(2):
        cached = curation_service._resolve_published_analyzer(
            folder,
            inputs["sorting_id"],
            expected,
            "metric",
            build,
            validate_analyzer=validate,
        )
        assert (
            cached.get_extension("templates").get_data()[0, :, 0].max() == 100
        )
    assert len(builds) == 1


@pytest.mark.parametrize("curation_cache", [False, True])
def test_unavailable_sorter_params_do_not_invalidate_a_sparse_cache(
    tmp_path, curation_cache
):
    from spyglass.spikesorting.v2._curation import analyzer as curation_service

    inputs = _legacy_cache(tmp_path, sign=-1, sparsity={})
    folder = inputs["analyzer_folder"]

    def unavailable_params():
        raise RuntimeError("temporary database connection failure")

    def must_not_build(*args):
        raise AssertionError("unavailable inputs must not invalidate the cache")

    validate = analyzer_service.analyzer_sparsity_validator(
        inputs["waveform_params"], unavailable_params
    )
    if curation_cache:
        cached = analyzer_cache.load_analyzer_folder(folder)
        expected = {
            "curated_unit_ids": (17,),
            "merged_spike_content_hash": curation_service.hash_sorting_spike_content(
                inputs["sorting"]
            ),
            "extension_request": {},
        }
        (folder / curation_service.CURATION_ANALYZER_MANIFEST).write_text(
            json.dumps(
                {
                    **expected,
                    "curated_unit_ids": [17],
                    "extension_inventory": curation_service._extension_inventory(
                        cached, "metric"
                    ),
                    "storage_fingerprint": curation_service._folder_storage_fingerprint(
                        folder
                    ),
                }
            )
        )

        def load():
            return curation_service._resolve_published_analyzer(
                folder,
                inputs["sorting_id"],
                expected,
                "metric",
                must_not_build,
                validate_analyzer=validate,
            )

    else:

        def load():
            return analyzer_service._load_analyzer_folder_or_rebuild(
                folder,
                sorting_id=inputs["sorting_id"],
                recipe_label="legacy",
                rebuild=True,
                rebuild_fn=must_not_build,
                validate_analyzer=validate,
            )

    before = analyzer_cache.analyzer_folder_storage_fingerprint(folder)
    with pytest.raises(
        RuntimeError, match="Could not resolve sorter parameters"
    ):
        load()
    assert analyzer_cache.analyzer_folder_storage_fingerprint(folder) == before
