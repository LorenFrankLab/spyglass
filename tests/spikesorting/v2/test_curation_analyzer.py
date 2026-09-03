"""Curation-scoped analyzer cache and routing contracts."""

from __future__ import annotations

import inspect
import hashlib
import multiprocessing
import time
import uuid
from pathlib import Path

import numpy as np
import pytest


def _concurrent_cache_resolve_worker(
    cache_root: str,
    sorting_id: str,
    folder: str,
    start_event,
    results,
) -> None:
    """Resolve one fake analyzer through the production cache transaction."""
    import datajoint as dj

    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = cache_root

    def _load(candidate, _expected, _role):
        marker = Path(candidate) / "complete.txt"
        return marker.read_text() if marker.exists() else None

    def _build(staging_folder):
        count_path = Path(cache_root) / "build_count.txt"
        count = int(count_path.read_text()) if count_path.exists() else 0
        count_path.write_text(str(count + 1))
        time.sleep(0.25)
        staging_folder = Path(staging_folder)
        staging_folder.mkdir()
        (staging_folder / "complete.txt").write_text("complete")

    resolver._load_valid_cached_analyzer = _load
    start_event.wait(timeout=10)
    try:
        value = resolver._resolve_published_analyzer(
            Path(folder), sorting_id, {}, "display", _build
        )
        results.put(("ok", value))
    except Exception:  # pragma: no cover - relayed to parent assertion
        import traceback

        results.put(("error", traceback.format_exc()))


def _folder_content_hash(folder: Path) -> str:
    """Hash relative paths and file bytes, ignoring filesystem metadata."""
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in folder.rglob("*") if candidate.is_file()
    ):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_curation_cache_path_covers_generation_recipe_role_and_si(
    monkeypatch, tmp_path
):
    from spyglass.spikesorting.v2 import _analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    sorting_id = uuid.uuid4()
    generation = uuid.uuid4()
    digest = "a" * 64
    base = cache.curation_analyzer_path(
        sorting_id, generation, "display", digest, "0.104.3"
    )
    assert base.parent == tmp_path
    parsed = cache.analyzer_cache_folder_identity(base.name)
    assert parsed is not None
    assert parsed.kind == "curation"
    assert parsed.sorting_id == sorting_id
    assert parsed.curation_uuid == generation
    assert parsed.role == "display"
    assert parsed.waveform_recipe_hash == digest
    assert cache.is_canonical_analyzer_folder_name(base.name)
    assert base != cache.curation_analyzer_path(
        sorting_id, uuid.uuid4(), "display", digest, "0.104.3"
    )
    assert base != cache.curation_analyzer_path(
        sorting_id, generation, "metric", digest, "0.104.3"
    )
    assert base != cache.curation_analyzer_path(
        sorting_id, generation, "display", "b" * 64, "0.104.3"
    )
    assert base != cache.curation_analyzer_path(
        sorting_id, generation, "display", digest, "0.105.0"
    )


def test_curation_cache_path_rejects_invalid_identity(tmp_path, monkeypatch):
    from spyglass.spikesorting.v2 import _analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="role"):
        cache.curation_analyzer_path(
            uuid.uuid4(), uuid.uuid4(), "other", "a" * 64, "0.104.3"
        )
    with pytest.raises(ValueError, match="SHA-256"):
        cache.curation_analyzer_path(
            uuid.uuid4(), uuid.uuid4(), "display", "short", "0.104.3"
        )


def test_reused_numeric_id_cannot_reuse_generation_cache(tmp_path, monkeypatch):
    """A delete/recreate generation changes path even if curation_id repeats."""
    from spyglass.spikesorting.v2 import _analyzer_cache as cache

    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    sorting_id = uuid.uuid4()
    recipe_hash = "a" * 64
    old_generation = cache.curation_analyzer_path(
        sorting_id, uuid.uuid4(), "display", recipe_hash, "0.104.3"
    )
    recreated_generation = cache.curation_analyzer_path(
        sorting_id, uuid.uuid4(), "display", recipe_hash, "0.104.3"
    )
    # curation_id is deliberately absent from the path function: two rows that
    # both receive reusable numeric id 7 still cannot collide.
    assert old_generation != recreated_generation


def test_spike_content_hash_covers_units_segments_and_frames():
    import spikeinterface as si

    from spyglass.spikesorting.v2._curation_analyzer import (
        hash_sorting_spike_content,
    )

    first = si.NumpySorting.from_unit_dict(
        [{0: np.array([10, 20]), 2: np.array([30])}],
        sampling_frequency=30_000,
    )
    same = si.NumpySorting.from_unit_dict(
        [{2: np.array([30]), 0: np.array([10, 20])}],
        sampling_frequency=30_000,
    )
    changed = si.NumpySorting.from_unit_dict(
        [{0: np.array([10, 21]), 2: np.array([30])}],
        sampling_frequency=30_000,
    )
    same_frames_different_frequency = si.NumpySorting.from_unit_dict(
        [{0: np.array([10, 20]), 2: np.array([30])}],
        sampling_frequency=29_999.999999999996,
    )
    assert hash_sorting_spike_content(first) == hash_sorting_spike_content(same)
    assert hash_sorting_spike_content(first) == hash_sorting_spike_content(
        same_frames_different_frequency
    )
    assert hash_sorting_spike_content(first) != hash_sorting_spike_content(
        changed
    )


def test_direct_curation_analyzer_access_is_detached(monkeypatch):
    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    detached = object()

    class _CacheBacked:
        def save_as(self, *, format):
            assert format == "memory"
            return detached

    monkeypatch.setattr(
        resolver, "_resolve_curation_analyzer", lambda *a, **k: _CacheBacked()
    )
    assert resolver.get_curation_analyzer({}, "recipe") is detached


def test_single_low_level_analyzer_builder():
    """Evaluation and the interactive merged wrapper share build_analyzer."""
    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    resolver_source = inspect.getsource(resolver.build_merged_analyzer)
    metric_source = (
        Path(__file__).parents[3]
        / "src"
        / "spyglass"
        / "spikesorting"
        / "v2"
        / "metric_curation.py"
    ).read_text()
    start = metric_source.index("    def make_compute(")
    end = metric_source.index("    def make_insert(", start)
    evaluation_source = metric_source[start:end]
    assert "build_analyzer(" in resolver_source
    assert "build_analyzer(" in evaluation_source
    assert "_resolve_curation_analyzer" not in evaluation_source
    assert "get_curation_analyzer" not in evaluation_source


def test_si_merge_parity_spike_recorded():
    evidence = (
        Path(__file__).parent / "resolver" / "si0104-merge-units-parity.md"
    ).read_text()
    assert "Decision: retain" in evidence
    assert "[500, 1499, 1500, 2000]" in evidence
    assert "[500, 1499, 2000]" in evidence
    assert "firing rate | 40 Hz | 30 Hz" in evidence


@pytest.mark.slow
def test_concurrent_resolve_builds_once(tmp_path):
    """Two processes serialize; the second sees the complete first publish."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("resolver cache concurrency test requires POSIX fork")
    context = multiprocessing.get_context("fork")
    sorting_id = str(uuid.uuid4())
    generation = uuid.uuid4().hex
    folder = tmp_path / (
        f"{sorting_id}__curation_{generation}_display_"
        f"{'a' * 64}_si_{'b' * 16}.zarr"
    )
    start_event = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_cache_resolve_worker,
            args=(str(tmp_path), sorting_id, str(folder), start_event, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive(), "concurrent resolver process hung"
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=2) for _ in processes)
    assert outcomes == [("ok", "complete"), ("ok", "complete")]
    assert (tmp_path / "build_count.txt").read_text() == "1"
    assert (folder / "complete.txt").read_text() == "complete"


@pytest.mark.slow
@pytest.mark.integration
def test_merged_unit_waveform_correlogram_and_ssviz_render(
    planted_two_unit_sort, curation_evaluation_defaults, monkeypatch
):
    """Merged reads render, remain immutable, and obey cache lifecycle."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2 import visualization as ssviz
    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
        curation_analyzer_with_extensions,
        curation_analyzer_cache_path,
        get_curation_analyzer,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(unit_id)
        for unit_id in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(sorting_key)
    folder = None
    try:
        root = CurationV2.insert_curation(sorting_key)
        recipe = (Sorting & sorting_key).fetch1("display_waveform_params_name")
        raw_curation_folder = curation_analyzer_cache_path(root, recipe)
        raw_analyzer = _resolve_curation_analyzer(root, recipe)
        assert list(raw_analyzer.unit_ids) == unit_ids
        assert not raw_curation_folder.exists()

        merged = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[unit_ids[:2]],
            parent_curation_id=root["curation_id"],
        )
        merged_id = int(merged["curation_id"])
        merged_uuid = str((CurationV2 & merged).fetch1("curation_uuid"))
        folder = curation_analyzer_cache_path(merged, recipe)
        analyzer = _resolve_curation_analyzer(merged, recipe)
        expected = CurationV2.get_sorting(merged)
        assert list(analyzer.unit_ids) == list(expected.unit_ids)
        for unit_id in expected.unit_ids:
            np.testing.assert_array_equal(
                analyzer.sorting.get_unit_spike_train(unit_id),
                expected.get_unit_spike_train(unit_id),
            )
        published_hash = _folder_content_hash(folder)

        # A second resolve reuses the validated generation instead of building
        # or changing the published folder.
        reused = _resolve_curation_analyzer(merged, recipe)
        assert list(reused.unit_ids) == list(expected.unit_ids)
        assert _folder_content_hash(folder) == published_hash

        # Direct access is detached. Even destructive mutation of that memory
        # copy cannot remove an extension from the published zarr analyzer.
        detached = get_curation_analyzer(merged, recipe)
        detached.delete_extension("correlograms")
        assert not detached.has_extension("correlograms")
        assert _resolve_curation_analyzer(merged, recipe).has_extension(
            "correlograms"
        )
        assert _folder_content_hash(folder) == published_hash

        # An unusual plot-only extension is computed on a temporary in-memory
        # derivative, never into the immutable published cache.
        with curation_analyzer_with_extensions(
            merged,
            recipe,
            extra_extensions={"spike_locations": {}},
        ) as derivative:
            assert derivative.has_extension("spike_locations")
        assert not _resolve_curation_analyzer(merged, recipe).has_extension(
            "spike_locations"
        )
        assert _folder_content_hash(folder) == published_hash

        selection = CurationEvaluationSelection.insert_selection(
            {
                **merged,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
            }
        )
        waveforms = CurationEvaluation().get_waveforms(selection)
        merged_unit_id = int(expected.unit_ids[-1])
        assert waveforms.get_waveforms(merged_unit_id).shape[0] > 0
        ccgs, _bins, resolved_ids = CurationEvaluation().get_correlograms(
            selection
        )
        assert ccgs.shape[:2] == (len(expected.unit_ids),) * 2
        assert list(resolved_ids) == list(expected.unit_ids)
        figure = CurationEvaluation().plot_correlograms(selection)
        assert figure.axes
        plt.close(figure)

        metrics = pd.DataFrame(
            {"snr": np.arange(len(expected.unit_ids), dtype=float)},
            index=np.asarray(expected.unit_ids, dtype=int),
        )
        monkeypatch.setattr(
            CurationEvaluation,
            "get_metrics",
            classmethod(lambda cls, key: metrics),
        )
        qc_axes = CurationEvaluation().plot_units_qc(selection)
        assert qc_axes
        plt.close("all")
        widget = ssviz.plot_si_template_metrics(selection)
        assert widget is not None
        plt.close("all")

        # Every supported read above leaves the cache byte-identical.
        assert _folder_content_hash(folder) == published_hash

        # The unified orphan collector recognizes this live curation reference.
        live_report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(folder) not in live_report["disk_side"]

        # Missing/invalid manifest and incomplete extensions are never surfaced;
        # each state is rebuilt from the committed curation.
        (folder / "spyglass_curation_analyzer_manifest.json").write_text("{}")
        repaired = _resolve_curation_analyzer(merged, recipe)
        assert repaired.has_extension("correlograms")
        manifest_path = folder / "spyglass_curation_analyzer_manifest.json"
        import json

        manifest = json.loads(manifest_path.read_text())
        expected_recipe_hash = manifest["waveform_recipe_hash"]
        manifest["waveform_recipe_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        repaired = _resolve_curation_analyzer(merged, recipe)
        assert repaired.has_extension("correlograms")
        assert (
            json.loads(manifest_path.read_text())["waveform_recipe_hash"]
            == expected_recipe_hash
        )
        repaired.delete_extension("correlograms")
        repaired = _resolve_curation_analyzer(merged, recipe)
        assert repaired.has_extension("correlograms")

        # Once the owning curation is deleted, the same folder is classified
        # and reclaimed through the one raw+curation cache entry point.
        clear_curations_for(sorting_key)
        recreated_root = CurationV2.insert_curation(sorting_key)
        recreated = CurationV2.create_merged_curation(
            sorting_key,
            merge_groups=[unit_ids[:2]],
            parent_curation_id=recreated_root["curation_id"],
        )
        assert int(recreated["curation_id"]) == merged_id
        assert (
            str((CurationV2 & recreated).fetch1("curation_uuid")) != merged_uuid
        )
        assert curation_analyzer_cache_path(recreated, recipe) != folder
        orphan_report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(folder) in orphan_report["disk_side"]
        import datajoint as dj

        monkeypatch.setattr(dj.utils, "user_choice", lambda message: "yes")
        Sorting.find_orphaned_analyzer_folders(dry_run=False)
        assert not folder.exists()
        clear_curations_for(sorting_key)
        folder = None
    finally:
        clear_curations_for(sorting_key)
        if folder is not None and folder.exists():
            import shutil

            shutil.rmtree(folder)


@pytest.mark.slow
@pytest.mark.integration
def test_preview_curation_requires_commit(
    planted_two_unit_sort, curation_evaluation_defaults
):
    """A proposed merge has no final analyzer namespace until committed."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting_key = dict(planted_two_unit_sort)
    unit_ids = sorted(
        int(unit_id)
        for unit_id in (Sorting.Unit & sorting_key).fetch("unit_id")
    )
    clear_curations_for(sorting_key)
    try:
        root = CurationV2.insert_curation(sorting_key)
        preview = CurationV2.propose_merge_curation(
            sorting_key,
            merge_groups=[unit_ids[:2]],
            parent_curation_id=root["curation_id"],
        )
        recipe = (Sorting & sorting_key).fetch1("display_waveform_params_name")
        with pytest.raises(ValueError, match="Commit the merge first"):
            _resolve_curation_analyzer(preview, recipe)
    finally:
        clear_curations_for(sorting_key)
