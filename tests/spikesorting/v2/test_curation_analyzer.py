"""Curation-scoped analyzer cache and routing contracts."""

from __future__ import annotations

import hashlib
import inspect
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


def test_extension_inventory_never_reads_payload(monkeypatch):
    """Cache hits validate small metadata without decoding extension arrays."""
    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    class _Extension:
        def __init__(self):
            self.params = {"operator": "average"}

        def get_data(self):
            raise AssertionError("extension payload must not be read")

    class _Analyzer:
        extension = _Extension()

        def get_saved_extension_names(self):
            return ["templates"]

        def get_extension(self, name):
            assert name == "templates"
            return self.extension

    monkeypatch.setattr(
        resolver,
        "_expected_extensions",
        lambda _role, _extra=(): ("templates",),
    )
    analyzer = _Analyzer()
    original = resolver._extension_inventory(analyzer, "display")
    analyzer.extension.params = {"operator": "median"}
    modified = resolver._extension_inventory(analyzer, "display")
    assert modified != original


def test_folder_storage_fingerprint_tracks_file_stats_and_ignores_manifest(
    tmp_path,
):
    """The cheap disk fingerprint covers analyzer files, not its own manifest."""
    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    payload = tmp_path / "extensions" / "templates" / "data.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"stored extension bytes")
    original = resolver._folder_storage_fingerprint(tmp_path)

    (tmp_path / resolver.CURATION_ANALYZER_MANIFEST).write_text("{}")
    assert resolver._folder_storage_fingerprint(tmp_path) == original

    payload.write_bytes(b"modified extension bytes with a new size")
    assert resolver._folder_storage_fingerprint(tmp_path) != original


def test_cache_rejects_storage_drift_before_loading_analyzer(
    monkeypatch, tmp_path
):
    """Changed chunks invalidate a cache without opening their array payloads."""
    import json

    import spikeinterface as si

    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    expected = {
        "sorting_id": "sorting-id",
        "curation_uuid": "curation-uuid",
        "curation_id": 1,
        "role": "display",
        "curated_unit_ids": (1,),
        "contributor_map": {},
        "merged_spike_content_hash": "spike-hash",
        "merge_policy_version": "policy",
        "source_artifact_hashes": {},
        "waveform_recipe_hash": "recipe-hash",
        "spikeinterface_version": "si-version",
    }
    payload = tmp_path / "extensions" / "templates" / "data.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"original")
    manifest = {
        **expected,
        "curated_unit_ids": [1],
        "extension_inventory": {"templates": "params-hash"},
        "storage_fingerprint": resolver._folder_storage_fingerprint(tmp_path),
    }
    (tmp_path / resolver.CURATION_ANALYZER_MANIFEST).write_text(
        json.dumps(manifest)
    )
    payload.write_bytes(b"externally modified with a different size")
    loads = []
    monkeypatch.setattr(
        si, "load_sorting_analyzer", lambda _folder: loads.append(_folder)
    )
    assert (
        resolver._load_valid_cached_analyzer(tmp_path, expected, "display")
        is None
    )
    assert loads == []


def test_open_curation_analyzer_yields_disk_backed_working_copy(
    monkeypatch, tmp_path
):
    """Expert access is a temp-dir ``binary_folder`` copy, removed on exit.

    Never a memory copy (unbounded for long sorts) and never the published
    cache itself (mutable SI object).
    """
    from spyglass import settings
    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    seen: dict = {}

    class _Working:
        def has_recording(self):
            return True

    class _Published:
        def save_as(self, *, format, folder):
            seen["format"] = format
            seen["folder"] = Path(folder)
            Path(folder).mkdir(parents=True)
            return _Working()

        def has_recording(self):
            return True

    monkeypatch.setattr(settings, "temp_dir", str(tmp_path))
    monkeypatch.setattr(
        resolver, "_resolve_curation_analyzer", lambda *a, **k: _Published()
    )
    with resolver.open_curation_analyzer({}, "recipe") as working:
        assert isinstance(working, _Working)
        assert seen["format"] == "binary_folder"
        assert seen["folder"].is_relative_to(tmp_path)
        assert seen["folder"].exists()
    assert not seen["folder"].parent.exists()


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
        f"{'a' * 64}_si_{'b' * 16}.analyzer"
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

    from spyglass.spikesorting.v2 import visualization as ssviz
    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
        curation_analyzer_cache_path,
        curation_analyzer_with_extensions,
        derived_extension_request_hash,
        open_curation_analyzer,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

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

        # Raw curations share the sort analyzer, so a requested extension is
        # persisted once rather than recomputed on an in-memory copy per view.
        with curation_analyzer_with_extensions(
            root,
            recipe,
            extra_extensions={"spike_locations": {}},
        ) as raw_with_locations:
            assert raw_with_locations.has_extension("spike_locations")
        assert _resolve_curation_analyzer(root, recipe).has_extension(
            "spike_locations"
        )

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

        # Expert access is a disk-backed working copy. Even destructive
        # mutation of that copy cannot remove an extension from the published
        # analyzer, and the copy is gone once the context exits.
        with open_curation_analyzer(merged, recipe) as working:
            working_folder = Path(working.folder)
            assert working_folder != folder
            working.delete_extension("correlograms")
            assert not working.has_extension("correlograms")
        assert not working_folder.exists()
        assert _resolve_curation_analyzer(merged, recipe).has_extension(
            "correlograms"
        )
        assert _folder_content_hash(folder) == published_hash

        # An unusual plot-only extension is computed ONCE into a disk-backed
        # derivative keyed by the exact request, never into the immutable
        # published base cache, and reused on the next identical request.
        request = {"spike_locations": {}}
        derived_folder = curation_analyzer_cache_path(merged, recipe).with_name(
            folder.name.replace(
                folder.suffix,
                f"_ext_{derived_extension_request_hash(request)}{folder.suffix}",
            )
        )
        with curation_analyzer_with_extensions(
            merged,
            recipe,
            extra_extensions=request,
        ) as derivative:
            assert derivative.has_extension("spike_locations")
            assert Path(derivative.folder) == derived_folder
        assert derived_folder.exists()
        derived_hash = _folder_content_hash(derived_folder)
        with curation_analyzer_with_extensions(
            merged,
            recipe,
            extra_extensions=request,
        ) as reused_derivative:
            assert Path(reused_derivative.folder) == derived_folder
        assert _folder_content_hash(derived_folder) == derived_hash
        # Different extension parameters are a different derivative.
        other_request = {"spike_locations": {"method": "center_of_mass"}}
        assert derived_extension_request_hash(
            other_request
        ) != derived_extension_request_hash(request)
        assert not _resolve_curation_analyzer(merged, recipe).has_extension(
            "spike_locations"
        )
        assert _folder_content_hash(folder) == published_hash

        # A PRESENT extension with different parameters is not "there": the
        # base carries correlograms at the standard bin; requesting a finer
        # bin must yield an analyzer whose correlograms use that bin, served
        # from a derivative, while the base keeps its own.
        base_bin = (
            _resolve_curation_analyzer(merged, recipe)
            .get_extension("correlograms")
            .params["bin_ms"]
        )
        assert base_bin != 0.2
        with curation_analyzer_with_extensions(
            merged, recipe, extra_extensions={"correlograms": {"bin_ms": 0.2}}
        ) as fine:
            assert fine.get_extension("correlograms").params["bin_ms"] == 0.2
            assert Path(fine.folder) != folder
        assert (
            _resolve_curation_analyzer(merged, recipe)
            .get_extension("correlograms")
            .params["bin_ms"]
            == base_bin
        )
        assert _folder_content_hash(folder) == published_hash
        # ... and on the RAW (root) curation: an ABSENT extension is persisted
        # into the shared sort analyzer (SI default bin), while a request with
        # DIFFERENT parameters is served from a derivative and never rewrites
        # the shared analyzer's version.
        with curation_analyzer_with_extensions(
            root, recipe, extra_extensions={"correlograms": {}}
        ):
            pass
        raw_base_bin = (
            Sorting()
            .get_analyzer(sorting_key)
            .get_extension("correlograms")
            .params["bin_ms"]
        )
        assert raw_base_bin != 0.2
        with curation_analyzer_with_extensions(
            root, recipe, extra_extensions={"correlograms": {"bin_ms": 0.2}}
        ) as raw_fine:
            assert (
                raw_fine.get_extension("correlograms").params["bin_ms"] == 0.2
            )
            assert Path(raw_fine.folder) != Path(
                Sorting().get_analyzer(sorting_key).folder
            )
        assert (
            Sorting()
            .get_analyzer(sorting_key)
            .get_extension("correlograms")
            .params["bin_ms"]
            == raw_base_bin
        )

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

        # The unified orphan collector recognizes this live curation reference
        # and its derivative (keyed by the same live generation).
        live_report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(folder) not in live_report["disk_side"]
        assert str(derived_folder) not in live_report["disk_side"]

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
    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

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


def test_extension_params_match_semantics():
    """Partial requests constrain only their keys; mismatches are not present."""
    from spyglass.spikesorting.v2._curation_analyzer import (
        extension_params_match,
    )

    class _Ext:
        def __init__(self, params):
            self.params = params

    class _Analyzer:
        def __init__(self, exts):
            self._exts = exts

        def has_extension(self, name):
            return name in self._exts

        def get_extension(self, name):
            return _Ext(self._exts[name])

    analyzer = _Analyzer({"correlograms": {"bin_ms": 1.0, "window_ms": 50.0}})
    assert extension_params_match(analyzer, "correlograms", {})
    assert extension_params_match(analyzer, "correlograms", {"bin_ms": 1})
    assert not extension_params_match(analyzer, "correlograms", {"bin_ms": 0.2})
    assert not extension_params_match(
        analyzer, "correlograms", {"window_ms": 50.0, "bin_ms": 0.5}
    )
    assert not extension_params_match(analyzer, "spike_locations", {})
