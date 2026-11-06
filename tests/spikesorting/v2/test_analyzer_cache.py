"""Analyzer-cache path policy.

The folder location depends only on ``sorting_id`` and one configured root
(``spikesorting_v2_analyzer_dir``, else ``temp_dir``). DB-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import spikeinterface.full as si

from spyglass.spikesorting.v2._storage.analyzer_cache import (
    analyzer_cache_root,
    analyzer_path,
    publish_analyzer_atomically,
    remove_analyzer_cache,
)
from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

# Display recipe; build_analyzer requires explicit waveform params.
_DISPLAY_PARAMS = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 20000,
    "whiten": False,
    "sparsity": {},
    "purpose": "display",
    "schema_version": 1,
}


def test_analyzer_cache_root_honors_config(restore_custom_config):
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    assert analyzer_cache_root() == Path("/tmp/v2_custom_an")
    assert analyzer_path("abc", "rec") == Path(
        "/tmp/v2_custom_an/abc__rec.analyzer"
    )


def test_analyzer_cache_root_falls_back_to_temp_dir(restore_custom_config):
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"].pop("spikesorting_v2_analyzer_dir", None)
    expected = Path(temp_dir) / "spikesorting_v2" / "analyzers"
    assert analyzer_cache_root() == expected
    assert analyzer_path("s1", "rec") == expected / "s1__rec.analyzer"


def test_analyzer_path_includes_params_name(restore_custom_config):
    """The cache folder is keyed by ``(sorting_id, waveform_params_name)``.

    The whitened metric analyzer never overwrites the display one.
    """
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    root = Path("/tmp/v2_custom_an")
    display = analyzer_path("sid1", "franklab_hippocampus_actual_waveforms")
    metric = analyzer_path("sid1", "franklab_hippocampus_metric_waveforms")
    assert display != metric
    assert display.parent == root and metric.parent == root
    assert (
        display == root / "sid1__franklab_hippocampus_actual_waveforms.analyzer"
    )


def test_empty_config_value_falls_back(restore_custom_config):
    """An empty-string override is falsy -> fall back to temp_dir."""
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = ""
    assert (
        analyzer_cache_root()
        == Path(temp_dir) / "spikesorting_v2" / "analyzers"
    )


def test_remove_analyzer_cache_removes_all_recipes(
    tmp_path, restore_custom_config
):
    """``remove_analyzer_cache`` globs every ``{sid}__*.analyzer`` recipe folder.

    A different sort's folder is untouched.
    """
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
    sid = "deadbeef"

    # Absent folder: no-op with missing_ok (default), raises otherwise.
    assert remove_analyzer_cache(sid) is False
    with pytest.raises(FileNotFoundError):
        remove_analyzer_cache(sid, missing_ok=False)

    # Two recipe folders for this sort, plus one for a different sort.
    display = analyzer_path(sid, "franklab_hippocampus_actual_waveforms")
    metric = analyzer_path(sid, "franklab_hippocampus_metric_waveforms")
    other = analyzer_path("cafef00d", "franklab_cortex_actual_waveforms")
    for folder in (display, metric, other):
        folder.mkdir(parents=True)
        (folder / "waveforms.bin").write_text("scratch")

    assert remove_analyzer_cache(sid) is True
    assert not display.exists()
    assert not metric.exists()
    assert other.exists()


def test_analyzer_cache_import_pulls_no_db_layer_modules():
    """A cold import of ``_analyzer_cache`` pulls in neither
    ``spyglass.common`` nor any v2 *schema* module (it reads ``dj.config`` /
    ``temp_dir`` only at call time), so importing it opens no DB connection.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent("""
        import sys
        import spyglass.spikesorting.v2._storage.analyzer_cache as c
        assert hasattr(c, "analyzer_path")
        assert hasattr(c, "analyzer_cache_root")
        assert hasattr(c, "remove_analyzer_cache")
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common")
            or m in (
                "spyglass.spikesorting.v2.recording",
                "spyglass.spikesorting.v2.artifact",
                "spyglass.spikesorting.v2.sorting",
            )
        )
        assert not leaked, "cold import pulled in DB-layer modules: " + repr(leaked)
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "analyzer-cache helper must import without pulling in the DB-layer "
        f"(spyglass.common) or v2 schema modules\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


class TestAnalyzerCacheLock:
    """Per-sort analyzer-cache lock: reentrant in one thread, exclusive
    across threads and processes, independent across sorts.
    """

    def test_lock_is_memoized_per_sort(self, tmp_path, restore_custom_config):
        import datajoint as dj

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        # Same instance per sort makes nested acquisition reentrant.
        assert analyzer_cache_lock("sid-A") is analyzer_cache_lock("sid-A")
        assert analyzer_cache_lock("sid-A") is not analyzer_cache_lock("sid-B")

    def test_nested_acquisition_does_not_deadlock(
        self, tmp_path, restore_custom_config
    ):
        """A nested acquisition of the same sort's lock must not block.

        The finite inner timeout turns a non-reentrant lock into a
        ``Timeout`` instead of a hang.
        """
        import datajoint as dj

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        with analyzer_cache_lock("sid-A"):
            with analyzer_cache_lock("sid-A").acquire(timeout=5):
                pass

    def test_concurrent_thread_cannot_acquire_held_lock(
        self, tmp_path, restore_custom_config
    ):
        """A different concurrent job (thread) of the same sort is excluded.

        Reentrancy is same-thread only.
        """
        import threading

        import datajoint as dj
        from filelock import Timeout

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        contender_result = {}

        def contend():
            try:
                with analyzer_cache_lock("sid-A").acquire(timeout=0):
                    contender_result["acquired"] = True
            except Timeout:
                contender_result["acquired"] = False

        with analyzer_cache_lock("sid-A"):
            thread = threading.Thread(target=contend)
            thread.start()
            thread.join()
        assert contender_result["acquired"] is False

    def test_different_sorts_do_not_block(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        with analyzer_cache_lock("sid-A"):
            with analyzer_cache_lock("sid-B").acquire(timeout=0):
                pass

    def test_lock_file_lives_under_cache_root(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj

        from spyglass.spikesorting.v2._storage.analyzer_cache import (
            analyzer_cache_lock,
            analyzer_cache_root,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        lock = analyzer_cache_lock("sid-A")
        assert Path(lock.lock_file).parent == analyzer_cache_root()


class TestPublishAnalyzerAtomically:
    """``publish_analyzer_atomically`` builds into a private temp folder and
    then moves it into the canonical slot.

    ``rename(2)`` needs an empty destination, so an existing folder is moved
    aside first and restored if the swap fails.
    """

    @staticmethod
    def _configure_root(tmp_path):
        import datajoint as dj

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)

    @staticmethod
    def _writer(content):
        def build_into(folder):
            folder.mkdir(parents=True)
            (folder / "data.bin").write_text(content)

        return build_into

    def _staging_is_empty(self, canonical):
        # Hidden ``.{stem}.build|trash-*.analyzer`` siblings.
        return not any(canonical.parent.glob(f".{canonical.stem}.*"))

    def test_absent_canonical_publishes_into_slot(
        self, tmp_path, restore_custom_config
    ):
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        result = publish_analyzer_atomically(canonical, self._writer("v1"))
        assert result == canonical
        assert (canonical / "data.bin").read_text() == "v1"
        assert self._staging_is_empty(canonical)

    def test_publish_holds_per_sort_lock(
        self, tmp_path, monkeypatch, restore_custom_config
    ):
        """The publisher acquires the sort-level lock around build + swap."""
        from spyglass.spikesorting.v2._storage import (
            analyzer_cache as cache_mod,
        )

        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        seen = []

        class _FakeLock:
            held = False

            def __enter__(self):
                self.held = True

            def __exit__(self, exc_type, exc, tb):
                self.held = False

        lock = _FakeLock()

        def _fake_lock(sorting_id):
            seen.append(sorting_id)
            return lock

        def _build_into(folder):
            assert lock.held, "build must run while the analyzer lock is held"
            folder.mkdir(parents=True)
            (folder / "data.bin").write_text("v1")

        monkeypatch.setattr(cache_mod, "analyzer_cache_lock", _fake_lock)
        publish_analyzer_atomically(canonical, _build_into)

        assert seen == ["sidP"]
        assert lock.held is False
        assert (canonical / "data.bin").read_text() == "v1"

    def test_existing_canonical_is_replaced(
        self, tmp_path, restore_custom_config
    ):
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        publish_analyzer_atomically(canonical, self._writer("v1"))
        publish_analyzer_atomically(canonical, self._writer("v2"))
        assert (canonical / "data.bin").read_text() == "v2"
        assert self._staging_is_empty(canonical)

    def test_publish_uses_os_replace_and_clears_temp(
        self, tmp_path, monkeypatch, restore_custom_config
    ):
        """The publish moves via ``os.replace`` and leaves no temp behind."""
        import os as os_mod

        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        calls = []
        real_replace = os_mod.replace

        def _spy(src, dst, *a, **k):
            calls.append((str(src), str(dst)))
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os_mod, "replace", _spy)
        publish_analyzer_atomically(canonical, self._writer("v1"))
        assert any(dst == str(canonical) for _, dst in calls)
        assert self._staging_is_empty(canonical)

    def test_build_failure_leaves_existing_canonical_untouched(
        self, tmp_path, restore_custom_config
    ):
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        publish_analyzer_atomically(canonical, self._writer("v1"))

        def _boom(folder):
            folder.mkdir(parents=True)
            (folder / "partial.bin").write_text("partial")
            raise RuntimeError("build boom")

        with pytest.raises(RuntimeError, match="build boom"):
            publish_analyzer_atomically(canonical, _boom)
        assert (canonical / "data.bin").read_text() == "v1"
        assert self._staging_is_empty(canonical)

    def test_zero_unit_build_leaves_slot_absent(
        self, tmp_path, restore_custom_config
    ):
        """When ``build_into`` creates no folder (zero-unit sort), the canonical
        slot stays absent and no error is raised."""
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")

        def _build_nothing(folder):
            return  # zero-unit short-circuit: no folder written

        result = publish_analyzer_atomically(canonical, _build_nothing)
        assert result == canonical
        assert not canonical.exists()
        assert self._staging_is_empty(canonical)

    def test_failed_swap_rolls_back_canonical(
        self, tmp_path, monkeypatch, restore_custom_config
    ):
        """If the temp -> canonical move fails mid-publish over an existing
        folder, the original is restored (the slot is never left empty)."""
        import os as os_mod

        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        publish_analyzer_atomically(canonical, self._writer("v1"))

        real_replace = os_mod.replace

        def _fail_temp_into_slot(src, dst, *a, **k):
            # Fail the install ("build-" src), not the rollback ("trash-").
            if str(dst) == str(canonical) and "build-" in str(src):
                raise OSError("swap boom")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os_mod, "replace", _fail_temp_into_slot)
        with pytest.raises(OSError, match="swap boom"):
            publish_analyzer_atomically(canonical, self._writer("v2"))
        assert canonical.exists()
        assert (canonical / "data.bin").read_text() == "v1"
        assert self._staging_is_empty(canonical)

    def test_failed_swap_and_failed_rollback_preserve_trash(
        self, tmp_path, monkeypatch, restore_custom_config
    ):
        """If both the install and the rollback fail, the moved-aside original
        (the only surviving copy) stays on disk."""
        import os as os_mod

        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        publish_analyzer_atomically(canonical, self._writer("v1"))

        real_replace = os_mod.replace

        def _fail_into_slot(src, dst, *a, **k):
            # Fail both the install and the rollback.
            if str(dst) == str(canonical):
                raise OSError("swap boom")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os_mod, "replace", _fail_into_slot)
        with pytest.raises(OSError, match="swap boom"):
            publish_analyzer_atomically(canonical, self._writer("v2"))

        trash = list(canonical.parent.glob(f".{canonical.stem}.trash-*"))
        assert trash, "the moved-aside original must be preserved for recovery"
        assert (trash[0] / "data.bin").read_text() == "v1"

    def test_temp_build_is_a_hidden_sibling_of_canonical(
        self, tmp_path, restore_custom_config
    ):
        """The temp build is a hidden sibling of the canonical slot, so the
        relative recording path survives the move and the orphan scan skips
        in-flight builds."""
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        seen = {}

        def _capture(folder):
            seen["temp"] = folder
            folder.mkdir(parents=True)
            (folder / "data.bin").write_text("v1")

        publish_analyzer_atomically(canonical, _capture)
        temp = seen["temp"]
        assert temp.parent == canonical.parent, "temp must be a sibling"
        assert temp.name.startswith("."), "temp must be hidden"
        assert temp.suffix == ".analyzer"
        assert temp.name != canonical.name

    def test_published_real_analyzer_keeps_its_recording(
        self, tmp_path, restore_custom_config
    ):
        """A real SI binary_folder analyzer published via the move keeps its recording.

        SI stores the recording path relative to the analyzer folder, so a
        non-sibling build would detach it.
        """
        self._configure_root(tmp_path)
        recording, sorting = si.generate_ground_truth_recording(
            durations=[2.0], num_channels=8, num_units=3, seed=0
        )
        # Path-backed, like the production NWB-backed recording.
        recording = recording.save(
            folder=tmp_path / "rec_on_disk", overwrite=True
        )
        canonical = analyzer_path("sidR", "rec")

        def build_into(folder):
            analyzer = si.create_sorting_analyzer(
                sorting,
                recording,
                sparse=True,
                format="binary_folder",
                folder=folder,
            )
            analyzer.compute(["random_spikes", "templates", "waveforms"])

        publish_analyzer_atomically(canonical, build_into)
        reloaded = si.load_sorting_analyzer(canonical)
        assert (
            reloaded.has_recording()
        ), "the published analyzer lost its recording after the move"
        # Fails on a detached recording.
        reloaded.compute(["noise_levels", "spike_amplitudes"])
        assert self._staging_is_empty(canonical)


# --------------------------------------------------------------------------- #
# ``build_analyzer`` projects a 3D probe to 2D; ``unit_locations`` and the
# spikeinterface-gui probe view require 2D. DB-free.
# --------------------------------------------------------------------------- #


@pytest.fixture
def recording_3d_and_sorting():
    """A synthetic recording (forced to a 3D probe) and its sorting."""
    recording, sorting = si.generate_ground_truth_recording(
        durations=[10.0], num_channels=8, num_units=4, seed=0
    )
    recording = recording.set_probe(recording.get_probe().to_3d(axes="xy"))
    assert recording.get_probe().ndim == 3
    return recording, sorting


class TestBuildAnalyzerProbeProjection:
    def test_analyzer_probe_is_2d(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=_DISPLAY_PARAMS,
        )

        analyzer = si.load_sorting_analyzer(folder)
        assert analyzer.get_probe().ndim == 2

    def test_unit_locations_computes_after_projection(
        self, recording_3d_and_sorting, tmp_path
    ):
        # A 3D probe raises "could not broadcast ... (3,) into shape (2,)".
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=_DISPLAY_PARAMS,
        )

        analyzer = si.load_sorting_analyzer(folder)
        analyzer.compute("unit_locations")
        unit_locations = analyzer.get_extension("unit_locations").get_data()
        assert unit_locations.shape[0] == sorting.get_num_units()


class TestBuildAnalyzerWaveformParams:
    """``build_analyzer`` reads window / subsample from the resolved params.

    Each region's resolved dict is passed directly; no DB read.
    """

    @staticmethod
    def _build(recording, sorting, folder, waveform_params):
        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-window"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
            waveform_params=waveform_params,
        )
        return si.load_sorting_analyzer(folder)

    def test_hippocampus_window(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        analyzer = self._build(
            recording,
            sorting,
            tmp_path / "hippo.analyzer",
            {
                "ms_before": 0.5,
                "ms_after": 0.5,
                "max_spikes_per_unit": 20000,
                "whiten": False,
                "sparsity": {},
                "purpose": "display",
                "schema_version": 1,
            },
        )
        wf = analyzer.get_extension("waveforms").params
        rs = analyzer.get_extension("random_spikes").params
        assert wf["ms_before"] == 0.5 and wf["ms_after"] == 0.5
        assert rs["max_spikes_per_unit"] == 20000

    def test_cortex_window(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        analyzer = self._build(
            recording,
            sorting,
            tmp_path / "cortex.analyzer",
            {
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": False,
                "sparsity": {},
                "purpose": "display",
                "schema_version": 1,
            },
        )
        wf = analyzer.get_extension("waveforms").params
        rs = analyzer.get_extension("random_spikes").params
        assert wf["ms_before"] == 1.0 and wf["ms_after"] == 2.0
        assert rs["max_spikes_per_unit"] == 20000

    def test_analyzer_folder_required(self, recording_3d_and_sorting):
        """Omitting ``analyzer_folder`` raises (the caller must resolve it)."""
        recording, sorting = recording_3d_and_sorting
        with pytest.raises(ValueError, match="analyzer_folder is required"):
            build_analyzer(
                sorting,
                recording,
                key={"sorting_id": "test-window"},
                sorter_row={"job_kwargs": {}},
                job_kwargs={},
            )

    def test_metric_analyzer_is_whitened(
        self, recording_3d_and_sorting, tmp_path
    ):
        """A metric (``whiten=True``) recipe builds a WHITENED analyzer with
        ``return_in_uV=False``; the display (``whiten=False``) recipe stays
        unwhitened with ``return_in_uV=True``.

        Same window for both, so the template difference is the whitening.
        PC/NN metrics need the decorrelated space.
        """
        import numpy as np

        recording, sorting = recording_3d_and_sorting
        window = {
            "ms_before": 1.0,
            "ms_after": 2.0,
            "max_spikes_per_unit": 20000,
            "sparsity": {},
            "schema_version": 1,
        }
        display = self._build(
            recording,
            sorting,
            tmp_path / "display.analyzer",
            {**window, "whiten": False, "purpose": "display"},
        )
        metric = self._build(
            recording,
            sorting,
            tmp_path / "metric.analyzer",
            {**window, "whiten": True, "purpose": "metric"},
        )
        assert display.return_in_uV is True
        assert metric.return_in_uV is False
        disp_t = display.get_extension("templates").get_data()
        met_t = metric.get_extension("templates").get_data()
        assert disp_t.shape == met_t.shape
        assert not np.allclose(disp_t, met_t)

    def test_metric_analyzer_whitened_under_unequal_gains(self, tmp_path):
        """Under unequal channel gains the whitened analyzer's noise is ~1 per
        channel.

        ``sip.whiten`` keeps the parent's gains, so ``return_in_uV=True``
        would re-apply them and noise would track [1, 5, 0.2, 3].
        """
        import numpy as np

        rec, sort = si.generate_ground_truth_recording(
            durations=[10.0],
            num_channels=4,
            num_units=3,
            seed=0,
            sampling_frequency=30000.0,
        )
        rec.set_channel_gains(np.array([1.0, 5.0, 0.2, 3.0], dtype="float32"))
        rec.set_channel_offsets(np.zeros(4, dtype="float32"))
        analyzer = self._build(
            rec,
            sort,
            tmp_path / "metric.analyzer",
            {
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": True,
                "sparsity": {},
                "purpose": "metric",
                "schema_version": 1,
            },
        )
        noise = analyzer.get_extension("noise_levels").get_data()
        assert np.allclose(noise, 1.0, atol=0.5), noise
        assert noise.max() < 2.0 and noise.min() > 0.4, noise
