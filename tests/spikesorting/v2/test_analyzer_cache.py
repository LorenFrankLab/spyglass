"""Analyzer-cache path policy.

The SortingAnalyzer folder is regeneratable scratch whose location is a pure
function of ``sorting_id`` + one configured root. These tests pin that
policy: the ``spikesorting_v2_analyzer_dir`` config override, the
``temp_dir`` fallback (unchanged default), the ``sorting_id``-keyed path, and
the ``remove_analyzer_cache`` semantics. They do not touch the database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import spikeinterface.full as si

from spyglass.spikesorting.v2._analyzer_cache import (
    analyzer_cache_root,
    analyzer_path,
    publish_analyzer_atomically,
    remove_analyzer_cache,
)
from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

# A display (unwhitened) recipe blob -- build_analyzer requires the resolved
# waveform params (it never picks a default), so the probe-projection tests
# that exercise the build path pass this explicitly.
_DISPLAY_PARAMS = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 20000,
    "whiten": False,
    "purpose": "display",
    "schema_version": 1,
}


def test_analyzer_cache_root_honors_config(restore_custom_config):
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    assert analyzer_cache_root() == Path("/tmp/v2_custom_an")
    assert analyzer_path("abc", "rec") == Path(
        "/tmp/v2_custom_an/abc__rec.zarr"
    )


def test_analyzer_cache_root_falls_back_to_temp_dir(restore_custom_config):
    import datajoint as dj

    from spyglass.settings import temp_dir

    dj.config["custom"].pop("spikesorting_v2_analyzer_dir", None)
    expected = Path(temp_dir) / "spikesorting_v2" / "analyzers"
    assert analyzer_cache_root() == expected
    assert analyzer_path("s1", "rec") == expected / "s1__rec.zarr"


def test_analyzer_path_includes_params_name(restore_custom_config):
    """The cache folder is keyed by ``(sorting_id, waveform_params_name)``.

    Two recipes for one sort resolve to distinct folders (so a whitened metric
    analyzer never overwrites the unwhitened display one), both under the
    configured root.
    """
    import datajoint as dj

    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = "/tmp/v2_custom_an"
    root = Path("/tmp/v2_custom_an")
    display = analyzer_path("sid1", "franklab_hippocampus_actual_waveforms")
    metric = analyzer_path("sid1", "franklab_hippocampus_metric_waveforms")
    assert display != metric
    assert display.parent == root and metric.parent == root
    assert display == root / "sid1__franklab_hippocampus_actual_waveforms.zarr"


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
    """``remove_analyzer_cache`` globs every ``{sid}__*.zarr`` recipe folder.

    A sort has multiple analyzer recipes on disk (display + metric); deleting
    the sort must orphan every one. A different sort's folder is untouched.
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
    # A different sort's folder is left in place.
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
        import spyglass.spikesorting.v2._analyzer_cache as c
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
    """Per-sort lock serializing concurrent analyzer-cache mutation.

    Every canonical-folder reader/writer/deleter (the populate build, the
    self-healing read/rebuild path, ``remove_analyzer_cache``, the recompute
    delete gate, and the ``CurationEvaluation`` raw-sort fast path) holds
    ``analyzer_cache_lock(sorting_id)`` so the on-disk analyzer is mutated by one
    job at a time and a reader never observes the brief move-aside window of an
    atomic publish. The lock is keyed on ``sorting_id`` (shared by every recipe
    folder of that sort), so jobs for DIFFERENT sorts still run concurrently.

    The lock is **process-reentrant**: the read path acquires it while the
    compute path already holds it (the fast path loads inside its own lock), so
    a same-thread nested acquisition must NOT block. This is achieved by
    memoizing one ``FileLock`` instance per lock-file path; filelock's instance
    counter makes the nested ``with`` reentrant while separate processes (the
    ``populate(processes=N)`` case) and separate threads still contend on the
    OS-level lock. DB-free: only ``analyzer_cache_root`` resolves.
    """

    def test_lock_is_memoized_per_sort(self, tmp_path, restore_custom_config):
        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        # Same sort -> the SAME instance (so nested acquisition is reentrant).
        assert analyzer_cache_lock("sid-A") is analyzer_cache_lock("sid-A")
        # Different sorts -> distinct instances (and distinct lock files).
        assert analyzer_cache_lock("sid-A") is not analyzer_cache_lock("sid-B")

    def test_nested_acquisition_does_not_deadlock(
        self, tmp_path, restore_custom_config
    ):
        """A nested acquisition of the same sort's lock must not block.

        The read/load path acquires the lock while the compute path (e.g. the
        ``CurationEvaluation`` fast path) already holds it for the same sort. A
        per-call ``FileLock`` instance would self-deadlock here; the memoized,
        reentrant lock returns instantly. The inner acquire uses a finite
        timeout so a regression (reverting to non-memoized locks) fails as a
        ``Timeout`` instead of hanging the suite.
        """
        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
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

        Reentrancy is same-thread only: a second thread acquiring the same
        sort's lock while it is held must time out, so two parallel curation
        jobs of one sort never mutate the shared analyzer concurrently. (Mirrors
        the cross-process serialization of ``populate(processes=N)``.)
        """
        import threading

        import datajoint as dj
        from filelock import Timeout

        from spyglass.spikesorting.v2._analyzer_cache import (
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

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        # Both held simultaneously without a Timeout -> independent locks.
        with analyzer_cache_lock("sid-A"):
            with analyzer_cache_lock("sid-B").acquire(timeout=0):
                pass

    def test_lock_file_lives_under_cache_root(
        self, tmp_path, restore_custom_config
    ):
        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
            analyzer_cache_root,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        lock = analyzer_cache_lock("sid-A")
        assert Path(lock.lock_file).parent == analyzer_cache_root()

    def test_curation_lock_is_compatibility_alias(
        self, tmp_path, restore_custom_config
    ):
        """``analyzer_curation_lock`` is retained as an alias for the renamed
        ``analyzer_cache_lock`` (its callers predate the rename)."""
        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
            analyzer_curation_lock,
        )

        dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(tmp_path)
        assert analyzer_curation_lock is analyzer_cache_lock
        # The alias resolves to the same memoized instance.
        assert analyzer_curation_lock("sid-A") is analyzer_cache_lock("sid-A")


class TestPublishAnalyzerAtomically:
    """``publish_analyzer_atomically`` builds into a private temp folder, then
    moves it into the canonical slot -- never writing the build straight into
    the canonical path.

    A directory rename is not a clean atomic swap (POSIX ``rename(2)`` needs an
    empty destination), so a rebuild over an existing ``.zarr`` is published by
    moving the existing folder aside, moving the temp in, and removing the
    aside copy; a failed swap restores the original. Callers hold the per-sort
    lock, so the brief move-aside window is reader-safe. These DB-free tests
    drive the publisher with a trivial ``build_into`` that writes a marker file.
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
        # The build/trash siblings are hidden ``.{stem}.build|trash-*.zarr``
        # folders next to the canonical slot; none should survive a publish.
        return not any(canonical.parent.glob(f".{canonical.stem}.*"))

    def test_absent_canonical_publishes_into_slot(
        self, tmp_path, restore_custom_config
    ):
        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        result = publish_analyzer_atomically(canonical, self._writer("v1"))
        assert result == canonical
        assert (canonical / "data.bin").read_text() == "v1"
        # No leftover temp under the hidden staging area after success.
        assert self._staging_is_empty(canonical)

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
        # The temp -> canonical move went through os.replace into the slot.
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
        # The canonical slot still holds v1; the partial temp is cleaned up.
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
            # The publish does: replace(canonical -> trash); replace(temp ->
            # canonical). Fail only the temp -> slot move (src is a "build-"
            # staging dir), NOT the trash -> slot rollback ("trash-"), so the
            # rollback can actually restore the original.
            if str(dst) == str(canonical) and "build-" in str(src):
                raise OSError("swap boom")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os_mod, "replace", _fail_temp_into_slot)
        with pytest.raises(OSError, match="swap boom"):
            publish_analyzer_atomically(canonical, self._writer("v2"))
        # Rolled back: the original v1 is restored, not left missing.
        assert canonical.exists()
        assert (canonical / "data.bin").read_text() == "v1"
        assert self._staging_is_empty(canonical)

    def test_failed_swap_and_failed_rollback_preserve_trash(
        self, tmp_path, monkeypatch, restore_custom_config
    ):
        """If the temp -> canonical install fails AND the trash -> canonical
        rollback ALSO fails, the moved-aside original must be LEFT on disk for
        manual recovery -- it is the only surviving copy, so the publisher must
        not delete it."""
        import os as os_mod

        self._configure_root(tmp_path)
        canonical = analyzer_path("sidP", "rec")
        publish_analyzer_atomically(canonical, self._writer("v1"))

        real_replace = os_mod.replace

        def _fail_into_slot(src, dst, *a, **k):
            # Fail BOTH moves whose destination is the canonical slot: the
            # temp -> canonical install and the trash -> canonical rollback.
            if str(dst) == str(canonical):
                raise OSError("swap boom")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(os_mod, "replace", _fail_into_slot)
        with pytest.raises(OSError, match="swap boom"):
            publish_analyzer_atomically(canonical, self._writer("v2"))

        # The original v1 survives in a trash sibling (canonical was moved aside
        # and could not be restored); it must NOT have been deleted.
        trash = list(canonical.parent.glob(f".{canonical.stem}.trash-*"))
        assert trash, "the moved-aside original must be preserved for recovery"
        assert (trash[0] / "data.bin").read_text() == "v1"

    def test_temp_build_is_a_hidden_sibling_of_canonical(
        self, tmp_path, restore_custom_config
    ):
        """The temp build folder is a HIDDEN ``.zarr`` SIBLING of the canonical
        slot (same parent directory), so (a) the move preserves the analyzer's
        recording path, relative to the analyzer folder, and (b) the leading
        ``.`` keeps the orphan scan from mistaking an in-flight build for a
        stray ``{sid}__*.zarr`` analyzer."""
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
        assert temp.suffix == ".zarr"
        assert temp.name != canonical.name

    def test_published_real_analyzer_keeps_its_recording(
        self, tmp_path, restore_custom_config
    ):
        """A real SI zarr analyzer published via the move keeps its recording.

        SI stores the recording reference as a path RELATIVE to the analyzer
        folder, so the temp build folder MUST be a sibling of the canonical slot
        -- otherwise the move silently detaches the recording and a later
        recording-dependent extension (``spike_amplitudes``) cannot compute.
        This is the regression guard for that detachment (a marker-file
        ``build_into`` cannot catch it).
        """
        self._configure_root(tmp_path)
        recording, sorting = si.generate_ground_truth_recording(
            durations=[2.0], num_channels=8, num_units=3, seed=0
        )
        # A path-backed recording (like the production NWB-backed one): its
        # provenance is a path stored relative to the analyzer folder.
        recording = recording.save(
            folder=tmp_path / "rec_on_disk", overwrite=True
        )
        canonical = analyzer_path("sidR", "rec")

        def build_into(folder):
            analyzer = si.create_sorting_analyzer(
                sorting, recording, sparse=True, format="zarr", folder=folder
            )
            analyzer.compute(["random_spikes", "templates", "waveforms"])

        publish_analyzer_atomically(canonical, build_into)
        reloaded = si.load_sorting_analyzer(canonical)
        assert reloaded.has_recording(), (
            "the published analyzer lost its recording after the move"
        )
        # The recording-dependent extension the detachment bug broke.
        reloaded.compute(["noise_levels", "spike_amplitudes"])
        assert self._staging_is_empty(canonical)


# --------------------------------------------------------------------------- #
# ``build_analyzer`` projects a 3D probe to 2D before building the analyzer.
#
# Spyglass stores electrode geometry in 3D (z is typically 0), but the
# SortingAnalyzer ``unit_locations`` extension and the spikeinterface-gui probe
# view assume 2D contact positions and raise
# ``could not broadcast input array from shape (3,) into shape (2,)`` on a 3D
# probe. ``build_analyzer`` projects the probe to 2D so both work; this is the
# regression guard. It is a fast, DB-free unit test: it drives ``build_analyzer``
# directly with a synthetic recording carrying a 3D probe, passing ``sorter_row``
# / ``job_kwargs`` / ``analyzer_folder`` so no database read happens.
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
        folder = tmp_path / "sort.zarr"

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
        # Without the 2D projection this raises ValueError:
        # "could not broadcast input array from shape (3,) into shape (2,)".
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.zarr"

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

    The window + subsample are no longer hardcoded -- they come from the
    resolved ``AnalyzerWaveformParameters`` blob the caller threads in. These
    drive ``build_analyzer`` directly with each region's resolved dict (no name
    lookup, no DB read), then read the window / cap back off the built
    analyzer's ``waveforms`` / ``random_spikes`` extensions.
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
            tmp_path / "hippo.zarr",
            {
                "ms_before": 0.5,
                "ms_after": 0.5,
                "max_spikes_per_unit": 20000,
                "whiten": False,
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
            tmp_path / "cortex.zarr",
            {
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": False,
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

        Same window for both, differing only in ``whiten`` -- so the template
        difference is the whitening, not the window. The display recipe carries
        real µV amplitudes (``return_in_uV=True``); the metric recipe must read
        back the decorrelated space (``return_in_uV=False``) for PC/NN metrics.
        """
        import numpy as np

        recording, sorting = recording_3d_and_sorting
        window = {
            "ms_before": 1.0,
            "ms_after": 2.0,
            "max_spikes_per_unit": 20000,
            "schema_version": 1,
        }
        display = self._build(
            recording,
            sorting,
            tmp_path / "display.zarr",
            {**window, "whiten": False, "purpose": "display"},
        )
        metric = self._build(
            recording,
            sorting,
            tmp_path / "metric.zarr",
            {**window, "whiten": True, "purpose": "metric"},
        )
        assert display.return_in_uV is True
        assert metric.return_in_uV is False
        disp_t = display.get_extension("templates").get_data()
        met_t = metric.get_extension("templates").get_data()
        assert disp_t.shape == met_t.shape
        # Whitening changes the templates; same window, so this is the whiten.
        assert not np.allclose(disp_t, met_t)

    def test_metric_analyzer_whitened_under_unequal_gains(self, tmp_path):
        """Under NON-uniform channel gains the whitened analyzer's per-channel
        noise is ~1 (unit variance) -- it would TRACK the gains if
        ``return_in_uV=True`` re-applied them.

        ``sip.whiten`` preserves the parent's per-channel gains, so a
        ``return_in_uV=True`` readback multiplies the unit-variance whitened
        traces back by those gains and un-normalizes the space. ``build_analyzer``
        builds the metric recipe with ``return_in_uV=False``; here the gains are
        [1, 5, 0.2, 3], so a gains-reapplied build would give noise ~[1, 5, 0.2,
        3] (max ~5, min ~0.15), which the bounds below reject.
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
            tmp_path / "metric.zarr",
            {
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": True,
                "purpose": "metric",
                "schema_version": 1,
            },
        )
        noise = analyzer.get_extension("noise_levels").get_data()
        # Whitened, gains NOT re-applied -> ~unit variance on every channel.
        assert np.allclose(noise, 1.0, atol=0.5), noise
        # Gains-reapplied (return_in_uV=True) would track [1, 5, 0.2, 3];
        # assert the 5x / 0.2x channels are NOT near their gains.
        assert noise.max() < 2.0 and noise.min() > 0.4, noise
