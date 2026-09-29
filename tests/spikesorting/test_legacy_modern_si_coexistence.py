"""v0/v1 spike-sorting stays importable + queryable under modern SpikeInterface.

The default install pins SpikeInterface 0.104 (the v2 baseline). Active v0/v1
*compute* (``make`` / waveform extraction) is gated behind
``_require_legacy_si_environment`` (raises under SI >= 0.101), but **read
access** must keep working: a user with existing v0/v1 sorts has to be able to
import the table modules and query / fetch their rows without a legacy
SpikeInterface install. This pins that contract so a future change that adds a
top-level removed-API import (breaking import, hence table access) fails here
instead of silently locking users out of their data.

Runs in the main ``run-tests`` job (SI 0.104); the legacy job tests the
compute paths under SI 0.99. The genuinely-removed SI APIs
(``WaveformExtractor`` / ``extract_waveforms`` / ``load_waveforms``) are
lazy-imported inside the guarded compute functions, so module import does not
touch them.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version

# The SI-bearing v0/v1 table modules (those with top-level SpikeInterface
# imports and ``_require_legacy_si_environment`` guards on their compute paths).
# These are the modules whose import would break first if a removed SI API were
# imported at module scope. The figurl / sortingview modules are intentionally
# omitted: they gate on optional sharing extras, not on the SI version.
_LEGACY_TABLE_MODULES = [
    "spyglass.spikesorting.v1.artifact",
    "spyglass.spikesorting.v1.curation",
    "spyglass.spikesorting.v1.metric_curation",
    "spyglass.spikesorting.v1.burst_curation",
    "spyglass.spikesorting.v1.recording",
    "spyglass.spikesorting.v1.sorting",
    "spyglass.spikesorting.v1.recompute",
    "spyglass.spikesorting.v0.spikesorting_artifact",
    "spyglass.spikesorting.v0.spikesorting_curation",
    "spyglass.spikesorting.v0.spikesorting_burst",
    "spyglass.spikesorting.v0.spikesorting_recording",
    "spyglass.spikesorting.v0.spikesorting_sorting",
    "spyglass.spikesorting.v0.spikesorting_recompute",
]


def test_compat_loader_dispatches_to_available_si_api(monkeypatch):
    """The loader wrapper selects the API exposed by the installed SI."""
    import spikeinterface as si

    from spyglass.spikesorting import _si_compat

    source = {"serialized": "extractor"}
    loaded = object()
    calls = []

    def _load(value):
        calls.append(value)
        return loaded

    loader_name = (
        "load_extractor"
        if callable(getattr(si, "load_extractor", None))
        else "load"
    )
    monkeypatch.setattr(si, loader_name, _load)

    assert _si_compat.load_extractor(source) is loaded
    assert calls == [source]


def test_compat_numpy_sorting_constructor_preserves_unit_ids():
    """The constructor wrapper works across SI names and keeps empty units."""
    from spyglass.spikesorting._si_compat import (
        numpy_sorting_from_samples_and_labels,
    )

    samples = np.asarray([5, 10, 15], dtype=np.int64)
    labels = np.asarray([2, 2, 10], dtype=np.int64)
    sorting = numpy_sorting_from_samples_and_labels(
        samples,
        labels,
        sampling_frequency=30_000.0,
        unit_ids=[2, 10, 99],
    )

    assert list(sorting.get_unit_ids()) == [2, 10, 99]
    np.testing.assert_array_equal(
        sorting.get_unit_spike_train(unit_id=2), [5, 10]
    )
    np.testing.assert_array_equal(
        sorting.get_unit_spike_train(unit_id=10), [15]
    )
    np.testing.assert_array_equal(sorting.get_unit_spike_train(unit_id=99), [])


_SI099_DIR = Path(__file__).parent / "fixtures" / "si099"
_PLANT_NAME = "si099_read_path"
_PLANT_SORT_GROUP_ID = 9099


@pytest.fixture(scope="module")
def v0_rows_over_si099_folders(
    dj_conn, mini_copy_name, team_name, tmp_path_factory
):
    """Plant v0 recording/sorting/curation rows over SI 0.99-written folders.

    The rows point at a temporary copy of ``fixtures/si099`` (see its README),
    so the loaders read real SpikeInterface 0.99 folders and cannot write into
    the repository. Every planted row is removed afterwards.
    """
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("Modern-SI read-path regression test")

    from spyglass.spikesorting.v0 import spikesorting_artifact as art
    from spyglass.spikesorting.v0 import spikesorting_curation as cur
    from spyglass.spikesorting.v0 import spikesorting_recording as rec
    from spyglass.spikesorting.v0 import spikesorting_sorting as srt

    folders = tmp_path_factory.mktemp("v0_si099") / "si099"
    shutil.copytree(_SI099_DIR, folders)
    with np.load(folders / "reference.npz") as data:
        reference = {key: data[key] for key in data.files}

    session = {"nwb_file_name": mini_copy_name}
    rec_key = {
        **session,
        "sort_group_id": _PLANT_SORT_GROUP_ID,
        "sort_interval_name": _PLANT_NAME,
        "preproc_params_name": _PLANT_NAME,
        "team_name": team_name,
    }
    artifact_key = {**rec_key, "artifact_params_name": _PLANT_NAME}
    sorter_key = {"sorter": _PLANT_NAME, "sorter_params_name": _PLANT_NAME}
    sorting_key = {
        **rec_key,
        **sorter_key,
        "artifact_removed_interval_list_name": _PLANT_NAME,
    }
    curation_key = {**sorting_key, "curation_id": 0}
    times = np.load(folders / "recording" / "times_cached_seg0.npy")
    sort_interval = np.asarray([times[0], times[-1]])

    # Insertion order follows the foreign keys; rows are removed in reverse.
    planted = [
        (rec.SortGroup(), {**session, "sort_group_id": _PLANT_SORT_GROUP_ID}),
        (
            rec.SortInterval(),
            {
                **session,
                "sort_interval_name": _PLANT_NAME,
                "sort_interval": sort_interval,
            },
        ),
        (
            rec.SpikeSortingPreprocessingParameters(),
            {"preproc_params_name": _PLANT_NAME, "preproc_params": {}},
        ),
        (
            rec.SpikeSortingRecordingSelection(),
            {**rec_key, "interval_list_name": "01_s1"},
        ),
        (
            rec.SpikeSortingRecording(),
            {
                **rec_key,
                "recording_path": str(folders / "recording"),
                "sort_interval_list_name": "01_s1",
            },
        ),
        (
            art.ArtifactDetectionParameters(),
            {"artifact_params_name": _PLANT_NAME, "artifact_params": {}},
        ),
        (art.ArtifactDetectionSelection(), artifact_key),
        (
            art.ArtifactRemovedIntervalList(),
            {
                **artifact_key,
                "artifact_removed_interval_list_name": _PLANT_NAME,
                "artifact_removed_valid_times": sort_interval[None, :],
                "artifact_times": np.empty((0, 2)),
            },
        ),
        (srt.SpikeSorterParameters(), {**sorter_key, "sorter_params": {}}),
        (srt.SpikeSortingSelection(), sorting_key),
        (
            srt.SpikeSorting(),
            {
                **sorting_key,
                "sorting_path": str(folders / "sorting"),
                "time_of_sort": 0,
            },
        ),
        (
            cur.Curation(),
            {
                **curation_key,
                "curation_labels": {},
                "merge_groups": [],
                "quality_metrics": {},
                "time_of_creation": 0,
            },
        ),
    ]
    inserted = []
    try:
        for table, row in planted:
            table.insert1(row, allow_direct_insert=True)
            inserted.append((table, {k: row[k] for k in table.primary_key}))
        yield {
            "recording_key": rec_key,
            "curation_key": curation_key,
            "reference": reference,
        }
    finally:
        for table, restriction in reversed(inserted):
            (table & restriction).delete_quick()


def test_v0_read_paths_under_modern_si(v0_rows_over_si099_folders):
    """v0 recording and curated-sorting reads load SI 0.99-written folders.

    ``SpikeSortingRecording.load_recording`` and
    ``Curation.get_curated_sorting`` run unpatched on planted v0 rows whose
    paths point at folders SpikeInterface 0.99 wrote; the loaded traces, channel
    ids, unit ids and spike trains must equal what 0.99 read back.

    v1 reads analysis NWB files, not extractor folders; its only folder read is
    ``_si_compat.load_waveforms`` (via ``MetricCuration.get_waveforms``), which
    ``test_si_compat_cross_generation.py`` covers.
    """
    from spyglass.spikesorting.v0.spikesorting_curation import Curation
    from spyglass.spikesorting.v0.spikesorting_recording import (
        SpikeSortingRecording,
    )

    planted = v0_rows_over_si099_folders
    reference = planted["reference"]

    recording = SpikeSortingRecording().load_recording(planted["recording_key"])
    np.testing.assert_array_equal(
        recording.get_channel_ids(), reference["channel_ids"]
    )
    traces = recording.get_traces(return_in_uV=False)
    assert traces.dtype == reference["traces"].dtype
    np.testing.assert_array_equal(traces, reference["traces"])

    sorting = Curation().get_curated_sorting(planted["curation_key"])
    unit_ids = list(sorting.get_unit_ids())
    assert unit_ids == list(reference["unit_ids"])
    for unit_id in unit_ids:
        np.testing.assert_array_equal(
            sorting.get_unit_spike_train(unit_id=unit_id),
            reference[f"spike_train_{unit_id}"],
        )


@pytest.mark.parametrize("module_name", _LEGACY_TABLE_MODULES)
def test_legacy_table_module_imports_under_modern_si(dj_conn, module_name):
    """Each SI-bearing v0/v1 table module imports under the modern SI pin.

    A top-level import of an API removed in SI 0.101+ would raise here -- which
    would also block read access to that table's rows. ``dj_conn`` is required
    because importing a ``@schema`` module declares its tables.
    """
    try:
        importlib.import_module(module_name)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - report the offending module clearly
        import spikeinterface

        pytest.fail(
            f"{module_name} failed to import under SpikeInterface "
            f"{spikeinterface.__version__}: {type(exc).__name__}: {exc}. "
            "v0/v1 table modules must stay importable so existing rows remain "
            "queryable; only compute is gated."
        )


def test_legacy_tables_are_queryable_under_modern_si(dj_conn):
    """Existing v0/v1 rows stay readable: a table query does not trip the guard.

    Read access (``len`` / ``fetch``) must work under SI 0.104 -- the legacy
    guard fires only on compute (``make`` / waveform extraction), never on a
    query. An empty test database is fine; the assertion is that the query
    returns rather than raising ``RuntimeError`` from the legacy guard.
    """
    from spyglass.spikesorting.v0.spikesorting_curation import (
        CuratedSpikeSorting,
    )
    from spyglass.spikesorting.v1.sorting import SpikeSorting

    # Declaring + counting (an empty result is expected) exercises schema
    # declaration and a fetch without touching the guarded compute path.
    assert len(SpikeSorting()) >= 0
    assert len(CuratedSpikeSorting()) >= 0
