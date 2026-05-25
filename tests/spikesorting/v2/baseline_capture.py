"""Capture a v1 spike-sorting baseline on a real lab dataset.

Runs the v1 pipeline -- recording materialization, artifact detection,
``clusterless_thresholder`` sort, ``CurationV1`` -- against a real NWB file and
writes small artifacts that v1 ↔ v2 parity tests will compare the modern (v2)
pipeline output against.

Must be invoked under the **v1 (SpikeInterface 0.99) runtime environment**, not
the v2 SI 0.104 environment, because v1's sort runs through
``si.extract_waveforms`` and related pre-0.101 APIs.

The script always targets the isolated Docker test database via
``bootstrap_v2_test_environment``; it writes only under a temporary
``SPYGLASS_BASE_DIR`` and on a ``pytests``-prefixed schema. Any production
metadata lookup (for example to locate the source NWB) is read-only and
requires ``SPYGLASS_ALLOW_PRODUCTION_SMOKE=1``.

Skipped with a clear message if no NWB path is supplied via ``--nwb-file`` or
the ``SPIKESORTING_V2_REAL_NWB_PATH`` environment variable.

Usage
-----
    SPIKESORTING_V2_REAL_NWB_PATH=/path/to/session.nwb \\
    python tests/spikesorting/v2/baseline_capture.py \\
        --team-name "Loren Frank Lab" \\
        --sort-group-id 0 \\
        --interval-list-name 01_s1 \\
        --output-dir tests/spikesorting/v2/baselines \\
        --base-dir tests/_data/spikesorting_v2 \\
        --database-prefix pytests
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

# The bootstrap must run before importing Spyglass, so resolve and import the
# standalone test-environment helper by path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _ingest_helpers import copy_and_insert_nwb  # noqa: E402
from test_env import bootstrap_v2_test_environment  # noqa: E402

_DEFAULT_BASE_DIR = "tests/_data/spikesorting_v2"
_DEFAULT_PREFIX = "pytests"
_DEFAULT_OUTPUT_DIR = "tests/spikesorting/v2/baselines"
_NWB_PATH_ENV = "SPIKESORTING_V2_REAL_NWB_PATH"


def _resolve_nwb(args: argparse.Namespace) -> Path | None:
    """Return the resolved NWB path, or None if no source was supplied."""
    candidate = args.nwb_file or os.environ.get(_NWB_PATH_ENV)
    if not candidate:
        return None
    return Path(candidate).expanduser()


def _ensure_sort_group(nwb_file_name: str, sort_group_id: int) -> None:
    """Make sure the sort-group layout exists for this NWB.

    ``SortGroup.set_group_by_shank`` is idempotent; calling it on an
    already-grouped session is a no-op.
    """
    from spyglass.spikesorting.v1 import SortGroup

    SortGroup.set_group_by_shank(nwb_file_name=nwb_file_name)
    if not (
        SortGroup
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    ):
        raise RuntimeError(
            f"sort_group_id={sort_group_id} is not present after "
            f"set_group_by_shank({nwb_file_name!r}). Pass a sort group that "
            "exists for this NWB."
        )


def _populate_recording(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
) -> dict:
    """Populate ``SpikeSortingRecording`` and return the recording PK."""
    from spyglass.spikesorting.v1 import (
        SpikeSortingPreprocessingParameters,
        SpikeSortingRecording,
        SpikeSortingRecordingSelection,
    )

    SpikeSortingPreprocessingParameters().insert_default()
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "preproc_param_name": "default",
        "interval_list_name": interval_list_name,
        "team_name": team_name,
    }
    SpikeSortingRecordingSelection.insert_selection(selection_key)
    ssr_pk = (
        (SpikeSortingRecordingSelection & selection_key).proj().fetch1("KEY")
    )
    SpikeSortingRecording.populate(ssr_pk)
    if not SpikeSortingRecording() & ssr_pk:
        raise RuntimeError(
            f"SpikeSortingRecording failed to populate for {ssr_pk}"
        )
    return ssr_pk


def _populate_artifact_detection(recording_id: str) -> dict:
    """Populate ``ArtifactDetection`` with default params and return its PK."""
    from spyglass.spikesorting.v1 import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters().insert_default()
    selection_key = {
        "recording_id": recording_id,
        "artifact_param_name": "default",
    }
    ArtifactDetectionSelection.insert_selection(selection_key)
    ArtifactDetection.populate(selection_key)
    art_pk = (
        (ArtifactDetectionSelection & selection_key).proj().fetch1("KEY")
    )
    return art_pk


def _populate_sorting(
    nwb_file_name: str,
    recording_id: str,
    artifact_id: str,
) -> dict:
    """Populate ``SpikeSorting`` with ``clusterless_thresholder``.

    Returns the populated sorting PK.
    """
    from spyglass.spikesorting.v1 import (
        SpikeSorterParameters,
        SpikeSorting,
        SpikeSortingSelection,
    )

    SpikeSorterParameters().insert_default()
    selection_key = {
        "nwb_file_name": nwb_file_name,
        "recording_id": recording_id,
        "interval_list_name": str(artifact_id),
        "sorter": "clusterless_thresholder",
        "sorter_param_name": "default_clusterless",
    }
    SpikeSortingSelection.insert_selection(selection_key)
    sort_pk = (SpikeSortingSelection & selection_key).proj().fetch1("KEY")
    SpikeSorting.populate(sort_pk)
    if not SpikeSorting() & sort_pk:
        raise RuntimeError(f"SpikeSorting failed to populate for {sort_pk}")
    return sort_pk


def _insert_curation(sorting_id: str) -> dict:
    """Insert the initial ``CurationV1`` row and return its PK."""
    from spyglass.spikesorting.v1 import CurationV1

    CurationV1.insert_curation(
        sorting_id=sorting_id,
        description="v1 baseline for modern-spike-sorting parity",
        parent_curation_id=-1,
    )
    return (
        CurationV1
        & {"sorting_id": sorting_id, "parent_curation_id": -1}
    ).fetch1("KEY")


def _read_spike_times(curation_key: dict) -> tuple[dict, float]:
    """Return per-unit spike times in seconds and the sampling rate.

    Parameters
    ----------
    curation_key : dict
        Primary key into ``CurationV1``.

    Returns
    -------
    spike_times : dict
        Mapping ``{unit_id: numpy.ndarray of spike times in seconds}``.
    sampling_frequency : float
    """
    import numpy as np

    from spyglass.spikesorting.v1 import CurationV1, SpikeSortingRecording

    sorting = CurationV1.get_sorting(curation_key)
    recording = SpikeSortingRecording.get_recording(curation_key)
    sampling_frequency = float(recording.get_sampling_frequency())
    spike_times = {
        int(unit_id): np.asarray(sorting.get_unit_spike_train(unit_id))
        / sampling_frequency
        for unit_id in sorting.get_unit_ids()
    }
    return spike_times, sampling_frequency


def _write_artifacts(
    output_dir: Path,
    spike_times: dict,
    metadata: dict,
) -> tuple[Path, Path]:
    """Write the two baseline artifacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spikes_path = output_dir / "baseline_v1_spike_times.pkl"
    meta_path = output_dir / "baseline_v1_recording_meta.json"
    with open(spikes_path, "wb") as handle:
        pickle.dump(spike_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return spikes_path, meta_path


def run_baseline_capture(
    nwb_source: Path,
    *,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Run the full v1 pipeline on ``nwb_source`` and write baseline artifacts.

    Parameters
    ----------
    nwb_source : pathlib.Path
        Path to a real lab NWB file (copied into the isolated raw directory).
    sort_group_id : int
        Sort group to capture (typically 0 for a tetrode-style first group).
    interval_list_name : str
        Interval list name on the session (typically the first epoch, e.g.
        ``"01_s1"``).
    team_name : str
        Lab team name; must exist in ``LabTeam`` before invocation.
    output_dir : pathlib.Path
        Destination for the small committed baseline artifacts.

    Returns
    -------
    spikes_path, meta_path : pathlib.Path
        The two written artifacts.
    """
    nwb_file_name = copy_and_insert_nwb(nwb_source)
    _ensure_sort_group(nwb_file_name, sort_group_id)
    recording_pk = _populate_recording(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
    )
    artifact_pk = _populate_artifact_detection(recording_pk["recording_id"])
    sorting_pk = _populate_sorting(
        nwb_file_name=nwb_file_name,
        recording_id=recording_pk["recording_id"],
        artifact_id=artifact_pk["artifact_id"],
    )
    curation_pk = _insert_curation(sorting_pk["sorting_id"])

    spike_times, sampling_frequency = _read_spike_times(curation_pk)
    duration_s = max(
        (times[-1] for times in spike_times.values() if len(times)),
        default=0.0,
    )

    metadata = {
        "nwb_file_name": nwb_file_name,
        "nwb_source_path": str(nwb_source),
        "sort_group_id": int(sort_group_id),
        "interval_list_name": interval_list_name,
        "team_name": team_name,
        "recording_id": str(recording_pk["recording_id"]),
        "artifact_id": str(artifact_pk["artifact_id"]),
        "sorting_id": str(sorting_pk["sorting_id"]),
        "curation_id": int(curation_pk["curation_id"]),
        "n_units": len(spike_times),
        "sampling_frequency_hz": sampling_frequency,
        "approx_last_spike_s": float(duration_s),
        "sorter": "clusterless_thresholder",
        "sorter_param_name": "default_clusterless",
        "preproc_param_name": "default",
        "artifact_param_name": "default",
        "spikeinterface_version": _pkg_version("spikeinterface"),
        "spyglass_version": _pkg_version("spyglass-neuro"),
    }
    spikes_path, meta_path = _write_artifacts(
        output_dir, spike_times, metadata
    )

    print(f"baseline spikes -> {spikes_path}")
    print(f"baseline meta   -> {meta_path}")
    print(f"  nwb_file_name  : {nwb_file_name}")
    print(f"  recording_id   : {metadata['recording_id']}")
    print(f"  artifact_id    : {metadata['artifact_id']}")
    print(f"  sorting_id     : {metadata['sorting_id']}")
    print(f"  curation_id    : {metadata['curation_id']}")
    print(f"  n_units        : {metadata['n_units']}")
    return spikes_path, meta_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Capture a v1 spike-sorting baseline on a real lab dataset."
        )
    )
    parser.add_argument(
        "--nwb-file",
        default=None,
        help="Path to a real NWB file (falls back to "
        f"{_NWB_PATH_ENV}).",
    )
    parser.add_argument(
        "--sort-group-id",
        type=int,
        default=0,
        help="Sort group to capture (default: 0).",
    )
    parser.add_argument(
        "--interval-list-name",
        default="01_s1",
        help="Interval list name on the session (default: 01_s1).",
    )
    parser.add_argument(
        "--team-name",
        required=True,
        help="Lab team name (must already exist in LabTeam).",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Where to write baseline artifacts.",
    )
    parser.add_argument(
        "--base-dir",
        default=_DEFAULT_BASE_DIR,
        help="Spyglass base directory (must be under tests/_data/ or a "
        "temporary directory).",
    )
    parser.add_argument(
        "--database-prefix",
        default=_DEFAULT_PREFIX,
        help="DataJoint schema prefix (default: pytests).",
    )
    parser.add_argument(
        "--dj-config",
        default=None,
        help="Optional path to an existing DataJoint config file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: bootstrap the test environment and run the v1 baseline."""
    args = _parse_args(argv)
    nwb_source = _resolve_nwb(args)
    if nwb_source is None:
        print(
            f"No NWB path supplied (--nwb-file or {_NWB_PATH_ENV}); "
            "skipping v1 baseline capture.",
            file=sys.stderr,
        )
        return 0
    if not nwb_source.exists():
        print(
            f"NWB path {nwb_source} does not exist.",
            file=sys.stderr,
        )
        return 2

    # Read-only production lookups (for example to locate the source NWB)
    # require an explicit gate; writes always go to the test prefix.
    allow_production_smoke = (
        os.environ.get("SPYGLASS_ALLOW_PRODUCTION_SMOKE") == "1"
    )
    bootstrap_v2_test_environment(
        base_dir=args.base_dir,
        database_prefix=args.database_prefix,
        dj_config=args.dj_config,
        allow_production_smoke=allow_production_smoke,
    )

    run_baseline_capture(
        nwb_source=nwb_source,
        sort_group_id=args.sort_group_id,
        interval_list_name=args.interval_list_name,
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
