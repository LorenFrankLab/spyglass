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
import hashlib
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


def _ensure_lab_team(team_name: str) -> None:
    """Insert ``LabTeam(team_name)`` if missing under the active schema.

    Each ``--database-prefix`` runs against a freshly-bootstrapped
    DataJoint schema, so ``LabTeam`` is empty unless we insert here.
    ``SpikeSortingRecordingSelection`` FKs ``team_name`` to
    ``common_lab.lab_team(team_name)``, so the insert MUST happen
    before ``_populate_recording``.
    """
    from spyglass.common.common_lab import LabTeam

    LabTeam.insert1(
        {"team_name": team_name, "team_description": "v1 baseline capture"},
        skip_duplicates=True,
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


def _populate_artifact_detection(
    recording_id: str,
    artifact_param_name: str = "default",
) -> dict:
    """Populate ``ArtifactDetection`` and return its PK.

    Parameters
    ----------
    artifact_param_name : str, optional
        Existing ``ArtifactDetectionParameters`` row name. Defaults to
        ``"default"`` (v1: 3000 µV threshold). For the polymer parity
        matrix pass ``"none"`` so the artifact stage is a typed
        no-detect on both pipelines; v1's ``"default"`` and v2's
        ``"default"`` rows ship different ``amplitude_thresh_uV`` (3000
        vs 500 after the v1 unit-conversion bug fix), which would
        otherwise FAIL the ``canonical_artifact_params`` fingerprint
        check.
    """
    from spyglass.spikesorting.v1 import (
        ArtifactDetection,
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )

    ArtifactDetectionParameters().insert_default()
    selection_key = {
        "recording_id": recording_id,
        "artifact_param_name": artifact_param_name,
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
    sorter_param_name: str = "default_clusterless",
) -> dict:
    """Populate ``SpikeSorting`` with ``clusterless_thresholder``.

    Returns the populated sorting PK.

    Parameters
    ----------
    sorter_param_name : str, optional
        Name of an existing ``SpikeSorterParameters`` row to use.
        Defaults to ``"default_clusterless"`` (100 uV threshold, the
        Frank-lab production value). For synthetic fixtures whose
        template amplitudes never reach 100 uV (e.g. the MEArec
        smoke fixture), the caller must pre-insert a lower-
        threshold row and pass its name here.
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
        "sorter_param_name": sorter_param_name,
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

    from spyglass.spikesorting.v1 import (
        CurationV1,
        SpikeSorting,
        SpikeSortingRecording,
    )

    sorting = CurationV1.get_sorting(curation_key)
    # ``SpikeSortingRecording.get_recording(curation_key)`` calls
    # ``(cls & curation_key).fetch1(...)``. DataJoint's ``&`` does NOT
    # auto-restrict via FK chain across distant tables -- the curation
    # key has ``sorting_id``/``curation_id`` but no ``recording_id``,
    # and ``SpikeSortingRecording`` has no ``sorting_id`` column, so
    # the restriction is vacuous and fetch1 raises whenever the shared
    # ``spikesorting_v1_recording`` schema accumulates >1 row (which
    # happens immediately on the second capture against the shared
    # schema, see ``capture_polymer_clusterless.sh`` concurrency note).
    # Resolve recording_id explicitly via SpikeSorting (which IS keyed
    # by sorting_id) and pass the recording_id pk to get_recording.
    recording_id = (
        SpikeSorting & {"sorting_id": curation_key["sorting_id"]}
    ).fetch1("recording_id")
    recording = SpikeSortingRecording.get_recording(
        {"recording_id": recording_id}
    )
    sampling_frequency = float(recording.get_sampling_frequency())
    spike_times = {
        int(unit_id): np.asarray(sorting.get_unit_spike_train(unit_id))
        / sampling_frequency
        for unit_id in sorting.get_unit_ids()
    }
    return spike_times, sampling_frequency


def _compute_invariant_fingerprints(
    *,
    nwb_source: Path,
    nwb_file_name: str,
    sort_group_id: int,
    artifact_id: str,
    sorter: str,
    sorter_param_name: str,
    preproc_param_name: str,
    artifact_param_name: str,
) -> dict:
    """Compute the seven invariant fingerprints for v1↔v2 parity verification.

    The fingerprints encode the inputs that v1 and v2 MUST agree on
    for a parity comparison to be meaningful (same NWB bytes, same
    sort-group electrodes, same bad-channel mask, same effective
    preproc / artifact / sorter kwargs, same artifact-removed
    valid times). The v2 parity test reconstructs its own state and
    asserts each fingerprint matches before comparing spike times.
    Contract: see ``.claude/docs/plans/spikesorting-v2/parity-extensions.md``
    § "Invariant fingerprinting".
    """
    import numpy as np

    from spyglass.common import Electrode, IntervalList
    from spyglass.spikesorting.v1 import (
        ArtifactDetectionParameters,
        SortGroup,
        SpikeSorterParameters,
        SpikeSortingPreprocessingParameters,
    )

    # Lazy import: ``_parity_canonical`` lives next to this module so the
    # capture script remains self-contained on the v1 worktree after sync.
    from tests.spikesorting.v2._parity_canonical import (
        canonical_artifact,
        canonical_preproc,
        canonical_sorter,
    )

    nwb_sha256 = hashlib.sha256(nwb_source.read_bytes()).hexdigest()

    sort_group_electrode_ids = sorted(
        int(eid)
        for eid in (
            SortGroup.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch("electrode_id")
    )

    # JSON has no native int-key support; serialize with stringified keys
    # and let the v2 reader re-canonicalize via ``{int(k): v for k, v in ...}``.
    bad_channel_by_electrode_id = {
        str(int(row["electrode_id"])): bool(row["bad_channel"] == "True")
        for row in (
            Electrode & {"nwb_file_name": nwb_file_name}
        ).fetch("KEY", "electrode_id", "bad_channel", as_dict=True)
    }

    canonical_preproc_params = canonical_preproc(
        (
            SpikeSortingPreprocessingParameters
            & {"preproc_param_name": preproc_param_name}
        ).fetch1("preproc_params")
    )

    canonical_artifact_params = canonical_artifact(
        (
            ArtifactDetectionParameters
            & {"artifact_param_name": artifact_param_name}
        ).fetch1("artifact_params")
    )

    # v1's ``ArtifactDetection`` writes the artifact-removed times to
    # ``IntervalList`` keyed by ``interval_list_name=str(artifact_id)``
    # (``src/spyglass/spikesorting/v1/artifact.py:200``). Pin dtype and
    # include shape so the hash is comparison-stable across writers.
    valid_times_raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": str(artifact_id),
        }
    ).fetch1("valid_times")
    valid_times = np.ascontiguousarray(valid_times_raw, dtype="<f8")
    artifact_valid_times_hash = hashlib.sha256(
        valid_times.tobytes()
        + np.asarray(valid_times.shape, dtype="<i8").tobytes()
    ).hexdigest()

    canonical_sorter_params = canonical_sorter(
        sorter,
        (
            SpikeSorterParameters
            & {"sorter": sorter, "sorter_param_name": sorter_param_name}
        ).fetch1("sorter_params"),
    )

    return {
        "nwb_sha256": nwb_sha256,
        "sort_group_electrode_ids": sort_group_electrode_ids,
        "bad_channel_by_electrode_id": bad_channel_by_electrode_id,
        "canonical_preproc_params": canonical_preproc_params,
        "canonical_artifact_params": canonical_artifact_params,
        "artifact_valid_times_hash": artifact_valid_times_hash,
        "canonical_sorter_params": canonical_sorter_params,
    }


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
    sorter_param_name: str = "default_clusterless",
    artifact_param_name: str = "default",
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
    sorter_param_name : str, optional
        Name of an existing ``SpikeSorterParameters`` row. Defaults to
        ``"default_clusterless"`` (Frank-lab 100 uV production
        threshold). For synthetic fixtures with smaller template
        amplitudes the caller must pre-insert a lower-threshold row.

    Returns
    -------
    spikes_path, meta_path : pathlib.Path
        The two written artifacts.
    """
    nwb_file_name = copy_and_insert_nwb(nwb_source)
    _ensure_lab_team(team_name)
    _ensure_sort_group(nwb_file_name, sort_group_id)
    recording_pk = _populate_recording(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name=interval_list_name,
        team_name=team_name,
    )
    artifact_pk = _populate_artifact_detection(
        recording_pk["recording_id"],
        artifact_param_name=artifact_param_name,
    )
    sorting_pk = _populate_sorting(
        nwb_file_name=nwb_file_name,
        recording_id=recording_pk["recording_id"],
        artifact_id=artifact_pk["artifact_id"],
        sorter_param_name=sorter_param_name,
    )
    curation_pk = _insert_curation(sorting_pk["sorting_id"])

    spike_times, sampling_frequency = _read_spike_times(curation_pk)

    # Refuse to write an uninformative baseline. A zero-unit or
    # all-empty-units bundle cannot serve as a v2 parity reference
    # (the parity test rejects it downstream, but failing here
    # avoids committing a useless artifact and gives the operator
    # an immediate signal that the sort parameters need tuning).
    if not spike_times:
        raise RuntimeError(
            "baseline_capture: v1 sort produced zero units on "
            f"{nwb_source.name} with sorter_param_name="
            f"{sorter_param_name!r}. Refusing to write a useless "
            "baseline. Lower the detection threshold (or pass a "
            "smoke-friendly --sorter-param-name) and re-run."
        )
    empty_units = sorted(
        int(uid) for uid, times in spike_times.items() if len(times) == 0
    )
    if empty_units:
        raise RuntimeError(
            "baseline_capture: v1 sort produced unit(s) with zero "
            f"spikes: {empty_units}. A baseline with empty units "
            "would make the per-unit parity drift check vacuous. "
            "Re-run on a longer / higher-amplitude segment."
        )

    duration_s = max(
        (times[-1] for times in spike_times.values() if len(times)),
        default=0.0,
    )

    fingerprints = _compute_invariant_fingerprints(
        nwb_source=nwb_source,
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        artifact_id=str(artifact_pk["artifact_id"]),
        sorter="clusterless_thresholder",
        sorter_param_name=sorter_param_name,
        preproc_param_name="default",
        artifact_param_name=artifact_param_name,
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
        "sorter_param_name": sorter_param_name,
        "preproc_param_name": "default",
        "artifact_param_name": artifact_param_name,
        "spikeinterface_version": _pkg_version("spikeinterface"),
        "spyglass_version": _pkg_version("spyglass-neuro"),
        **fingerprints,
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
    parser.add_argument(
        "--sorter-param-name",
        default="default_clusterless",
        help=(
            "Name of the SpikeSorterParameters row to sort with. "
            "Defaults to 'default_clusterless' (Frank-lab 100 uV "
            "production threshold). For synthetic fixtures with "
            "smaller template amplitudes (e.g. MEArec smoke), pre-"
            "insert a lower-threshold row and pass its name here."
        ),
    )
    parser.add_argument(
        "--artifact-param-name",
        default="default",
        help=(
            "Name of the ArtifactDetectionParameters row to use. "
            "Defaults to 'default' (v1: 3000 uV threshold). For the "
            "polymer parity matrix pass 'none' so the artifact stage "
            "is a typed no-detect on both pipelines -- v1's 'default' "
            "and v2's 'default' rows differ in amplitude_thresh_uV "
            "(3000 vs 500), which would FAIL the canonical_artifact_params "
            "fingerprint check on synthetic fixtures with no real artifacts."
        ),
    )
    return parser.parse_args(argv)


def _ensure_smoke_sorter_param_row(sorter_param_name: str) -> None:
    """Refresh the smoke-fixture ``clusterless_thresholder`` row.

    Frank-lab's 100 uV default rejects every peak the synthetic
    MEArec smoke fixture produces (its templates max out around a
    few uV); without a calibrated row the v1 sort produces zero
    units and the curation step crashes on the empty units table.
    A 5 uV threshold reliably surfaces planted peaks while staying
    above the simulator's noise floor.

    Always deletes any existing row with the same name before
    inserting -- a stale row from an earlier capture (possibly
    written with different ``noise_levels`` / ``detect_threshold``
    semantics under a prior schema) would silently be reused and
    flip the v1 baseline's contract without the caller knowing.
    The v2 parity test does the same delete-then-insert dance for
    the same reason.
    """
    from spyglass.spikesorting.v1 import SpikeSorterParameters
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAMS,
    )

    key = {
        "sorter": "clusterless_thresholder",
        "sorter_param_name": sorter_param_name,
    }
    # ``noise_levels`` intentionally omitted so SI 0.99 computes it
    # from the recording itself; passing a Python list fails inside
    # SI's ``detect_peaks`` with ``can't multiply sequence by
    # non-int of type 'float'``. v1's ``SpikeSorterParameters``
    # schema also expects the legacy ``outputs`` / ``random_chunk_kwargs``
    # keys that v2's pydantic-validated schema drops; supply them
    # here only.
    v1_payload = dict(SMOKE_CLUSTERLESS_PARAMS)
    v1_payload.update({"outputs": "sorting", "random_chunk_kwargs": {}})

    # v1 ``SpikeSorterParameters`` lives in a schema that does NOT honor
    # ``dj.config["database.prefix"]`` (verified empirically: schema name
    # remains ``spikesorting_v1_sorting`` regardless of prefix), so
    # parallel captures share this table. The row name is the same
    # across all sessions, but the params blob might drift if an
    # earlier capture wrote a stale schema. Two-step refresh:
    # (1) if an existing row's payload mismatches, delete it; otherwise
    # leave it intact so concurrent sessions don't race delete↔insert.
    # (2) insert with ``skip_duplicates=True`` so a concurrent session
    # that already inserted the row is a no-op, not an error.
    existing = (SpikeSorterParameters & key).fetch("sorter_params", limit=1)
    if len(existing) > 0 and existing[0] != v1_payload:
        (SpikeSorterParameters & key).delete(safemode=False)
    SpikeSorterParameters().insert1(
        {**key, "sorter_params": v1_payload},
        skip_duplicates=True,
    )


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

    # Pre-insert a smoke-friendly sorter row if the caller asked for a
    # name other than the shipped default; the test runner expects
    # the row to exist before run_baseline_capture references it.
    if args.sorter_param_name != "default_clusterless":
        _ensure_smoke_sorter_param_row(args.sorter_param_name)

    run_baseline_capture(
        nwb_source=nwb_source,
        sort_group_id=args.sort_group_id,
        interval_list_name=args.interval_list_name,
        team_name=args.team_name,
        output_dir=Path(args.output_dir),
        artifact_param_name=args.artifact_param_name,
        sorter_param_name=args.sorter_param_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
