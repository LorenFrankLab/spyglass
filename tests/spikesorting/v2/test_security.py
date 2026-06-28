"""Security / trust-model hardening for the v2 pipeline.

Defense-in-depth under the trusted-compute-operator model: artifacts default to
owner-write (not world-writable), the sorter scratch is only world-writable for
a container backend (the UID-mismatch case), and caller-supplied NWB file names
are confined to a bare basename before being joined onto a managed directory.
"""

import os
import stat
import uuid

import pytest


# --------------------------------------------------------------------------
# Caller-supplied NWB file-name path confinement (DB-free).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape.nwb",
        "sub/dir.nwb",
        "/absolute/path.nwb",
        "..",
        ".",
        "",
        "a/../b.nwb",
    ],
)
def test_nwb_filename_traversal_rejected(bad_name):
    """A non-basename ``nwb_file_name`` (separator / ``..`` / absolute) is
    rejected at the filename acceptance boundary."""
    from spyglass.utils.nwb_helper_fn import assert_safe_nwb_file_name

    with pytest.raises(ValueError, match="bare file name"):
        assert_safe_nwb_file_name(bad_name)


def test_nwb_filename_basename_accepted():
    """A legitimate bare basename (incl. one with dots/underscores) passes --
    the guard must not over-reject."""
    from spyglass.utils.nwb_helper_fn import assert_safe_nwb_file_name

    for good in ("session1_.nwb", "mearec_smoke.nwb", "a.b.c.nwb"):
        assert_safe_nwb_file_name(good)  # does not raise


# --------------------------------------------------------------------------
# Sorter scratch permissions: container-only world-writable.
# --------------------------------------------------------------------------


def _scratch_mode_for_backend(monkeypatch, execution_params):
    """Run ``Sorting._run_si_sorter`` with a stubbed ``run_sorter`` that
    records the mode of the per-sort scratch dir, and return that mode."""
    import spikeinterface as si
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2.test_sorting_dispatch import _tiny_numpy_sorting

    rec = si.generate_recording(
        num_channels=4, durations=[1.0], sampling_frequency=30_000.0
    )
    captured = {}

    def _capture(*args, **kwargs):
        # ``folder`` is a child of the per-sort scratch dir (the chmod target).
        scratch = os.path.dirname(kwargs["folder"])
        captured["mode"] = stat.S_IMODE(os.stat(scratch).st_mode)
        return _tiny_numpy_sorting()

    monkeypatch.setattr(sis, "run_sorter", _capture)
    Sorting._run_si_sorter(
        "mountainsort5", {}, rec, uuid.uuid4(), None, execution_params
    )
    return captured["mode"]


@pytest.mark.usefixtures("dj_conn")
def test_local_sorter_scratch_not_world_writable(monkeypatch):
    """A LOCAL sorter run leaves its scratch non-world-writable (no gratuitous
    0o777); a CONTAINER backend run still chmods 0o777 (the container process
    may run as a different uid and must write into the scratch)."""
    local_mode = _scratch_mode_for_backend(monkeypatch, None)
    assert not (local_mode & 0o002), (
        "local sorter scratch must not be world-writable; got "
        f"{oct(local_mode)}"
    )

    container_mode = _scratch_mode_for_backend(
        monkeypatch,
        {"backend": "docker", "container_image": "spikeinterface/test:latest"},
    )
    assert container_mode == 0o777, (
        "container-backend scratch must remain 0o777 (UID-mismatch case); got "
        f"{oct(container_mode)}"
    )


# --------------------------------------------------------------------------
# Materialized v2 artifacts are owner-write only (0o644), not world-writable.
# --------------------------------------------------------------------------


def test_all_v2_writers_restrict_permission():
    """Every ``AnalysisNwbfile().create(...)`` in the v2 package passes
    ``restrict_permission=True``.

    ``AnalysisNwbfile.create`` defaults to world-writable ``0o666``; a v2 writer
    that forgets the flag silently ships a world-writable artifact. This static
    guard covers all writers (Recording / Sorting units / CurationEvaluation /
    UnitMatch) -- present and future -- without populating each artifact type.
    """
    import ast
    from pathlib import Path

    v2_dir = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spyglass"
        / "spikesorting"
        / "v2"
    )
    offenders = []
    for py in sorted(v2_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            obj = getattr(func, "value", None)
            is_create = (
                isinstance(func, ast.Attribute)
                and func.attr == "create"
                and isinstance(obj, ast.Call)
                and isinstance(obj.func, ast.Name)
                and obj.func.id == "AnalysisNwbfile"
            )
            if not is_create:
                continue
            rp = next(
                (kw for kw in node.keywords if kw.arg == "restrict_permission"),
                None,
            )
            if not (
                rp is not None
                and isinstance(rp.value, ast.Constant)
                and rp.value.value is True
            ):
                offenders.append(f"{py.name}:{node.lineno}")

    assert not offenders, (
        "v2 AnalysisNwbfile().create() call(s) missing "
        f"restrict_permission=True (artifacts would be world-writable): "
        f"{offenders}"
    )


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_v2_artifacts_not_world_writable(populated_sorting):
    """A populated recording artifact is written 0o644, not the default
    world-writable 0o666 (the v2 writer passes ``restrict_permission=True``)."""
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # SortingSelection is source-polymorphic: recording_id lives on the
    # RecordingSource part, not the master.
    recording_id = (
        SortingSelection.RecordingSource & populated_sorting
    ).fetch1("recording_id")
    analysis_file_name = (
        Recording & {"recording_id": recording_id}
    ).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    mode = stat.S_IMODE(os.stat(abs_path).st_mode)

    assert mode == 0o644, (
        f"v2 recording artifact must be 0o644 (owner-write only); got "
        f"{oct(mode)}"
    )
