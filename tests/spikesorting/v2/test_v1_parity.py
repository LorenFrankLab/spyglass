"""v2 matches v1's contract where it could silently diverge.

Pure-Python or DB-tier without populate; integration tests live in
``single_session/``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from spyglass.spikesorting.v2._params.preprocessing import (
    CommonReferenceParams,
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import (
    ClusterlessThresholderSchema,
    Kilosort4Schema,
)

# v2 schema modules connect on import.
pytestmark = pytest.mark.usefixtures("dj_conn")


# ---------- schema defaults ------------------------------------------------


def test_artifact_defaults_match_v1_effective_threshold():
    """``amplitude_threshold_uv=500.0`` µV; ``proportion_above_threshold=1.0``.

    v1's nominal 3000 is ~585 µV on 0.195 µV/count Intan probes;
    proportion 1.0 keeps v1's "all channels must exceed" rule.
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    blob = ArtifactDetectionParamsSchema().model_dump()
    assert blob["amplitude_threshold_uv"] == 500.0
    assert blob["proportion_above_threshold"] == 1.0


def test_common_reference_field_removed_from_schema():
    """``common_reference`` has ``operator`` and no ``reference`` field."""
    assert "reference" not in CommonReferenceParams.model_fields
    assert "operator" in CommonReferenceParams.model_fields
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CommonReferenceParams(reference="local")


def test_default_has_no_recording_stage_whiten():
    """``default`` preset carries no ``whiten`` key.

    Whitening is owned by the sorter / analyzer rows; the recording stage
    does not accept an inert ``whiten`` field.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )

    contents = {
        name: params
        for (
            name,
            params,
            _ver,
            _job,
        ) in PreprocessingParameters._DEFAULT_CONTENTS
    }
    assert "default" in contents
    assert "whiten" not in contents["default"]


def test_phase_shift_preset_neuropixels_on_franklab_off():
    """Phase-shift ships ON in ``default_neuropixels``, OFF in ``default``.

    The two presets differ only by the phase-shift step.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )

    contents = {
        name: params
        for (
            name,
            params,
            _ver,
            _job,
        ) in PreprocessingParameters._DEFAULT_CONTENTS
    }

    assert contents["default"]["phase_shift"] is None

    np_params = contents["default_neuropixels"]
    assert np_params["phase_shift"] == {"margin_ms": 100.0}
    assert np_params["bandpass_filter"] == {
        "freq_min": 300.0,
        "freq_max": 6000.0,
    }

    assert {k: v for k, v in np_params.items() if k != "phase_shift"} == {
        k: v for k, v in contents["default"].items() if k != "phase_shift"
    }


def test_min_segment_length_field_present():
    """``PreprocessingParamsSchema`` carries ``min_segment_length``."""
    blob = PreprocessingParamsSchema().model_dump()
    assert "min_segment_length" in blob
    assert blob["min_segment_length"] == 1.0


def test_artifact_min_length_s_field_present():
    """``ArtifactDetectionParamsSchema`` carries ``min_length_s``."""
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    blob = ArtifactDetectionParamsSchema().model_dump()
    assert "min_length_s" in blob
    assert blob["min_length_s"] == 1.0


# ---------- schema escape hatches -------------------------------------------


def test_kilosort4_schema_accepts_extra_kwargs_v1_parity():
    """KS4 schema mirrors v1's escape hatch (extra='allow')."""
    blob = Kilosort4Schema.model_validate(
        {"Th_universal": 9.0, "batch_size": 60_000, "nearest_chans": 10}
    ).model_dump()
    assert blob["batch_size"] == 60_000
    assert blob["nearest_chans"] == 10


def test_optional_matching_extra_resolution():
    """The ``spikesorting-v2-matching`` extra installs without breaking
    the v2 pins.

    Skips when the extra is absent. Otherwise ``mat73`` imports,
    ``UnitMatchPy`` imports or raises an error naming ``_tkinter``, and
    NumPy stays >= 2 (UnitMatchPy 3.2.8+ requires ``numpy<2``).
    """
    import importlib
    import importlib.util
    import os

    mat73_available = importlib.util.find_spec("mat73") is not None
    unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None

    matching_required = bool(
        os.environ.get("SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED")
    )

    if matching_required:
        # The matching CI job requires both packages.
        missing = []
        if not mat73_available:
            missing.append("mat73")
        if not unitmatch_available:
            missing.append("UnitMatchPy")
        if missing:
            pytest.fail(
                "SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED=1 but "
                f"the following matching-extra deps are not "
                f"importable: {missing}. The matching-extra install "
                "in this env must have failed silently. Check the "
                "preceding ``uv pip install`` step's log."
            )
    elif not (mat73_available and unitmatch_available):
        pytest.skip(
            "spikesorting-v2-matching extra is not fully installed "
            "in this environment (mat73="
            f"{mat73_available}, UnitMatchPy={unitmatch_available}); "
            "this test activates when the user runs "
            "``pip install spyglass-neuro[spikesorting-v2-matching]``."
        )

    import mat73  # noqa: F401

    # An import error must name ``_tkinter`` so users can diagnose it.
    try:
        importlib.import_module("UnitMatchPy")
    except ImportError as exc:
        assert "_tkinter" in str(exc) or "tkinter" in str(exc), (
            f"UnitMatchPy import failed with an opaque "
            f"ImportError that does not cite the known "
            f"``_tkinter`` cause: {exc!r}. The resolver-evidence "
            "contract requires the import path to either succeed "
            "or surface the ``_tkinter`` dependency clearly."
        )

    np_major = int(np.__version__.split(".")[0])
    assert np_major >= 2, (
        f"NumPy {np.__version__} is below the v2-supported >=2.0 "
        "pin; spikesorting-v2-matching install forced an "
        "environment downgrade -- the matching-extra resolver "
        "evidence needs re-verification."
    )


def test_ms4_default_row_only_inserted_when_ms4_installed():
    """Local MS4 default rows insert only if MS4 is installed.

    ``_DEFAULT_CONTENTS`` keeps every row; ``_gated_default_rows`` checks
    ``installed_sorters()``. The containerized MS4 row always inserts.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._sorting.dispatch import is_container_backend
    from spyglass.spikesorting.v2.sorting import SorterParameters

    # The gating decision, independent of live table state.
    insertable, skipped = SorterParameters._gated_default_rows()
    insertable_rows = {row[:2] for row in insertable}
    skipped_rows = {row[:2] for row in skipped}

    # clusterless_thresholder is Spyglass-internal -> never gated.
    clusterless = {
        row[:2]
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] == "clusterless_thresholder"
    }
    assert clusterless and clusterless <= insertable_rows
    assert not clusterless & skipped_rows

    ms4_rows = [
        row
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] == "mountainsort4"
    ]
    local_ms4 = {
        row[:2] for row in ms4_rows if not is_container_backend(row[5])
    }
    container_ms4 = {
        row[:2] for row in ms4_rows if is_container_backend(row[5])
    }
    assert local_ms4 and container_ms4

    assert container_ms4 <= insertable_rows
    if "mountainsort4" in sis.installed_sorters():
        assert local_ms4 <= insertable_rows
        assert not local_ms4 & skipped_rows
    else:
        assert local_ms4 <= skipped_rows
        assert not local_ms4 & insertable_rows


def test_clusterless_schema_documents_dead_fields_or_drops_them():
    """The clusterless schema has no ``outputs`` / ``random_chunk_kwargs``."""
    fields = ClusterlessThresholderSchema.model_fields
    # ``noise_levels`` is forwarded to detect_peaks.
    assert "noise_levels" in fields
    assert "outputs" not in fields
    assert "random_chunk_kwargs" not in fields


# ---------- user-facing helpers ---------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_get_spike_sorting_v2_merge_ids_resolves_restriction(
    populated_sorting,
):
    """The notebook helper resolves a restriction to its v2 merge_id(s).

    ``as_dict`` toggles between UUIDs and ``{"merge_id": ...}`` dicts.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import (
        get_spike_sorting_v2_merge_ids,
    )

    # A fresh root registers exactly one v2 merge row.
    curation_keys = (CurationV2 & populated_sorting).fetch("KEY", as_dict=True)
    if curation_keys:
        for mid in (SpikeSortingOutput.CurationV2 & curation_keys).fetch(
            "merge_id"
        ):
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & populated_sorting).super_delete(warn=False)

    pk = CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
    expected = {
        str(m) for m in (SpikeSortingOutput.CurationV2 & pk).fetch("merge_id")
    }
    assert expected, "insert_curation must register a v2 merge row"

    restriction = {"sorting_id": populated_sorting["sorting_id"]}
    ids = get_spike_sorting_v2_merge_ids(restriction)
    assert {str(m) for m in ids} == expected

    dicts = get_spike_sorting_v2_merge_ids(restriction, as_dict=True)
    assert all(set(d.keys()) == {"merge_id"} for d in dicts)
    assert {str(d["merge_id"]) for d in dicts} == expected


# ---------- phase-label leakage --------------------------------------------


def test_no_phase_label_leakage_in_runtime_code():
    """Zero plan-phase / review-code identifier hits in v2 artifacts.

    Forbidden: ``Phase N`` / ``Task N`` labels, single-letter review codes
    (``A4``, ``N19``) and hyphenated codes (``OP-3``). Scans v2 source, v2
    guides, tutorial scripts, and the CHANGELOG (phase/task only). Tests
    are not scanned (unit names like ``A1``).
    """
    import re

    # A wrong parents index would scan nothing; roots are checked below.
    repo_root = Path(__file__).resolve().parents[3]
    v2_src = repo_root / "src" / "spyglass" / "spikesorting" / "v2"
    user_docs = (
        repo_root / "docs" / "src" / "Features" / "SpikeSortingV2_Migration.md",
        repo_root
        / "docs"
        / "src"
        / "ForDevelopers"
        / "SpikeSortingV2Architecture.md",
    )
    notebook_scripts = tuple(
        sorted(
            (repo_root / "notebooks" / "py_scripts").glob(
                "10_Spike_SortingV2*.py"
            )
        )
    )
    changelog = repo_root / "CHANGELOG.md"
    assert v2_src.is_dir(), (
        f"phase-label scan resolved to {v2_src!r}, which is not a "
        "directory; the test would scan zero files and produce a vacuous "
        "pass. Check the parents[N] index."
    )

    # ``Phase``/``Task`` + a number (any sub-label like ``1b``) or the
    # plan-directory form ``phase-3``; case-insensitive.
    phase_re = re.compile(r"\b(?:phase|task)[\s-]+\d", re.IGNORECASE)
    # Narrow letter set so ``MS5``, ``KS4`` and ``V1`` do not match.
    audit_re = re.compile(r"\b[ABCDNQRT]\d{1,2}\b")
    hyphen_re = re.compile(r"\b[A-Z]{2,}-\d{1,2}\b")
    offenders: list[str] = []

    def _scan(path, label, patterns):
        if not path.is_file():
            return
        text = path.read_text()
        for pat in patterns:
            for match in pat.finditer(text):
                offenders.append(f"{label}: {match.group()!r}")

    for py_path in v2_src.rglob("*.py"):
        _scan(
            py_path,
            str(py_path.relative_to(repo_root)),
            (phase_re, audit_re, hyphen_re),
        )
    for doc in user_docs:
        _scan(
            doc,
            str(doc.relative_to(repo_root)),
            (phase_re, audit_re, hyphen_re),
        )
    for notebook_script in notebook_scripts:
        _scan(
            notebook_script,
            str(notebook_script.relative_to(repo_root)),
            (phase_re, audit_re, hyphen_re),
        )
    _scan(changelog, "CHANGELOG.md", (phase_re,))

    assert not offenders, (
        "plan-phase / review-code leakage in v2 runtime or user-facing "
        f"docs: {offenders}"
    )


# ---------- merge dispatch --------------------------------------------------


def test_merge_dispatch_raises_on_unknown_restriction_keys():
    """A deliberate v2 query (strict) raises on unknown keys instead of
    silently dropping them."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    with pytest.raises(ValueError, match="bogus_field"):
        SpikeSortingOutput()._get_restricted_merge_ids_v2(
            {"nwb_file_name": "x.nwb", "bogus_field": "y"}
        )


def test_merge_dispatch_lenient_on_non_v2_keys_in_default_path(dj_conn):
    """In the multi-source default path an unknown key yields no v2 rows,
    so a v1-only key does not raise; an explicit v2 query still raises.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    assert (
        len(
            SpikeSortingOutput()._get_restricted_merge_ids_v2(
                {"nwb_file_name": "x.nwb", "bogus_field": "y"}, strict=False
            )
        )
        == 0
    )
    with pytest.raises(ValueError, match="bogus_field"):
        SpikeSortingOutput()._get_restricted_merge_ids_v2(
            {"nwb_file_name": "x.nwb", "bogus_field": "y"}, strict=True
        )


def test_get_restricted_merge_ids_default_sources_includes_v2():
    """When ``sources`` is unspecified, v2 is among the resolved sources.

    The default is ``None``, resolved at call time so v0/v1-only
    deployments do not require v2.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        _available_merge_sources,
    )

    sig = inspect.signature(SpikeSortingOutput.get_restricted_merge_ids)
    assert sig.parameters["sources"].default is None

    resolved = _available_merge_sources()
    assert "v0" in resolved
    assert "v1" in resolved
    assert "v2" in resolved


# ---------- Tri-part dispatch + make_compute purity ------------------------


def _make_compute_db_write_violations(src):
    """The source gate used by both real methods and planted fault controls."""
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    forbidden_attrs = {"insert", "insert1", "insert_many"}
    receiver_types = {"IntervalList", "AnalysisNwbfile"}
    tainted = set(receiver_types)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Name) and callee.id in receiver_types:
                tainted.update(
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        func = node.func
        receiver = func.value
        name = receiver.id if isinstance(receiver, ast.Name) else None
        instance_type = (
            receiver.func.id
            if isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            else None
        )
        writes_rows = func.attr in forbidden_attrs and (
            name == "self" or name in tainted or instance_type in receiver_types
        )
        writes_analysis = func.attr == "add" and (
            name in tainted or instance_type == "AnalysisNwbfile"
        )
        controls_transaction = (
            isinstance(receiver, ast.Attribute)
            and isinstance(receiver.value, ast.Name)
            and receiver.value.id == "self"
            and receiver.attr == "connection"
        )
        if writes_rows or writes_analysis or controls_transaction:
            violations.append(ast.unparse(node))
    return violations


def test_make_compute_is_pure():
    """make_compute on Recording / RecordingArtifactDetection / Sorting writes no DB rows."""
    from spyglass.spikesorting.v2 import artifact, recording, sorting

    for module, name in (
        (recording, "Recording"),
        (artifact, "RecordingArtifactDetection"),
        (sorting, "Sorting"),
    ):
        violations = _make_compute_db_write_violations(
            inspect.getsource(getattr(module, name).make_compute)
        )
        assert (
            not violations
        ), f"{name}.make_compute must perform no DB writes: {violations}"


def test_make_compute_purity_guard_actually_catches_regressions():
    """The AST purity guard flags each planted DB-write pattern."""
    REGRESSION_SAMPLES = {
        "self.insert1": "def make_compute(self, key):\n    self.insert1({'x': 1})\n",
        "IntervalList.insert1 direct": (
            "def make_compute(self, key):\n"
            "    IntervalList.insert1({'x': 1})\n"
        ),
        "IntervalList().insert1 instance-call": (
            "def make_compute(self, key):\n"
            "    IntervalList().insert1({'x': 1})\n"
        ),
        "aliased IntervalList.insert1": (
            "def make_compute(self, key):\n"
            "    tbl = IntervalList()\n"
            "    tbl.insert1({'x': 1})\n"
        ),
        "AnalysisNwbfile().add": (
            "def make_compute(self, key):\n"
            "    AnalysisNwbfile().add('a', 'b')\n"
        ),
        "aliased AnalysisNwbfile.add": (
            "def make_compute(self, key):\n"
            "    nwb = AnalysisNwbfile()\n"
            "    nwb.add('a', 'b')\n"
        ),
        "self.connection.<anything>": (
            "def make_compute(self, key):\n"
            "    self.connection.commit_transaction()\n"
        ),
    }

    # Exercise the helper that inspects the actual runtime methods above.
    for label, src in REGRESSION_SAMPLES.items():
        assert _make_compute_db_write_violations(src), (
            f"AST guard failed to catch regression sample {label!r}; "
            "test_make_compute_is_pure would silently pass on this "
            "pattern. Widen the guard."
        )
    assert (
        _make_compute_db_write_violations(
            "def make_compute(self, key):\n"
            "    arrays = compute_arrays(key)\n"
            "    return arrays\n"
        )
        == []
    )


def test_curation_v2_nwb_write_outside_transaction():
    """CurationV2 stages the curated-units NWB before and outside the
    transaction, keeping the transaction short.

    * ``_insert_curation_rows_transaction`` makes no NWB-write call inside
      ``_safe_context()``.
    * ``_stage_curation_artifact`` performs the NWB staging call.
    * ``insert_curation`` stages before calling the transaction helper.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    NWB_WRITE_NAMES = {"write_curated_units_nwb"}

    def _is_forbidden_call(node):
        """True if node is a Call that writes NWB or stages units."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        # Bare or attribute ``write_curated_units_nwb(...)``.
        if isinstance(func, ast.Name) and func.id in NWB_WRITE_NAMES:
            return func.id
        if isinstance(func, ast.Attribute) and func.attr in NWB_WRITE_NAMES:
            return func.attr
        # ``AnalysisNwbfile().create``
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "create"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "AnalysisNwbfile"
        ):
            return "AnalysisNwbfile_create"
        # ``pynwb.NWBHDF5IO(...)``
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "NWBHDF5IO"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pynwb"
        ):
            return "pynwb_NWBHDF5IO"
        # Bare ``NWBHDF5IO(...)`` (imported directly)
        if isinstance(func, ast.Name) and func.id == "NWBHDF5IO":
            return "NWBHDF5IO_direct"
        return None

    def _callee_name(node):
        """Return the called function/method name of a Call node."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    # --- The transaction lives in _insert_curation_rows_transaction. ---
    txn_src = inspect.getsource(CurationV2._insert_curation_rows_transaction)
    txn_tree = ast.parse(inspect.cleandoc(txn_src))
    # Find every ``with`` block whose context manager is a
    # ``cls._safe_context()`` call.
    txn_blocks = []
    for node in ast.walk(txn_tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                cm_func = ctx.func
                cm_name = (
                    cm_func.id
                    if isinstance(cm_func, ast.Name)
                    else (
                        cm_func.attr
                        if isinstance(cm_func, ast.Attribute)
                        else None
                    )
                )
                if cm_name == "_safe_context":
                    txn_blocks.append(node)
                    break

    assert len(txn_blocks) >= 1, (
        "_insert_curation_rows_transaction must use _safe_context() "
        "for atomic master+part inserts; no such ``with`` block "
        "was found."
    )

    # No forbidden NWB-write calls allowed inside any txn block.
    for txn_block in txn_blocks:
        for body_node in txn_block.body:
            for sub in ast.walk(body_node):
                forbidden = _is_forbidden_call(sub)
                if forbidden is not None:
                    pytest.fail(
                        f"_insert_curation_rows_transaction calls "
                        f"{forbidden!r} INSIDE a _safe_context() "
                        "block; the heavy NWB write must happen "
                        "OUTSIDE the transaction to keep it short."
                    )

    # --- The staging call lives in _stage_curation_artifact. ---
    stage_src = inspect.getsource(CurationV2._stage_curation_artifact)
    stage_tree = ast.parse(inspect.cleandoc(stage_src))
    has_stage_call = any(
        _is_forbidden_call(n) in NWB_WRITE_NAMES
        for n in ast.walk(stage_tree)
        if isinstance(n, ast.Call)
    )
    assert has_stage_call, (
        "_stage_curation_artifact does not call write_curated_units_nwb; "
        "the NWB staging step is missing."
    )

    # --- The orchestrator stages BEFORE it commits. ---
    orch_src = inspect.getsource(CurationV2.insert_curation)
    orch_tree = ast.parse(inspect.cleandoc(orch_src))
    stage_linenos = [
        n.lineno
        for n in ast.walk(orch_tree)
        if _callee_name(n) == "_stage_curation_artifact"
    ]
    txn_linenos = [
        n.lineno
        for n in ast.walk(orch_tree)
        if _callee_name(n) == "_insert_curation_rows_transaction"
    ]
    assert stage_linenos, (
        "insert_curation does not call _stage_curation_artifact; the NWB "
        "staging step is missing from the orchestrator."
    )
    assert txn_linenos, (
        "insert_curation does not call _insert_curation_rows_transaction; "
        "the atomic insert step is missing from the orchestrator."
    )
    assert min(stage_linenos) < min(txn_linenos), (
        "insert_curation must stage the NWB (_stage_curation_artifact) "
        "BEFORE committing rows (_insert_curation_rows_transaction) so the "
        "heavy filesystem write stays outside the transaction."
    )
