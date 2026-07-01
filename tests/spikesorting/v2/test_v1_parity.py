"""v1-parity validation slice.

Tests in this module verify that v2 behavior matches v1's documented
contract on points where the earlier v2 work silently diverged. Each
test is short, focused, and either pure-Python or DB-tier without
populate -- the heavier integration / regression tests live in
the ``single_session/`` suite.

Where a test pins down the fix for a specific v1↔v2 divergence the
docstring cites the v1 source line so a future reviewer can confirm
we did not drift away from v1's intent without justification.
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

# Most tests below import from v2 runtime modules whose ``dj.schema``
# activations require a live Docker MySQL connection. The
# ``dj_conn`` fixture (from the project conftest) starts the
# container once per session.
pytestmark = pytest.mark.usefixtures("dj_conn")


# ---------- schema defaults ------------------------------------------------


def test_artifact_defaults_match_b1_revised():
    """``amplitude_threshold_uv=500.0`` µV; ``proportion_above_threshold=1.0``.

    The amplitude default is the v1-effective Frank-lab Intan
    threshold post-unit-conversion-fix (v1's 3000 nominal == ~585 µV
    on 0.195 µV/count Intan probes). proportion=1.0 reverts an
    earlier silent flip from v1's "all channels must exceed".
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    blob = ArtifactDetectionParamsSchema().model_dump()
    assert blob["amplitude_threshold_uv"] == 500.0
    assert blob["proportion_above_threshold"] == 1.0


def test_common_reference_field_removed_from_schema():
    """The dead ``reference`` field is gone; ``operator`` stays."""
    assert "reference" not in CommonReferenceParams.model_fields
    assert "operator" in CommonReferenceParams.model_fields
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CommonReferenceParams(reference="local")


def test_default_whiten_none():
    """``default`` preset ships ``whiten=None``.

    Matches the other two presets and the deferred-to-sorter
    reality. The WhitenParams schema is preserved as
    forward-compat scaffolding.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )

    # Locate the default row in _DEFAULT_CONTENTS.
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
    assert contents["default"]["whiten"] is None


def test_phase_shift_preset_neuropixels_on_franklab_off():
    """Phase-shift ships ON in ``default_neuropixels``, OFF in ``default``.

    The franklab headline default stays a no-op (``phase_shift is None``);
    the blessed Neuropixels recipe enables it (``margin_ms == 100.0``) on top
    of the same 300-6000 Hz bandpass, so the two presets differ ONLY by the
    phase-shift step.
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

    # franklab: phase-shift OFF.
    assert contents["default"]["phase_shift"] is None

    # neuropixels: phase-shift ON (margin 100 ms) + the same bandpass.
    np_params = contents["default_neuropixels"]
    assert np_params["phase_shift"] == {"margin_ms": 100.0}
    assert np_params["bandpass_filter"] == {
        "freq_min": 300.0,
        "freq_max": 6000.0,
    }

    # The two presets differ ONLY by the phase-shift step.
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
    """``spikesorting-v2-matching`` extra resolves cleanly with the
    v2 NumPy + Python pins and a ``UnitMatchPy`` import does not
    fail loudly on the ``_tkinter`` import path that older
    versions trip on.

    The extra ships ``UnitMatchPy>=3.2.6,<3.2.8`` + ``mat73``. The
    test:

    1. If the extra is NOT installed (the default test
       environment), skip cleanly so this test acts as
       documentation rather than a hard requirement -- it
       activates only when a user installs ``pip install
       "spyglass-neuro[spikesorting-v2-matching]"``.
    2. If installed, verify:
       a. ``mat73`` imports.
       b. ``UnitMatchPy`` imports OR raises a *clear* import
          error citing ``_tkinter`` (the known ``UnitMatchPy``
          GUI dependency that broke on minimal Python builds);
          a bare ``ImportError`` without ``_tkinter`` in the
          message is the actual regression we guard against.
       c. NumPy version remains in the v2-supported range
          (``>=2.0``). The extra is pinned to UnitMatchPy 3.2.7
          (declares ``numpy<3,>=2``); 3.2.8+ reactively flipped to
          ``numpy<2`` (upstream #134), which would force a
          downgrade. The resolver-evidence contract requires the
          pin to stay on a numpy>=2-compatible release.
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
        # In CI's dedicated matching-extra job (which sets
        # ``SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED=1`` AFTER
        # running ``uv pip install ...[spikesorting-v2-matching]``)
        # the extra contract is BOTH ``mat73`` AND ``UnitMatchPy``.
        # An or-gate (passing on mat73 alone) would silently let a
        # broken UnitMatchPy declaration through whenever mat73
        # happens to resolve.
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
        # Default env: the extra is not installed and we don't
        # require it. Skip cleanly so the test acts as
        # documentation. (Use AND here too -- if only one of the
        # two is importable in the default env, that's an unusual
        # state worth noting but not a CI failure outside the
        # dedicated job.)
        pytest.skip(
            "spikesorting-v2-matching extra is not fully installed "
            "in this environment (mat73="
            f"{mat73_available}, UnitMatchPy={unitmatch_available}); "
            "this test activates when the user runs "
            "``pip install spyglass-neuro[spikesorting-v2-matching]``."
        )

    # mat73 is the small ScilabHDF5 .mat reader; failure here would
    # be a packaging regression on the extra itself.
    import mat73  # noqa: F401

    # UnitMatchPy: a clean import IS the desired contract. If it
    # raises, the error message MUST cite ``_tkinter`` (the known
    # GUI-import dependency) so a user can diagnose; a bare
    # ``ImportError`` without a clue is the failure mode we guard
    # against.
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

    # NumPy compatibility: the v2 supported pin is ``>=2.0``. A
    # matching-extra install MUST NOT have forced a downgrade.
    np_major = int(np.__version__.split(".")[0])
    assert np_major >= 2, (
        f"NumPy {np.__version__} is below the v2-supported >=2.0 "
        "pin; spikesorting-v2-matching install forced an "
        "environment downgrade -- the matching-extra resolver "
        "evidence needs re-verification."
    )


def test_ms4_default_row_only_inserted_when_ms4_installed():
    """MS4 runtime guard: the default row inserts iff MS4 is installed.

    ``_DEFAULT_CONTENTS`` keeps the full catalog (MS4 rows included) for
    introspection, but ``insert_default`` gates each SI sorter on
    ``spikeinterface.sorters.installed_sorters()`` via
    ``_gated_default_rows``. On a platform without MS4 (the v2 SI-0.104
    test image) the MS4 rows are routed to "skipped"; on a platform with
    MS4 they are routed to "insertable". Without the gate, a v2 user who
    inserts an uninstalled sorter's default row and then populates
    ``Sorting`` hits an unhelpful SI "sorter not registered" error.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    # Assert the gating DECISION (independent of the live table state,
    # which other tests may have populated).
    insertable, skipped = SorterParameters._gated_default_rows()
    insertable_sorters = {row[0] for row in insertable}
    skipped_sorters = {row[0] for row in skipped}

    # clusterless_thresholder is Spyglass-internal -> never gated.
    assert "clusterless_thresholder" in insertable_sorters
    assert "clusterless_thresholder" not in skipped_sorters

    # The catalog always advertises MS4 (introspection), regardless of
    # install status.
    assert any(
        row[0] == "mountainsort4" for row in SorterParameters._DEFAULT_CONTENTS
    )

    if "mountainsort4" in sis.installed_sorters():
        assert "mountainsort4" in insertable_sorters
        assert "mountainsort4" not in skipped_sorters
    else:
        assert "mountainsort4" in skipped_sorters
        assert "mountainsort4" not in insertable_sorters


def test_clusterless_schema_documents_dead_fields_or_drops_them():
    """Dead ``outputs`` / ``random_chunk_kwargs`` fields are gone."""
    fields = ClusterlessThresholderSchema.model_fields
    # ``noise_levels`` STAYS (the runtime forwards it to detect_peaks);
    # the other two are gone.
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

    Behavioral replacement for the prior signature-only check: a root
    curation registers a ``SpikeSortingOutput.CurationV2`` merge row, and
    the helper must return that merge_id for a restriction on the sorting
    -- with ``as_dict`` toggling between a list of UUIDs and a list of
    ``{"merge_id": ...}`` dicts (the v1-parity / enhancement contract).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.utils import get_spike_sorting_v2_merge_ids

    # Clear any prior curation (master-before-part) then mint a root,
    # which registers exactly one v2 merge row for this sorting.
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

    Shipped runtime code AND user-facing docs must not reference
    internal plan-process identifiers:
      - plan-phase / plan-task labels (``Phase N``, ``phase-N``, ``Task N``)
      - review/audit codes (``A4``, ``R5``, ``C3``, ``N19``, ``B3`` ...) --
        a single uppercase letter from the review namespaces followed by a
        number; an earlier version only caught ``Phase/Task N`` and let
        these slip through.
    The scan covers the v2 runtime source, the user-facing migration guide,
    and the CHANGELOG (phase/task only there -- the audit-code regex would
    false-positive on unrelated historical release entries). Tests and
    baseline-fixture machinery may still reference these labels because they
    describe an immutable historical baseline the test corpus compares
    against, so only the surfaces above are scanned.
    """
    import re

    # parents[3] = repo root (parents[2] is ``tests/``, NOT ``tests/src/``).
    # The earlier ``parents[2]`` form resolved to a nonexistent
    # ``tests/src/...`` path so ``rglob`` walked zero files and the test
    # silently passed; verify the resolved roots exist before scanning.
    repo_root = Path(__file__).resolve().parents[3]
    v2_src = repo_root / "src" / "spyglass" / "spikesorting" / "v2"
    migration_doc = (
        repo_root / "docs" / "src" / "Features" / "SpikeSortingV2_Migration.md"
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
    # Review/audit codes: one uppercase letter from the review namespaces
    # (A/B/C/D/N/Q/R/T) + 1-2 digits, as a standalone token. Narrow on
    # purpose so it does not match sorter names (``MS5``, ``KS4``) or
    # version tokens (``V1``). If a legitimate token ever trips this,
    # reword it -- a review code in shipped code/docs is the thing we forbid.
    audit_re = re.compile(r"\b[ABCDNQRT]\d{1,2}\b")
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
            (phase_re, audit_re),
        )
    _scan(migration_doc, "SpikeSortingV2_Migration.md", (phase_re, audit_re))
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
    """The v2 resolver is lenient in the multi-source DEFAULT path: a key it
    doesn't recognize yields no v2 rows instead of raising. So when
    ``get_restricted_merge_ids`` auto-defaults its sources, a key handled by
    v1 (which the dispatcher runs first) no longer trips the v2 branch's
    'unknown key' ValueError. The strict raise is preserved for a deliberate
    v2 query (sources=['v2'] / a direct strict resolve).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    # Lenient (what the auto-default dispatch passes to the v2 branch):
    # no raise, no v2 rows (an empty fetch result).
    assert (
        len(
            SpikeSortingOutput()._get_restricted_merge_ids_v2(
                {"nwb_file_name": "x.nwb", "bogus_field": "y"}, strict=False
            )
        )
        == 0
    )
    # Strict (an explicit v2 query) still raises on the unknown key.
    with pytest.raises(ValueError, match="bogus_field"):
        SpikeSortingOutput()._get_restricted_merge_ids_v2(
            {"nwb_file_name": "x.nwb", "bogus_field": "y"}, strict=True
        )


def test_get_restricted_merge_ids_default_sources_includes_v2():
    """When ``sources`` is unspecified, v2 is among the resolved sources.

    The ``sources`` parameter default is ``None`` (resolved at call time
    to the available sources via ``_available_merge_sources``) rather
    than a mutable ``["v0", "v1", "v2"]`` literal. That avoids the
    mutable-default-argument antipattern and avoids forcing ``"v2"`` in
    v0/v1-only deployments where the v2 module never imported. The
    behavior that matters -- a caller who omits ``sources`` sees v2 rows
    when v2 is importable -- is what this test pins, rather than the
    signature literal.
    """
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
        _available_merge_sources,
    )

    # The signature default is intentionally None (runtime-resolved),
    # NOT a mutable list literal.
    sig = inspect.signature(SpikeSortingOutput.get_restricted_merge_ids)
    assert sig.parameters["sources"].default is None

    # In an environment where the v2 module imported, the resolved
    # default sources include v2 (and always v0 + v1).
    resolved = _available_merge_sources()
    assert "v0" in resolved
    assert "v1" in resolved
    assert "v2" in resolved


# ---------- Tri-part dispatch + make_compute purity ------------------------


def test_make_compute_is_pure():
    """make_compute on Recording / ArtifactDetection / Sorting writes no DB rows.

    Static AST guard. The forbidden surface is:

    1. **Any** call whose attribute name is ``insert1`` / ``insert``
       on a name or call expression (catches ``self.insert1``,
       ``IntervalList.insert1``, ``IntervalList().insert1``,
       and aliased forms like ``tbl = IntervalList(); tbl.insert1``).
    2. ``AnalysisNwbfile().add`` and the aliased
       ``nwb = AnalysisNwbfile(); nwb.add`` form.
    3. Any access on ``self.connection`` (transaction control belongs
       in ``make_insert``).

    Local-alias receivers are caught by walking assigns: a binding
    like ``tbl = IntervalList()`` taints ``tbl`` for the rest of
    the function body. The previous narrow check only flagged
    direct ``IntervalList.insert1(...)`` calls; this widened
    version surfaces refactors that route through a local alias.

    This is a defense-in-depth AST guard, not the load-bearing test:
    the behavioral counterparts that actually prove a failed populate
    leaves no orphaned NWB/DB state are
    ``single_session/test_sorting.py::test_sorting_make_rollback_cleans_units_nwb``
    and its siblings (``test_recording.py::test_recording_make_rollback_cleans_analysis_nwb``,
    ``test_curation_insert.py::test_curation_v2_insert_rollback_cleans_units_nwb``).
    """
    from spyglass.spikesorting.v2 import artifact, recording, sorting

    forbidden_attrs = {"insert", "insert1", "insert_many"}
    forbidden_receiver_types = {"IntervalList", "AnalysisNwbfile"}

    for mod, cls_name in [
        (recording, "Recording"),
        (artifact, "ArtifactDetection"),
        (sorting, "Sorting"),
    ]:
        cls = getattr(mod, cls_name)
        src = inspect.getsource(cls.make_compute)
        tree = ast.parse(inspect.cleandoc(src))

        # Walk assigns to build the set of local names that bind a
        # forbidden-receiver-type instance (``x = IntervalList()``,
        # ``y = AnalysisNwbfile()``). Receivers tainted this way
        # propagate the same write-forbidden semantics.
        tainted_names = set(forbidden_receiver_types)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(
                node.value, ast.Call
            ):
                callee = node.value.func
                callee_name = (
                    callee.id if isinstance(callee, ast.Name) else None
                )
                if callee_name in forbidden_receiver_types:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            tainted_names.add(tgt.id)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            attr = func.attr

            # (1) Any insert*/insert1/insert_many on self, a tainted
            # local alias, or directly on a forbidden receiver type.
            if attr in forbidden_attrs:
                # ``self.insert1``
                if isinstance(func.value, ast.Name) and func.value.id == "self":
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        f"self.{attr}; make_compute must be pure "
                        "(no DB writes)."
                    )
                # ``IntervalList.insert1`` / ``aliased.insert1``
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id in tainted_names
                ):
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        f"{func.value.id}.{attr}; that DB write "
                        "must move to make_insert."
                    )
                # ``IntervalList().insert1`` (instance-call form)
                if (
                    isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id in forbidden_receiver_types
                ):
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        f"{func.value.func.id}().{attr}; that DB "
                        "write must move to make_insert."
                    )

            # (2) ``AnalysisNwbfile().add`` and aliased ``nwb.add``.
            if attr == "add":
                if (
                    isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id == "AnalysisNwbfile"
                ):
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        "AnalysisNwbfile().add; that must run in "
                        "make_insert."
                    )
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id in tainted_names
                ):
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        f"{func.value.id}.add (likely aliased "
                        "AnalysisNwbfile); must move to make_insert."
                    )

            # (3) ``self.connection.<anything>`` (transaction control).
            if (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "self"
                and func.value.attr == "connection"
            ):
                pytest.fail(
                    f"{cls_name}.make_compute calls self.connection"
                    f".{attr}; transaction control belongs in "
                    "make_insert."
                )


def test_make_compute_purity_guard_actually_catches_regressions():
    """Tripwire: the widened AST guard catches local-alias regressions.

    Builds a synthetic source string with each of the regression
    patterns the widened guard is supposed to flag, runs the same
    AST walk against it, and asserts each pattern fires the
    pytest.fail. Without this meta-test, a future "simplification"
    that narrows the guard would silently leave production
    refactors un-guarded.
    """
    import ast
    import textwrap

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

    # Re-derive the same forbidden surface the guard uses; if the
    # guard ever drifts, this meta-test naturally drifts with it
    # because both reference the same constants.
    forbidden_attrs = {"insert", "insert1", "insert_many"}
    forbidden_receiver_types = {"IntervalList", "AnalysisNwbfile"}

    def _guard_walks(src: str) -> bool:
        """Return True iff the AST walk (mirror of the real guard)
        would have failed on this source."""
        tree = ast.parse(textwrap.dedent(src))
        tainted = set(forbidden_receiver_types)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                cf = n.value.func
                if (
                    isinstance(cf, ast.Name)
                    and cf.id in forbidden_receiver_types
                ):
                    for tgt in n.targets:
                        if isinstance(tgt, ast.Name):
                            tainted.add(tgt.id)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if not isinstance(f, ast.Attribute):
                continue
            if f.attr in forbidden_attrs:
                if isinstance(f.value, ast.Name) and f.value.id == "self":
                    return True
                if isinstance(f.value, ast.Name) and f.value.id in tainted:
                    return True
                if (
                    isinstance(f.value, ast.Call)
                    and isinstance(f.value.func, ast.Name)
                    and f.value.func.id in forbidden_receiver_types
                ):
                    return True
            if f.attr == "add":
                if (
                    isinstance(f.value, ast.Call)
                    and isinstance(f.value.func, ast.Name)
                    and f.value.func.id == "AnalysisNwbfile"
                ):
                    return True
                if isinstance(f.value, ast.Name) and f.value.id in tainted:
                    return True
            if (
                isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "self"
                and f.value.attr == "connection"
            ):
                return True
        return False

    for label, src in REGRESSION_SAMPLES.items():
        assert _guard_walks(src), (
            f"AST guard failed to catch regression sample {label!r}; "
            "test_make_compute_is_pure would silently pass on this "
            "pattern. Widen the guard."
        )


def test_curation_v2_nwb_write_outside_transaction():
    """CurationV2 stages the curated-units NWB OUTSIDE / BEFORE the txn.

    Manual-table insert_curation has no framework transaction; the heavy
    NWB write must happen OUTSIDE the explicit ``transaction_or_noop``
    block to keep the inner transaction short. ``insert_curation`` is a
    thin orchestrator that delegates staging to
    ``_stage_curation_artifact`` and the atomic inserts to
    ``_insert_curation_rows_transaction``; this test pins the invariant
    across those helpers:

      * ``_insert_curation_rows_transaction`` opens the
        ``transaction_or_noop`` block and performs NO NWB-write /
        ``AnalysisNwbfile().create`` / ``NWBHDF5IO`` call inside it.
      * ``_stage_curation_artifact`` performs the NWB staging call.
      * ``insert_curation`` calls staging BEFORE the transaction helper.

    Strategy: walk the AST of each method. The transaction-block scan and
    the forbidden-call check run on ``_insert_curation_rows_transaction``;
    the staging-call check runs on ``_stage_curation_artifact``; the
    ordering check runs on ``insert_curation``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    NWB_WRITE_NAMES = {"write_curated_units_nwb"}

    def _is_forbidden_call(node):
        """True if node is a Call that writes NWB or stages units."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        # Direct staging call: ``write_curated_units_nwb(...)`` (the
        # DB-free service function ``_stage_curation_artifact`` calls
        # directly), or an attribute form ``x.write_curated_units_nwb(...)``.
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
    # ``transaction_or_noop(...)`` call (or attribute thereof).
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
                if cm_name == "transaction_or_noop":
                    txn_blocks.append(node)
                    break

    assert len(txn_blocks) >= 1, (
        "_insert_curation_rows_transaction must use transaction_or_noop "
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
                        f"{forbidden!r} INSIDE a transaction_or_noop "
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
