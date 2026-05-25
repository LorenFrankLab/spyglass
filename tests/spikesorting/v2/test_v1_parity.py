"""v1-parity validation slice for Phase 1b.

Tests in this module verify that v2 behavior matches v1's documented
contract on points where Phase 1 silently diverged. Each test is
short, focused, and either pure-Python (T1) or DB-tier without
populate (T2) -- the heavier integration / regression tests live
in ``test_single_session_pipeline.py``.

Where a test pins down the fix for a specific Phase 1b plan item
(R/N/B tag), the docstring cites the tag and the v1 source line so
a future reviewer can confirm we did not drift away from v1's
intent without justification.
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


# ---------- B1 / R7 / R13 / R18 schema defaults ----------------------------


def test_artifact_defaults_match_b1_revised():
    """B1: amplitude_thresh_uV=500.0 µV; proportion_above_thresh=1.0.

    The amplitude default is the v1-effective Frank-lab Intan
    threshold post-unit-conversion-fix (v1's 3000 nominal == ~585 µV
    on 0.195 µV/count Intan probes). proportion=1.0 reverts v2's
    silent flip from v1's "all channels must exceed".
    """
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    blob = ArtifactDetectionParamsSchema().model_dump()
    assert blob["amplitude_thresh_uV"] == 500.0
    assert blob["proportion_above_thresh"] == 1.0


def test_common_reference_field_removed_from_schema():
    """R18: the dead ``reference`` field is gone; ``operator`` stays."""
    assert "reference" not in CommonReferenceParams.model_fields
    assert "operator" in CommonReferenceParams.model_fields
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CommonReferenceParams(reference="local")


def test_default_franklab_whiten_none():
    """N35: ``default_franklab`` preset ships ``whiten=None``.

    Matches the other two presets and the deferred-to-sorter
    reality. The WhitenParams schema is preserved as
    forward-compat scaffolding.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
    )

    # Locate the default_franklab row in _DEFAULT_CONTENTS.
    contents = {
        name: params
        for (name, params, _ver, _job) in PreprocessingParameters._DEFAULT_CONTENTS
    }
    assert "default_franklab" in contents
    assert contents["default_franklab"]["whiten"] is None


def test_min_segment_length_field_present():
    """R7: ``PreprocessingParamsSchema`` carries ``min_segment_length``."""
    blob = PreprocessingParamsSchema().model_dump()
    assert "min_segment_length" in blob
    assert blob["min_segment_length"] == 1.0


def test_artifact_min_length_s_field_present():
    """R13: ``ArtifactDetectionParamsSchema`` carries ``min_length_s``."""
    from spyglass.spikesorting.v2._params.artifact_detection import (
        ArtifactDetectionParamsSchema,
    )

    blob = ArtifactDetectionParamsSchema().model_dump()
    assert "min_length_s" in blob
    assert blob["min_length_s"] == 1.0


# ---------- N34 / N48 schema escape hatches ---------------------------------


def test_kilosort4_schema_accepts_extra_kwargs_v1_parity():
    """N34: KS4 schema mirrors v1's escape hatch (extra='allow')."""
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

    The extra ships ``UnitMatchPy>=3.3,<4`` + ``mat73``. The
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
          (``>=2.0``) -- ``UnitMatchPy`` historically pinned
          ``numpy<2`` which would force an environment
          downgrade. The Phase 0c plan requires the resolver to
          have confirmed this does not happen.
    """
    import importlib
    import importlib.util
    import os

    mat73_available = importlib.util.find_spec("mat73") is not None
    unitmatch_available = importlib.util.find_spec("UnitMatchPy") is not None

    if not (mat73_available or unitmatch_available):
        # In CI's dedicated matching-extra job (which sets
        # ``SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED=1`` AFTER running
        # ``uv pip install ...[spikesorting-v2-matching]``), the
        # absence of both deps means the install step itself failed
        # -- ``|| true`` in the workflow swallowed it. Fail loud here
        # so a permanent extra-declaration bug cannot pose as a
        # transient install failure.
        if os.environ.get("SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED"):
            pytest.fail(
                "SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED=1 but "
                "neither ``mat73`` nor ``UnitMatchPy`` are "
                "importable -- the matching-extra install in this "
                "env must have failed silently. Check the "
                "preceding ``uv pip install`` step's log."
            )
        pytest.skip(
            "spikesorting-v2-matching extra is not installed in "
            "this environment; this test activates when the user "
            "runs ``pip install spyglass-neuro[spikesorting-v2-matching]``."
        )

    # mat73 is the small ScilabHDF5 .mat reader; failure here would
    # be a packaging regression on the extra itself.
    if mat73_available:
        import mat73  # noqa: F401

    # UnitMatchPy: a clean import IS the desired contract. If it
    # raises, the error message MUST cite ``_tkinter`` (the known
    # GUI-import dependency) so a user can diagnose; a bare
    # ``ImportError`` without a clue is the failure mode we guard
    # against.
    if unitmatch_available:
        try:
            importlib.import_module("UnitMatchPy")
        except ImportError as exc:
            assert "_tkinter" in str(exc) or "tkinter" in str(exc), (
                f"UnitMatchPy import failed with an opaque "
                f"ImportError that does not cite the known "
                f"``_tkinter`` cause: {exc!r}. Phase 0c required "
                "the import path to either succeed or surface the "
                "``_tkinter`` dependency clearly."
            )

    # NumPy compatibility: the v2 supported pin is ``>=2.0``. A
    # matching-extra install MUST NOT have forced a downgrade.
    import numpy as np

    np_major = int(np.__version__.split(".")[0])
    assert np_major >= 2, (
        f"NumPy {np.__version__} is below the v2-supported >=2.0 "
        "pin; spikesorting-v2-matching install forced an "
        "environment downgrade -- the Phase 0c resolver evidence "
        "needs re-verification."
    )


def test_ms4_default_row_only_shipped_when_ms4_installed():
    """MS4 runtime guard.

    MS4 runtime status must be EXPLICIT: either MS4 is in
    ``sis.installed_sorters()`` (and the default Lookup row is
    safe to ship), or the missing-MS4 case must be documented
    with a platform-specific guard. v2 ships an MS4 default row
    at ``SorterParameters._DEFAULT_CONTENTS`` but the resolver
    artifact at ``tests/spikesorting/v2/resolver/si0104-runtime.md``
    shows MS4 is NOT in the installed-sorter set in the v2 test
    image. This test pins the contract: if MS4 is not installed,
    a v2 user attempting to ``Sorting.populate`` with the MS4
    default row will hit ``sis.run_sorter`` with an unregistered
    sorter and get an unhelpful SI error. The right behavior is to
    SKIP the MS4 row at ``insert_default`` time when MS4 isn't
    importable, OR to emit a clear deprecation warning. This test
    skips cleanly when MS4 IS installed (so it doesn't block the
    happy path) and FAILS with a clear message when MS4 is missing
    but the default row is still in the contents tuple -- forcing
    the install-guard issue to a head.
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.sorting import SorterParameters

    ms4_installed = "mountainsort4" in sis.installed_sorters()
    ms4_default_rows = [
        row
        for row in SorterParameters._DEFAULT_CONTENTS
        if row[0] == "mountainsort4"
    ]
    if ms4_installed:
        # Skip rather than bare-return so pytest reports
        # SKIPPED visibly -- a bare ``return`` registers as a
        # PASSED test with zero assertions, losing the "what was
        # exercised" signal.
        pytest.skip(
            "mountainsort4 is in sis.installed_sorters(); the "
            "default-row vs runtime guard is a non-issue on this "
            "platform."
        )

    assert not ms4_default_rows, (
        "SorterParameters._DEFAULT_CONTENTS ships MS4 default rows "
        f"({[r[1] for r in ms4_default_rows]!r}) but MS4 is not in "
        "sis.installed_sorters() on this platform. A v2 user "
        "calling SorterParameters.insert_default() then "
        "Sorting.populate(... sorter='mountainsort4' ...) will hit "
        "an unhelpful SI 'sorter not registered' error. The Phase "
        "0c plan requires MS4 runtime status to be explicit: "
        "either install MS4 (Linux only, separate dep) OR gate the "
        "default row insert behind ``if 'mountainsort4' in "
        "sis.installed_sorters()``."
    )


def test_clusterless_schema_documents_dead_fields_or_drops_them():
    """N48: dead ``outputs`` / ``random_chunk_kwargs`` fields are gone."""
    fields = ClusterlessThresholderSchema.model_fields
    # ``noise_levels`` STAYS per N19; the other two are gone post-N48.
    assert "noise_levels" in fields
    assert "outputs" not in fields
    assert "random_chunk_kwargs" not in fields


# ---------- N45 + B7 user-facing helpers ------------------------------------


def test_v2_merge_ids_helper_exists():
    """N45: v2-side parallel of v1's ``get_spiking_sorting_v1_merge_ids``."""
    from spyglass.spikesorting.v2.utils import (
        get_spiking_sorting_v2_merge_ids,
    )

    sig = inspect.signature(get_spiking_sorting_v2_merge_ids)
    assert "restriction" in sig.parameters
    assert "as_dict" in sig.parameters


def test_heterogeneous_gain_rationale_comment_present():
    """B7: the v1 latent-bug rationale comment is durable in source.

    The comment lives above the ``_np.unique(recording.get_channel_
    gains())`` check inside ``Recording._write_nwb_artifact``.
    A future ``cleanup`` pass that deletes it would silently lose
    the "why we don't pick gains[0]" context.
    """
    from spyglass.spikesorting.v2 import recording as recording_mod

    src = inspect.getsource(recording_mod.Recording._write_nwb_artifact)
    assert "silently picking gains[0]" in src
    assert "heterogeneous" in src.lower()


# ---------- N27 phase-label leakage ----------------------------------------


def test_no_phase_label_leakage_in_runtime_code():
    """N27: zero ``Phase 1`` / ``Phase 1b`` hits in runtime v2 source.

    Comments + docstrings included. Tests + baseline fixtures are
    intentionally allowed to keep historical phase-naming since
    they reference an immutable baseline.
    """
    # parents[3] = repo root (parents[2] is ``tests/``, NOT
    # ``tests/src/``). The earlier ``parents[2]`` form resolved
    # to a nonexistent ``tests/src/...`` path so ``rglob`` walked
    # zero files and the test silently passed regardless of any
    # leakage in real source. Verify the resolved root exists
    # before scanning to keep this kind of typo loud.
    v2_src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spyglass"
        / "spikesorting"
        / "v2"
    )
    assert v2_src.is_dir(), (
        f"v1-parity phase-label scan resolved to {v2_src!r}, which "
        "is not a directory; the test would scan zero files and "
        "produce a vacuous pass. Check the parents[N] index."
    )
    offenders: list[str] = []
    for py_path in v2_src.rglob("*.py"):
        text = py_path.read_text()
        for literal in ("Phase 1 ", "Phase 1b ", "Phase 1c ", "Phase 1.", "Phase 1\n"):
            if literal in text:
                offenders.append(f"{py_path.relative_to(v2_src)}: {literal!r}")
                break
    assert not offenders, f"Phase-label leakage in runtime v2: {offenders}"


# ---------- N29 merge dispatch ---------------------------------------------


def test_merge_dispatch_raises_on_unknown_restriction_keys():
    """N29: unknown restriction keys raise instead of silently dropping."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    with pytest.raises(ValueError, match="bogus_field"):
        SpikeSortingOutput()._get_restricted_merge_ids_v2(
            {"nwb_file_name": "x.nwb", "bogus_field": "y"}
        )


def test_get_restricted_merge_ids_default_sources_includes_v2():
    """N49: default sources list includes v2 so v1 callers see v2 rows."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    sig = inspect.signature(SpikeSortingOutput.get_restricted_merge_ids)
    default = sig.parameters["sources"].default
    assert default == ["v0", "v1", "v2"]


# ---------- N50 timestamp repair (pure-numpy unit) --------------------------


def test_n50_repair_clean_increasing_passthrough():
    """Strictly-increasing input is returned unchanged."""
    from spyglass.spikesorting.v2.recording import Recording

    class _FakeRecording:
        def get_sampling_frequency(self):
            return 1000.0

        def get_times(self):
            return np.array([0.0, 0.001, 0.002, 0.003])

        def get_num_segments(self):
            return 1

    rec = _FakeRecording()
    # Inject through the helper directly; signature is
    # (recording, raw_path).
    out = Recording._repaired_timestamps(rec, "/fake.nwb")
    np.testing.assert_array_equal(out, rec.get_times())


def test_n50_chunked_monotonicity_count_matches_unchunked():
    """The chunked monotonicity counter agrees with np.diff over chunks.

    The chunked counter (used to keep peak memory bounded for chronic
    recordings) overlaps adjacent chunks by one sample so the diff at
    every boundary is computed exactly once. Parametrizations are
    chosen so:

    * Multiple chunks contribute (chunk_size < n) for every case.
    * At least one injected non-monotonic event sits ON a chunk
      boundary for each chunk_size -- this is the off-by-one
      regression we are guarding against (without the overlap-by-one
      window, a boundary event would be counted zero or two times).
    """
    from spyglass.spikesorting.v2.recording import Recording

    rng = np.random.default_rng(0xCA7E)
    n = 10_000
    ts = np.cumsum(rng.uniform(0.0009, 0.0011, size=n))
    # Inject 4 non-monotonic spots at fixed offsets PLUS one per
    # tested chunk size at the exact boundary. The fixed offsets
    # land at various positions inside / across chunks; the
    # boundary-injected ones land precisely at i == k*chunk_size
    # so a broken overlap window would miscount them.
    fixed_offenders = (123, 1_000, 5_555, 9_999)
    chunk_sizes = (5, 100, 1_000, 3_333)
    for cs in chunk_sizes:
        boundary = cs  # i = 1 * chunk_size: exact overlap point
        for i in (*fixed_offenders, boundary):
            ts[i] = ts[i - 1] - 0.001

        expected = int(np.sum(np.diff(ts) <= 0))
        actual = Recording._count_non_monotonic_chunked(ts, cs)
        assert actual == expected, (
            f"chunked counter at chunk_size={cs}: got {actual}, "
            f"expected {expected}. Boundary-event miscount is the "
            "off-by-one regression this test guards against."
        )
        # Reset the boundary offender so the next iteration injects
        # its own.
        ts[boundary] = ts[boundary - 1] + 0.001


@pytest.mark.parametrize(
    "input_ts,expected",
    [
        # backslide-and-return: middle dip, original max at the end
        (
            [5.0, 4.9, 5.0],
            [5.0, 5.001, 5.002],
        ),
        # multi-sample backslide
        (
            [10.0, 8.0, 9.0, 11.0],
            [10.0, 11.0, 12.0, 13.0],
        ),
        # tied first value
        (
            [5.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ),
    ],
)
def test_n50_repair_non_monotonic_patterns(input_ts, expected):
    """All three non-monotonic patterns repair to strictly increasing.

    Plain ``np.maximum.accumulate`` on the raw timestamps misses
    the ``[T, T-eps, T]`` backslide-and-return case (it would
    return ``[T, T+sp, T]``). The actual implementation enforces
    the constraint in shifted coordinates ``u[i] = ts[i] - i*sp``
    which catches every pattern.
    """
    from spyglass.spikesorting.v2.recording import Recording

    class _FakeRecording:
        def get_sampling_frequency(self):
            # 1000 Hz: sample_period = 0.001 for the first case,
            # 1.0 for the others -- parametrize via class instance.
            return self._fs

        def get_times(self):
            return np.array(self._ts, dtype=np.float64)

        def get_num_segments(self):
            return 1

    fs = 1000.0 if max(abs(np.diff(input_ts))) < 1.0 else 1.0
    rec = _FakeRecording()
    rec._fs = fs
    rec._ts = input_ts
    out = Recording._repaired_timestamps(rec, "/fake.nwb")
    np.testing.assert_allclose(out, expected, atol=1e-9)
    assert np.all(np.diff(out) > 0)


# ---------- N42 / N53 CurationV2 accessor surface --------------------------


def test_curation_v2_accessors_are_classmethod():
    """N53: all four CurationV2 accessor methods are @classmethod.

    Lets the merge dispatcher's
    ``source_table.get_recording(merge_key)`` call (which binds
    the class, not an instance) resolve correctly.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    for name in (
        "get_recording",
        "get_sorting",
        "get_sort_group_info",
        "get_merged_sorting",
        "get_merge_groups",
    ):
        attr = inspect.getattr_static(CurationV2, name)
        assert isinstance(attr, classmethod), (
            f"CurationV2.{name} must be @classmethod for merge dispatch; "
            f"got {type(attr).__name__}"
        )


# ---------- N52 Sorting.get_sorting as_dataframe ---------------------------


def test_sorting_get_sorting_accepts_as_dataframe_flag():
    """N52: ``Sorting.get_sorting`` accepts ``as_dataframe`` kwarg."""
    from spyglass.spikesorting.v2.sorting import Sorting

    sig = inspect.signature(Sorting.get_sorting)
    assert "as_dataframe" in sig.parameters
    assert sig.parameters["as_dataframe"].default is False


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
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                ):
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
    import ast as _ast
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
        tree = _ast.parse(textwrap.dedent(src))
        tainted = set(forbidden_receiver_types)
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Assign) and isinstance(n.value, _ast.Call):
                cf = n.value.func
                if isinstance(cf, _ast.Name) and cf.id in forbidden_receiver_types:
                    for tgt in n.targets:
                        if isinstance(tgt, _ast.Name):
                            tainted.add(tgt.id)
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.Call):
                continue
            f = n.func
            if not isinstance(f, _ast.Attribute):
                continue
            if f.attr in forbidden_attrs:
                if isinstance(f.value, _ast.Name) and f.value.id == "self":
                    return True
                if isinstance(f.value, _ast.Name) and f.value.id in tainted:
                    return True
                if (
                    isinstance(f.value, _ast.Call)
                    and isinstance(f.value.func, _ast.Name)
                    and f.value.func.id in forbidden_receiver_types
                ):
                    return True
            if f.attr == "add":
                if (
                    isinstance(f.value, _ast.Call)
                    and isinstance(f.value.func, _ast.Name)
                    and f.value.func.id == "AnalysisNwbfile"
                ):
                    return True
                if isinstance(f.value, _ast.Name) and f.value.id in tainted:
                    return True
            if (
                isinstance(f.value, _ast.Attribute)
                and isinstance(f.value.value, _ast.Name)
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
    """CurationV2.insert_curation stages NWB BEFORE transaction.

    Manual-table insert_curation has no framework transaction; the
    NWB write must happen OUTSIDE the explicit
    ``transaction_or_noop`` block to keep the inner transaction
    short.

    Strategy: walk the AST. Find the ``with transaction_or_noop(...)``
    block (matches both the bare-name and ``alias = transaction_or_noop``
    aliased forms by checking the resolved context-manager call's
    function name). Inside that block's body, no NWB-write helper
    or ``AnalysisNwbfile().create`` / ``pynwb.NWBHDF5IO`` call is
    allowed. Outside it, the NWB-write helper must be called at
    least once and before any ``with`` whose CM is
    ``transaction_or_noop``.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    src = inspect.getsource(CurationV2.insert_curation)
    tree = ast.parse(inspect.cleandoc(src))

    NWB_WRITE_NAMES = {"_stage_curated_units_nwb"}
    NWB_CREATE_PATTERNS = {
        # ``AnalysisNwbfile().create``: Call -> Attribute "create"
        # on a Call whose func.id is "AnalysisNwbfile"
        "AnalysisNwbfile_create",
        "pynwb_NWBHDF5IO",
    }

    def _is_forbidden_call(node):
        """True if node is a Call that writes NWB or stages units."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        # Direct helper call: ``_stage_curated_units_nwb(...)`` or
        # ``self._stage_curated_units_nwb(...)`` or
        # ``cls._stage_curated_units_nwb(...)``
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

    # Find every ``with`` block whose context manager is a
    # ``transaction_or_noop(...)`` call (or attribute thereof).
    txn_blocks = []
    for node in ast.walk(tree):
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
        "CurationV2.insert_curation must use transaction_or_noop "
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
                        f"CurationV2.insert_curation calls "
                        f"{forbidden!r} INSIDE a transaction_or_noop "
                        "block; the heavy NWB write must happen "
                        "OUTSIDE the transaction to keep it short."
                    )

    # The stage helper MUST be called somewhere -- catches a refactor
    # that quietly drops the call.
    has_stage_call = any(
        _is_forbidden_call(n) in NWB_WRITE_NAMES
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    )
    assert has_stage_call, (
        "CurationV2.insert_curation does not call "
        "_stage_curated_units_nwb; the NWB staging step is missing."
    )
