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
    v2_src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "spyglass"
        / "spikesorting"
        / "v2"
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

    Static AST guard: each ``make_compute`` body does not call
    ``self.insert1``, ``AnalysisNwbfile().add``,
    ``IntervalList.insert1``, or any method on ``self.connection``.
    Guards against future refactors that re-introduce monolithic
    ``make`` patterns.
    """
    from spyglass.spikesorting.v2 import artifact, recording, sorting

    forbidden_calls = {
        ("self", "insert1"),
        ("IntervalList", "insert1"),
        ("self", "connection"),
    }
    for mod, cls_name in [
        (recording, "Recording"),
        (artifact, "ArtifactDetection"),
        (sorting, "Sorting"),
    ]:
        cls = getattr(mod, cls_name)
        src = inspect.getsource(cls.make_compute)
        tree = ast.parse(inspect.cleandoc(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                ):
                    pair = (func.value.id, func.attr)
                    assert pair not in forbidden_calls, (
                        f"{cls_name}.make_compute calls "
                        f"{pair[0]}.{pair[1]}; make_compute must be "
                        "pure (no DB writes)."
                    )
                # Also catch the (Cls).add pattern where Cls is an
                # ``AnalysisNwbfile`` call expression.
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "add"
                    and isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id == "AnalysisNwbfile"
                ):
                    pytest.fail(
                        f"{cls_name}.make_compute calls "
                        "AnalysisNwbfile().add; that must run in "
                        "make_insert."
                    )


def test_curation_v2_nwb_write_outside_transaction():
    """CurationV2.insert_curation stages NWB BEFORE transaction.

    Manual-table insert_curation has no framework transaction; the
    NWB write must happen OUTSIDE the explicit ``transaction_or_noop``
    block to keep the inner transaction short.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    src = inspect.getsource(CurationV2.insert_curation)
    stage_idx = src.find("_stage_curated_units_nwb")
    txn_idx = src.find("with transaction_or_noop")
    assert stage_idx != -1, "missing _stage_curated_units_nwb call"
    assert txn_idx != -1, "missing transaction_or_noop block"
    assert stage_idx < txn_idx, (
        "_stage_curated_units_nwb must be called BEFORE the "
        "transaction_or_noop block."
    )
    # And nothing inside the transaction block invokes the stage helper
    # or pynwb.NWBHDF5IO / AnalysisNwbfile.create.
    txn_body = src[txn_idx:]
    for forbidden in (
        "_stage_curated_units_nwb(",
        "pynwb.NWBHDF5IO(",
        "AnalysisNwbfile().create",
    ):
        assert forbidden not in txn_body, (
            f"{forbidden!r} must not appear inside the transaction; "
            "it would re-introduce the long-transaction problem."
        )
