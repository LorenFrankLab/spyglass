"""DB-free tests for parameter-content fingerprints (``_parameter_identity``).

The fingerprint covers table, sorter, schema version, ``params`` and
``job_kwargs``, but not the row name.
"""

from __future__ import annotations

import numpy as np
import pytest

from spyglass.spikesorting.v2._core.parameter_identity import (
    parameter_fingerprint,
    short_fingerprint,
)


def test_fingerprint_is_stable_and_dict_order_independent():
    """Same content in any key order yields the same 64-char hex digest."""
    fp1 = parameter_fingerprint(
        "PreprocessingParameters",
        params={"a": 1, "b": {"x": 1.0, "y": 2.0}},
        params_schema_version=4,
        job_kwargs=None,
    )
    fp2 = parameter_fingerprint(
        "PreprocessingParameters",
        params={"b": {"y": 2.0, "x": 1.0}, "a": 1},
        params_schema_version=4,
        job_kwargs=None,
    )
    assert fp1 == fp2
    assert len(fp1) == 64
    assert all(c in "0123456789abcdef" for c in fp1)


def test_fingerprint_changes_with_params():
    """Different blob content -> different fingerprint."""
    fp_a = parameter_fingerprint(
        "ArtifactDetectionParameters",
        params={"amplitude_threshold_uv": 100.0},
        params_schema_version=2,
        job_kwargs=None,
    )
    fp_b = parameter_fingerprint(
        "ArtifactDetectionParameters",
        params={"amplitude_threshold_uv": 50.0},
        params_schema_version=2,
        job_kwargs=None,
    )
    assert fp_a != fp_b


def test_fingerprint_includes_sorter_context():
    """Same params under different sorters -> different fingerprint.

    Duplicate detection is scoped per sorter.
    """
    params = {"detect_threshold": 3.0}
    fp_ms4 = parameter_fingerprint(
        "SorterParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
        sorter="mountainsort4",
    )
    fp_ms5 = parameter_fingerprint(
        "SorterParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
        sorter="mountainsort5",
    )
    assert fp_ms4 != fp_ms5


def test_fingerprint_includes_table():
    """Identical blob in two different tables -> different fingerprint."""
    params = {"x": 1.0}
    fp_a = parameter_fingerprint(
        "PreprocessingParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
    )
    fp_b = parameter_fingerprint(
        "ArtifactDetectionParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
    )
    assert fp_a != fp_b


def test_fingerprint_includes_job_kwargs():
    """Same params, different job_kwargs -> different fingerprint."""
    params = {"x": 1.0}
    fp_none = parameter_fingerprint(
        "SorterParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
        sorter="mountainsort4",
    )
    fp_jk = parameter_fingerprint(
        "SorterParameters",
        params=params,
        params_schema_version=1,
        job_kwargs={"n_jobs": 4},
        sorter="mountainsort4",
    )
    assert fp_none != fp_jk


def test_schema_version_is_part_of_identity():
    """The schema version distinguishes otherwise-identical blobs."""
    params = {"x": 1.0}
    fp_v1 = parameter_fingerprint(
        "PreprocessingParameters",
        params=params,
        params_schema_version=1,
        job_kwargs=None,
    )
    fp_v3 = parameter_fingerprint(
        "PreprocessingParameters",
        params=params,
        params_schema_version=3,
        job_kwargs=None,
    )
    assert fp_v1 != fp_v3


def test_short_fingerprint_is_a_prefix():
    """The display hash is a deterministic prefix of the full digest."""
    fp = parameter_fingerprint(
        "PreprocessingParameters",
        params={"x": 1.0},
        params_schema_version=1,
        job_kwargs=None,
    )
    short = short_fingerprint(fp)
    assert fp.startswith(short)
    assert len(short) == 12


def test_shipped_recipe_fingerprints_are_locked():
    """The June 2026 production recipes have frozen content fingerprints.

    A recipe change must ship under a new dated name so name-derived
    selection ids stay reproducible.
    """
    from tests.spikesorting.v2 import _recipe_constants as rc

    specs = [
        (
            "PreprocessingParameters",
            rc.FRANKLAB_HIPPOCAMPUS_2026_06,
            rc.FRANKLAB_HIPPOCAMPUS_2026_06_PARAMS,
            4,
            None,
        ),
        (
            "PreprocessingParameters",
            rc.FRANKLAB_CORTEX_2026_06,
            rc.FRANKLAB_CORTEX_2026_06_PARAMS,
            4,
            None,
        ),
        (
            "ArtifactDetectionParameters",
            rc.FRANKLAB_100UV_P07_2026_06,
            rc.FRANKLAB_100UV_P07_2026_06_PARAMS,
            2,
            None,
        ),
        (
            "ArtifactDetectionParameters",
            rc.FRANKLAB_50UV_P07_2026_06,
            rc.FRANKLAB_50UV_P07_2026_06_PARAMS,
            2,
            None,
        ),
        (
            "SorterParameters",
            rc.FRANKLAB_30KHZ_MS4_2026_06,
            rc.FRANKLAB_30KHZ_MS4_2026_06_PARAMS,
            1,
            "mountainsort4",
        ),
        (
            "SorterParameters",
            rc.FRANKLAB_20KHZ_MS4_2026_06,
            rc.FRANKLAB_20KHZ_MS4_2026_06_PARAMS,
            1,
            "mountainsort4",
        ),
    ]
    expected = {
        rc.FRANKLAB_HIPPOCAMPUS_2026_06: "a23acc3c06a9",
        rc.FRANKLAB_CORTEX_2026_06: "2872d9dcb497",
        rc.FRANKLAB_100UV_P07_2026_06: "6e6ddbc7e5b7",
        rc.FRANKLAB_50UV_P07_2026_06: "a4a9158675e8",
        rc.FRANKLAB_30KHZ_MS4_2026_06: "7bd8d640c764",
        rc.FRANKLAB_20KHZ_MS4_2026_06: "6ae617642d8a",
    }
    for table, name, params, version, sorter in specs:
        got = short_fingerprint(
            parameter_fingerprint(
                table,
                params=params,
                params_schema_version=version,
                job_kwargs=None,
                sorter=sorter,
            )
        )
        assert got == expected[name], f"{name}: {got} != {expected[name]}"


def test_int_and_float_param_values_share_a_fingerprint():
    """``9`` and ``9.0`` must produce the same content fingerprint.

    ``extra="allow"`` sorter schemas pass user keys through uncoerced.
    """
    common = dict(params_schema_version=1, job_kwargs=None, sorter="kilosort4")
    fp_int = parameter_fingerprint(
        "SorterParameters", params={"batch_size": 60000}, **common
    )
    fp_float = parameter_fingerprint(
        "SorterParameters", params={"batch_size": 60000.0}, **common
    )
    assert fp_int == fp_float


def test_numeric_normalization_is_recursive():
    """Int/float equivalence holds inside nested dicts and lists."""
    common = dict(params_schema_version=1, job_kwargs=None)
    fp_floats = parameter_fingerprint(
        "PreprocessingParameters",
        params={"a": {"b": 5.0}, "c": [1.0, 2.0]},
        **common,
    )
    fp_ints = parameter_fingerprint(
        "PreprocessingParameters",
        params={"a": {"b": 5}, "c": [1, 2]},
        **common,
    )
    assert fp_floats == fp_ints


def test_bool_is_not_collapsed_to_int_in_fingerprint():
    """``True`` and ``1`` fingerprint differently."""
    common = dict(params_schema_version=1, job_kwargs=None, sorter="kilosort4")
    fp_true = parameter_fingerprint(
        "SorterParameters", params={"flag": True}, **common
    )
    fp_one = parameter_fingerprint(
        "SorterParameters", params={"flag": 1}, **common
    )
    assert fp_true != fp_one


def test_nan_or_inf_param_is_rejected_not_silently_encoded():
    """A NaN/Inf param value raises at fingerprint time rather than emitting
    the invalid-JSON tokens ``NaN``/``Infinity`` no strict reader accepts."""
    import math

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            parameter_fingerprint(
                "SorterParameters",
                params={"some_kwarg": bad},
                params_schema_version=1,
                job_kwargs=None,
                sorter="kilosort4",
            )


# ---------------------------------------------------------------------------
# QualityMetricParameters duplicate-content guard (content spans columns).
# ---------------------------------------------------------------------------


def _qmp_row(name, *, metric_names, **overrides):
    row = {
        "metric_params_name": name,
        "metric_names": metric_names,
        "metric_kwargs": {},
        "observed_presence_bin_duration_s": 60.0,
        "template_metric_columns": [],
        "skip_pc_metrics": True,
        "params_schema_version": 1,
        "job_kwargs": None,
    }
    row.update(overrides)
    return row


def test_quality_metric_duplicate_content_rejected():
    """A second NAME for identical QMP content forks provenance -> rejected."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_duplicate_quality_metric_content,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    stored = [_qmp_row("preset_a", metric_names=["snr", "isi_violation"])]
    incoming = [_qmp_row("preset_b", metric_names=["snr", "isi_violation"])]
    with pytest.raises(DuplicateParameterContentError, match="preset_a"):
        reject_duplicate_quality_metric_content(stored, incoming)


def test_quality_metric_same_name_reinsert_is_idempotent():
    """Re-inserting the same name+content never trips the guard."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_duplicate_quality_metric_content,
    )

    stored = [_qmp_row("preset_a", metric_names=["snr"])]
    reject_duplicate_quality_metric_content(
        stored, [_qmp_row("preset_a", metric_names=["snr"])]
    )  # no raise


def test_quality_metric_different_content_allowed():
    """A new name with DIFFERENT content is fine."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_duplicate_quality_metric_content,
    )

    stored = [_qmp_row("preset_a", metric_names=["snr"])]
    incoming = [_qmp_row("preset_b", metric_names=["snr", "firing_rate"])]
    reject_duplicate_quality_metric_content(stored, incoming)  # no raise


def test_quality_metric_duplicate_escape_hatch():
    """``allow_duplicate_params=True`` opts out of the guard."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_duplicate_quality_metric_content,
    )

    stored = [_qmp_row("preset_a", metric_names=["snr"])]
    incoming = [_qmp_row("preset_b", metric_names=["snr"])]
    reject_duplicate_quality_metric_content(
        stored, incoming, allow_duplicate_params=True
    )  # no raise


def test_reject_stale_quality_metric_defaults_flags_edited_row():
    """A stored row under a shipped default's name but different content raises."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_stale_quality_metric_defaults,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    stored = [_qmp_row("franklab_default", metric_names=["snr"])]
    shipped = [
        _qmp_row("franklab_default", metric_names=["snr", "isi_violation"])
    ]
    with pytest.raises(
        DuplicateParameterContentError, match="franklab_default"
    ):
        reject_stale_quality_metric_defaults(stored, shipped)


def test_reject_stale_quality_metric_defaults_matches_identical_content():
    """A stored row identical to the shipped default is not stale."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_stale_quality_metric_defaults,
    )

    stored = [_qmp_row("franklab_default", metric_names=["snr"])]
    shipped = [_qmp_row("franklab_default", metric_names=["snr"])]
    reject_stale_quality_metric_defaults(stored, shipped)  # no raise


@pytest.mark.parametrize(
    "stored_overrides, shipped_overrides",
    [
        ({"params_schema_version": 1}, {"params_schema_version": 2}),
        (
            {"observed_presence_bin_duration_s": float(np.float32(1 / 3))},
            {"observed_presence_bin_duration_s": 1 / 3},
        ),
    ],
    ids=["schema_version", "float32_bin_duration"],
)
def test_reject_stale_quality_metric_defaults_compares_exactly(
    stored_overrides, shipped_overrides
):
    """A different ``params_schema_version`` or a float32-rounded bin duration
    under a shipped default's name is stale content, not a match."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_stale_quality_metric_defaults,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )

    stored = [
        _qmp_row("franklab_default", metric_names=["snr"], **stored_overrides)
    ]
    shipped = [
        _qmp_row("franklab_default", metric_names=["snr"], **shipped_overrides)
    ]
    with pytest.raises(
        DuplicateParameterContentError, match="franklab_default"
    ):
        reject_stale_quality_metric_defaults(stored, shipped)


def test_reject_stale_quality_metric_defaults_ignores_other_names():
    """A stored row under a name outside the shipped catalog is untouched."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        reject_stale_quality_metric_defaults,
    )

    stored = [_qmp_row("my_custom_recipe", metric_names=["snr"])]
    shipped = [
        _qmp_row("franklab_default", metric_names=["snr", "isi_violation"])
    ]
    reject_stale_quality_metric_defaults(stored, shipped)  # no raise


# ---------------------------------------------------------------------------
# Duplicate-content guard + describe_parameter_rows (DB-backed).
# ---------------------------------------------------------------------------


@pytest.mark.database
def test_quality_metric_insert_rejects_duplicate_content(dj_conn):
    """End-to-end QMP dedup: a new name with content identical to a shipped row
    is rejected; the intentional franklab/neuropixels duplicate in
    insert_default still succeeds via the escape hatch."""
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    # franklab_default and neuropixels_default share content on purpose.
    QualityMetricParameters.insert_default()
    shipped = (
        QualityMetricParameters & {"metric_params_name": "minimal"}
    ).fetch1()

    dup = {
        "metric_params_name": "minimal_copy",
        "metric_names": shipped["metric_names"],
        "metric_kwargs": shipped["metric_kwargs"],
        "template_metric_columns": shipped["template_metric_columns"],
        "skip_pc_metrics": shipped["skip_pc_metrics"],
    }
    key = {"metric_params_name": "minimal_copy"}
    (QualityMetricParameters & key).delete_quick()
    try:
        with pytest.raises(
            DuplicateParameterContentError, match="duplicates the content"
        ):
            QualityMetricParameters().insert(dup)
        QualityMetricParameters().insert(dup, allow_duplicate_params=True)
        assert QualityMetricParameters & key
    finally:
        (QualityMetricParameters & key).delete_quick()


@pytest.mark.database
def test_qmp_duplicate_content_detected_after_double_column(dj_conn):
    """A fractional ``observed_presence_bin_duration_s`` that needs more than
    single precision still fingerprints identically once fetched back, so a
    second name for the same content is caught.

    The fingerprint hashes the stored value exactly; a float32 column
    would read ``1 / 3`` back as ``0.3333333432674408`` and miss the match.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    first = {
        "metric_params_name": "bin_duration_fraction_a",
        "metric_names": ["snr"],
        "metric_kwargs": {"snr": {"peak_sign": "neg"}},
        "template_metric_columns": [],
        "skip_pc_metrics": True,
        "observed_presence_bin_duration_s": 1 / 3,
    }
    dup = {**first, "metric_params_name": "bin_duration_fraction_b"}
    keys = [
        {"metric_params_name": first["metric_params_name"]},
        {"metric_params_name": dup["metric_params_name"]},
    ]
    (QualityMetricParameters & keys).delete_quick()
    try:
        QualityMetricParameters().insert(first)
        with pytest.raises(
            DuplicateParameterContentError, match="duplicates the content"
        ):
            QualityMetricParameters().insert(dup)
    finally:
        (QualityMetricParameters & keys).delete_quick()


@pytest.mark.database
def test_insert_default_is_idempotent(dj_conn):
    """Two consecutive calls succeed and leave exactly one shipped row."""
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    QualityMetricParameters.insert_default()
    QualityMetricParameters.insert_default()
    assert (
        len(
            QualityMetricParameters & {"metric_params_name": "franklab_default"}
        )
        == 1
    )


@pytest.mark.database
def test_insert_default_raises_on_stale_same_name_row(dj_conn):
    """A stored ``franklab_default`` whose content differs from the shipped
    default makes ``insert_default`` raise."""
    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    QualityMetricParameters.insert_default()
    original = (
        QualityMetricParameters & {"metric_params_name": "franklab_default"}
    ).fetch1()
    edited = dict(original, metric_kwargs={"snr": {"peak_sign": "pos"}})
    free_table = dj.FreeTable(
        dj.conn(), QualityMetricParameters.full_table_name
    )
    try:
        free_table.update1(edited)
        with pytest.raises(
            DuplicateParameterContentError, match="franklab_default"
        ):
            QualityMetricParameters.insert_default()
    finally:
        free_table.update1(original)
    assert (
        QualityMetricParameters & {"metric_params_name": "franklab_default"}
    ).fetch1("metric_kwargs") == original["metric_kwargs"]


@pytest.mark.database
def test_insert_default_rejects_other_schema_version_row(dj_conn):
    """A stored row identical in content but at another
    ``params_schema_version`` is stale: ``insert_default`` raises instead of
    keeping it under the shipped name."""
    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )

    QualityMetricParameters.insert_default()
    original = (
        QualityMetricParameters & {"metric_params_name": "franklab_default"}
    ).fetch1()
    downgraded = dict(
        original, params_schema_version=original["params_schema_version"] - 1
    )
    free_table = dj.FreeTable(
        dj.conn(), QualityMetricParameters.full_table_name
    )
    try:
        free_table.update1(downgraded)
        with pytest.raises(
            DuplicateParameterContentError, match="franklab_default"
        ):
            QualityMetricParameters.insert_default()
    finally:
        free_table.update1(original)


@pytest.mark.database
def test_duplicate_parameter_content_rejected(dj_conn):
    """A second name for an existing preproc blob is rejected by default."""
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    shipped = (
        PreprocessingParameters & {"preprocessing_params_name": "default"}
    ).fetch1()

    with pytest.raises(
        DuplicateParameterContentError, match="duplicates the content"
    ):
        PreprocessingParameters.insert1(
            {
                "preprocessing_params_name": "my_copy_of_default",
                "params": shipped["params"],
                "params_schema_version": shipped["params_schema_version"],
                "job_kwargs": shipped["job_kwargs"],
            }
        )


@pytest.mark.database
def test_duplicate_parameter_content_escape_hatch(dj_conn):
    """``allow_duplicate_params=True`` inserts the dup and marks duplicate_of."""
    from spyglass.spikesorting.v2.pipeline import describe_parameter_rows
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    shipped = (
        PreprocessingParameters & {"preprocessing_params_name": "default"}
    ).fetch1()
    dup_name = "my_copy_of_default"
    try:
        PreprocessingParameters.insert1(
            {
                "preprocessing_params_name": dup_name,
                "params": shipped["params"],
                "params_schema_version": shipped["params_schema_version"],
                "job_kwargs": shipped["job_kwargs"],
            },
            allow_duplicate_params=True,
        )
        assert PreprocessingParameters & {"preprocessing_params_name": dup_name}
        df = describe_parameter_rows()
        dup = df.loc[df["parameter_name"] == dup_name].iloc[0]
        assert dup["duplicate_of"] == "default"
        assert "duplicate content" in dup["name_warnings"]
    finally:
        (
            PreprocessingParameters & {"preprocessing_params_name": dup_name}
        ).delete(safemode=False)


@pytest.mark.database
def test_sorter_duplicate_rejected_scoped_by_sorter(dj_conn):
    """A second name for an existing sorter blob (same sorter) is rejected.

    Uses the always-available ``clusterless_thresholder`` row so the test does
    not depend on an SI sorter binary being installed.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    shipped = (
        SorterParameters
        & {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        }
    ).fetch1()

    with pytest.raises(
        DuplicateParameterContentError, match="duplicates the content"
    ):
        SorterParameters.insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": "my_clusterless_copy",
                "params": shipped["params"],
                "params_schema_version": shipped["params_schema_version"],
                "job_kwargs": shipped["job_kwargs"],
            }
        )


@pytest.mark.database
def test_params_lookup_rejects_replace(dj_conn):
    """``insert``/``insert1`` with ``replace=True`` raise on every
    ``ImmutableParamsLookup`` subclass; the stored row is unchanged.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.metric_curation import (
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import SorterParameters

    QualityMetricParameters.insert_default()
    qmp_key = {"metric_params_name": "franklab_default"}
    qmp_before = (QualityMetricParameters & qmp_key).fetch1()
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        QualityMetricParameters().insert(dict(qmp_before), replace=True)
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        QualityMetricParameters().insert1(dict(qmp_before), replace=True)
    assert (QualityMetricParameters & qmp_key).fetch1() == qmp_before

    SorterParameters.insert_default()
    sorter_key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }
    sorter_before = (SorterParameters & sorter_key).fetch1()
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        SorterParameters().insert([dict(sorter_before)], replace=True)
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        SorterParameters().insert1(dict(sorter_before), replace=True)
    assert (SorterParameters & sorter_key).fetch1() == sorter_before


@pytest.mark.database
def test_params_lookup_rejects_positional_replace(dj_conn):
    """A positional ``replace`` is rejected like ``replace=True``.

    Uses ``MotionCorrectionParameters``, which has no ``insert`` override.
    Later positional flags keep their meaning.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.motion import MotionCorrectionParameters

    MotionCorrectionParameters.insert_default()
    key = {"motion_correction_params_name": "dredge_v1"}
    before = (MotionCorrectionParameters & key).fetch1()
    overwrite = {**before, "motion_estimation_params_name": "dredge_fast_v1"}
    assert overwrite != before
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        MotionCorrectionParameters().insert([overwrite], True)
    assert (MotionCorrectionParameters & key).fetch1() == before

    MotionCorrectionParameters().insert([dict(before)], False, True)
    assert (MotionCorrectionParameters & key).fetch1() == before


@pytest.mark.database
def test_describe_parameter_rows_columns_and_usage(dj_conn):
    """Documented columns + correct ``used_by_pipeline_presets`` per row."""
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2._orchestration.reporting import (
        _PARAMETER_ROW_COLUMNS,
        describe_parameter_rows,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()

    df = describe_parameter_rows()
    assert list(df.columns) == _PARAMETER_ROW_COLUMNS

    def _cell(table, name, col):
        sub = df[(df["table"] == table) & (df["parameter_name"] == name)]
        assert len(sub) == 1, f"{table}/{name}: expected one row"
        return sub.iloc[0][col]

    # The 100 uV production artifact row is bundled by every franklab MS4/MS5
    # preset; the 50 uV row ships but no preset uses it.
    used_100 = _cell(
        "ArtifactDetectionParameters",
        "franklab_100uv_p07_2026_06",
        "used_by_pipeline_presets",
    )
    assert "franklab_tetrode_hippocampus_30khz_ms4_2026_06" in used_100
    assert (
        _cell(
            "ArtifactDetectionParameters",
            "franklab_50uv_p07_2026_06",
            "used_by_pipeline_presets",
        )
        == []
    )

    # The clusterless preset is the only one on the generic 500 uV "default"
    # artifact + "default" preproc rows.
    assert _cell(
        "ArtifactDetectionParameters", "default", "used_by_pipeline_presets"
    ) == ["franklab_clusterless_2026_06"]
    assert _cell(
        "PreprocessingParameters", "default", "used_by_pipeline_presets"
    ) == ["franklab_clusterless_2026_06"]

    # The hippocampus region preproc row is a shipped catalog default.
    assert bool(
        _cell(
            "PreprocessingParameters",
            "franklab_hippocampus_2026_06",
            "is_shipped_default",
        )
    )

    # Every preset scores with a metric row and an auto-curation rule row.
    QualityMetricParameters.insert_default()
    AutoCurationRules.insert_default()
    df = describe_parameter_rows()
    assert "franklab_tetrode_hippocampus_30khz_ms4_2026_06" in _cell(
        "QualityMetricParameters",
        "franklab_default",
        "used_by_pipeline_presets",
    )
    assert _cell("AutoCurationRules", "none", "used_by_pipeline_presets") == [
        "franklab_clusterless_2026_06"
    ]


def test_describe_parameter_rows_covers_all_seeded_tables(dj_conn):
    """The report lists every parameter Lookup ``initialize_v2_defaults`` seeds.

    All ten seeded tables appear, including those no preset references.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.reporting import (
        describe_parameter_rows,
    )

    initialize_v2_defaults()
    tables = set(describe_parameter_rows()["table"])
    assert {
        "PreprocessingParameters",
        "ArtifactDetectionParameters",
        "SorterParameters",
        "AnalyzerWaveformParameters",
        "QualityMetricParameters",
        "AutoCurationRules",
        "MatcherParameters",
        "MotionEstimationParameters",
        "MotionInterpolationParameters",
        "MotionCorrectionParameters",
    } <= tables


def test_describe_parameter_rows_marks_dynamic_default_shipped(dj_conn):
    """Tables with code-built defaults report shipped 'franklab' rows as
    ``is_shipped_default=True``, with no 'non-catalog row' warning.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.reporting import (
        describe_parameter_rows,
    )

    initialize_v2_defaults()
    df = describe_parameter_rows()
    row = df[
        (df["table"] == "QualityMetricParameters")
        & (df["parameter_name"] == "franklab_default")
    ]
    assert len(row) == 1
    assert bool(row.iloc[0]["is_shipped_default"]) is True
    assert "non-catalog" not in (row.iloc[0]["name_warnings"] or "")

    matcher = df[
        (df["table"] == "MatcherParameters")
        & (df["parameter_name"] == "unitmatch_default")
    ]
    assert len(matcher) == 1
    assert bool(matcher.iloc[0]["is_shipped_default"]) is True


def test_describe_parameter_rows_autocuration_duplicate_name_insensitive(
    dj_conn,
):
    """Two AutoCurationRules rule sets with IDENTICAL Rule rows under different
    names share a fingerprint (the name FK is excluded from the Rule content) and
    surface as duplicate_of each other.
    """
    from spyglass.spikesorting.v2._orchestration.reporting import (
        describe_parameter_rows,
    )
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    rules = [
        {
            "rule_index": 0,
            "rule_name": "nn_noise",
            "metric_name": "nn_noise_overlap",
            "operator": ">",
            "threshold": 0.1,
            "label": "noise",
        },
    ]
    name_a, name_b = "audit_rules_dup_a", "audit_rules_dup_b"
    keys = [
        {"auto_curation_rules_name": name_a},
        {"auto_curation_rules_name": name_b},
    ]
    try:
        AutoCurationRules.insert_rules(
            {**keys[0], "auto_merge_preset": "none"}, [dict(r) for r in rules]
        )
        AutoCurationRules.insert_rules(
            {**keys[1], "auto_merge_preset": "none"}, [dict(r) for r in rules]
        )
        sub = describe_parameter_rows().pipe(
            lambda df: df[df["table"] == "AutoCurationRules"]
        )
        a = sub[sub["parameter_name"] == name_a].iloc[0]
        b = sub[sub["parameter_name"] == name_b].iloc[0]
        assert a["duplicate_of"] == name_b
        assert b["duplicate_of"] == name_a
    finally:
        (AutoCurationRules & keys).delete(safemode=False)


@pytest.mark.database
def test_rule_missing_policy_persists_and_changes_identity(dj_conn):
    """Policy round-trips and is semantic in idempotency/reporting identity."""
    from spyglass.spikesorting.v2._orchestration.reporting import (
        describe_parameter_rows,
    )
    from spyglass.spikesorting.v2.metric_curation import AutoCurationRules

    names = ["missing_policy_error", "missing_policy_pass"]
    keys = [{"auto_curation_rules_name": name} for name in names]

    def rule(policy):
        return {
            "rule_index": 0,
            "rule_name": "snr_noise",
            "metric_name": "snr",
            "operator": "<",
            "threshold": 2.0,
            "label": "noise",
            "missing_policy": policy,
        }

    (AutoCurationRules & keys).delete(safemode=False)
    try:
        AutoCurationRules.insert_rules(
            {**keys[0], "auto_merge_preset": "none"}, [rule("error")]
        )
        AutoCurationRules.insert_rules(
            {**keys[1], "auto_merge_preset": "none"}, [rule("pass")]
        )
        assert (AutoCurationRules.Rule & keys[0]).fetch1(
            "missing_policy"
        ) == "error"
        assert (AutoCurationRules.Rule & keys[1]).fetch1(
            "missing_policy"
        ) == "pass"
        with pytest.raises(
            ValueError, match="different auto-merge/rule payload"
        ):
            AutoCurationRules.insert_rules(
                {**keys[0], "auto_merge_preset": "none"}, [rule("pass")]
            )

        rows = describe_parameter_rows()
        rows = rows[
            (rows["table"] == "AutoCurationRules")
            & rows["parameter_name"].isin(names)
        ].set_index("parameter_name")
        assert (
            rows.loc[names[0], "fingerprint"]
            != rows.loc[names[1], "fingerprint"]
        )
        assert not rows.loc[names[0], "duplicate_of"]
        assert not rows.loc[names[1], "duplicate_of"]
    finally:
        (AutoCurationRules & keys).delete(safemode=False)


@pytest.mark.database
def test_within_batch_duplicate_content_rejected(dj_conn):
    """Two same-content/different-name rows in ONE insert() call collide.

    Rejected even when neither row is stored yet.
    """
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    # freq_min=777 matches no shipped row.
    blob = PreprocessingParamsSchema.model_validate(
        {"bandpass_filter": {"freq_min": 777.0, "freq_max": 6000.0}}
    ).model_dump()
    rows = [
        {
            "preprocessing_params_name": "batch_dup_a",
            "params": blob,
            "params_schema_version": 4,
            "job_kwargs": None,
        },
        {
            "preprocessing_params_name": "batch_dup_b",
            "params": blob,
            "params_schema_version": 4,
            "job_kwargs": None,
        },
    ]
    # Raises before super().insert, so neither row lands (no cleanup needed).
    with pytest.raises(DuplicateParameterContentError, match="batch_dup_a"):
        PreprocessingParameters.insert(rows)


@pytest.mark.database
def test_duplicate_rejected_when_schema_version_column_omitted(dj_conn):
    """A duplicate that OMITS params_schema_version is still caught.

    The guard reads the version from the blob's ``schema_version``.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    shipped = (
        PreprocessingParameters & {"preprocessing_params_name": "default"}
    ).fetch1()
    with pytest.raises(DuplicateParameterContentError, match="default"):
        PreprocessingParameters.insert1(
            {
                "preprocessing_params_name": "omit_version_copy",
                "params": shipped["params"],
                # params_schema_version deliberately omitted
                "job_kwargs": shipped["job_kwargs"],
            }
        )


# ---- generalized outer params_schema_version backfill -----------------------

_VALIDATED_LOOKUP_SCHEMAS = [
    ("preprocessing", "PreprocessingParamsSchema"),
    ("artifact_detection", "ArtifactDetectionParamsSchema"),
    ("analyzer_waveform", "AnalyzerWaveformParamsSchema"),
    ("matcher", "UnitMatchParamsSchema"),
    ("sorter", "GenericSorterParamsSchema"),
]


@pytest.mark.parametrize("module_name, schema_name", _VALIDATED_LOOKUP_SCHEMAS)
def test_outer_version_backfilled_for_all_lookups(module_name, schema_name):
    """``validate_lookup_rows`` fills an omitted ``params_schema_version``
    from the blob for every Lookup that uses it.
    """
    import importlib

    from spyglass.spikesorting.v2._core.lookup_validation import (
        validate_lookup_rows,
    )

    schema = getattr(
        importlib.import_module(
            f"spyglass.spikesorting.v2._params.{module_name}"
        ),
        schema_name,
    )
    current = schema.model_fields["schema_version"].default
    row = {"params": {"schema_version": current}}
    validated = validate_lookup_rows(
        [row],
        ["name", "params", "params_schema_version", "job_kwargs"],
        schema_for=lambda _row: schema,
        table_name="BackfillProbe",
    )
    assert validated[0]["params_schema_version"] == current


def test_outer_version_drift_still_trips_after_backfill():
    """An explicit ``params_schema_version`` that disagrees with the blob
    still raises."""
    from spyglass.spikesorting.v2._core.lookup_validation import (
        validate_lookup_rows,
    )
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )

    current = PreprocessingParamsSchema().schema_version
    row = {
        "params": {"schema_version": current},
        "params_schema_version": current - 1,
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_lookup_rows(
            [row],
            ["name", "params", "params_schema_version", "job_kwargs"],
            schema_for=lambda _row: PreprocessingParamsSchema,
            table_name="BackfillProbe",
        )
