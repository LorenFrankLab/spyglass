"""DB-free tests for parameter-content fingerprints (``_parameter_identity``).

The fingerprint identifies a parameter-Lookup row by its *content* (table,
sorter context, schema version, validated ``params`` blob, and
``job_kwargs``) with the row NAME excluded -- so two rows with identical
content under different names share a fingerprint. That is the signal the
duplicate-content guard and ``describe_parameter_rows`` rely on.

These tests are pure-stdlib and must not need a database.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._parameter_identity import (
    parameter_fingerprint,
    short_fingerprint,
)


def test_fingerprint_is_stable_and_dict_order_independent():
    """Same content in any key order yields the same 64-char hex digest."""
    fp1 = parameter_fingerprint(
        "PreprocessingParameters",
        params={"a": 1, "b": {"x": 1.0, "y": 2.0}},
        params_schema_version=3,
        job_kwargs=None,
    )
    fp2 = parameter_fingerprint(
        "PreprocessingParameters",
        params={"b": {"y": 2.0, "x": 1.0}, "a": 1},
        params_schema_version=3,
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


def test_fingerprint_excludes_row_name():
    """The function takes no name; identical content fingerprints identically.

    This is the whole point: a row called ``franklab_cortex_2026_06`` and a
    duplicate called ``my_copy`` with the same blob collide, which is what
    the duplicate-content guard detects.
    """
    content = dict(
        params={"freq_min": 600.0}, params_schema_version=3, job_kwargs=None
    )
    assert parameter_fingerprint(
        "PreprocessingParameters", **content
    ) == parameter_fingerprint("PreprocessingParameters", **content)


def test_fingerprint_includes_sorter_context():
    """Same params under different sorters -> different fingerprint.

    SorterParameters duplicate detection is scoped per sorter, so the sorter
    must be part of the content identity.
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

    Pinning the fingerprint of each dated recipe (computed DB-free from the
    independent ``_recipe_constants`` literals) makes an in-place edit to a
    shipped recipe fail loudly: a recipe change must ship under a NEW dated
    name so a name-derived selection id stays reproducible, never mutate an
    existing one. The parity tests separately tie these constants to the
    shipped ``_DEFAULT_CONTENTS`` blobs, so the pair locks both the canonical
    recipe and the rows that ship it.
    """
    from tests.spikesorting.v2 import _recipe_constants as rc

    specs = [
        (
            "PreprocessingParameters",
            rc.FRANKLAB_HIPPOCAMPUS_2026_06,
            rc.FRANKLAB_HIPPOCAMPUS_2026_06_PARAMS,
            3,
            None,
        ),
        (
            "PreprocessingParameters",
            rc.FRANKLAB_CORTEX_2026_06,
            rc.FRANKLAB_CORTEX_2026_06_PARAMS,
            3,
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
    # These frozen digests were regenerated when the canonical form gained
    # numeric normalization (int-valued floats collapse to ints so 9 and 9.0
    # fingerprint alike) + ``allow_nan=False``. The lock still serves its
    # purpose: an in-place edit to a shipped recipe changes the recipe content
    # and so the digest, failing here.
    expected = {
        rc.FRANKLAB_HIPPOCAMPUS_2026_06: "370e76eab686",
        rc.FRANKLAB_CORTEX_2026_06: "9c3db4b5b525",
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

    The ``extra="allow"`` sorter schemas (Kilosort4, SpykingCircus2,
    Tridesclous2, generic) pass user keys through uncoerced, so a blob may
    carry ``60000`` under one name and ``60000.0`` under another for identical
    science. A plain ``json.dumps`` renders those differently and forks the
    fingerprint, defeating the duplicate-content guard. The canonical form
    normalizes int-valued floats so the guard still fires.
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
    """``True`` and ``1`` stay distinct -- different JSON types, different
    intent (a boolean toggle is not the integer 1)."""
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
# Duplicate-content guard + describe_parameter_rows (DB-backed). Each imports
# the v2 schema modules lazily and takes ``dj_conn`` so the schema's
# ``dj.schema(...)`` import has a live connection.
# ---------------------------------------------------------------------------


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
        assert PreprocessingParameters & {
            "preprocessing_params_name": dup_name
        }
        df = describe_parameter_rows()
        dup = df.loc[df["parameter_name"] == dup_name].iloc[0]
        assert dup["duplicate_of"] == "default"
        assert "duplicate content" in dup["name_warnings"]
    finally:
        (
            PreprocessingParameters
            & {"preprocessing_params_name": dup_name}
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
def test_describe_parameter_rows_columns_and_usage(dj_conn):
    """Documented columns + correct ``used_by_pipeline_presets`` per row."""
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2._pipeline_reporting import (
        _PARAMETER_ROW_COLUMNS,
        describe_parameter_rows,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()

    df = describe_parameter_rows()
    assert list(df.columns) == _PARAMETER_ROW_COLUMNS

    def _cell(table, name, col):
        sub = df[
            (df["table"] == table) & (df["parameter_name"] == name)
        ]
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


def test_describe_parameter_rows_covers_all_seeded_tables(dj_conn):
    """The report lists every parameter Lookup ``initialize_v2_defaults`` seeds.

    Pins the report against operational drift: it must cover all EIGHT seeded
    parameter tables, not just the three preset-referenced ones. The five
    downstream / cross-session tables carry blank preset-fold columns but still
    appear so a user can audit every row they can populate.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_reporting import (
        describe_parameter_rows,
    )

    initialize_v2_defaults()
    tables = set(describe_parameter_rows()["table"])
    assert {
        "PreprocessingParameters",
        "ArtifactDetectionParameters",
        "SorterParameters",
        "AnalyzerWaveformParameters",
        "MotionCorrectionParameters",
        "QualityMetricParameters",
        "AutoCurationRules",
        "MatcherParameters",
    } <= tables


@pytest.mark.database
def test_within_batch_duplicate_content_rejected(dj_conn):
    """Two same-content/different-name rows in ONE insert() call collide.

    Distinct from the incoming-vs-stored path: the guard also tracks the
    fingerprints accumulated WITHIN the batch, so a single bulk insert of two
    identical-content rows under different names is rejected even when neither
    row exists in the table yet.
    """
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    # Content that matches no shipped row (freq_min=777), so the only possible
    # collision is between the two batch rows themselves.
    blob = PreprocessingParamsSchema.model_validate(
        {"bandpass_filter": {"freq_min": 777.0, "freq_max": 6000.0}}
    ).model_dump()
    rows = [
        {
            "preprocessing_params_name": "batch_dup_a",
            "params": blob,
            "params_schema_version": 3,
            "job_kwargs": None,
        },
        {
            "preprocessing_params_name": "batch_dup_b",
            "params": blob,
            "params_schema_version": 3,
            "job_kwargs": None,
        },
    ]
    # Raises before super().insert, so neither row lands (no cleanup needed).
    with pytest.raises(DuplicateParameterContentError, match="batch_dup_a"):
        PreprocessingParameters.insert(rows)


@pytest.mark.database
def test_duplicate_rejected_when_schema_version_column_omitted(dj_conn):
    """A duplicate that OMITS params_schema_version is still caught.

    Preproc / Artifact dict inserts may omit the ``params_schema_version``
    column (the DataJoint column default fills it). The guard then resolves the
    version from the validated blob's inner ``schema_version``, so the
    fingerprint still matches the stored row and the duplicate is rejected --
    exercising the column-omitted fallback that a KeyError would otherwise hit.
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
