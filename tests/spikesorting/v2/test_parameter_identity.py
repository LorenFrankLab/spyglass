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
    expected = {
        rc.FRANKLAB_HIPPOCAMPUS_2026_06: "65a601c2cbe6",
        rc.FRANKLAB_CORTEX_2026_06: "154a714d0114",
        rc.FRANKLAB_100UV_P07_2026_06: "596d7894f2d0",
        rc.FRANKLAB_50UV_P07_2026_06: "f26f082f82ea",
        rc.FRANKLAB_30KHZ_MS4_2026_06: "a82767b225c3",
        rc.FRANKLAB_20KHZ_MS4_2026_06: "0f29e8800e84",
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
    from spyglass.spikesorting.v2.pipeline import (
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
