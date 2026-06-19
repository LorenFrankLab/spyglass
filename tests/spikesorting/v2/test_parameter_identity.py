"""DB-free tests for parameter-content fingerprints (``_parameter_identity``).

The fingerprint identifies a parameter-Lookup row by its *content* (table,
sorter context, schema version, validated ``params`` blob, and
``job_kwargs``) with the row NAME excluded -- so two rows with identical
content under different names share a fingerprint. That is the signal the
duplicate-content guard and ``describe_parameter_rows`` rely on.

These tests are pure-stdlib and must not need a database.
"""

from __future__ import annotations

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
