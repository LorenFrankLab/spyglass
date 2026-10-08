"""Parameter-batch insertion contracts without a database connection."""

import copy
from types import SimpleNamespace

import datajoint as dj
import pytest
from pydantic import BaseModel, Field, ValidationError

from spyglass.spikesorting.v2.exceptions import DuplicateParameterContentError
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    _insert_parameter_rows,
)
from spyglass.spikesorting.v2._core.lookup_validation import (
    validate_lookup_rows,
)

pytestmark = pytest.mark.unit


class _Params(BaseModel):
    schema_version: int = 1
    threshold: float = Field(gt=0)


class _InsertSink:
    def insert(self, rows, replace=False, *args, **kwargs):
        self.writes.append((copy.deepcopy(rows), replace, args, kwargs))


class _Table(ImmutableParamsLookup, _InsertSink):
    heading = SimpleNamespace(
        names=["params_name", "params", "params_schema_version", "job_kwargs"]
    )

    def __init__(self, stored=()):
        self.stored = list(stored)
        self.writes = []

    def fetch(self, *, as_dict):
        assert as_dict
        return copy.deepcopy(self.stored)


def _row(name, threshold=1, **extra):
    return {"params_name": name, "params": {"threshold": threshold}, **extra}


def _insert(table, rows, **kwargs):
    _insert_parameter_rows(
        table,
        rows,
        insert_rows=table.insert,
        table_name="TestParameters",
        name_attr="params_name",
        schema_for=lambda _row: _Params,
        **kwargs,
    )


def test_generator_and_positional_rows_normalize_before_one_insert():
    table = _Table()
    original = _row("dict", "2")
    positional = ("tuple", {"threshold": "3"}, 0, None)
    _insert(
        table,
        (row for row in (original, positional)),
        skip_duplicates=True,
        ignore_extra_fields=True,
    )
    assert len(table.writes) == 1
    rows, replace, args, flags = table.writes[0]
    assert [row["params"]["threshold"] for row in rows] == [2.0, 3.0]
    assert [row["params_schema_version"] for row in rows] == [1, 1]
    assert flags == {"skip_duplicates": True, "ignore_extra_fields": True}
    assert replace is False and args == ()
    assert original == _row("dict", "2")


@pytest.mark.parametrize(
    "invalid,error",
    [
        (_row("bad", -1), ValidationError),
        (_row("bad", 2, params_schema_version=9), ValueError),
    ],
)
def test_invalid_batch_does_not_insert_valid_siblings(invalid, error):
    table = _Table()
    with pytest.raises(error):
        _insert(table, [_row("valid"), invalid])
    assert table.writes == []


def test_table_hook_rejection_happens_before_insertion():
    table = _Table()

    def reject_job_seed(row, _schema):
        if "random_seed" in (row.get("job_kwargs") or {}):
            raise ValueError("seed belongs in params")

    with pytest.raises(ValueError, match="seed belongs in params"):
        _insert(
            table,
            [_row("valid"), _row("bad", 2, job_kwargs={"random_seed": 1})],
            per_row_hook=reject_job_seed,
        )
    assert table.writes == []


def test_duplicate_content_in_one_batch_prevents_any_insert():
    table = _Table()
    with pytest.raises(DuplicateParameterContentError):
        _insert(table, [_row("first"), _row("second")])
    assert table.writes == []


def test_reinsertion_and_duplicate_escape_keep_datajoint_flags():
    normalized = {
        "params_name": "existing",
        "params": {"schema_version": 1, "threshold": 1.0},
        "params_schema_version": 1,
    }
    table = _Table([normalized])
    _insert(table, [_row("existing")], skip_duplicates=True)
    with pytest.raises(DuplicateParameterContentError):
        _insert(table, [_row("copy")])
    _insert(
        table, [_row("copy")], allow_duplicate_params=True, skip_duplicates=True
    )
    assert len(table.writes) == 2
    assert all(write[3] == {"skip_duplicates": True} for write in table.writes)


@pytest.mark.parametrize("scope", ["sorter", "matcher"])
def test_duplicate_content_is_scoped_to_its_backend(scope):
    table = _Table()
    scope_flag = {f"{scope}_keyed": True}
    _insert(
        table,
        [
            _row("one", **{scope: "backend_a"}),
            _row("two", **{scope: "backend_b"}),
        ],
        **scope_flag,
    )
    with pytest.raises(DuplicateParameterContentError):
        _insert(
            table,
            [
                _row("one", **{scope: "backend_a"}),
                _row("two", **{scope: "backend_a"}),
            ],
            **scope_flag,
        )
    assert len(table.writes) == 1


def test_custom_batch_validation_precedes_duplicate_guard():
    table = _Table()

    def validate_batch(rows, names):
        normalized = validate_lookup_rows(
            rows,
            names,
            schema_for=lambda _row: _Params,
            table_name="TestParameters",
        )
        if any(row.get("job_kwargs") for row in normalized):
            raise ValueError("custom job policy")
        return normalized

    with pytest.raises(ValueError, match="custom job policy"):
        _insert(
            table,
            [_row("one"), _row("copy", job_kwargs={"n_jobs": 2})],
            validate_rows=validate_batch,
        )
    assert table.writes == []


def test_replace_still_reaches_immutable_parameter_guard():
    table = _Table()
    with pytest.raises(dj.errors.DataJointError, match="replace=True"):
        _insert(table, [_row("one")], replace=True)
    assert table.writes == []


@pytest.mark.database
@pytest.mark.parametrize(
    "module_name,table_name,name_attr",
    [
        ("recording", "PreprocessingParameters", "preprocessing_params_name"),
        (
            "artifact",
            "ArtifactDetectionParameters",
            "artifact_detection_params_name",
        ),
        ("sorting", "AnalyzerWaveformParameters", "waveform_params_name"),
        ("sorting", "SorterParameters", "sorter_params_name"),
        (
            "motion",
            "MotionEstimationParameters",
            "motion_estimation_params_name",
        ),
        (
            "motion",
            "MotionInterpolationParameters",
            "motion_interpolation_params_name",
        ),
        ("unit_matching", "MatcherParameters", "matcher_params_name"),
    ],
)
def test_parameter_table_adapters_preserve_insert_contract(
    dj_conn, module_name, table_name, name_attr
):
    import importlib

    table = getattr(
        importlib.import_module(f"spyglass.spikesorting.v2.{module_name}"),
        table_name,
    )()
    table.insert_default()
    original = table.fetch(as_dict=True)[0]
    first = {**copy.deepcopy(original), name_attr: "_pytest_shared_insert_a"}
    second = {**copy.deepcopy(original), name_attr: "_pytest_shared_insert_b"}
    keys = [
        {key: row[key] for key in table.primary_key} for row in (first, second)
    ]
    (table & keys).delete_quick()
    try:
        invalid = {
            **copy.deepcopy(second),
            "params_schema_version": original["params_schema_version"] + 99,
        }
        with pytest.raises(ValueError, match="schema_version"):
            table.insert([first, invalid], allow_duplicate_params=True)
        assert not (table & keys)

        positional = tuple(first[name] for name in table.heading.names)
        with_extra_column = {**second, "_ignored_column": "not a table field"}
        for _ in range(2):
            table.insert(
                (row for row in (positional, with_extra_column)),
                allow_duplicate_params=True,
                skip_duplicates=True,
                ignore_extra_fields=True,
            )
        assert len(table & keys) == 2

        for insert in (table.insert, table.insert1):
            rows = [first] if insert.__name__ == "insert" else first
            with pytest.raises(dj.errors.DataJointError, match="replace=True"):
                insert(rows, allow_duplicate_params=True, replace=True)
        assert (table & keys[0]).fetch1("params") == original["params"]
    finally:
        (table & keys).delete_quick()
