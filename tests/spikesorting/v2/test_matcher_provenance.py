"""Producer and asset provenance survives matching's NWB boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from spyglass.spikesorting.v2._matching.preflight import (
    mixed_motion_treatment_warning,
)
from spyglass.spikesorting.v2._matching.provenance import (
    matcher_provenance,
    validate_matcher_provenance,
)


class Backend:
    @staticmethod
    def backend_version():
        return "1.2.3"

    @staticmethod
    def provenance_fingerprints(params):
        return {"model_sha256": params["model_sha256"]}


class Preparer:
    @staticmethod
    def preparer_version():
        return "4.5.6"

    @staticmethod
    def provenance_fingerprints(params):
        return {"feature_definition": params["feature_definition"]}


def test_matcher_runtime_receipt_records_resolved_jobs():
    from spyglass.spikesorting.v2._core.runtime import (
        validate_runtime_provenance,
    )

    values = matcher_provenance(
        object(), object(), {}, job_kwargs={"n_jobs": 3, "chunk_duration": "2s"}
    )
    validate_runtime_provenance(values)
    assert values["runtime_environment"]["job_kwargs"] == {
        "n_jobs": 3,
        "chunk_duration": "2s",
    }
    assert (
        values["runtime_environment"]["execution_params"]["stage"] == "matching"
    )


@pytest.mark.parametrize("damage", ["missing", "tampered", "incomplete"])
def test_matcher_provenance_requires_intact_runtime_receipt(damage):
    values = matcher_provenance(object(), object(), {})
    if damage == "missing":
        del values["runtime_environment"]
    elif damage == "tampered":
        values["runtime_environment"]["job_kwargs"]["n_jobs"] = 71
    else:
        del values["runtime_environment"]["packages"]["numpy"]
    with pytest.raises(ValueError):
        validate_matcher_provenance(values)


def test_provenance_identifies_both_producers_and_parameter_assets():
    params = {"model_sha256": "a" * 64, "feature_definition": "templates-v2"}
    got = matcher_provenance(Backend(), Preparer(), params)
    assert got["schema_version"] == 1
    for name, cls, version in (
        ("backend", Backend, "1.2.3"),
        ("preparer", Preparer, "4.5.6"),
    ):
        assert (
            got[name]["qualified_name"]
            == f"{cls.__module__}.{cls.__qualname__}"
        )
        assert got[name]["version"] == version
    assert got["backend"]["fingerprints"] == {"model_sha256": "a" * 64}
    assert got["preparer"]["fingerprints"] == {
        "feature_definition": "templates-v2"
    }


def test_unknown_plugin_versions_are_not_guessed():
    got = matcher_provenance(object(), object(), {})
    assert got["backend"]["version"] is None
    assert got["preparer"]["version"] is None
    assert got["backend"]["fingerprints"] == {}
    validate_matcher_provenance(got)


def test_missing_spyglass_distribution_records_unknown(monkeypatch):
    from spyglass.spikesorting.v2._matching import provenance

    def missing(name):
        raise provenance.PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", missing)
    got = matcher_provenance(object(), object(), {})
    assert got["spyglass_version"] is None
    validate_matcher_provenance(got)


@pytest.mark.parametrize("component", ["backend", "preparer"])
def test_matcher_provenance_requires_both_producer_identities(component):
    snapshot = matcher_provenance(object(), object(), {})
    for absent in (component, "qualified_name", "version", "fingerprints"):
        incomplete = deepcopy(snapshot)
        if absent == component:
            del incomplete[component]
        else:
            del incomplete[component][absent]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_matcher_provenance(incomplete)


@pytest.mark.parametrize(
    "component,field,value,match",
    [
        (
            "backend",
            "qualified_name",
            "",
            "qualified_name must be a nonempty string",
        ),
        (
            "preparer",
            "qualified_name",
            None,
            "qualified_name must be a nonempty string",
        ),
        ("backend", "version", 2, "version must be a string or None"),
        (
            "preparer",
            "fingerprints",
            {"model_sha256": 123},
            "nonempty string names and fingerprints",
        ),
    ],
)
def test_matcher_provenance_rejects_malformed_producer_metadata(
    component, field, value, match
):
    snapshot = matcher_provenance(object(), object(), {})
    snapshot[component][field] = value
    with pytest.raises(ValueError, match=match):
        validate_matcher_provenance(snapshot)


@pytest.mark.parametrize("schema", [None, True, "1", 0, 2])
def test_matcher_provenance_requires_current_schema(schema):
    snapshot = matcher_provenance(object(), object(), {})
    snapshot["schema_version"] = schema
    with pytest.raises(ValueError, match="requires schema_version=1"):
        validate_matcher_provenance(snapshot)


def test_matcher_tables_reject_incomplete_producers_before_building(
    monkeypatch,
):
    from spyglass.spikesorting.v2._matching.compute import (
        unit_match_provenance_tables,
    )
    from spyglass.spikesorting.v2._storage import provenance

    def refuse_table(*args, **kwargs):
        pytest.fail("An incomplete producer snapshot reached an NWB builder")

    monkeypatch.setattr(provenance, "build_provenance_table", refuse_table)
    incomplete = matcher_provenance(object(), object(), {})
    del incomplete["preparer"]
    with pytest.raises(ValueError, match="missing required fields.*preparer"):
        unit_match_provenance_tables(
            {"unitmatch_id": "00000000-0000-0000-0000-000000000004"},
            [],
            session_group_owner=None,
            session_group_name=None,
            matcher_params_name="unitmatch_default",
            matcher_backend="builtins",
            matcher_backend_version=None,
            spikeinterface_version="test",
            matcher_provenance=incomplete,
        )


@pytest.mark.parametrize(
    "fingerprints", [None, [], {"model": 4}, {"": "x"}, {"x": ""}]
)
def test_invalid_asset_provenance_is_rejected(fingerprints):
    backend = SimpleNamespace(
        provenance_fingerprints=lambda params: fingerprints
    )
    with pytest.raises(TypeError, match="nonempty string"):
        matcher_provenance(backend, object(), {})


def test_version_hook_failures_propagate():
    def unavailable():
        raise OSError("Cannot read producer metadata")

    with pytest.raises(OSError, match="producer metadata"):
        matcher_provenance(
            object(), SimpleNamespace(preparer_version=unavailable), {}
        )
    with pytest.raises(TypeError, match="str or None"):
        matcher_provenance(
            SimpleNamespace(backend_version=lambda: 123), object(), {}
        )


def test_matcher_asset_provenance_round_trips_in_pairs_nwb(tmp_path):
    from spyglass.spikesorting.v2._storage.provenance import (
        UNITMATCH_INPUTS,
        UNITMATCH_PROVENANCE,
        read_long_provenance,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2._matching.compute import (
        unit_match_provenance_tables,
    )
    from spyglass.spikesorting.v2._storage.matches_nwb import write_pairs_table
    from tests.spikesorting.v2.test_nwb_provenance import _new_nwbfile, _write
    from tests.spikesorting.v2._provenance_helpers import (
        CURATION_UUID,
        RECORDING_ID,
        SORTING_ID,
    )

    params = {"model_sha256": "b" * 64, "feature_definition": "templates-v3"}
    expected = matcher_provenance(Backend(), Preparer(), params)
    plan = {
        "input_index": 0,
        "sorting_id": SORTING_ID,
        "curation_id": 2,
        "curation_uuid": CURATION_UUID,
        "source_kind": "recording",
        "source_id": RECORDING_ID,
        "input_start_time": "2026-01-01T00:00:00+00:00",
        "waveform_traces": "recording",
        "motion_corrected_recording_id": None,
        "selection_snapshot": {"policy": "all", "unit_ids": [1]},
        "recordings": [
            {
                "recording_index": 0,
                "recording_id": RECORDING_ID,
                "nwb_file_name": "source.nwb",
                "interval_list_name": "sorting interval",
                "session_start_time": "2026-01-01T00:00:00+00:00",
                "start_sample": 0,
                "end_sample": 100,
                "source_coverage": {
                    "version": 1,
                    "source_object_id": "raw",
                    "sample_ranges": [[0, 100]],
                    "time_ranges": [[0.0, 0.0033]],
                },
            }
        ],
    }
    tables = unit_match_provenance_tables(
        {"unitmatch_id": "00000000-0000-0000-0000-000000000004"},
        [plan],
        session_group_owner=None,
        session_group_name=None,
        matcher_params_name="custom",
        matcher_backend=Backend.__module__,
        matcher_backend_version="1.2.3",
        spikeinterface_version="test",
        matcher_provenance=expected,
    )
    path = _write(_new_nwbfile(), tmp_path)
    write_pairs_table(path, [], provenance_tables=tables)
    header = read_provenance_values(path, UNITMATCH_PROVENANCE)
    assert header["matcher_provenance"] == expected
    [input_row] = read_long_provenance(path, UNITMATCH_INPUTS)
    assert input_row["sorting_id"] == SORTING_ID
    assert input_row["curation_uuid"] == CURATION_UUID
    assert input_row["curation_id"] == 2
    assert input_row["source_id"] == RECORDING_ID


def test_run_diagnostics_join_the_producer_provenance():
    import json

    from spyglass.spikesorting.v2._matching.provenance import (
        with_run_diagnostics,
    )

    base = matcher_provenance(object(), object(), {})
    diagnostics = {
        "n_units": 11,
        "n_units_per_input": [6, 5],
        "n_expected_matches": 17,
    }
    got = with_run_diagnostics(base, diagnostics)
    validate_matcher_provenance(got)
    assert got["run_diagnostics"] == diagnostics
    assert "run_diagnostics" not in base
    json.dumps(got, default=str)
    assert with_run_diagnostics(base, None) == base


def test_matcher_run_diagnostics_reads_the_registered_backend():
    from spyglass.spikesorting.v2._matching import compute
    from spyglass.spikesorting.v2.matcher_protocol import get_matcher

    backend = get_matcher("unitmatch")
    backend.last_run_diagnostics = {"n_units": 3}
    try:
        assert compute.matcher_run_diagnostics("unitmatch") == {"n_units": 3}
    finally:
        backend.last_run_diagnostics = None
    assert compute.matcher_run_diagnostics("unitmatch") is None


def _unitmatch_backend_with_defaults(monkeypatch, **overrides):
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2._matching import unitmatch_backend as module

    um = module._require_unitmatch()
    real = um.default_params.get_default_param

    def patched(*args, **kwargs):
        return {**real(*args, **kwargs), **overrides}

    monkeypatch.setattr(um.default_params, "get_default_param", patched)
    return module.UnitMatchBackend()


def test_unitmatch_fingerprints_change_with_a_unitmatch_default(monkeypatch):
    """The effective UnitMatchPy defaults are fingerprinted, so an upgrade
    that changes one is visible in the run's provenance."""
    same = _unitmatch_backend_with_defaults(monkeypatch)
    base = same.provenance_fingerprints({})
    assert set(base) == {"default_params_sha256", "niter", "numpy_arange_shim"}
    assert base["niter"] == "2"
    assert base["numpy_arange_shim"] == "true"
    assert same.provenance_fingerprints({}) == base

    changed = _unitmatch_backend_with_defaults(monkeypatch, channel_radius=151)
    assert (
        changed.provenance_fingerprints({})["default_params_sha256"]
        != base["default_params_sha256"]
    )


def test_unitmatch_fingerprints_ignore_derived_arrays(monkeypatch):
    import numpy as np

    backend = _unitmatch_backend_with_defaults(monkeypatch)
    base = backend.provenance_fingerprints({})
    other = _unitmatch_backend_with_defaults(
        monkeypatch, bins=np.arange(3), waveidx=np.arange(5)
    )
    assert other.provenance_fingerprints({}) == base


def test_unitmatch_fingerprints_reach_the_producer_provenance():
    pytest.importorskip("UnitMatchPy")
    from spyglass.spikesorting.v2._matching.unitmatch_backend import (
        UnitMatchBackend,
    )

    values = matcher_provenance(UnitMatchBackend(), object(), {})
    validate_matcher_provenance(values)
    assert "default_params_sha256" in values["backend"]["fingerprints"]


def test_unitmatch_fingerprints_are_empty_without_unitmatchpy(monkeypatch):
    from spyglass.spikesorting.v2._matching import unitmatch_backend as module

    def missing():
        raise ImportError("UnitMatchPy is required")

    monkeypatch.setattr(module, "_require_unitmatch", missing)
    assert module.UnitMatchBackend().provenance_fingerprints({}) == {}


@pytest.mark.parametrize(
    "treatment",
    [
        {0: None, 1: None},
        {0: "a" * 64, 1: "a" * 64, 2: "a" * 64},
        {0: None},
    ],
)
def test_same_motion_treatment_gives_no_warning(treatment):
    assert mixed_motion_treatment_warning(treatment) is None


def test_corrected_and_uncorrected_inputs_warn_and_name_the_uncorrected():
    text = mixed_motion_treatment_warning({0: None, 1: "a" * 64, 2: "a" * 64})
    assert "different motion treatment" in text
    assert "input(s) [0] have no motion correction" in text
    assert "input(s) [1, 2] use interpolation recipe aaaaaaaaaaaa" in text


def test_two_interpolation_recipes_warn_and_group_inputs_by_recipe():
    text = mixed_motion_treatment_warning({0: "a" * 64, 1: "b" * 64})
    assert "no motion correction" not in text
    assert "input(s) [0] use interpolation recipe aaaaaaaaaaaa" in text
    assert "input(s) [1] use interpolation recipe bbbbbbbbbbbb" in text


def test_motion_treatment_by_input_maps_uncorrected_to_none():
    from spyglass.spikesorting.v2._matching.preflight import (
        motion_treatment_by_input,
    )

    got = motion_treatment_by_input(
        [
            {"input_index": 0, "motion_corrected_recording_id": None},
            {"input_index": 1, "motion_corrected_recording_id": "m1"},
            {"input_index": 2, "motion_corrected_recording_id": "m2"},
        ],
        recipe_hash_of={"m1": "h1", "m2": "h2"}.__getitem__,
    )
    assert got == {0: None, 1: "h1", 2: "h2"}
