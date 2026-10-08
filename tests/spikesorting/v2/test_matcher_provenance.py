"""Producer and asset provenance survives matching's NWB boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

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
        "recordings": [
            {
                "recording_index": 0,
                "recording_id": RECORDING_ID,
                "nwb_file_name": "source.nwb",
                "interval_list_name": "sorting interval",
                "session_start_time": "2026-01-01T00:00:00+00:00",
                "start_sample": 0,
                "end_sample": 100,
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
