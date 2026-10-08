"""Producer and asset provenance survives matching's NWB boundary."""

from types import SimpleNamespace

import pytest

from spyglass.spikesorting.v2._matching.provenance import matcher_provenance


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


def test_missing_spyglass_distribution_records_unknown(monkeypatch):
    from spyglass.spikesorting.v2._matching import provenance

    def missing(name):
        raise provenance.PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", missing)
    got = matcher_provenance(object(), object(), {})
    assert got["spyglass_version"] is None


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
        UNITMATCH_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2._matching.compute import (
        unit_match_provenance_tables,
    )
    from spyglass.spikesorting.v2._storage.matches_nwb import write_pairs_table
    from tests.spikesorting.v2.test_nwb_provenance import _new_nwbfile, _write

    params = {"model_sha256": "b" * 64, "feature_definition": "templates-v3"}
    expected = matcher_provenance(Backend(), Preparer(), params)
    tables = unit_match_provenance_tables(
        {"unitmatch_id": "test-run"},
        [],
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
