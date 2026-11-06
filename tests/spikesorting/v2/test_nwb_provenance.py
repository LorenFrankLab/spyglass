"""DB-free unit tests for the NWB provenance scratch helper.

These exercise the pure (de)serialization in
``spyglass.spikesorting.v2._storage.provenance`` -- the ``key``/``value_json``
scalar provenance container and the typed long-table container -- with no
DataJoint server and no SpikeInterface analyzer. The helper writes into and
reads back from a real (tiny) NWB file on disk, so the round-trip assertions
prove the values survive an HDF5 write, not just an in-memory build.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


@pytest.mark.parametrize("kind", ["sorting", "curation"])
@pytest.mark.parametrize("damage", ["missing", "tampered", "incomplete"])
def test_current_artifact_headers_require_intact_runtime_receipts(kind, damage):
    from tests.spikesorting.v2._provenance_helpers import (
        curation_header,
        sorting_provenance,
    )
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_curation_header,
        validate_sorting_provenance,
    )

    values = sorting_provenance() if kind == "sorting" else curation_header()
    if damage == "missing":
        del values["runtime_environment"]
    elif damage == "tampered":
        values["runtime_environment"]["job_kwargs"]["n_jobs"] = 71
    else:
        del values["runtime_environment"]["packages"]["numpy"]
    validate = (
        validate_sorting_provenance
        if kind == "sorting"
        else validate_curation_header
    )
    with pytest.raises(ValueError):
        validate(values)


pytestmark = [pytest.mark.integration, pytest.mark.nwb, pytest.mark.io_heavy]


def _new_nwbfile():
    """A minimal, valid in-memory NWBFile (tz-aware start time)."""
    import pynwb

    return pynwb.NWBFile(
        session_description="provenance helper test",
        identifier="provenance-test",
        session_start_time=datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
    )


def _write(nwbfile, tmp_path):
    import pynwb

    path = str(tmp_path / "provenance.nwb")
    with pynwb.NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)
    return path


def test_scalar_provenance_round_trips_mixed_types(tmp_path):
    """A scalar bundle (str/int/float/bool/None/dict/list) survives HDF5."""
    from spyglass.spikesorting.v2._storage.provenance import (
        PROVENANCE_SCHEMA_VERSION,
        build_provenance_table,
        read_provenance_values,
    )

    values = {
        "recording_id": "abc-123",
        "sort_group_id": 7,
        "threshold_uv": 0.1,
        "merges_applied": True,
        "sorter_version": None,  # nullable provenance field
        "sorter_params": {"detect_sign": -1, "freq_min": 300.0},  # nested blob
        "members": [0, 1, 2],  # list blob
    }

    nwbfile = _new_nwbfile()
    nwbfile.add_scratch(
        build_provenance_table("spyglass_v2_test_provenance", values)
    )
    path = _write(nwbfile, tmp_path)

    read = read_provenance_values(path, "spyglass_v2_test_provenance")

    for key, expected in values.items():
        assert read[key] == expected, key
    # Every provenance container records its schema version.
    assert read["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_scalar_provenance_coerces_datajoint_types(tmp_path):
    """Numpy + UUID (DataJoint-deserialized types) serialize + read back.

    Provenance bundles re-emit values fetched from DataJoint (metric kwargs,
    rule thresholds, hash manifests, row ids), which deserialize as numpy
    scalars/arrays and ``uuid.UUID``; plain ``json.dumps`` cannot encode those,
    so the helper coerces them (UUIDs to their canonical string form).
    """
    import uuid

    import numpy as np

    from spyglass.spikesorting.v2._storage.provenance import (
        build_provenance_table,
        read_provenance_values,
    )

    rec_id = uuid.uuid4()
    values = {
        "threshold": np.float64(1.5),
        "count": np.int64(7),
        "flags": np.array([1, 2, 3]),
        "ok": np.bool_(True),
        "recording_id": rec_id,
    }

    nwbfile = _new_nwbfile()
    nwbfile.add_scratch(
        build_provenance_table("spyglass_v2_test_dj_types", values)
    )
    path = _write(nwbfile, tmp_path)

    read = read_provenance_values(path, "spyglass_v2_test_dj_types")
    assert read["threshold"] == 1.5
    assert read["count"] == 7
    assert read["flags"] == [1, 2, 3]
    assert read["ok"] is True
    assert read["recording_id"] == str(rec_id)


def test_long_provenance_table_round_trips_typed_rows(tmp_path):
    """A typed long table (one row per member) survives HDF5 with native types."""
    from spyglass.spikesorting.v2._storage.provenance import (
        PROVENANCE_SCHEMA_VERSION,
        build_long_provenance_table,
        read_long_provenance,
    )

    rows = [
        {"member_index": 0, "recording_id": "rec-a", "end_sample": 1000},
        {"member_index": 1, "recording_id": "rec-b", "end_sample": 2500},
    ]
    columns = [
        ("member_index", int),
        ("recording_id", str),
        ("end_sample", int),
    ]

    nwbfile = _new_nwbfile()
    nwbfile.add_scratch(
        build_long_provenance_table("spyglass_v2_test_members", rows, columns)
    )
    path = _write(nwbfile, tmp_path)

    read = read_long_provenance(path, "spyglass_v2_test_members")

    assert len(read) == 2
    for got, expected in zip(read, rows):
        for col, _dtype in columns:
            assert got[col] == expected[col], col
            # Native Python types, not numpy scalars, for downstream use.
            assert type(got[col]) is type(expected[col]), col
        assert got["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_long_provenance_table_handles_empty_rows(tmp_path):
    """A zero-row long table still writes and reads back as an empty list."""
    from spyglass.spikesorting.v2._storage.provenance import (
        build_long_provenance_table,
        read_long_provenance,
    )

    columns = [("member_index", int), ("recording_id", str)]

    nwbfile = _new_nwbfile()
    nwbfile.add_scratch(
        build_long_provenance_table("spyglass_v2_test_empty", [], columns)
    )
    path = _write(nwbfile, tmp_path)

    read = read_long_provenance(path, "spyglass_v2_test_empty")

    assert read == []


@pytest.mark.parametrize(
    "overrides,missing,match",
    [
        ({}, "sorting_id", "missing required fields.*sorting_id"),
        ({}, "sorter_params", "missing required fields.*sorter_params"),
        ({}, "statistics_spans", "missing required fields.*statistics_spans"),
        ({"sorting_id": "unknown"}, None, "sorting_id must be a UUID"),
        ({"recording_id": None}, None, "exactly one recording source"),
        (
            {"concat_recording_id": "00000000-0000-0000-0000-000000000005"},
            None,
            "exactly one recording source",
        ),
        ({"recording_id": "unknown"}, None, "recording_id must be a UUID"),
        ({"sorter_params": []}, None, "sorter_params must be a mapping"),
        (
            {"effective_random_seed": True},
            None,
            "effective_random_seed must be an integer",
        ),
    ],
)
def test_sorting_provenance_requires_identified_source_and_resolved_recipe(
    overrides, missing, match
):
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_sorting_provenance,
    )
    from tests.spikesorting.v2._provenance_helpers import sorting_provenance

    values = sorting_provenance(**overrides)
    if missing is not None:
        del values[missing]
    with pytest.raises(ValueError, match=match):
        validate_sorting_provenance(values)


@pytest.mark.parametrize("version", [123, {"version": "1.2"}, ""])
def test_sorting_provenance_rejects_malformed_external_versions(version):
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_sorting_provenance,
    )
    from tests.spikesorting.v2._provenance_helpers import sorting_provenance

    with pytest.raises(
        ValueError, match="sorter_version must be a string or None"
    ):
        validate_sorting_provenance(sorting_provenance(sorter_version=version))


@pytest.mark.parametrize(
    "spans",
    [
        [[0, 5], [4, 10]],
        [[5, 10], [0, 5]],
        [[-1, 5]],
        [[5, 5]],
        [[0.0, 5.0]],
        [[False, 5]],
        [[0, 5, 10]],
        [0, 5],
    ],
)
def test_sorting_provenance_rejects_invalid_statistics_frame_spans(spans):
    """Statistics must exclude gaps and cannot overlap or coerce frame IDs."""
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_sorting_provenance,
    )
    from tests.spikesorting.v2._provenance_helpers import sorting_provenance

    with pytest.raises(ValueError, match="statistics_spans"):
        validate_sorting_provenance(sorting_provenance(statistics_spans=spans))


@pytest.mark.parametrize("spans", [[], [[0, 5], [5, 10]], [[0, 5], [10, 20]]])
def test_sorting_provenance_accepts_concat_empty_or_disjoint_statistics(spans):
    """Empty results and known join/gap boundaries keep complete provenance."""
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_sorting_provenance,
    )
    from tests.spikesorting.v2._provenance_helpers import (
        RECORDING_ID,
        sorting_provenance,
    )

    values = sorting_provenance(
        recording_id=None,
        concat_recording_id=RECORDING_ID,
        statistics_spans=spans,
        sorter_version=None,
        effective_random_seed=None,
    )
    validate_sorting_provenance(values)


@pytest.mark.parametrize(
    "overrides,missing,match",
    [
        ({}, "curation_uuid", "missing required fields.*curation_uuid"),
        (
            {},
            "parent_curation_id",
            "missing required fields.*parent_curation_id",
        ),
        ({"curation_uuid": "unknown"}, None, "curation_uuid must be a UUID"),
        ({"sorting_id": "unknown"}, None, "sorting_id must be a UUID"),
        ({"curation_id": -1}, None, "curation_id must be an integer"),
        ({"curation_id": True}, None, "curation_id must be an integer"),
        (
            {"parent_curation_id": 1.5},
            None,
            "parent_curation_id must be an integer",
        ),
        ({"merges_applied": 1}, None, "merges_applied must be a boolean"),
    ],
)
def test_curation_header_requires_generation_and_operation_identity(
    overrides, missing, match
):
    from spyglass.spikesorting.v2._storage.provenance import (
        validate_curation_header,
    )
    from tests.spikesorting.v2._provenance_helpers import curation_header

    values = curation_header(**overrides)
    if missing is not None:
        del values[missing]
    with pytest.raises(ValueError, match=match):
        validate_curation_header(values)


@pytest.mark.parametrize("writer", ["sorting", "curation"])
@pytest.mark.parametrize("state", ["absent", "incomplete", "malformed"])
def test_units_writers_reject_invalid_identity_before_staging(
    writer, state, monkeypatch
):
    """An incomplete run cannot allocate a file or query its source tables."""
    import pynwb

    from spyglass.spikesorting.v2._storage.units_nwb import (
        write_curated_units_nwb,
        write_sorting_units_nwb,
    )
    from tests.spikesorting.v2._provenance_helpers import (
        SORTING_ID,
        curation_header,
        sorting_provenance,
    )

    def refuse_file_io(*args, **kwargs):
        pytest.fail("Invalid provenance reached NWB file I/O")

    monkeypatch.setattr(pynwb, "NWBHDF5IO", refuse_file_io)
    # The unit tier also rejects DataJoint access. Both public writers must
    # validate before importing or querying their table adapters.
    if writer == "sorting":
        values = sorting_provenance()
        if state == "absent":
            values = None
        elif state == "incomplete":
            del values["recording_id"]
        else:
            values["recording_id"] = "unknown"
        with pytest.raises(ValueError, match="sorting provenance"):
            write_sorting_units_nwb(
                None, None, "unused.nwb", source_provenance=values
            )
    else:
        values = curation_header()
        if state == "absent":
            values = None
        elif state == "incomplete":
            del values["curation_uuid"]
        else:
            values["curation_uuid"] = "unknown"
        with pytest.raises(ValueError, match="curation provenance"):
            write_curated_units_nwb(
                SORTING_ID, {}, False, {}, curation_header=values
            )


def test_current_sort_and_curation_headers_round_trip_generation_and_recipe(
    tmp_path,
):
    """The export identifies its source, recipe and exact curation generation."""
    import uuid

    import numpy as np

    from spyglass.spikesorting.v2._storage.provenance import (
        CURATION_PROVENANCE,
        SORTING_PROVENANCE,
        build_provenance_table,
        read_provenance_values,
        validate_curation_header,
        validate_sorting_provenance,
    )
    from tests.spikesorting.v2._provenance_helpers import (
        CURATION_UUID,
        RECORDING_ID,
        SORTING_ID,
        curation_header,
        sorting_provenance,
    )

    source = sorting_provenance(
        sorting_id=uuid.UUID(SORTING_ID),
        recording_id=uuid.UUID(RECORDING_ID),
        sorter="mountainsort5",
        sorter_params_name="ms5_test",
        sorter_params={"schema_version": 1, "detect_threshold": 5.5},
        execution_params={"n_jobs": 2},
        effective_random_seed=np.int64(17),
        statistics_spans=np.asarray([[0, 50], [60, 100]], dtype=np.int64),
        sorter_version=None,
    )
    generation = curation_header(
        sorting_id=uuid.UUID(SORTING_ID),
        curation_id=np.int64(2),
        curation_uuid=uuid.UUID(CURATION_UUID),
        parent_curation_id=np.int64(1),
        curation_source="figpack",
        merges_applied=np.bool_(True),
        description="reviewed merge",
    )
    validate_sorting_provenance(source)
    validate_curation_header(generation)
    nwbfile = _new_nwbfile()
    nwbfile.add_scratch(build_provenance_table(SORTING_PROVENANCE, source))
    nwbfile.add_scratch(build_provenance_table(CURATION_PROVENANCE, generation))
    path = _write(nwbfile, tmp_path)

    read_source = read_provenance_values(path, SORTING_PROVENANCE)
    read_generation = read_provenance_values(path, CURATION_PROVENANCE)
    assert (
        read_source["sorting_id"] == read_generation["sorting_id"] == SORTING_ID
    )
    assert read_source["recording_id"] == RECORDING_ID
    assert read_source["concat_recording_id"] is None
    assert read_source["sorter_params"] == source["sorter_params"]
    assert read_source["execution_params"] == {"n_jobs": 2}
    assert read_source["effective_random_seed"] == 17
    assert read_source["statistics_spans"] == [[0, 50], [60, 100]]
    assert read_source["sorter_version"] is None
    assert read_generation["curation_uuid"] == CURATION_UUID
    assert read_generation["curation_id"] == 2
    assert read_generation["parent_curation_id"] == 1
    assert read_generation["merges_applied"] is True
    assert read_generation["curation_source"] == "figpack"
