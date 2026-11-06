"""Sorting wire-contracts and table-shape invariants.

Tri-part NamedTuples are splatted positionally, so their field order must
match the ``make_*`` signatures; a mismatch of str-typed slots would not raise.
"""

from __future__ import annotations

import pytest


def test_to_int_unit_id_raises_typed_error_on_non_integer():
    """A non-integer sorter unit_id raises ``NonIntegerUnitIDError`` (a
    ValueError subclass) naming the id and the remap guidance.
    """
    from spyglass.spikesorting.v2._sorting.units import _to_int_unit_id
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError

    assert _to_int_unit_id(3) == 3
    assert _to_int_unit_id("5") == 5

    with pytest.raises(NonIntegerUnitIDError, match="noise_3"):
        _to_int_unit_id("noise_3")

    assert issubclass(NonIntegerUnitIDError, ValueError)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_master_has_no_analyzer_folder_column():
    """The ``Sorting`` master has no ``analyzer_folder`` column (the path
    derives from ``sorting_id``); ``n_units`` is a column."""
    from spyglass.spikesorting.v2.sorting import Sorting

    attrs = Sorting.heading.attributes
    assert "analyzer_folder" not in attrs
    assert "n_units" in attrs


@pytest.mark.usefixtures("dj_conn")
def test_sorting_computed_matches_make_insert_signature():
    """``SortingComputed`` field order matches ``make_insert``."""
    import inspect

    from spyglass.spikesorting.v2.sorting import Sorting, SortingComputed

    params = list(inspect.signature(Sorting.make_insert).parameters)
    assert params[0] == "self"
    assert params[1] == "key"
    assert tuple(params[2:]) == SortingComputed._fields


@pytest.mark.usefixtures("dj_conn")
def test_sorting_fetched_matches_make_compute_signature():
    """``SortingFetched`` field order matches ``make_compute``."""
    import inspect

    from spyglass.spikesorting.v2.sorting import Sorting, SortingFetched

    params = list(inspect.signature(Sorting.make_compute).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == SortingFetched._fields


@pytest.mark.usefixtures("dj_conn")
def test_curation_evaluation_tuples_match_make_signatures():
    """CurationEvaluation's Fetched/Computed field orders match
    ``make_compute``/``make_insert``; the fetched carrier is four records.
    """
    import inspect

    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationComputed,
        CurationEvaluationFetched,
    )

    compute_params = list(
        inspect.signature(CurationEvaluation.make_compute).parameters
    )
    assert compute_params[:2] == ["self", "key"]
    assert tuple(compute_params[2:]) == CurationEvaluationFetched._fields

    insert_params = list(
        inspect.signature(CurationEvaluation.make_insert).parameters
    )
    assert insert_params[:2] == ["self", "key"]
    assert tuple(insert_params[2:]) == CurationEvaluationComputed._fields


@pytest.mark.usefixtures("dj_conn")
def test_recording_fetched_matches_make_compute_signature():
    """``RecordingFetched`` field order matches ``make_compute``."""
    import inspect

    from spyglass.spikesorting.v2.recording import Recording, RecordingFetched

    params = list(inspect.signature(Recording.make_compute).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == RecordingFetched._fields


def test_recording_artifact_result_field_contract():
    """``RecordingArtifactResult`` fields equal ``RecordingComputed``'s
    first eight, in order.
    """
    from spyglass.spikesorting.v2._recording.types import (
        RecordingArtifactResult,
        RecordingComputed,
    )

    assert RecordingArtifactResult._fields == (
        "analysis_file_name",
        "object_id",
        "content_hash",
        "saved_start",
        "saved_end",
        "sampling_frequency",
        "n_channels",
        "duration_s",
    )
    assert RecordingArtifactResult._fields == RecordingComputed._fields[:8]


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_make_fetch_heals_the_traces_file_idempotently(
    populated_sorting,
):
    """``make_fetch`` rebuilds a missing traces file and returns the same
    carrier, so DataJoint's second fetch matches the first."""
    from pathlib import Path

    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._tripart_helpers import fetch_hash

    fetched = Sorting().make_fetch(populated_sorting)
    assert Path(fetched.traces_abs_path).exists()
    Path(fetched.traces_abs_path).unlink()

    healed = Sorting().make_fetch(populated_sorting)
    assert Path(healed.traces_abs_path).exists()
    assert fetch_hash(healed) == fetch_hash(fetched)


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_make_compute_queries_db_only_to_stage_output(
    populated_sorting, monkeypatch
):
    """``make_compute`` sorts the traces file ``make_fetch`` resolved and
    queries the DB only while staging its units NWB (no traces-row fetch,
    self-heal or input path lookup)."""
    from spyglass.spikesorting.v2._storage.staged_outputs import (
        unlink_staged_analysis_file as _unlink_staged_analysis_file,
    )
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._tripart_helpers import forbid_db_queries

    table = Sorting()
    fetched = table.make_fetch(populated_sorting)
    with forbid_db_queries(
        monkeypatch, "Sorting.make_compute", allow_staging=True
    ):
        computed = table.make_compute(populated_sorting, *fetched)
    try:
        assert computed.analysis_file_name
        assert len(computed.unit_rows) == int(
            (Sorting & populated_sorting).fetch1("n_units")
        )
    finally:
        computed.staged_analyzer.close()
        _unlink_staged_analysis_file(
            computed.analysis_file_name, context="test"
        )
