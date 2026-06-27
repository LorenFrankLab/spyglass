"""Sorting wire-contracts and table-shape invariants.

Covers the typed ``NonIntegerUnitIDError`` on a non-integer unit_id, the
``Sorting`` master no longer declaring an ``analyzer_folder`` column, and the
positional NamedTuple-to-signature contracts (``SortingComputed`` vs
``make_insert``; ``RecordingArtifactResult`` vs ``RecordingComputed``).
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("dj_conn")
def test_to_int_unit_id_raises_typed_error_on_non_integer():
    """A sorter unit_id that does not convert to int raises the typed
    ``NonIntegerUnitIDError`` (a ValueError subclass), naming the offending id
    and the remap guidance -- not a bare ``int()`` ValueError or a silent
    coercion.
    """
    from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError
    from spyglass.spikesorting.v2.sorting import _to_int_unit_id

    # Convertible ids pass through.
    assert _to_int_unit_id(3) == 3
    assert _to_int_unit_id("5") == 5

    # A non-convertible label raises the TYPED error, naming the bad id.
    with pytest.raises(NonIntegerUnitIDError, match="noise_3"):
        _to_int_unit_id("noise_3")

    # It remains a ValueError subclass so existing ``except ValueError``
    # handlers still catch it.
    assert issubclass(NonIntegerUnitIDError, ValueError)


@pytest.mark.usefixtures("dj_conn")
def test_sorting_master_has_no_analyzer_folder_column():
    """The ``Sorting`` master no longer declares an ``analyzer_folder``
    column (the analyzer cache path is computed from ``sorting_id``, not
    persisted); ``n_units`` is still a column."""
    from spyglass.spikesorting.v2.sorting import Sorting

    attrs = Sorting.heading.attributes
    assert "analyzer_folder" not in attrs
    assert "n_units" in attrs


@pytest.mark.usefixtures("dj_conn")
def test_sorting_computed_matches_make_insert_signature():
    """The tri-part dispatch unpacks ``SortingComputed`` POSITIONALLY into
    ``make_insert(key, *computed)``, so the NamedTuple field order is a wire
    contract. Pin that it matches ``make_insert``'s parameter order -- a
    reorder would silently mis-bind the several str-adjacent slots
    (analysis_file_name / units_object_id / nwb_file_name)
    without a TypeError.
    """
    import inspect

    from spyglass.spikesorting.v2.sorting import Sorting, SortingComputed

    params = list(inspect.signature(Sorting.make_insert).parameters)
    assert params[0] == "self"
    assert params[1] == "key"
    assert tuple(params[2:]) == SortingComputed._fields


@pytest.mark.usefixtures("dj_conn")
def test_sorting_fetched_matches_make_compute_signature():
    """The tri-part dispatch splats ``SortingFetched`` POSITIONALLY into
    ``make_compute(key, *fetched)``, so the NamedTuple field order is a wire
    contract. ``execution_params`` was appended to both ``SortingFetched`` and
    ``make_compute`` -- a misalignment would silently mis-bind the str-adjacent
    slots (nwb_file_name / display_waveform_params_name) without a TypeError.
    """
    import inspect

    from spyglass.spikesorting.v2.sorting import Sorting, SortingFetched

    params = list(inspect.signature(Sorting.make_compute).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == SortingFetched._fields


@pytest.mark.usefixtures("dj_conn")
def test_curation_evaluation_tuples_match_make_signatures():
    """CurationEvaluation's tri-part NamedTuples are POSITIONAL wire contracts.

    The dispatch splats ``make_fetch`` -> ``make_compute(key, *fetched)`` and
    ``make_compute`` -> ``make_insert(key, *computed)`` positionally, so the
    field order of ``CurationEvaluationFetched`` / ``CurationEvaluationComputed``
    must match the corresponding parameter order. The fetched carrier is long
    (30 fields, several str-adjacent: abs paths / folders / recipe blobs), so a
    reorder -- or a dropped/added field on only one side -- would silently
    mis-bind later slots without a TypeError. Pin it.
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
    """The tri-part dispatch splats ``RecordingFetched`` POSITIONALLY into
    ``make_compute(key, *fetched)``, so the NamedTuple field order is a wire
    contract. ``raw_object_id`` was appended to both ``RecordingFetched`` and
    ``make_compute`` -- a misalignment would silently mis-bind the str-adjacent
    slots (e.g. ``preprocessing_job_kwargs`` / ``raw_object_id``) without a
    TypeError.
    """
    import inspect

    from spyglass.spikesorting.v2.recording import Recording, RecordingFetched

    params = list(inspect.signature(Recording.make_compute).parameters)
    assert params[:2] == ["self", "key"]
    assert tuple(params[2:]) == RecordingFetched._fields


def test_recording_artifact_result_field_contract():
    """``_compute_recording_artifact`` returns a typed ``RecordingArtifactResult``
    (NamedTuple) rather than a bare 8-tuple, so ``make_compute`` and
    ``_rebuild_nwb_artifact`` read its fields by name instead of by position.

    Pin the field set in order, AND pin that it equals ``RecordingComputed``'s
    first eight fields. ``make_compute`` transfers them by name into the tri-part
    contract NamedTuple, so the prefix-equality keeps that mapping order-safe and
    guarantees a future positional ``RecordingComputed(*artifact, ...)`` splat
    would still bind correctly; a reorder/rename in either type fails here.
    """
    from spyglass.spikesorting.v2.recording import (
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
