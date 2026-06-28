"""Unit tests for the pure ``resolve_restriction`` source-routing classifier.

``CurationV2.resolve_restriction`` delegates its pure half to
``_curation_routing.classify_and_normalize_restriction``: validate the
restriction keys, map the ``artifact_detection_{uuid}`` interval-name
convention back to ``artifact_detection_id``, normalize that id to a UUID, and
split the keys into the per-source restriction dicts the join assembly
consumes. These tests drive the classifier directly -- no DataJoint, no DB --
so the routing branches (unknown-key strict/lenient, source classification,
artifact-id normalization + unresolved-name warning, anti-join vs. wildcard)
are pinned cheaply.

``_curation_routing`` imports only stdlib + the DB-free ``_artifact_naming``
parser, so this file needs no database fixture; the import-boundary contract is
enforced separately by ``test_service_import_contracts``.
"""

from __future__ import annotations

import uuid

import pytest

from spyglass.spikesorting.v2._artifact_naming import (
    artifact_detection_interval_list_name,
)
from spyglass.spikesorting.v2._curation_routing import (
    NO_ARTIFACT_RESTRICTION,
    classify_and_normalize_restriction,
)

pytestmark = pytest.mark.unit


def _classify(key, *, restrict_by_artifact=True, strict=True):
    return classify_and_normalize_restriction(
        key, restrict_by_artifact=restrict_by_artifact, strict=strict
    )


def test_classify_rejects_unknown_key_when_strict():
    """An unknown key raises in strict mode, returns None in lenient mode."""
    with pytest.raises(ValueError, match="unknown restriction keys"):
        _classify({"bogus": 1}, strict=True)
    assert _classify({"bogus": 1}, strict=False) is None


def test_classify_splits_recording_concat_sort_curation_keys():
    """A mixed recording+sort+curation key splits into the right per-source
    dicts; mixing a concat key with a recording key is contradictory."""
    plan = _classify(
        {
            "nwb_file_name": "x.nwb",
            "sort_group_id": 0,
            "sorter": "mountainsort5",
            "curation_id": 2,
        }
    )
    assert plan.rec_restriction == {
        "nwb_file_name": "x.nwb",
        "sort_group_id": 0,
    }
    assert plan.sort_restriction == {"sorter": "mountainsort5"}
    assert plan.curation_restriction == {"curation_id": 2}
    assert plan.concat_restriction == {}
    # No artifact key named -> the wildcard sentinel, NOT the None anti-join.
    assert plan.artifact_detection_id is NO_ARTIFACT_RESTRICTION

    # A concat-source key combined with a recording key is contradictory: a
    # sort has exactly one input source.
    with pytest.raises(ValueError, match="cannot combine concat-source"):
        _classify({"concat_recording_id": "c", "nwb_file_name": "x.nwb"})


def test_classify_routes_concat_only_keys():
    """Concat-source keys route into ``concat_restriction`` with empty rec."""
    plan = _classify({"session_group_name": "g", "session_group_owner": "o"})
    assert plan.concat_restriction == {
        "session_group_name": "g",
        "session_group_owner": "o",
    }
    assert plan.rec_restriction == {}


def test_classify_preproc_is_cross_source_shared_key():
    """``preprocessing_params_name`` lives on both RecordingSelection and
    ConcatenatedRecordingSelection, so it classifies as a shared (cross-source)
    restriction -- not recording-only -- and triggers no contradiction, letting
    a broad preproc query reach both source families."""
    plan = _classify({"preprocessing_params_name": "default"})
    assert plan.shared_restriction == {"preprocessing_params_name": "default"}
    assert plan.rec_restriction == {}
    assert plan.concat_restriction == {}


def test_classify_accepts_concat_recording_id_with_preproc():
    """``{concat_recording_id, preprocessing_params_name}`` is a valid concat
    restriction (the recipe is on ConcatenatedRecordingSelection), NOT a
    contradiction -- the concat key routes, the shared preproc filters it."""
    plan = _classify(
        {"concat_recording_id": "c", "preprocessing_params_name": "default"}
    )
    assert plan.concat_restriction == {"concat_recording_id": "c"}
    assert plan.shared_restriction == {"preprocessing_params_name": "default"}
    assert plan.rec_restriction == {}


def test_classify_recording_id_with_preproc_splits_rec_and_shared():
    """A single-recording key + preproc keeps the recording key in the rec
    restriction and the cross-source preproc in the shared restriction."""
    plan = _classify(
        {"recording_id": "r", "preprocessing_params_name": "default"}
    )
    assert plan.rec_restriction == {"recording_id": "r"}
    assert plan.shared_restriction == {"preprocessing_params_name": "default"}


def test_classify_normalizes_artifact_id_and_warning():
    """Artifact-name -> UUID id; raw uuid string -> UUID; unresolved name ->
    captured warning (not raised); explicit None preserved for the anti-join."""
    aid = uuid.uuid4()

    # 1. An artifact-detection interval name normalizes to the UUID id and the
    #    interval name is consumed (not left as a recording restriction).
    name = artifact_detection_interval_list_name(aid)
    plan = _classify({"interval_list_name": name}, restrict_by_artifact=True)
    assert plan.artifact_detection_id == aid
    assert isinstance(plan.artifact_detection_id, uuid.UUID)
    assert "interval_list_name" not in plan.rec_restriction
    assert plan.unresolved_name_warning is None

    # 2. A raw uuid string passed directly normalizes to a uuid.UUID.
    plan = _classify({"artifact_detection_id": str(aid)})
    assert plan.artifact_detection_id == aid
    assert isinstance(plan.artifact_detection_id, uuid.UUID)

    # 3. restrict_by_artifact=True but a non-artifact interval name and no
    #    artifact id -> capture an unresolved-name warning, do NOT raise, and
    #    leave the interval name as a recording restriction (unrestricted).
    plan = _classify(
        {"interval_list_name": "raw data valid times"},
        restrict_by_artifact=True,
    )
    assert plan.unresolved_name_warning is not None
    assert "will NOT be" in plan.unresolved_name_warning
    assert plan.rec_restriction == {
        "interval_list_name": "raw data valid times"
    }
    assert plan.artifact_detection_id is NO_ARTIFACT_RESTRICTION

    # 4. An explicit artifact_detection_id=None is preserved as the anti-join
    #    signal -- distinct from the wildcard sentinel.
    plan = _classify({"artifact_detection_id": None})
    assert plan.artifact_detection_id is None
    assert plan.sort_restriction == {}


def test_classify_emits_warning_before_contradiction_raise(caplog):
    """On the doubly-degenerate input (a concat key + an unresolved interval
    name under restrict_by_artifact), the unresolved-name warning is emitted
    before the contradiction ValueError -- matching the original inline order
    (this path raises before reaching the table-boundary emission)."""
    with caplog.at_level("WARNING", logger="spyglass"):
        with pytest.raises(ValueError, match="cannot combine concat-source"):
            _classify(
                {
                    "interval_list_name": "raw data valid times",
                    "concat_recording_id": "c",
                },
                restrict_by_artifact=True,
            )
    assert any(
        "will NOT be artifact-restricted" in r.getMessage()
        for r in caplog.records
        if r.name == "spyglass"
    )


def test_classify_leaves_interval_name_as_recording_key_when_not_restricting():
    """``restrict_by_artifact=False`` leaves a non-artifact interval name as a
    plain recording restriction and emits no warning."""
    plan = _classify(
        {"interval_list_name": "raw data valid times"},
        restrict_by_artifact=False,
    )
    assert plan.rec_restriction == {
        "interval_list_name": "raw data valid times"
    }
    assert plan.unresolved_name_warning is None
    assert plan.artifact_detection_id is NO_ARTIFACT_RESTRICTION


def test_classify_does_not_emit_include_exclude_directive():
    """The plan carries only normalized restrictions + an artifact id + a
    warning flag -- no invented exclude/include vocabulary."""
    plan = _classify({"sorting_id": "s"})
    assert set(plan._fields) == {
        "rec_restriction",
        "concat_restriction",
        "shared_restriction",
        "sort_restriction",
        "curation_restriction",
        "artifact_detection_id",
        "restrict_by_artifact",
        "unresolved_name_warning",
    }
