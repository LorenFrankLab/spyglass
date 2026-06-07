"""Regression tests for v2 merge-id resolution by artifact (the A1 fix).

``SpikeSortingOutput._get_restricted_merge_ids_v2`` (and its public wrapper
``get_spiking_sorting_v2_merge_ids``) advertises ``artifact_id`` as a
restriction key, but ``artifact_id`` is NOT on ``SortingSelection`` -- it
lives on the optional ``SortingSelection.ArtifactSource`` part. Restricting
the recording/sorter join (``sort_master``) by ``artifact_id`` silently
dropped the key (verified: the compiled SQL is identical with and without
it), so two sorts differing only by artifact resolved to BOTH merge_ids.
The fix resolves ``artifact_id`` through the part and intersects on
``sorting_id`` when a UUID is requested. An absent ``artifact_id`` key is a
wildcard (no filter); an explicit ``artifact_id=None`` means "no artifact
pass" and anti-joins to sorts with no ``ArtifactSource`` row (the v2 design
makes part-row presence the artifact identity, so None is not a wildcard).

It also normalizes a str/UUID ``artifact_id`` to ``uuid.UUID`` (the column
is a uuid); the same str-vs-UUID class is fixed in
``SortingSelection.insert_selection``'s idempotency dedup (Task 4).

These tests build a second sort on the same recording as the package-scoped
``populated_sorting`` (which is artifact-backed) so the two differ only by
artifact state, then curate both into ``SpikeSortingOutput``.
"""

from __future__ import annotations

import pytest


def _clear_curations_for(sorting_key):
    """Drop a sorting's CurationV2 rows + merge masters (shared helper)."""
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    clear_curations_for(sorting_key)


@pytest.fixture(scope="module")
def two_sorts_one_recording(populated_sorting):
    """Two curated sorts on one recording, differing only by artifact state.

    ``sort_artifact`` is the package-scoped ``populated_sorting`` (an
    ArtifactSource part row is present); ``sort_no_artifact`` is a fresh
    selection on the same recording + sorter + params with NO artifact pass.
    Both are populated and curated into ``SpikeSortingOutput``.

    Yields a dict of recording id, the artifact id, both sorting ids, and
    both merge ids.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        Sorting,
        SortingSelection,
    )

    sort_artifact = populated_sorting
    recording_id = SortingSelection.resolve_source(sort_artifact).key[
        "recording_id"
    ]
    artifact_id = SortingSelection.resolve_artifact(sort_artifact)
    assert artifact_id is not None, "fixture sort must be artifact-backed"
    sorter, sorter_params_name = (SortingSelection & sort_artifact).fetch1(
        "sorter", "sorter_params_name"
    )

    sort_no_artifact = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
        }
    )
    if not (Sorting & sort_no_artifact):
        Sorting.populate(sort_no_artifact, reserve_jobs=False)

    # Fresh single root curation per sort -> one merge row each.
    _clear_curations_for(sort_artifact)
    _clear_curations_for(sort_no_artifact)
    cur_artifact = CurationV2.insert_curation(sorting_key=sort_artifact)
    cur_no_artifact = CurationV2.insert_curation(sorting_key=sort_no_artifact)
    merge_id_artifact = (SpikeSortingOutput.CurationV2 & cur_artifact).fetch1(
        "merge_id"
    )
    merge_id_no_artifact = (
        SpikeSortingOutput.CurationV2 & cur_no_artifact
    ).fetch1("merge_id")

    yield {
        "recording_id": recording_id,
        "artifact_id": artifact_id,
        "sorting_id_artifact": sort_artifact["sorting_id"],
        "sorting_id_no_artifact": sort_no_artifact["sorting_id"],
        "merge_id_artifact": merge_id_artifact,
        "merge_id_no_artifact": merge_id_no_artifact,
    }

    _clear_curations_for(sort_artifact)
    _clear_curations_for(sort_no_artifact)
    (Sorting & sort_no_artifact).super_delete(warn=False)
    (SortingSelection & sort_no_artifact).super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_artifact_restriction_is_exclusive(two_sorts_one_recording):
    """Restricting by ``artifact_id`` returns only the matching sort's
    merge_id -- not both sorts on the recording (the A1 fix)."""
    from spyglass.spikesorting.v2.utils import (
        get_spiking_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    # Sanity: with no artifact restriction, the recording matches both sorts.
    both = get_spiking_sorting_v2_merge_ids(
        {"recording_id": ctx["recording_id"]}
    )
    assert set(both) == {
        ctx["merge_id_artifact"],
        ctx["merge_id_no_artifact"],
    }

    # With the artifact restriction, only the artifact-backed sort resolves.
    restricted = get_spiking_sorting_v2_merge_ids(
        {
            "recording_id": ctx["recording_id"],
            "artifact_id": ctx["artifact_id"],
        }
    )
    assert list(restricted) == [ctx["merge_id_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_restrict_by_artifact_interval_name(
    two_sorts_one_recording,
):
    """The ``interval_list_name='artifact_{uuid}'`` convention resolves to
    the single matching merge_id (exercises the str->UUID cast)."""
    from spyglass.spikesorting.v2.utils import (
        get_spiking_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording
    interval_list_name = f"artifact_{ctx['artifact_id']}"

    resolved = get_spiking_sorting_v2_merge_ids(
        {
            "recording_id": ctx["recording_id"],
            "interval_list_name": interval_list_name,
        }
    )
    assert list(resolved) == [ctx["merge_id_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_artifact_id_none_selects_no_artifact_sort(
    two_sorts_one_recording,
):
    """``artifact_id=None`` means "no artifact pass" (anti-join), not a
    wildcard: it returns only the sort with NO ArtifactSource row, not both
    sorts on the recording."""
    from spyglass.spikesorting.v2.utils import (
        get_spiking_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    resolved = get_spiking_sorting_v2_merge_ids(
        {"recording_id": ctx["recording_id"], "artifact_id": None}
    )
    assert list(resolved) == [ctx["merge_id_no_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_no_artifact_sort_unaffected(two_sorts_one_recording):
    """A sort with NO ArtifactSource still resolves by rec/curation keys --
    the optional-part intersection is skipped when no artifact is asked
    for, so it is not dropped."""
    from spyglass.spikesorting.v2.utils import (
        get_spiking_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    resolved = get_spiking_sorting_v2_merge_ids(
        {"sorting_id": ctx["sorting_id_no_artifact"]}
    )
    assert list(resolved) == [ctx["merge_id_no_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_insert_selection_dedup_accepts_str_artifact_id(populated_sorting):
    """``insert_selection`` with a str ``artifact_id`` is idempotent.

    The find-existing dedup compares ``resolve_artifact`` (a ``uuid.UUID``)
    against the supplied ``artifact_id``; a str would never equal the stored
    UUID, so the second insert would miss its match and create a duplicate
    sort. The fix normalizes the supplied id to a UUID first.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    artifact_id = SortingSelection.resolve_artifact(populated_sorting)
    sorter, sorter_params_name = (SortingSelection & populated_sorting).fetch1(
        "sorter", "sorter_params_name"
    )

    before = {
        str(sid)
        for sid in (SortingSelection & {"sorter": sorter}).fetch("sorting_id")
    }

    found = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_id": str(artifact_id),  # str, not UUID
        }
    )

    after = {
        str(sid)
        for sid in (SortingSelection & {"sorter": sorter}).fetch("sorting_id")
    }
    # Idempotent: resolved to the existing sort, created no new row.
    assert found == populated_sorting
    assert after == before
