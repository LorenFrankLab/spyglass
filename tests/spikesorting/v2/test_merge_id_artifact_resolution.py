"""Regression tests for v2 merge-id resolution by artifact detection.

``SpikeSortingOutput._get_restricted_merge_ids_v2`` (and its public wrapper
``get_spike_sorting_v2_merge_ids``) advertises ``artifact_detection_id`` as a
restriction key, but ``artifact_detection_id`` is NOT on ``SortingSelection`` -- it
lives on the optional ``SortingSelection.ArtifactDetectionSource`` part. Restricting
the recording/sorter join (``sort_master``) by ``artifact_detection_id`` silently
dropped the key (verified: the compiled SQL is identical with and without
it), so two sorts differing only by artifact resolved to BOTH merge_ids.
The fix resolves ``artifact_detection_id`` through the part and intersects on
``sorting_id`` when a UUID is requested. An absent ``artifact_detection_id`` key is a
wildcard (no filter); an explicit ``artifact_detection_id=None`` means
"no artifact-detection pass" and anti-joins to sorts with no
``ArtifactDetectionSource`` row (the v2 design makes part-row presence the
artifact-detection identity, so None is not a wildcard).

It also normalizes a str/UUID ``artifact_detection_id`` to ``uuid.UUID`` (the column
is a uuid); the same str-vs-UUID class is fixed in
``SortingSelection.insert_selection``'s idempotency dedup.

These tests build a second sort on the same recording as the package-scoped
``populated_sorting`` (which is artifact-detection-backed) so the two differ
only by artifact-detection state, then curate both into ``SpikeSortingOutput``.
"""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import (
    clear_curations_for as _clear_curations_for,
)


@pytest.fixture(scope="module")
def two_sorts_one_recording(populated_sorting):
    """Two curated sorts on one recording, differing only by artifact-detection state.

    ``sort_artifact`` is the package-scoped ``populated_sorting`` (an
    ArtifactDetectionSource part row is present); ``sort_no_artifact`` is a fresh
    selection on the same recording + sorter + params with NO artifact-detection pass.
    Both are populated and curated into ``SpikeSortingOutput``.

    Yields a dict of recording id, the artifact-detection id, both sorting
    ids, and both merge ids.
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
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        sort_artifact
    )
    assert (
        artifact_detection_id is not None
    ), "fixture sort must be artifact-detection-backed"
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
        "artifact_detection_id": artifact_detection_id,
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
    """Restricting by ``artifact_detection_id`` returns only the matching sort's
    merge_id -- not both sorts on the recording."""
    from spyglass.spikesorting.v2.utils import (
        get_spike_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    # Sanity: with no artifact-detection restriction, the recording matches both sorts.
    both = get_spike_sorting_v2_merge_ids({"recording_id": ctx["recording_id"]})
    assert set(both) == {
        ctx["merge_id_artifact"],
        ctx["merge_id_no_artifact"],
    }

    # With the artifact-detection restriction, only the matching sort resolves.
    restricted = get_spike_sorting_v2_merge_ids(
        {
            "recording_id": ctx["recording_id"],
            "artifact_detection_id": ctx["artifact_detection_id"],
        }
    )
    assert list(restricted) == [ctx["merge_id_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_distinguish_two_artifact_backed_sorts(
    two_sorts_one_recording,
):
    """Two artifact-backed sorts on one recording with DIFFERENT artifact ids:
    restricting by one ``artifact_detection_id`` returns ONLY that sort.

    Regression for the ``CurationV2.resolve_restriction`` bug where the
    ``ArtifactDetectionSource`` part was restricted by ``artifact_detection_id``
    -- a key dropped after the merge-FK rename to ``artifact_detection_merge_id``
    -- collapsing the filter to "any artifact-backed sort". The base
    ``two_sorts_one_recording`` fixture (one artifact + one no-artifact sort)
    masks the bug because "all artifact-backed" then coincides with "the one
    artifact"; this test adds a SECOND, differently-detected artifact sort so a
    dropped-key restriction returns two merge ids where one is required.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.artifact_output import (
        ArtifactDetectionOutput,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import get_spike_sorting_v2_merge_ids

    ctx = two_sorts_one_recording
    recording_id = ctx["recording_id"]
    id_none = ctx["artifact_detection_id"]  # the fixture's "none" artifact
    sorter, sorter_params_name = (
        SortingSelection & {"sorting_id": ctx["sorting_id_artifact"]}
    ).fetch1("sorter", "sorter_params_name")

    # A SECOND artifact detection on the same recording, different params ->
    # a distinct artifact_detection_id -> a second artifact-backed sort.
    ArtifactDetectionParameters.insert_default()
    art2 = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": recording_id,
            "artifact_detection_params_name": "default",
        }
    )
    id_default = art2["artifact_detection_id"]
    assert str(id_default) != str(id_none)
    if not (RecordingArtifactDetection & art2):
        RecordingArtifactDetection.populate(art2, reserve_jobs=False)
    sort2 = SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": sorter,
            "sorter_params_name": sorter_params_name,
            "artifact_detection_id": id_default,
        }
    )
    try:
        if not (Sorting & sort2):
            Sorting.populate(sort2, reserve_jobs=False)
        _clear_curations_for(sort2)
        cur2 = CurationV2.insert_curation(sorting_key=sort2)
        merge_id_default = (SpikeSortingOutput.CurationV2 & cur2).fetch1(
            "merge_id"
        )

        # Restricting by the "none" artifact must return ONLY its sort -- the
        # dropped-key bug would also return the "default" sort.
        by_none = set(
            get_spike_sorting_v2_merge_ids(
                {
                    "recording_id": recording_id,
                    "artifact_detection_id": id_none,
                }
            )
        )
        assert ctx["merge_id_artifact"] in by_none
        assert merge_id_default not in by_none, (
            "resolve_restriction matched a DIFFERENT artifact's sort -- the "
            "ArtifactDetectionSource restriction is keyed on the wrong column"
        )
        # ...and restricting by "default" returns ONLY the default sort.
        by_default = set(
            get_spike_sorting_v2_merge_ids(
                {
                    "recording_id": recording_id,
                    "artifact_detection_id": id_default,
                }
            )
        )
        assert by_default == {merge_id_default}
    finally:
        _clear_curations_for(sort2)
        (Sorting & sort2).super_delete(warn=False)
        (SortingSelection & sort2).super_delete(warn=False)
        try:
            _mid = ArtifactDetectionOutput.get_merge_id(art2)
            (ArtifactDetectionOutput & {"merge_id": _mid}).super_delete(
                warn=False, force_masters=True
            )
        except KeyError:
            pass
        (RecordingArtifactDetection & art2).super_delete(warn=False)
        (RecordingArtifactSelection & art2).super_delete(warn=False)


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_restrict_by_artifact_interval_name(
    two_sorts_one_recording,
):
    """The ``interval_list_name='artifact_detection_{uuid}'`` convention resolves to
    the single matching merge_id (exercises the str->UUID cast)."""
    from spyglass.spikesorting.v2.utils import (
        get_spike_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording
    interval_list_name = f"artifact_detection_{ctx['artifact_detection_id']}"

    resolved = get_spike_sorting_v2_merge_ids(
        {
            "recording_id": ctx["recording_id"],
            "interval_list_name": interval_list_name,
        }
    )
    assert list(resolved) == [ctx["merge_id_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_artifact_detection_id_none_selects_no_artifact_sort(
    two_sorts_one_recording,
):
    """``artifact_detection_id=None`` means "no artifact-detection pass".

    It anti-joins: returns only the sort with NO ArtifactDetectionSource row,
    not both sorts on the recording.
    """
    from spyglass.spikesorting.v2.utils import (
        get_spike_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    resolved = get_spike_sorting_v2_merge_ids(
        {"recording_id": ctx["recording_id"], "artifact_detection_id": None}
    )
    assert list(resolved) == [ctx["merge_id_no_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_merge_ids_no_artifact_sort_unaffected(two_sorts_one_recording):
    """A sort with NO ArtifactDetectionSource still resolves by rec/curation keys --
    the optional-part intersection is skipped when no artifact detection is asked
    for, so it is not dropped."""
    from spyglass.spikesorting.v2.utils import (
        get_spike_sorting_v2_merge_ids,
    )

    ctx = two_sorts_one_recording

    resolved = get_spike_sorting_v2_merge_ids(
        {"sorting_id": ctx["sorting_id_no_artifact"]}
    )
    assert list(resolved) == [ctx["merge_id_no_artifact"]]


@pytest.mark.slow
@pytest.mark.integration
def test_insert_selection_dedup_accepts_str_artifact_detection_id(
    populated_sorting,
):
    """``insert_selection`` with a str ``artifact_detection_id`` is idempotent.

    The find-existing dedup compares ``resolve_artifact_detection`` (a ``uuid.UUID``)
    against the supplied ``artifact_detection_id``; a str would never equal the stored
    UUID, so the second insert would miss its match and create a duplicate
    sort. The fix normalizes the supplied id to a UUID first.
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
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
            "artifact_detection_id": str(
                artifact_detection_id
            ),  # str, not UUID
        }
    )

    after = {
        str(sid)
        for sid in (SortingSelection & {"sorter": sorter}).fetch("sorting_id")
    }
    # Idempotent: resolved to the existing sort, created no new row.
    assert found == populated_sorting
    assert after == before


@pytest.mark.slow
@pytest.mark.integration
def test_v2_merge_ids_warn_on_multi_curation_fanout(
    populated_sorting, monkeypatch
):
    """When a sorting has >1 CurationV2 curation and no curation_id is given,
    get_restricted_merge_ids returns one merge_id PER curation AND warns -- so
    a caller does not silently double-count the same units in a
    SortedSpikesGroup / decode."""
    import spyglass.spikesorting.spikesorting_merge as merge_mod
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clear_curations_for(populated_sorting)
    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    if not unit_ids:
        pytest.skip("need >=1 unit to build a distinct child curation")

    # Two curations on ONE sorting: a root + a labeled child.
    CurationV2.insert_curation(sorting_key=populated_sorting)
    CurationV2.insert_curation(
        sorting_key=populated_sorting,
        parent_curation_id=0,
        labels={unit_ids[0]: ["mua"]},
    )
    sid = populated_sorting["sorting_id"]

    seen: list[str] = []
    monkeypatch.setattr(
        merge_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )

    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"sorting_id": sid}, sources=["v2"]
    )
    try:
        assert (
            len(merge_ids) == 2
        ), f"two curations should fan out to two merge_ids; got {merge_ids}"
        assert any(
            "multiple CurationV2 curations" in s for s in seen
        ), f"expected a multi-curation fan-out warning; got {seen}"
    finally:
        _clear_curations_for(populated_sorting)


@pytest.mark.slow
@pytest.mark.integration
def test_v2_merge_ids_single_curation_does_not_warn(
    populated_sorting, monkeypatch
):
    """A sorting with exactly ONE curation must NOT emit the fan-out warning
    (negative control for test_v2_merge_ids_warn_on_multi_curation_fanout)."""
    import spyglass.spikesorting.spikesorting_merge as merge_mod
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    _clear_curations_for(populated_sorting)
    CurationV2.insert_curation(sorting_key=populated_sorting)
    sid = populated_sorting["sorting_id"]

    seen: list[str] = []
    monkeypatch.setattr(
        merge_mod.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    merge_ids = SpikeSortingOutput().get_restricted_merge_ids(
        {"sorting_id": sid}, sources=["v2"]
    )
    try:
        assert len(merge_ids) == 1
        assert not any(
            "multiple CurationV2 curations" in s for s in seen
        ), f"single-curation sort must not warn about fan-out; got {seen}"
    finally:
        _clear_curations_for(populated_sorting)


@pytest.mark.slow
@pytest.mark.integration
def test_assert_decoding_merge_ids_raises_on_fanout(populated_sorting):
    """The consumer-boundary validator RAISES when >1 merge_id maps to one
    sorting (the fan-out get_restricted_merge_ids only warns about)."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = sorted(
        int(u) for u in (Sorting.Unit & populated_sorting).fetch("unit_id")
    )
    if not unit_ids:
        pytest.skip("need >=1 unit for a distinct child curation")

    _clear_curations_for(populated_sorting)
    CurationV2.insert_curation(sorting_key=populated_sorting)
    CurationV2.insert_curation(
        sorting_key=populated_sorting,
        parent_curation_id=0,
        labels={unit_ids[0]: ["mua"]},
    )
    sid = populated_sorting["sorting_id"]
    merge_ids = list(
        SpikeSortingOutput().get_restricted_merge_ids(
            {"sorting_id": sid}, sources=["v2"]
        )
    )
    try:
        assert len(merge_ids) == 2
        with pytest.raises(ValueError, match="same sorting"):
            SpikeSortingOutput.assert_decoding_merge_ids_ok(merge_ids)
        # A single (one-curation) merge_id is accepted.
        SpikeSortingOutput.assert_decoding_merge_ids_ok([merge_ids[0]])
    finally:
        _clear_curations_for(populated_sorting)
