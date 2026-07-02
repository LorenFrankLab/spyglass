"""Unit tests for the pure curation insert-plan builder.

``CurationV2.insert_curation`` delegates its pure half to ``_curation_plan``:
shape the post-merge ``Unit`` rows (via ``_curation_transforms``) and
validate the supplied label keys against the source/written unit sets.
These tests drive the builder + the stray-label decision matrix directly --
no DataJoint, no transaction -- so the misconfiguration paths (truly-stray
vs absorbed-contributor labels, raise vs warn-and-drop) are pinned cheaply.

``_curation_plan`` imports only ``_curation_transforms`` plus stdlib logging,
so this file needs no database fixture; the import-boundary contract is
separately enforced by ``test_service_import_contracts`` (``_curation_plan`` is
in ``_DB_FREE_SERVICE_MODULES``).
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._curation_plan import (
    build_curation_insert_plan,
    build_curation_summary,
    validate_label_unit_ids,
)
from spyglass.spikesorting.v2._curation_transforms import (
    normalize_curation_payload,
)

pytestmark = pytest.mark.unit


def _unit(uid, amp=10.0, n=100, eid=0):
    """A minimal ``Sorting.Unit`` row dict for the row builder."""
    return {
        "unit_id": uid,
        "nwb_file_name": "x.nwb",
        "electrode_group_name": "0",
        "electrode_id": eid,
        "peak_amplitude_uv": amp,
        "n_spikes": n,
    }


# ---------- normalize_curation_payload (FigURL/FigPack transport) ----------


def test_normalize_curation_payload_accepts_figurl_spellings():
    """The manual/FigURL-shaped payload normalizes to v2 insert inputs."""
    labels, merge_groups = normalize_curation_payload(
        {
            "labelsByUnit": {"1": ["noise", "reject"], "2": []},
            "mergeGroups": [["1", 3], [4, "5"]],
        }
    )
    assert labels == {1: ["noise", "reject"], 2: []}
    assert merge_groups == [[1, 3], [4, 5]]


def test_normalize_curation_payload_accepts_python_spellings_and_kwargs():
    """Snake-case payload keys and explicit kwargs share one normalizer."""
    labels, merge_groups = normalize_curation_payload(
        {"labels_by_unit": {"1": ["mua"]}, "merge_groups": [["1", "2"]]}
    )
    assert labels == {1: ["mua"]}
    assert merge_groups == [[1, 2]]

    labels, merge_groups = normalize_curation_payload(
        labels={3: ["accept"]}, merge_groups=[[3, 4]]
    )
    assert labels == {3: ["accept"]}
    assert merge_groups == [[3, 4]]


def test_normalize_curation_payload_accepts_association_merge_map():
    """The old per-unit association map deduplicates to full merge groups."""
    _labels, merge_groups = normalize_curation_payload(
        {"mergeGroups": {"1": ["2"], "2": ["1"], "3": []}}
    )
    assert merge_groups == [[1, 2]]


def test_normalize_curation_payload_unions_intersecting_associations():
    """A transitive association chain collapses to ONE group (v1 parity).

    v1's ``_merge_dict_to_list`` unions intersecting associations, so
    ``{1: [2], 2: [3]}`` is one group ``[1, 2, 3]`` -- not overlapping
    ``[1, 2]`` / ``[2, 3]`` that would later double-assign unit 2 and fail.
    Disjoint chains stay separate; pure singletons drop.
    """
    _labels, chain = normalize_curation_payload(
        {"mergeGroups": {"1": ["2"], "2": ["3"]}}
    )
    assert chain == [[1, 2, 3]]

    _labels, disjoint = normalize_curation_payload(
        merge_groups={1: [2], 4: [5]}
    )
    assert sorted(disjoint) == [[1, 2], [4, 5]]

    _labels, singletons = normalize_curation_payload(
        merge_groups={1: [], 2: []}
    )
    assert singletons == []


def test_normalize_curation_payload_rejects_duplicate_sources():
    """Payload and kwarg sources cannot both set the same semantic field."""
    with pytest.raises(ValueError, match="both inside payload and via labels"):
        normalize_curation_payload({"labelsByUnit": {"1": ["mua"]}}, labels={})

    with pytest.raises(
        ValueError, match="both inside payload and via merge_groups"
    ):
        normalize_curation_payload({"mergeGroups": [[1, 2]]}, merge_groups=[])


def test_normalize_curation_payload_rejects_scalar_label_value():
    """A string label must be wrapped in a list; do not split characters."""
    with pytest.raises(ValueError, match="must be a list of labels"):
        normalize_curation_payload({"labelsByUnit": {"1": "noise"}})


# ---------- validate_label_unit_ids (stray-label matrix) -------------------


def _validate(labels, source, written, *, permissive=False):
    validate_label_unit_ids(
        labels,
        source_unit_ids=set(source),
        written_unit_ids=set(written),
        sorting_id="s",
        apply_merge=False,
        allow_unknown_unit_ids=permissive,
    )


def test_labels_on_written_units_pass(caplog):
    """Keys that are all in the written set never raise or warn."""
    with caplog.at_level("WARNING", logger="spyglass"):
        _validate({0: ["mua"], 1: ["noise"]}, source={0, 1}, written={0, 1})
    assert not [r for r in caplog.records if r.name == "spyglass"]


def test_truly_stray_label_raises_by_default():
    """A key in neither source nor written is a typo -> raise by default."""
    with pytest.raises(ValueError, match="neither in the source unit set"):
        _validate({9: ["mua"]}, source={0, 1}, written={0, 1})


def test_truly_stray_label_warn_and_drop_when_permissive(caplog):
    """``allow_unknown_unit_ids=True`` downgrades the typo raise to warn-and-drop."""
    with caplog.at_level("WARNING", logger="spyglass"):
        _validate({9: ["mua"]}, source={0, 1}, written={0, 1}, permissive=True)
    messages = [r.getMessage() for r in caplog.records if r.name == "spyglass"]
    assert any(
        "neither in the source unit set" in msg and "ignored" in msg
        for msg in messages
    )


def test_absorbed_contributor_label_warns_not_raises(caplog):
    """A key in the source but absorbed (not written) warns, never raises --
    even with allow_unknown_unit_ids=False."""
    with caplog.at_level("WARNING", logger="spyglass"):
        _validate({1: ["mua"]}, source={0, 1}, written={0}, permissive=False)
    messages = [r.getMessage() for r in caplog.records if r.name == "spyglass"]
    assert any(
        "absorbed contributor" in msg and "[1]" in msg for msg in messages
    )


# ---------- build_curation_insert_plan -------------------------------------


def test_plan_no_merge_passthrough_threads_curation_id():
    """No merge groups: every unit passes through 1:1 and the resolved
    curation_id is stamped on each row."""
    plan = build_curation_insert_plan(
        sorting_id="s",
        sorting_units=[_unit(0), _unit(1)],
        merge_groups=None,
        apply_merge=False,
        labels={},
        allow_unknown_unit_ids=False,
        curation_id=3,
    )
    assert plan.curation_id == 3
    assert {r["unit_id"] for r in plan.unit_rows} == {0, 1}
    assert all(r["curation_id"] == 3 for r in plan.unit_rows)
    assert plan.kept_unit_to_contributors == {0: [0], 1: [1]}


def test_plan_apply_merge_assigns_fresh_id_and_inherits_peak():
    """apply_merge=True: the merged unit gets max(source)+1, absorbs its
    contributors, and inherits the highest-amplitude contributor's peak."""
    plan = build_curation_insert_plan(
        sorting_id="s",
        sorting_units=[_unit(0, amp=5.0), _unit(1, amp=20.0)],
        merge_groups=[[0, 1]],
        apply_merge=True,
        labels={},
        allow_unknown_unit_ids=False,
        curation_id=0,
    )
    assert plan.kept_unit_to_contributors == {2: [0, 1]}
    assert {r["unit_id"] for r in plan.unit_rows} == {2}
    merged = plan.unit_rows[0]
    assert merged["peak_amplitude_uv"] == 20.0  # from the louder contributor
    assert merged["n_spikes"] == 200  # summed contributor trains


def test_plan_truly_stray_label_raises():
    """The builder surfaces the stray-label typo guard end-to-end."""
    with pytest.raises(ValueError, match="neither in the source unit set"):
        build_curation_insert_plan(
            sorting_id="s",
            sorting_units=[_unit(0)],
            merge_groups=None,
            apply_merge=False,
            labels={9: ["mua"]},
            allow_unknown_unit_ids=False,
            curation_id=0,
        )


# ---------- build_curation_summary -----------------------------------------


def _summary(**overrides):
    """A ``build_curation_summary`` call with sensible defaults."""
    kwargs = dict(
        sorting_id="s",
        curation_id=0,
        merges_applied=False,
        description="d",
        labels={0: ["mua"]},
        merge_id=None,
        unit_contributor_groups={0: [0], 1: [1]},
        n_units=2,
    )
    kwargs.update(overrides)
    return build_curation_summary(**kwargs)


def test_build_curation_summary_is_merge_preview():
    """is_merge_preview is True iff not applied AND a >1-contributor group."""
    # Not applied + a real (>1-contributor) merge group -> preview.
    assert (
        _summary(merges_applied=False, unit_contributor_groups={0: [1, 2]})[
            "is_merge_preview"
        ]
        is True
    )
    # Applied -> never a preview, even with a >1-contributor group.
    assert (
        _summary(merges_applied=True, unit_contributor_groups={0: [1, 2]})[
            "is_merge_preview"
        ]
        is False
    )
    # Only single-contributor (self-entry) groups -> not a preview.
    assert (
        _summary(
            merges_applied=False,
            unit_contributor_groups={0: [0], 1: [1]},
        )["is_merge_preview"]
        is False
    )


def test_build_curation_summary_passthrough_and_coercions():
    """Fields pass through; curation_id/merges_applied/description are coerced."""
    summary = _summary(
        sorting_id="sort-uuid",
        curation_id=True,  # bool is an int subtype; int() -> 1
        merges_applied=1,  # truthy non-bool -> bool() -> True
        description=7,  # non-str -> str() -> "7"
        labels={3: ["accept"]},
        merge_id="merge-uuid",
        unit_contributor_groups={3: [3], 7: [7, 9]},
        n_units=5,
    )
    assert summary == {
        "sorting_id": "sort-uuid",
        "curation_id": 1,
        "n_units": 5,
        "labels": {3: ["accept"]},
        "merge_groups": {7: [7, 9]},
        "unit_contributor_groups": {3: [3], 7: [7, 9]},
        "merges_applied": True,
        "is_merge_preview": False,
        "merge_id": "merge-uuid",
        "description": "7",
    }


# ---------- compose_curation_labels (inherit / replace) --------------------


def _compose(parent, supplied, kept, *, policy="inherit", apply_merge=False):
    from spyglass.spikesorting.v2._curation_transforms import (
        compose_curation_labels,
    )

    return compose_curation_labels(
        parent_labels=parent,
        supplied_labels=supplied,
        kept_unit_to_contributors=kept,
        label_policy=policy,
        apply_merge=apply_merge,
    )


def test_compose_labels_root_inherit_is_supplied_only():
    """A root (empty parent_labels) under 'inherit' yields the supplied set."""
    out = _compose({}, {0: ["mua"]}, {0: [0], 1: [1]})
    assert out == {0: ["mua"]}


def test_compose_labels_child_inherits_parent_per_unit():
    """A preview child inherits each parent unit's own labels (no union)."""
    out = _compose(
        {2: ["mua"], 3: ["noise"]},
        {},
        {2: [2], 3: [3]},
        apply_merge=False,
    )
    assert out == {2: ["mua"], 3: ["noise"]}


def test_compose_labels_committed_merge_unions_contributor_labels():
    """A committed merge inherits the UNION of its contributors' labels."""
    out = _compose(
        {2: ["mua"], 3: ["noise"]},
        {},
        {4: [2, 3]},
        apply_merge=True,
    )
    assert out == {4: ["mua", "noise"]}


def test_compose_labels_explicit_override_replaces_inherited_unit():
    """A supplied entry overrides the inherited value for that unit only."""
    out = _compose(
        {2: ["mua"], 3: ["noise"]},
        {4: ["accept"]},
        {4: [2, 3]},
        apply_merge=True,
    )
    assert out == {4: ["accept"]}


def test_compose_labels_explicit_empty_clears_inherited():
    """A supplied empty list clears the inherited labels for that unit."""
    out = _compose(
        {2: ["mua"], 3: ["noise"]},
        {2: []},
        {2: [2], 3: [3]},
        apply_merge=False,
    )
    assert out == {3: ["noise"]}


def test_compose_labels_replace_drops_inherited():
    """'replace' makes the supplied labels the whole child state."""
    out = _compose(
        {2: ["mua"], 3: ["noise"]},
        {2: ["accept"]},
        {2: [2], 3: [3]},
        policy="replace",
    )
    assert out == {2: ["accept"]}


def test_compose_labels_rejects_bad_policy():
    with pytest.raises(ValueError, match="label_policy"):
        _compose({}, {}, {0: [0]}, policy="merge")
