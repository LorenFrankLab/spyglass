"""Hermetic (no-DB, no-figpack) tests for the FigPack curation helpers.

Covers the DB-free logic in ``_figpack_curation``: the default label palette,
the content-addressed config hash, and the round trip between FigPack's
``sorting_curation`` annotation state and v2's ``(labels, merge_groups)`` form.
These need neither a DataJoint server nor the optional ``figpack`` packages.
"""

from __future__ import annotations

import json

import pytest

from spyglass.spikesorting.v2._figpack_curation import (
    FIGPACK_INSTALL_HINT,
    annotations_payload_hash,
    curation_annotations_to_labels_and_merges,
    default_label_options,
    figpack_config_hash,
    labels_and_merges_to_annotations,
    normalize_displayed_unit_properties,
    pack_display_config,
    unpack_display_config,
)

_SORTING_ID = "11111111-2222-3333-4444-555555555555"


@pytest.mark.parametrize("bad_id", [1.9, 1.0, True, False])
def test_annotations_reject_non_integer_merge_ids(bad_id):
    annotations = labels_and_merges_to_annotations({}, [[bad_id, 2]])
    with pytest.raises(ValueError, match="unit_id must be an integer"):
        curation_annotations_to_labels_and_merges(annotations)


def _hash(**overrides):
    base = dict(
        sorting_id=_SORTING_ID,
        curation_id=0,
        label_options=["accept", "mua", "noise"],
        displayed_unit_properties=None,
        upload=False,
        ephemeral=False,
    )
    base.update(overrides)
    return figpack_config_hash(**base)


def test_default_label_options_is_curation_order():
    """Default palette is the three primary labels, not FigURL-era 'good'."""
    assert default_label_options() == ["accept", "mua", "noise"]


def test_config_hash_is_deterministic():
    """Same configuration -> same 64-char digest."""
    digest = _hash()
    assert digest == _hash()
    assert len(digest) == 64


def test_config_hash_sensitive_to_every_field():
    """Each config field changes the hash (no silent aliasing)."""
    base = _hash()
    assert _hash(curation_id=1) != base
    assert _hash(curation_uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") != base
    assert _hash(label_options=["mua", "accept", "noise"]) != base  # order
    assert _hash(displayed_unit_properties=["x", "y"]) != base
    assert _hash(displayed_unit_properties=["y", "x"]) != _hash(
        displayed_unit_properties=["x", "y"]
    )
    assert _hash(displayed_unit_properties=[]) != base
    assert _hash(upload=True) != base
    assert _hash(ephemeral=True) != base
    assert _hash(sorting_id="99999999-2222-3333-4444-555555555555") != base
    assert _hash(review_config={"profile_hash": "a"}) != base
    assert _hash(review_config={"profile_hash": "b"}) != _hash(
        review_config={"profile_hash": "a"}
    )


def test_review_config_packs_without_changing_expert_storage_shape():
    """Review snapshots share the blob while legacy expert lists stay lists."""
    assert pack_display_config(["x", "y"]) == ["x", "y"]
    packed = pack_display_config(
        ["snr"], {"profile_hash": "abc", "spec": ("minimal", "none")}
    )
    properties, review = unpack_display_config(packed)
    assert properties == ["snr"]
    assert review == {
        "profile_hash": "abc",
        "spec": ["minimal", "none"],
    }


def test_annotations_hash_is_logical_not_json_spacing():
    """TOCTOU hash is stable across mapping order and serialization spacing."""
    first = {"annotations": {"/": {"b": 2, "a": 1}}}
    second = json.loads(json.dumps(first, indent=4, sort_keys=False))
    assert annotations_payload_hash(first) == annotations_payload_hash(second)
    second["annotations"]["/"]["a"] = 3
    assert annotations_payload_hash(first) != annotations_payload_hash(second)


def test_displayed_unit_properties_normalization():
    """None, explicit empty, and ordered lists remain distinct configs."""
    assert normalize_displayed_unit_properties(None) is None
    assert normalize_displayed_unit_properties([]) == []
    assert normalize_displayed_unit_properties(("x", "y")) == ["x", "y"]
    with pytest.raises(ValueError):
        normalize_displayed_unit_properties(["x", "x"])
    with pytest.raises(ValueError):
        normalize_displayed_unit_properties([""])
    with pytest.raises(TypeError):
        normalize_displayed_unit_properties("x")
    with pytest.raises(TypeError):
        normalize_displayed_unit_properties(["x", 1])


def test_curation_state_round_trips():
    """labels + merge groups survive the annotations encode/decode round trip."""
    labels = {0: ["noise"], 1: ["accept", "mua"]}
    merge_groups = [[2, 3], [4, 5, 6]]
    payload = labels_and_merges_to_annotations(
        labels, merge_groups, label_options=["accept", "mua", "noise"]
    )
    got_labels, got_merges = curation_annotations_to_labels_and_merges(payload)
    assert got_labels == labels
    assert got_merges == merge_groups


def test_parse_coerces_unit_ids_to_int():
    """FigPack stores labelsByUnit keys as strings; parse returns int ids."""
    payload = {
        "annotations": {
            "/": {
                "sorting_curation": (
                    '{"labelsByUnit": {"7": ["mua"]}, "mergeGroups": [["8","9"]]}'
                )
            }
        }
    }
    labels, merges = curation_annotations_to_labels_and_merges(payload)
    assert labels == {7: ["mua"]}
    assert merges == [[8, 9]]


def test_parse_empty_and_missing_yield_empty():
    """A pristine / absent / empty figure round-trips to ({}, [])."""
    assert curation_annotations_to_labels_and_merges(None) == ({}, [])
    assert curation_annotations_to_labels_and_merges({}) == ({}, [])
    assert curation_annotations_to_labels_and_merges(
        {"annotations": {"/": {}}}
    ) == ({}, [])
    assert curation_annotations_to_labels_and_merges(
        {"annotations": {"/": {"sorting_curation": "{}"}}}
    ) == ({}, [])


def test_parse_accepts_dict_state_not_only_json_string():
    """The sorting_curation value may already be a dict, not a JSON string."""
    payload = {
        "annotations": {
            "/": {"sorting_curation": {"labelsByUnit": {"1": ["accept"]}}}
        }
    }
    labels, merges = curation_annotations_to_labels_and_merges(payload)
    assert labels == {1: ["accept"]}
    assert merges == []


def test_install_hint_names_the_extra():
    """The install hint points at the optional curation extra."""
    assert "spikesorting-v2-curation" in FIGPACK_INSTALL_HINT
