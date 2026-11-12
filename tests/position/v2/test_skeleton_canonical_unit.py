"""Unit tests for body-part name canonicalization primitives.

Covers build_canonical_map and canonicalize in
spyglass.position.v2.utils.skeleton. All tests are pure (no database).
"""

import pytest


class TestBuildCanonicalMap:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.utils.skeleton import build_canonical_map

        self.fn = build_canonical_map

    def test_maps_normalized_key_to_raw_spelling(self):
        assert self.fn(["greenLED", "earR"]) == {
            "green led": "greenLED",
            "ear r": "earR",
        }

    def test_repeated_identical_spelling_is_not_a_collision(self):
        # Same raw spelling twice collapses to one entry, no error.
        assert self.fn(["greenLED", "greenLED"]) == {"green led": "greenLED"}

    def test_distinct_spellings_same_normalized_key_raise(self):
        import datajoint as dj

        with pytest.raises(dj.DataJointError) as exc:
            self.fn(["greenLED", "greenLed"])
        msg = str(exc.value)
        assert "greenLED" in msg and "greenLed" in msg

    def test_values_are_always_input_spellings(self):
        known = ["redLED_C", "tailBase"]
        result = self.fn(known)
        assert set(result.values()) == set(known)
        # keys are normalized, values are never the normalized form
        assert "red led c" in result and result["red led c"] == "redLED_C"


class TestCanonicalize:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.utils.skeleton import (
            build_canonical_map,
            canonicalize,
        )

        self.fn = canonicalize
        self.cmap = build_canonical_map(["greenLED", "earR", "redLED_C"])

    def test_resolves_camelcase_surface_form(self):
        assert self.fn("EarR", self.cmap) == "earR"

    def test_resolves_snake_and_spacing_variants(self):
        assert self.fn("green_led", self.cmap) == "greenLED"
        assert self.fn("  green   LED ", self.cmap) == "greenLED"
        assert self.fn("red-led-c", self.cmap) == "redLED_C"

    def test_miss_returns_default(self):
        assert self.fn("nose", self.cmap) is None
        assert self.fn("nose", self.cmap, default="nose") == "nose"

    def test_never_returns_normalized_form(self):
        # A hit always yields a canonical (input) spelling, never 'ear r'.
        result = self.fn("ear_r", self.cmap)
        assert result == "earR"
        assert result in self.cmap.values()


class TestShapeHashStability:
    """The shape hash is computed here, so a networkx release cannot move it.

    `nx.weisfeiler_lehman_graph_hash` changed its output in networkx 3.5,
    which left the constants in `Skeleton.contents` disagreeing with a fresh
    install on Python 3.11 or later.
    """

    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.utils.skeleton import shape_hash_from_edges

        self.fn = shape_hash_from_edges

    def test_known_shapes_have_pinned_values(self):
        """Hard-coded, so a change to the canonical form fails here first.

        These are the three values `Skeleton.contents` ships.
        """
        assert self.fn(["whiteLED"], []) == "c305e12973b3d9ed3c36d38dd0fb895b"
        assert (
            self.fn(["greenLED", "redLED_C"], [("greenLED", "redLED_C")])
            == "41b6347725857673189b4dd0d174abb5"
        )
        assert (
            self.fn(
                ["greenLED", "redLED_C", "redLED_L", "redLED_R"],
                [
                    ("greenLED", "redLED_C"),
                    ("greenLED", "redLED_L"),
                    ("greenLED", "redLED_R"),
                ],
            )
            == "4ab61d4314b09e9227229f9166f15135"
        )

    def test_hash_ignores_labels(self):
        """Topology only, so a renamed part keeps its bucket."""
        assert self.fn(["a", "b"], [("a", "b")]) == self.fn(
            ["x", "y"], [("x", "y")]
        )

    def test_same_degree_sequence_collides(self):
        """A collision is expected and is the caller's job to settle.

        Two disjoint triangles and one hexagon are both 2-regular on six
        nodes, so they share node count, edge count and degree sequence while
        being non-isomorphic. They share a bucket, and
        `is_duplicate_skeleton` separates them.
        """
        from spyglass.position.v2.utils.skeleton import is_duplicate_skeleton

        nodes = ["a", "b", "c", "d", "e", "f"]
        triangles = [
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "e"),
            ("e", "f"),
            ("f", "d"),
        ]
        hexagon = [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("f", "a"),
        ]
        assert self.fn(nodes, triangles) == self.fn(nodes, hexagon)
        assert not is_duplicate_skeleton(
            nodes, triangles, nodes, hexagon
        ), "the exact comparison must separate what the hash buckets together"

    def test_edge_count_separates_shapes(self):
        """Node count alone is not the key."""
        assert self.fn(["a", "b"], []) != self.fn(["a", "b"], [("a", "b")])
