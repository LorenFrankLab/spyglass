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
