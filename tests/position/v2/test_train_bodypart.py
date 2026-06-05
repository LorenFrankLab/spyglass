"""Tests for BodyPart table (train.py)."""

import datajoint as dj
import pytest


class TestBodyPartTable:
    """Test BodyPart lookup table functionality."""

    def test_bodypart_has_default_entries(self, bodypart):
        """Test that BodyPart table has default Frank Lab entries.

        Given: BodyPart table
        When: Fetching all bodyparts
        Then: Default bodyparts exist (LEDs, head parts, body parts)
        """
        all_parts = set(bodypart.fetch("bodypart"))

        # Check for key default bodyparts from train.py:119-147
        expected_parts = {
            "greenled",
            "redled c",
            "nose",
            "head",
            "tailbase",
        }

        # At least some default parts should exist
        assert len(all_parts & expected_parts) > 0


class TestBodyPartNormalization:
    """Test bodypart name normalization (used by Skeleton)."""

    def test_normalize_label_lowercase(self, skeleton):
        """Test normalization converts to lowercase."""
        assert skeleton._normalize_label("GreenLED") == "green led"
        assert skeleton._normalize_label("NOSE") == "nose"

    def test_normalize_label_camel_case(self, skeleton):
        """Test normalization handles camelCase properly.

        Given: Bodypart names in camelCase format
        When: Normalized
        Then: Spaces inserted before uppercase letters following lowercase
        """
        assert skeleton._normalize_label("FirstSecond") == "first second"
        assert skeleton._normalize_label("greenLED") == "green led"
        assert skeleton._normalize_label("leftEar") == "left ear"
        assert skeleton._normalize_label("tailBase") == "tail base"
        # Mixed camelCase and other separators
        assert skeleton._normalize_label("greenLED_C") == "green led c"
        assert skeleton._normalize_label("redLED-Left") == "red led left"

    def test_normalize_label_replaces(self, skeleton):
        """Test normalization replaces underscores/hyphens with spaces."""
        assert skeleton._normalize_label("green_led") == "green led"
        assert skeleton._normalize_label("left_ear") == "left ear"
        assert skeleton._normalize_label("green-led") == "green led"
        assert skeleton._normalize_label("tail-base") == "tail base"
        assert skeleton._normalize_label("  nose  ") == "nose"
        assert skeleton._normalize_label("\tgreen led\n") == "green led"
        assert skeleton._normalize_label("  Green_LED-C  ") == "green led c"
        assert skeleton._normalize_label("TAIL_BASE") == "tail base"


class TestBodyPartValidation:
    """Test bodypart validation in Skeleton._validate_bodyparts()."""

    def test_validate_bodyparts_accepts_valid(self, bodypart, skeleton):
        """Test validation passes for existing bodyparts."""
        # Get actual bodyparts from table
        valid_parts = set(bodypart.fetch("bodypart")[:3])
        skeleton._validate_bodyparts(valid_parts)

    def test_validate_bodyparts_rejects_invalid(self, skeleton):
        """Test validation fails for unknown bodyparts."""

        with pytest.raises(dj.DataJointError, match="Unknown bodypart"):
            skeleton._validate_bodyparts({"nonexistent_bodypart_xyz123"})

    def test_validate_bodyparts_error_message_lists_missing(self, skeleton):
        """Test error message lists missing bodyparts."""
        with pytest.raises(
            dj.DataJointError, match="invalid1|invalid2"
        ) as exc_info:
            skeleton._validate_bodyparts({"invalid1", "invalid2"})

        assert "admin" in str(exc_info.value).lower()

    def test_validate_bodyparts_mixed_valid_invalid(self, bodypart, skeleton):
        """Test validation with mix of valid and invalid bodyparts."""
        # Get one valid bodypart
        valid_part = bodypart.fetch("bodypart", limit=1)[0]
        valid_part_norm = skeleton._normalize_label(valid_part)

        mixed_parts = {valid_part_norm, "definitely_invalid_xyz"}

        with pytest.raises(dj.DataJointError, match="definitely invalid xyz"):
            skeleton._validate_bodyparts(mixed_parts)
