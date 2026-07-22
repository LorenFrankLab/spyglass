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


class TestBodyPartCollisionGuard:
    """Test BodyPart rejects a new spelling that collides with an existing one.

    A concept may have only one canonical spelling: inserting a different
    surface form of an existing part (same normalized key) must fail rather
    than create an ambiguous duplicate.
    """

    def test_colliding_spelling_raises(self, bodypart):
        """A new spelling of an existing part raises, naming both spellings.

        Uses a separator variant ('green_led'), which MySQL's case-insensitive
        collation treats as a distinct primary key from 'greenLED' but which
        normalizes to the same canonical key -- the case the guard must catch.
        """
        assert bodypart & {"bodypart": "greenLED"}
        with pytest.raises(dj.DataJointError) as exc:
            bodypart.insert1({"bodypart": "green_led"})
        msg = str(exc.value)
        assert "greenLED" in msg and "green_led" in msg

    def test_noncolliding_new_part_inserts(self, bodypart):
        """A genuinely new, non-colliding part still inserts (admin)."""
        name = "whiskerTipTest"
        assert not (bodypart & {"bodypart": name})
        try:
            bodypart.insert1({"bodypart": name})
            assert bodypart & {"bodypart": name}
        finally:
            (bodypart & {"bodypart": name}).delete(safemode=False)

    def test_exact_duplicate_warns_without_error(self, bodypart):
        """Re-inserting an existing exact spelling is idempotent (no raise)."""
        existing = bodypart.fetch("bodypart", limit=1)[0]
        # Should return quietly rather than raise a collision or FK error.
        bodypart.insert1({"bodypart": existing})

    def test_nonadmin_insert_raises_permission(self, bodypart, monkeypatch):
        """A non-admin inserting a novel part still hits the permission gate."""
        from spyglass.common import LabMember

        monkeypatch.setattr(
            LabMember, "user_is_admin", property(lambda self: False)
        )
        with pytest.raises(PermissionError):
            bodypart.insert1({"bodypart": "novelNonAdminPartXyz"})

    def test_canon_map_clean_table_returns_dict(self, bodypart):
        """canon_map() returns a normalized->canonical mapping when clean."""
        cmap = bodypart.canon_map()
        assert isinstance(cmap, dict)
        assert cmap.get("green led") == "greenLED"

    def test_canon_map_collision_gives_admin_guidance(self, bodypart):
        """A colliding pair in the table raises clear admin-actionable error.

        The duplicate is injected via bulk insert (bypassing the insert1
        guard) to mimic a pre-existing / admin-introduced inconsistency.
        """
        bodypart.insert(
            [{"bodypart": "green_led"}],
            allow_direct_insert=True,
            skip_duplicates=True,
        )
        try:
            with pytest.raises(dj.DataJointError) as exc:
                bodypart.canon_map()
            msg = str(exc.value)
            assert "admin" in msg.lower()
            assert "greenLED" in msg and "green_led" in msg
        finally:
            (bodypart & {"bodypart": "green_led"}).delete(safemode=False)


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
