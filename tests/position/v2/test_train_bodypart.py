"""Tests for BodyPart table (train.py)."""

import datajoint as dj
import pytest


@pytest.fixture
def skel_utils():
    """The skeleton primitives, called directly rather than via the table.

    `Skeleton` used to carry thin instance-method wrappers for these purely
    so tests could reach them; the wrappers are gone.
    """
    from spyglass.position.v2.utils import skeleton as skel

    return skel


class TestBodyPartTable:
    """Test BodyPart lookup table functionality."""

    def test_bodypart_has_default_entries(self, bodypart):
        """Test that BodyPart table has default Frank Lab entries.

        Given: BodyPart table
        When: Fetching all bodyparts
        Then: Default bodyparts exist (LEDs, head parts, body parts)
        """
        all_parts = set(bodypart.fetch("bodypart"))

        # Canonical default spellings from BodyPart.contents (train.py:100-128).
        # fetch returns the stored spelling, so compare against exact canonical
        # forms rather than normalized ones.
        expected_parts = {
            "greenLED",
            "redLED_C",
            "nose",
            "head",
            "tailBase",
        }

        assert expected_parts <= all_parts


class TestBodyPartNormalization:
    """Test bodypart name normalization (used by Skeleton)."""

    def test_normalize_label_lowercase(self, skel_utils):
        """Test normalization converts to lowercase."""
        assert skel_utils.normalize_label("GreenLED") == "green led"
        assert skel_utils.normalize_label("NOSE") == "nose"

    def test_normalize_label_camel_case(self, skel_utils):
        """Test normalization handles camelCase properly.

        Given: Bodypart names in camelCase format
        When: Normalized
        Then: Spaces inserted before uppercase letters following lowercase
        """
        assert skel_utils.normalize_label("FirstSecond") == "first second"
        assert skel_utils.normalize_label("greenLED") == "green led"
        assert skel_utils.normalize_label("leftEar") == "left ear"
        assert skel_utils.normalize_label("tailBase") == "tail base"
        # Mixed camelCase and other separators
        assert skel_utils.normalize_label("greenLED_C") == "green led c"
        assert skel_utils.normalize_label("redLED-Left") == "red led left"

    def test_normalize_label_replaces(self, skel_utils):
        """Test normalization replaces underscores/hyphens with spaces."""
        assert skel_utils.normalize_label("green_led") == "green led"
        assert skel_utils.normalize_label("left_ear") == "left ear"
        assert skel_utils.normalize_label("green-led") == "green led"
        assert skel_utils.normalize_label("tail-base") == "tail base"
        assert skel_utils.normalize_label("  nose  ") == "nose"
        assert skel_utils.normalize_label("\tgreen led\n") == "green led"
        assert skel_utils.normalize_label("  Green_LED-C  ") == "green led c"
        assert skel_utils.normalize_label("TAIL_BASE") == "tail base"


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
        normalizes to the same canonical key, the case the guard must catch.
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
        assert cmap.get("red led c") == "redLED_C"
        assert cmap.get("tail base") == "tailBase"
        assert cmap.get("nose") == "nose"

    def test_canon_map_collision_gives_admin_guidance(self, bodypart):
        """A colliding pair in the table raises clear admin-actionable error.

        The duplicate goes in through DataJoint's own `insert`, since no
        path on `BodyPart` produces one: `insert1` and the plural `insert`
        both refuse it. That models a row written before the guard existed,
        or by an admin working outside the table's API.
        """
        dj.Table.insert(
            bodypart,
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


class TestLookupBulkInsert:
    """`insert([...])` routes through `insert1` on the Lookup tables.

    `dj.Table.insert1` delegates to `insert`, so without the override a
    plural insert reaches the database with none of the validation. The
    declared `contents` are exempt: DataJoint fills a Lookup through
    `insert` at schema activation, before any user code runs.
    """

    def test_bulk_insert_rejects_a_colliding_spelling(self, bodypart):
        """The collision guard applies to every row of a bulk call."""
        assert bodypart & {"bodypart": "greenLED"}
        before = set(bodypart.fetch("bodypart"))

        with pytest.raises(dj.DataJointError, match="collides with existing"):
            bodypart.insert([{"bodypart": "green_led"}])

        assert set(bodypart.fetch("bodypart")) == before

    def test_bulk_insert_applies_the_admin_check(self, bodypart, monkeypatch):
        """A non-admin bulk insert is refused, as the singular one is."""
        from spyglass.common import LabMember

        monkeypatch.setattr(
            LabMember, "user_is_admin", property(lambda self: False)
        )
        with pytest.raises(PermissionError, match="Only admins"):
            bodypart.insert([{"bodypart": "bulkNovelPartXyz"}])

    def test_contents_are_exempt(self, bodypart, monkeypatch):
        """Schema activation inserts `contents`, so it skips the checks.

        A non-admin importing spyglass must not trip the admin check.
        """
        from spyglass.common import LabMember

        monkeypatch.setattr(
            LabMember, "user_is_admin", property(lambda self: False)
        )
        bodypart.insert(list(type(bodypart).contents), skip_duplicates=True)

        with pytest.raises(PermissionError):
            bodypart.insert([{"bodypart": "notContentsXyz"}])

    def test_skeleton_bulk_insert_validates(self, skeleton):
        """An unknown body part is refused through the plural path."""
        with pytest.raises(dj.DataJointError):
            skeleton.insert(
                [
                    {
                        "skeleton_id": "bulk_unknown_probe",
                        "bodyparts": ["definitelyNovelXyz", "nose"],
                        "edges": [("definitelyNovelXyz", "nose")],
                    }
                ],
                check_duplicates=False,
            )

    def test_model_params_bulk_insert_rejects_unknown_tool(self, model_params):
        """An unsupported tool is refused through the plural path."""
        with pytest.raises(ValueError, match="Tool not supported"):
            model_params.insert(
                [
                    {
                        "model_params_id": "bulk_bad_tool",
                        "tool": "NOT_A_TOOL",
                        "params": {},
                    }
                ]
            )


class TestBodyPartValidation:
    """Test bodypart validation in Skeleton._validate_bodyparts()."""

    def test_validate_bodyparts_accepts_valid(self, bodypart, skeleton):
        """Test validation passes for existing bodyparts."""
        # Get actual bodyparts from table
        valid_parts = set(bodypart.fetch("bodypart")[:3])
        assert skeleton._validate_bodyparts(valid_parts) is True

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

    def test_validate_bodyparts_mixed_valid_invalid(
        self, bodypart, skeleton, skel_utils
    ):
        """Test validation with mix of valid and invalid bodyparts."""
        # Get one valid bodypart
        valid_part = bodypart.fetch("bodypart", limit=1)[0]
        valid_part_norm = skel_utils.normalize_label(valid_part)

        mixed_parts = {valid_part_norm, "definitely_invalid_xyz"}

        with pytest.raises(dj.DataJointError, match="definitely invalid xyz"):
            skeleton._validate_bodyparts(mixed_parts)


class TestAcceptNewBodypartsScope:
    """`accept_new_bodyparts=True` must skip *only* the admin check.

    It auto-registers unresolved parts via ``BodyPart().insert()`` (plural),
    which ``BodyPart`` does not override, so it cannot reach the collision
    guard in ``insert1``. That is harmless while ``canonicalize`` and the guard
    derive their key from the same ``normalize_label``: a name that fails to
    resolve also cannot collide. These tests pin that equivalence, which is
    what keeps the flag honest. Scoping ``canon_map`` to curated-only parts
    opens the hole.
    """

    def test_unresolved_implies_no_collision(self, bodypart):
        """The invariant the flag leans on, asserted over the real table.

        For every curated part, a name that ``canonicalize`` cannot resolve
        must also have no collision under ``insert1``'s rule. If this ever
        fails, ``accept_new_bodyparts`` is inserting past the guard.
        """
        from spyglass.position.v2.utils.skeleton import (
            canonicalize,
            normalize_label,
        )

        names = [str(b) for b in bodypart.fetch("bodypart")]
        cmap = bodypart.canon_map()

        probes = ["mouse_center", "tail1", "left_ear_tip", "wholly_novel_xyz"]
        for probe in probes:
            if canonicalize(probe, cmap) is not None:
                continue  # resolves; never reaches the insert
            norm = normalize_label(probe)
            collisions = [n for n in names if normalize_label(n) == norm]
            assert not collisions, (
                f"{probe!r} does not resolve but collides with {collisions}: "
                "accept_new_bodyparts would insert past the collision guard"
            )

    def test_resolvable_variant_is_not_duplicated(self, bodypart, skeleton):
        """A separator variant maps onto the curated spelling, not a new row.

        ``tail_base`` normalizes onto ``tailBase``; the skeleton must reuse the
        curated identity rather than registering a second spelling.
        """
        assert bodypart & {"bodypart": "tailBase"}
        before = set(bodypart.fetch("bodypart"))

        skeleton.insert1(
            {
                "skeleton_id": "zoo_variant_probe",
                "bodyparts": ["tail_base", "nose"],
                "edges": [("tail_base", "nose")],
            },
            accept_new_bodyparts=True,
            check_duplicates=False,
        )

        assert set(bodypart.fetch("bodypart")) == before  # no new rows
        stored = (skeleton & {"skeleton_id": "zoo_variant_probe"}).fetch1(
            "bodyparts"
        )
        assert "tailBase" in stored and "tail_base" not in stored

    def test_novel_part_registered_for_nonadmin(
        self, bodypart, skeleton, monkeypatch
    ):
        """The admin bypass is the flag's purpose, and it must hold."""
        from spyglass.common import LabMember

        monkeypatch.setattr(
            LabMember, "user_is_admin", property(lambda self: False)
        )
        novel = "zooNovelPartXyz"
        assert not (bodypart & {"bodypart": novel})

        skeleton.insert1(
            {
                "skeleton_id": "zoo_nonadmin_probe",
                "bodyparts": [novel, "nose"],
                "edges": [(novel, "nose")],
            },
            accept_new_bodyparts=True,
            check_duplicates=False,
        )
        assert bodypart & {"bodypart": novel}
        (bodypart & {"bodypart": novel}).super_delete(
            warn=False, safemode=False
        )

    def test_without_flag_unknown_part_raises(self, skeleton):
        """Default stays strict: unknown parts are rejected, not registered."""
        with pytest.raises(dj.DataJointError):
            skeleton.insert1(
                {
                    "skeleton_id": "zoo_strict_probe",
                    "bodyparts": ["definitelyNovelXyz", "nose"],
                    "edges": [("definitelyNovelXyz", "nose")],
                },
                check_duplicates=False,
            )


class TestBodyPartSource:
    """`BodyPart.source` partitions the vocabulary so it keeps controlling.

    Zoo imports add ~119 names to a 27-entry curated table. Without a
    partition, the first import silently widens what a hand-built lab project
    may use. Resolution must still see everything (a zoo skeleton has to
    resolve its own parts); only *validation* narrows to curated.

    All RED until Z08.
    """

    def test_source_column_exists_and_defaults_curated(self, bodypart):
        """The seeded lab vocabulary stays `curated` after the migration.

        Scoped to the seed list on purpose: asserting *every* row is curated
        is a global claim about a shared table, and fails as soon as any test
        in the session imports a model that registers `imported` parts.
        """
        assert "source" in bodypart.heading.names
        sources = set(bodypart.fetch("source"))
        assert sources == {"curated"}

    def test_validation_ignores_imported_parts(self, bodypart, skeleton):
        """The whole point: a zoo-only name stays invalid for user projects."""
        imported = "zooOnlyPartXyz"
        bodypart.insert1(
            {"bodypart": imported, "source": "imported"}, warn=False
        )
        try:
            from spyglass.position.v2.utils.skeleton import normalize_label

            with pytest.raises(dj.DataJointError):
                skeleton._validate_bodyparts({normalize_label(imported)})
        finally:
            (bodypart & {"bodypart": imported}).super_delete(
                warn=False, safemode=False
            )

    def test_canon_map_still_resolves_imported_parts(self, bodypart):
        """Reads stay lenient, so a zoo skeleton reads its own pose."""
        from spyglass.position.v2.utils.skeleton import (
            canonicalize,
            normalize_label,
        )

        imported = "zooOnlyPartXyz"
        bodypart.insert1(
            {"bodypart": imported, "source": "imported"}, warn=False
        )
        try:
            cmap = bodypart.canon_map()
            assert cmap.get(normalize_label(imported)) == imported
            assert canonicalize("zoo_only_part_xyz", cmap) == imported
        finally:
            (bodypart & {"bodypart": imported}).super_delete(
                warn=False, safemode=False
            )
