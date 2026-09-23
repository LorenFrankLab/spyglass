"""Tests for ``SpyglassMixinPart.delete``. See issue #1555.

``(Part & restr).delete()`` used to call itself whenever the restriction could
not be applied to the master table. These tests pin down the replacement
behavior, which mirrors ``datajoint.user_tables.Part.delete``: a part-only
delete is refused unless ``force=True`` is passed.
"""

import datajoint as dj
import pytest
from datajoint.errors import DataJointError

SCHEMA_NAME = "test_part_delete"


@pytest.fixture(scope="module")
def part_schema(dj_conn, teardown):
    """Declare a minimal master/part pair for exercising part deletes."""
    from spyglass.utils import SpyglassMixin, SpyglassMixinPart

    class DeleteMaster(SpyglassMixin, dj.Manual):
        definition = """
        master_id: int
        """

        class Item(SpyglassMixinPart):
            definition = """
            -> DeleteMaster
            item_id: int
            """

    schema = dj.Schema(context=dict(DeleteMaster=DeleteMaster))
    schema(DeleteMaster)
    schema.activate(SCHEMA_NAME, connection=dj_conn)

    prev_safemode = dj.config.get("safemode", True)
    dj.config["safemode"] = False  # part-only deletes cannot forward safemode

    yield DeleteMaster

    dj.config["safemode"] = prev_safemode
    if teardown:
        schema.drop(force=True)


@pytest.fixture(autouse=True)
def no_reentry(monkeypatch):
    """Fail fast if ``SpyglassMixinPart.delete`` calls itself.

    The pre-fix implementation re-restricted itself on every pass, so the
    condition string grew exponentially and the runaway exhausted memory long
    before Python's recursion limit tripped. Capping re-entry turns the bug
    into an immediate, readable ``RecursionError``.
    """
    from spyglass.utils.dj_mixin import SpyglassMixinPart

    # __dict__ lookup: dj's TableMeta instantiates on class attribute access
    original = SpyglassMixinPart.__dict__["delete"]
    depth = dict(n=0)

    def guarded(self, *args, **kwargs):
        depth["n"] += 1
        try:
            if depth["n"] > 2:
                raise RecursionError(
                    "SpyglassMixinPart.delete re-entered itself "
                    + f"{depth['n']} deep. See issue #1555."
                )
            return original(self, *args, **kwargs)
        finally:
            depth["n"] -= 1

    monkeypatch.setattr(SpyglassMixinPart, "delete", guarded, raising=False)
    yield


@pytest.fixture
def populated(part_schema):
    """Reset master/part contents before each test.

    Yields
    ------
    tuple
        ``(master, part)`` table instances, holding 2 master entries with 2
        part entries apiece.
    """
    master, part = part_schema(), part_schema.Item()

    master.super_delete(warn=False, safemode=False)
    master.insert([{"master_id": i} for i in (1, 2)])
    part.insert(
        [{"master_id": m, "item_id": i} for m in (1, 2) for i in (1, 2)]
    )

    yield master, part


def test_part_only_str_restr_requires_force(populated):
    """A part-only string restriction refuses rather than recursing."""
    master, part = populated

    with pytest.raises(DataJointError) as err:
        (part & "item_id = 1").delete()

    assert "force" in str(err.value).lower(), "No force hint in error message"
    assert len(part) == 4, "Part entries deleted without force"
    assert len(master) == 2, "Master entries deleted without force"


def test_part_only_dict_restr_requires_force(populated):
    """A part-only dict restriction refuses rather than recursing.

    Dict restrictions are compiled to a condition string before they reach
    ``delete``, so they hit the same master-promotion failure as strings.
    """
    master, part = populated

    with pytest.raises(DataJointError) as err:
        (part & {"item_id": 1}).delete()

    assert "force" in str(err.value).lower(), "No force hint in error message"
    assert len(master) == 2, "Master entries deleted without force"
    assert len(part) == 4, "Part entries deleted without force"


def test_master_restr_deletes_master(populated):
    """A restriction valid on the master still cascades from the master."""
    master, part = populated

    (part & {"master_id": 1}).delete()

    assert len(master) == 1, "Master entry not deleted"
    assert len(part) == 2, "Part entries not cascaded"


def test_force_deletes_part_only(populated):
    """``force=True`` deletes the part entries and leaves the master."""
    master, part = populated

    (part & "item_id = 1").delete(force=True)

    assert len(part) == 2, "Part entries not deleted with force"
    assert len(master) == 2, "Master entries deleted with force"


def test_unrestricted_delete_removes_master(populated):
    """An unrestricted part delete still propagates to the master."""
    master, part = populated

    part.delete()

    assert len(master) == 0, "Master entries not deleted"
    assert len(part) == 0, "Part entries not deleted"
