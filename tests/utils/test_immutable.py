"""Tests for the row guards in ``spyglass.utils.mixins.immutable``."""

from types import SimpleNamespace

import datajoint as dj
import pytest

SCHEMA_NAME = "test_immutable_mixins"


@pytest.fixture(scope="module")
def guards():
    """Return the module under test, imported after the test config loads."""
    from spyglass.utils.mixins import immutable

    return immutable


@pytest.fixture(scope="module")
def tables(dj_conn, teardown, guards):
    """Declare a guarded parameter table and a factory-built master/part."""
    from spyglass.utils import SpyglassMixin, SpyglassMixinPart

    class GuardParams(SpyglassMixin, guards.ImmutableParamsLookup, dj.Lookup):
        definition = """
        params_name: varchar(32)
        ---
        threshold: float
        """
        contents = [("default", 1.0)]

    class GuardGroup(SpyglassMixin, guards.FactoryOnlyMaster, dj.Manual):
        _factory_create_call = "GuardGroup.create_group()"

        definition = """
        group_name: varchar(32)
        ---
        note = "": varchar(64)
        """

        class Member(guards.FactoryOnlyMaster, SpyglassMixinPart):
            _factory_create_call = "GuardGroup.create_group()"

            definition = """
            -> master
            member_id: int
            """

        @classmethod
        def create_group(cls, group_name, member_ids):
            with cls.connection.transaction:
                cls.insert1(
                    {"group_name": group_name}, allow_direct_insert=True
                )
                cls.Member.insert(
                    [
                        {"group_name": group_name, "member_id": member_id}
                        for member_id in member_ids
                    ],
                    allow_direct_insert=True,
                )

    schema = dj.Schema(
        context=dict(GuardParams=GuardParams, GuardGroup=GuardGroup)
    )
    schema(GuardParams)
    schema(GuardGroup)
    schema.activate(SCHEMA_NAME, connection=dj_conn)

    yield SimpleNamespace(params=GuardParams, group=GuardGroup)

    if teardown:
        schema.drop(force=True)


def _threshold(tables, name):
    return (tables.params & {"params_name": name}).fetch1("threshold")


def test_params_update1_rejected(tables, guards):
    with pytest.raises(guards.ImmutableRowError, match="allow_param_mutation"):
        tables.params().update1({"params_name": "default", "threshold": 2.0})
    assert _threshold(tables, "default") == 1.0


def test_params_insert_and_flagged_update1(tables):
    tables.params().insert1({"params_name": "editable", "threshold": 3.0})
    assert _threshold(tables, "editable") == 3.0

    tables.params().update1(
        {"params_name": "editable", "threshold": 4.0},
        allow_param_mutation=True,
    )
    assert _threshold(tables, "editable") == 4.0


@pytest.mark.parametrize(
    "call",
    [
        lambda table: table.insert([("default", 5.0)], replace=True),
        lambda table: table.insert([("default", 5.0)], True),
        lambda table: table.insert1(("default", 5.0), replace=True),
    ],
    ids=["keyword", "positional", "insert1"],
)
def test_params_insert_replace_rejected(tables, guards, call):
    with pytest.raises(guards.ImmutableRowError, match="replace=True"):
        call(tables.params())
    assert _threshold(tables, "default") == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda group: group().insert1({"group_name": "direct"}),
        lambda group: group().insert([{"group_name": "direct"}]),
        lambda group: group.Member().insert1(
            {"group_name": "direct", "member_id": 0}
        ),
    ],
    ids=["insert1", "insert", "part"],
)
def test_factory_master_direct_insert_rejected(tables, guards, call):
    with pytest.raises(
        guards.UnsupportedDirectInsertError, match=r"GuardGroup\.create_group"
    ):
        call(tables.group)
    assert not tables.group & {"group_name": "direct"}


def test_factory_master_allow_direct_insert(tables):
    tables.group().insert1({"group_name": "bypass"}, allow_direct_insert=True)
    assert tables.group & {"group_name": "bypass"}

    tables.group.create_group("built", [0, 1])
    assert len(tables.group.Member & {"group_name": "built"}) == 2


def test_factory_master_update1(tables, guards):
    tables.group.create_group("edited", [0])
    row = {"group_name": "edited", "note": "changed"}
    note = tables.group & {"group_name": "edited"}

    with pytest.raises(guards.ImmutableRowError, match="allow_master_mutation"):
        tables.group().update1(row)
    assert note.fetch1("note") == ""

    tables.group().update1(row, allow_master_mutation=True)
    assert note.fetch1("note") == "changed"


def _resolver(cls, name):
    """Return the class in ``cls``'s MRO whose own ``name`` is used."""
    return next(base for base in cls.__mro__ if name in vars(base))


def test_guards_define_nothing_spyglass_mixin_defines(guards):
    """A guard placed after ``SpyglassMixin`` is not shadowed by it."""
    from spyglass.utils import SpyglassMixin

    spyglass_names = {
        name
        for cls in SpyglassMixin.__mro__
        if cls is not object
        for name in vars(cls)
        if not name.startswith("__")
    }
    for guard in (
        guards._IdentityMasterGuard,
        guards.FactoryOnlyMaster,
        guards.ImmutableParamsLookup,
    ):
        defined = {name for name in vars(guard) if not name.startswith("__")}
        assert defined and not defined & spyglass_names, guard.__name__


def test_guards_precede_the_datajoint_tier(tables, guards):
    """Declared order resolves to the guard; tier-first order bypasses it."""
    from spyglass.utils import SpyglassMixin

    for table, guard in (
        (tables.params, guards.ImmutableParamsLookup),
        (tables.group, guards._IdentityMasterGuard),
        (tables.group.Member, guards._IdentityMasterGuard),
    ):
        for name in ("insert", "update1"):
            assert _resolver(table, name) is guard, (table.__name__, name)

    mro = tables.params.__mro__
    assert mro.index(SpyglassMixin) < mro.index(guards.ImmutableParamsLookup)

    class TierFirst(SpyglassMixin, dj.Lookup, guards.ImmutableParamsLookup):
        definition = """
        params_name: varchar(32)
        """

    assert _resolver(TierFirst, "update1") is dj.Table
