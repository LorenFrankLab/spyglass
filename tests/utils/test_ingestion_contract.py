"""Specification for the revised IngestionMixin contract.

Written before the implementation: these tests describe the contract the
mixin is moving to. Four structural problems in the previous one each
produced real defects --

  P1  entries carried in a bare dict, keyed by class *or* instance, with
      insert order enforced only by convention
  P2  one generation method serving two contracts (a container and its rows),
      recursing through itself
  P3  file-level state passed between mixin calls by mutating `self`
  P4  generation reaching into the database, so parsing is not side-effect free

The pieces below exist to make those unrepresentable rather than merely
discouraged. `tests/utils/test_ingestion_mixin.py` pins the *current*
behavior; this module pins the target.
"""

import pytest

from spyglass.data_import.ingestion_plan import FileContext, PlannedEntries
from spyglass.utils.mixins.helpers import HelperMixin

# ---------------------------------------------------------------------------
# PlannedEntries -- P1: identity, merging, ordering
# ---------------------------------------------------------------------------


class _Alpha(HelperMixin):
    """Stand-in for a table class.

    Inherits the helper mixin and names itself, as every ingestion target
    does -- the collection identifies a target by asking it those things.
    """

    full_table_name = "`stand_in`.`alpha`"


class _Beta(HelperMixin):
    """Stand-in for a second table class."""

    full_table_name = "`stand_in`.`beta`"


def test_class_and_instance_are_the_same_target():
    """An instance key must collapse to its class.

    Returning `{Table(): [...]}` reads naturally and was wrong: a fresh
    instance is a fresh dict key, so entries from a second row or source
    object landed under a different key than the first.
    """
    entries = PlannedEntries()
    entries.add(_Alpha, [{"n": 1}])
    entries.add(_Alpha(), [{"n": 2}])

    assert len(list(entries)) == 1, "One target expected, not two"
    assert entries.rows_for(_Alpha) == ({"n": 1}, {"n": 2})


def test_add_accumulates_rather_than_replacing():
    """Adding to a target already present appends to it."""
    entries = PlannedEntries()
    entries.add(_Alpha, [{"n": 1}])
    entries.add(_Alpha, [{"n": 2}, {"n": 3}])

    assert len(entries.rows_for(_Alpha)) == 3


def test_extend_merges_an_unseen_target():
    """Merging must accept a target the receiver has never seen.

    Both merge sites in the old mixin indexed the dict directly, so a later
    row or source object introducing a new target raised KeyError.
    """
    first = PlannedEntries()
    first.add(_Alpha, [{"n": 1}])

    second = PlannedEntries()
    second.add(_Beta, [{"m": 1}])
    second.add(_Alpha, [{"n": 2}])

    first.extend(second)

    assert [target for target, _ in first] == [
        _Alpha,
        _Beta,
    ], "First-seen order kept"
    assert len(first.rows_for(_Alpha)) == 2
    assert len(first.rows_for(_Beta)) == 1


def test_rows_for_unknown_target_is_empty():
    """Querying a target that was never added is not an error."""
    assert PlannedEntries().rows_for(_Alpha) == ()


def test_empty_entries_are_falsy():
    """An empty collection is falsy; a populated one is truthy."""
    entries = PlannedEntries()
    assert not entries

    entries.add(_Alpha, [{"n": 1}])
    assert entries


def test_iteration_yields_target_and_rows():
    """Iterating yields (target, rows) pairs in insertion order."""
    entries = PlannedEntries()
    entries.add(_Beta, [{"m": 1}])
    entries.add(_Alpha, [{"n": 1}])

    assert [target for target, _ in entries] == [_Beta, _Alpha]


def test_dependency_order_puts_parents_first(common):
    """Insert order is derived, not declared.

    Callers previously had to return a dict with the parent ahead of the
    child, enforced by a docstring. The collection sorts by the foreign-key
    graph instead, so a caller adding them in the wrong order still inserts
    correctly.
    """
    entries = PlannedEntries()
    entries.add(common.TaskEpoch, [{"epoch": 1}])  # child
    entries.add(common.Task, [{"task_name": "t"}])  # parent

    ordered = [target for target, _ in entries.in_dependency_order()]

    assert ordered.index(common.Task) < ordered.index(
        common.TaskEpoch
    ), "Task is TaskEpoch's parent and must be inserted first"


def test_freeze_is_immutable():
    """A frozen collection can be stored and compared, not mutated."""
    entries = PlannedEntries()
    entries.add(_Alpha, [{"n": 1}])

    frozen = entries.freeze()

    with pytest.raises((AttributeError, TypeError)):
        frozen.add(_Alpha, [{"n": 2}])


# ---------------------------------------------------------------------------
# FileContext -- P3: state, reads, problems
# ---------------------------------------------------------------------------


def test_context_cache_is_per_run():
    """Two contexts do not share scratch space.

    File-level lookups used to be cached on `self`, i.e. on a table object
    that outlives any one ingestion.
    """
    first = FileContext(nwb_file_name="a_.nwb", nwb_file=None)
    second = FileContext(nwb_file_name="b_.nwb", nwb_file=None)

    first.cache["cameras"] = {1: "cam"}

    assert "cameras" not in second.cache


def test_context_records_reads():
    """Objects a table reads are recorded, for per-object plan reuse.

    This read-set is what lets a later attempt re-parse only the tables whose
    source objects changed.
    """
    ctx = FileContext(nwb_file_name="a_.nwb", nwb_file=None)

    ctx.record_read("object-id-1")
    ctx.record_read("object-id-2")
    ctx.record_read("object-id-1")

    assert set(ctx.reads) == {"object-id-1", "object-id-2"}


def test_context_collects_problems_without_raising():
    """A problem is recorded and parsing continues.

    Failures must accumulate into a report rather than aborting the pass at
    the first one.
    """
    ctx = FileContext(nwb_file_name="a_.nwb", nwb_file=None)

    ctx.problem("soft", "no_source", "nothing to ingest")

    assert len(ctx.problems) == 1
    assert ctx.problems[0].severity == "soft"
    assert ctx.problems[0].code == "no_source"


# ---------------------------------------------------------------------------
# Mixin methods -- P2: one contract per method
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, value):
        self.value = value


class _Rows:
    """A source object that expands into rows, like a DynamicTable."""

    def __init__(self, values):
        self._values = values

    def to_dataframe(self):
        return self

    def itertuples(self):
        return iter([_Row(v) for v in self._values])


def _stand_in(**attrs):
    """Build a minimal IngestionMixin user with the given overrides."""
    from spyglass.utils.mixins.ingestion import IngestionMixin

    namespace = {
        "full_table_name": "`stand_in`.`mixin_user`",
        "table_key_to_obj_attr": property(
            lambda self: {"self": {"value": "value"}}
        ),
    }
    namespace.update(attrs)
    return type("_StandIn", (IngestionMixin,), namespace)()


def test_row_level_override_needs_no_container_guard():
    """Overriding the row contract must not also receive the container.

    Every override that touched rows needed a
    `hasattr(nwb_obj, "to_dataframe")` guard, because one method served both
    contracts. Omitting it multiplied VectorData columns together.
    """
    seen = []

    def entries_for_row(self, row, ctx):
        seen.append(row)
        entries = PlannedEntries()
        entries.add(type(self), [{"value": row.value}])
        return entries

    table = _stand_in(entries_for_row=entries_for_row)
    ctx = FileContext(nwb_file_name="a_.nwb", nwb_file=None)

    entries = table.entries_for_source(_Rows([1, 2, 3]), ctx)

    assert len(seen) == 3, "Row contract should see rows only"
    assert not any(
        hasattr(item, "to_dataframe") for item in seen
    ), "The container must not be passed to the row contract"
    assert len(entries.rows_for(type(table))) == 3


def test_source_level_override_receives_the_container():
    """Overriding the source contract receives the object itself."""
    seen = []

    def entries_for_source(self, source, ctx):
        seen.append(source)
        return PlannedEntries()

    table = _stand_in(entries_for_source=entries_for_source)
    ctx = FileContext(nwb_file_name="a_.nwb", nwb_file=None)
    source = _Rows([1])

    table.entries_for_source(source, ctx)

    assert seen == [source]


def test_default_row_mapping_applies_table_key_to_obj_attr():
    """The declarative mapping still drives the default row contract."""
    table = _stand_in()
    ctx = FileContext(nwb_file_name="a_.nwb", nwb_file=None)

    entries = table.entries_for_row(_Row(7), ctx)

    assert entries.rows_for(type(table)) == ({"value": 7},)


# ---------------------------------------------------------------------------
# Lifecycle hooks -- P4: purity, with named escape hatches
# ---------------------------------------------------------------------------


def test_hooks_run_around_the_parse(common, mini_copy_name, mini_insert):
    """before_parse runs before sources are found; after_insert follows it.

    Patched onto an instance rather than a subclass: subclassing a declared
    DataJoint table would try to declare another one.
    """
    from types import MethodType

    order = []
    table = common.SampleCount()

    def before_parse(self, ctx):
        order.append("before_parse")

    def find_sources(self, ctx):
        order.append("find_sources")
        return type(self).find_sources(self, ctx)

    def after_insert(self, ctx, inserted):
        order.append("after_insert")

    table.before_parse = MethodType(before_parse, table)
    table.find_sources = MethodType(find_sources, table)
    table.after_insert = MethodType(after_insert, table)

    # after_insert only fires when there is something to insert
    restr = {"nwb_file_name": mini_copy_name}
    (table & restr).super_delete(warn=False, safemode=False)
    try:
        table.insert_from_nwbfile(mini_copy_name)
    finally:
        if not len(table & restr):
            common.SampleCount().insert_from_nwbfile(mini_copy_name)

    assert order[:2] == [
        "before_parse",
        "find_sources",
    ], f"Hooks ran out of order: {order}"
    assert "after_insert" in order, f"after_insert did not run: {order}"


def _nwb_file_for(nwb_file_name):
    """Open the NWB file the way the mixin does."""
    from spyglass.common.common_nwbfile import Nwbfile

    return (Nwbfile & {"nwb_file_name": nwb_file_name}).fetch_nwb()[0]


def test_generation_does_not_write_to_the_database(
    common, mini_copy_name, mini_insert, monkeypatch
):
    """Parsing must not insert. The planner depends on it.

    ImportedLFP reached LFPElectrodeGroup.cautious_insert during generation;
    that has to move to a hook so a dry run stays read-only.
    """
    from spyglass.utils.mixins.ingestion import IngestionMixin

    writes = []

    def _record(self, *args, **kwargs):
        writes.append(type(self).__name__)

    for table_name in ("ImportedLFP", "PositionSource", "TaskEpoch"):
        table = getattr(common, table_name, None)
        if table is None:
            continue
        ctx = FileContext(
            nwb_file_name=mini_copy_name,
            nwb_file=_nwb_file_for(mini_copy_name),
            base_key={"nwb_file_name": mini_copy_name},
        )
        instance = table()
        monkeypatch.setattr(type(instance), "insert", _record, raising=False)
        instance.before_parse(ctx)
        for source in instance.find_sources(ctx):
            instance.entries_for_source(source, ctx)

    assert not writes, f"Generation wrote to {set(writes)}"
    assert issubclass(type(common.SampleCount()), IngestionMixin)
