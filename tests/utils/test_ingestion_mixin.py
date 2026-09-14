"""Unit tests for IngestionMixin's entry-composition paths.

The mixin's single-table declarative path is well exercised by the tables that
use it. Its *composition* paths -- merging entries across rows and source
objects, validating entries for tables other than `self`, and routing
`populate` -- were not, and every defect found while migrating tables to the
mixin lived in one of them.

Each test below pins one of those defects. They are deliberately close to the
metal: several drive the mixin through a stand-in object rather than a real
DataJoint table, so they stay fast and do not depend on database state.
"""

import numpy as np
import pytest

from spyglass.utils.mixins.ingestion import IngestionMixin


class _Stand_in(IngestionMixin):
    """Minimal IngestionMixin user, standing in for a DataJoint table.

    Carries only what the entry-composition methods touch, so the tests do not
    need a schema, a connection, or an NWB file.
    """

    def __init__(self, mapping=None, extra=None):
        self._mapping = mapping or {"value": "value"}
        self._extra = extra  # (table, rows) emitted alongside self's entries

    @property
    def table_key_to_obj_attr(self):
        return {"self": self._mapping}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Emit self's entries, plus a second table's when configured."""
        entries = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        if self._extra is not None and not hasattr(nwb_obj, "to_dataframe"):
            table, rows = self._extra
            entries.setdefault(table, []).extend(rows)
        return entries


class _Row:
    """One row of a stand-in source object."""

    def __init__(self, value):
        self.value = value


class _Rows:
    """A source object that expands into rows, like a DynamicTable."""

    def __init__(self, values):
        self._values = values

    def to_dataframe(self):
        """Return an object whose itertuples() yields the rows."""
        return self

    def itertuples(self):
        """Yield one _Row per value."""
        return iter([_Row(value) for value in self._values])


def test_row_loop_keeps_entries_for_other_tables():
    """Expanding rows must not discard the tables a row emits.

    The loop previously kept only `[self]` from each row's result, so a table
    that emitted a parent's entries per row silently lost them.
    """
    other = "other_table"
    table = _Stand_in(extra=(other, [{"parent": 1}]))

    entries = table.generate_entries_from_nwb_object(_Rows([1, 2, 3]))

    assert len(entries[table]) == 3, "One entry per row expected"
    assert (
        other in entries
    ), "Entries for a second table were dropped by the row loop"
    assert len(entries[other]) == 3, "Second table's entries should accumulate"


def test_row_loop_preserves_first_seen_order():
    """A parent emitted before self must stay ahead of it.

    Insert order follows dict order, and a child inserted before its parent
    fails on the foreign key.
    """

    class _ParentFirst(_Stand_in):
        def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
            if hasattr(nwb_obj, "to_dataframe"):
                return super(_Stand_in, self).generate_entries_from_nwb_object(
                    nwb_obj, base_key
                )
            self_entries = super(
                _Stand_in, self
            ).generate_entries_from_nwb_object(nwb_obj, base_key)
            return {"parent": [{"p": 1}], **self_entries}

    table = _ParentFirst()
    entries = table.generate_entries_from_nwb_object(_Rows([1, 2]))

    assert list(entries)[0] == "parent", "Parent should precede self"


def test_remove_null_from_dicts_takes_a_list():
    """The fallback adjuster must match the signature it stands in for.

    `_adjust_entries` passes a list of entries. This previously took a single
    dict and so raised AttributeError for every table lacking its own
    `_adjust_keys_for_entry` -- any part declared with SpyglassMixin.
    """
    table = _Stand_in()

    adjusted = table._remove_null_from_dicts(
        [{"a": 1, "b": None, "c": ""}, {"a": 2, "b": "keep"}]
    )

    assert adjusted == [{"a": 1}, {"a": 2, "b": "keep"}]


@pytest.mark.parametrize(
    "new, existing, expected",
    [
        (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), False),
        (np.array([[1.0, 2.0]]), np.array([[1.0, 3.0]]), True),
        (np.array([1, 2, 3]), None, True),
    ],
)
def test_unequal_vals_compares_arrays(new, existing, expected):
    """Array attributes must compare without an ambiguous truth value.

    Tables that emit a parent's entries alongside their own pass blob values
    (IntervalList.valid_times) through duplicate validation. Both `array or
    ''` and `array != other` raise ValueError on an array.
    """
    result = IngestionMixin._unequal_vals(
        "valid_times", {"valid_times": new}, {"valid_times": existing}
    )

    assert result is expected or result == expected


@pytest.mark.parametrize("falsy", [0, 0.0, False])
def test_unequal_vals_keeps_falsy_values_distinct_from_missing(falsy):
    """A stored 0 or False is a value, not a missing entry.

    Coalescing every falsy value to "" made `0` and `NULL` compare equal,
    hiding a real divergence during duplicate validation.
    """
    assert IngestionMixin._unequal_vals(
        "count", {"count": falsy}, {"count": None}
    ), f"{falsy!r} should differ from a missing value"

    assert not IngestionMixin._unequal_vals(
        "count", {"count": falsy}, {"count": falsy}
    ), f"{falsy!r} should equal itself"


def test_unequal_vals_treats_none_as_empty_string():
    """None and "" remain equal, the case the coalescing existed for."""
    assert not IngestionMixin._unequal_vals(
        "note", {"note": None}, {"note": ""}
    )


def test_multi_object_merge_accepts_a_new_table(
    common, mini_copy_name, mini_insert
):
    """A later source object may emit a table the first did not.

    `insert_from_nwbfile` merged each object's entries with
    `entries[table].extend(...)`, which raised KeyError the moment a second
    object introduced a table key the first had not used. TaskEpoch does
    exactly this: the test file holds two task tables, each emitting Task
    alongside its own entries.

    Asserted against a plan rather than an insert. The file is already
    ingested by the fixture, and TaskEpoch does not expect duplicates, so
    re-inserting raises by design -- see the 0.5.6 breaking change. The merge
    is what this test is about, and planning exercises it without the raise.
    """
    plan = common.TaskEpoch().plan_from_nwbfile(mini_copy_name)
    entries = plan.entries

    names = {
        getattr(table, "table_name", str(table)) for table, _ in entries or ()
    }

    assert entries, "TaskEpoch should generate entries for the test file"
    assert any(
        "task_epoch" in name for name in names
    ), f"Expected TaskEpoch's own entries, saw {names}"
    assert any(
        name == "#task" or name.endswith("task") for name in names
    ), f"Expected Task entries emitted alongside, saw {names}"


def test_populate_routes_to_ingestion(common, mini_copy_name, mini_insert):
    """populate() ingests rather than calling the deprecated `make`.

    `make` on an ingestion table raises NotImplementedError, so populate()
    would break for any registered file missing rows in that table. It now
    ingests each such file instead.
    """
    table = common.SampleCount()
    restr = {"nwb_file_name": mini_copy_name}

    before = len(table & restr)
    assert before, "Fixture should have ingested SampleCount"

    try:
        # super_delete: plain delete would remove the NWB file from disk
        (table & restr).super_delete(warn=False, safemode=False)
        assert not len(table & restr), "Rows should be cleared for the test"

        table.populate()
        after = len(table & restr)
    finally:
        # Restore the rows even if populate() failed, so a failure here does
        # not leave the database short for every later test.
        if not len(table & restr):
            table.insert_from_nwbfile(mini_copy_name)

    assert (
        after == before
    ), "populate() should have re-ingested the cleared rows"


def test_unequal_vals_tolerates_datetime_storage_rounding():
    """A datetime read back from storage still equals the one parsed.

    DataJoint stores datetimes without a timezone and at second resolution,
    so a strict comparison reports a divergence for every session on every
    re-parse.
    """
    from datetime import datetime, timedelta, timezone

    parsed = datetime(2023, 6, 22, 15, 59, 57, 888000, tzinfo=timezone.utc)
    stored = datetime(2023, 6, 22, 15, 59, 58)

    assert not IngestionMixin._unequal_vals(
        "session_start_time",
        {"session_start_time": parsed},
        {"session_start_time": stored},
    ), "Sub-second rounding is not a divergence"

    assert IngestionMixin._unequal_vals(
        "session_start_time",
        {"session_start_time": parsed},
        {"session_start_time": stored + timedelta(hours=1)},
    ), "A real difference should still be reported"
