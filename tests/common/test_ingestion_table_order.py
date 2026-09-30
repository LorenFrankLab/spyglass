"""The declared ingestion set: its order, and its coverage.

`ingestion_tables()` writes both out by hand rather than deriving them, because
the schema is fixed at import time and sorting on every call rediscovers an
answer that cannot change. The cost of writing them out is that a table added
in the wrong place, or a target left undeclared, is silent -- so it is checked
here instead, against DataJoint's own graph and against a real plan.
"""

import pytest


@pytest.fixture
def declared(common):
    """The declared ingestion set."""
    from spyglass.common.populate_all_common import ingestion_tables

    return ingestion_tables()


def test_ingest_list_is_dependency_ordered(common, declared):
    """A parent must be planned before any table that depends on it.

    The planner walks this order, so a table listed before its parent would
    resolve a cross-reference against a plan that has not reached the parent
    yet -- and would resolve it to nothing, quietly.
    """
    ingest = [t.as_instance for t in declared["ingest"]]
    position = {
        table.full_table_name: index for index, table in enumerate(ingest)
    }

    for index, table in enumerate(ingest):
        for parent in table.parents():
            if not isinstance(parent, str) or parent not in position:
                continue  # a parent outside the set is ingested elsewhere
            assert position[parent] < index, (
                f"{table.full_table_name} is listed at {index}, before its "
                f"parent {parent} at {position[parent]}"
            )


def test_ingest_and_targets_do_not_overlap(common, declared):
    """A table either parses the file or only receives rows, never both.

    An entry in both lists would be parsed once and then treated as a target
    of itself, which is not wrong so much as a sign the categories have
    stopped meaning anything.
    """
    ingest = {t.as_instance.full_table_name for t in declared["ingest"]}
    targets = {t.as_instance.full_table_name for t in declared["targets"]}

    assert not (ingest & targets), f"Declared as both: {ingest & targets}"


def test_every_plan_target_is_declared(common, mini_copy_name, mini_insert):
    """Whatever a plan names must appear in one of the two lists.

    This is the check that matters for reuse: a staged entry names its target
    by string, and rehydrating it means finding the class again. The declared
    set is that lookup, so a target missing from it is a target whose entries
    cannot be reloaded. Five were missing before they were declared -- `Task`,
    `RawPosition` and its part, `LFPElectrodeGroup` and its part -- each
    emitted by a table in `ingest`.
    """
    from spyglass.common.populate_all_common import ingestion_tables
    from spyglass.data_import.planner import plan_nwbfile

    tables = ingestion_tables()
    declared_names = {
        t.as_instance.full_table_name
        for group in tables.values()
        for t in group
    }
    # Parts of a declared table are declared by extension: a master's parts
    # come with it, and naming each one here would be noise.
    for group in tables.values():
        for table in group:
            declared_names |= set(table.as_instance.parts())

    plan = plan_nwbfile(mini_copy_name)
    targets = {
        getattr(target, "full_table_name", str(target))
        for table_plan in plan.table_plans
        for target, _ in table_plan.entries
    }

    assert targets, "The mini file should plan something"
    undeclared = targets - declared_names
    assert not undeclared, (
        "These tables receive planned rows but are declared nowhere in "
        f"`ingestion_tables()`: {sorted(undeclared)}"
    )


def test_targets_carry_a_real_heading(common, declared):
    """Targets are real classes, not names.

    The point of declaring them as classes: `check_planned_rows` needs a
    heading and parents to validate a row, and `insert_plan` needs something
    it can insert into. A name-to-class registry was the alternative, and this
    list is why it is not needed.
    """
    for table in declared["targets"]:
        instance = table.as_instance
        assert instance.heading.names, f"{table} has no heading"
        assert instance.primary_key, f"{table} has no primary key"
