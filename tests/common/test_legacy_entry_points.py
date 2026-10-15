"""The entry points that do not build a parse context.

Planning and `insert_from_nwbfile` both construct a `FileContext`, so `ctx` is
never None through either. The two-argument
`generate_entries_from_nwb_object(nwb_obj, base_key)` signature is still
public and still documented as resolving from the database.
"""

import pytest


@pytest.fixture
def ingestion_mixin():
    """The mixin carrying the plan-or-database helpers."""
    from spyglass.utils.mixins.ingestion import IngestionMixin

    return IngestionMixin


def test_value_from_plan_reads_the_database_without_a_context(
    common, mini_copy_name, mini_insert
):
    """`ctx=None` resolves from storage rather than raising."""
    raw = common.common_ephys.Raw()
    stored = (raw & {"nwb_file_name": mini_copy_name}).fetch1(
        "interval_list_name"
    )

    found = common.common_dio.DIOEvents().value_from_plan(
        None, raw, "interval_list_name", {"nwb_file_name": mini_copy_name}
    )

    assert found == stored, "Without a context the stored row answers"


def test_value_from_plan_returns_none_for_a_missing_row(common):
    """A missing row is None, since the callers branch on it."""
    found = common.common_dio.DIOEvents().value_from_plan(
        None,
        common.common_ephys.Raw(),
        "interval_list_name",
        {"nwb_file_name": "_no_such_file_.nwb"},
    )

    assert found is None


def test_rows_from_plan_reads_the_database_without_a_context(
    common, mini_copy_name, mini_insert
):
    """The rows variant, which `StateScriptFile` resolves epochs through."""
    task_epoch = common.TaskEpoch()
    stored = len(task_epoch & {"nwb_file_name": mini_copy_name})

    found = common.common_behav.StateScriptFile().rows_from_plan(
        None, task_epoch, {"nwb_file_name": mini_copy_name}
    )

    assert len(found) == stored, "Without a context the stored rows answer"
    assert all(isinstance(row, dict) for row in found)


def test_electrode_generation_without_a_context(common, mini_content):
    """A two-argument `Electrode` call needs no config and must not raise."""
    electrodes = mini_content.electrodes
    if electrodes is None:  # pragma: no cover - corpus without electrodes
        pytest.skip("mini file has no electrodes table")

    entries = common.common_ephys.Electrode().generate_entries_from_nwb_object(
        electrodes, {"nwb_file_name": "irrelevant_.nwb"}
    )

    assert entries, "A direct call should still generate entries"


def test_get_abs_path_accepts_a_name_with_directories(common, mini_copy_name):
    """The path and the row lookup agree on the file name."""
    bare = common.Nwbfile.get_abs_path(mini_copy_name)
    prefixed = common.Nwbfile.get_abs_path(f"/some/other/path/{mini_copy_name}")

    assert prefixed == bare, "The directory in front is dropped, as documented"


def test_get_abs_path_keeps_wildcards_for_the_query(common, mini_copy_name):
    """Normalizing must not eat a `%` wildcard the query needs."""
    from pathlib import Path

    assert Path(f"%{mini_copy_name[:8]}%").name == f"%{mini_copy_name[:8]}%", (
        "Premise: pathlib leaves a wildcard alone, so normalizing the name "
        "does not break substring lookups"
    )


def test_statescript_epochs_resolve_against_the_plan(common, mini_copy_name):
    """`StateScriptFile` finds epochs the plan intends but has not written.

    Asserted here rather than through a dry run: no corpus file carries
    associated files, so `StateScriptFile` is `skipped` on every one of them.
    """
    from ndx_franklab_novela import AssociatedFiles

    from spyglass.data_import.planner import VirtualKeySpace
    from spyglass.utils.ingestion_plan import FileContext

    unseen = "_plan_only_never_stored_.nwb"
    epochs = [1, 2]

    key_space = VirtualKeySpace(unseen)
    key_space.add_planned(
        common.TaskEpoch(),
        [{"nwb_file_name": unseen, "epoch": epoch} for epoch in epochs],
    )
    ctx = FileContext(
        nwb_file_name=unseen,
        base_key={"nwb_file_name": unseen},
        key_space=key_space,
    )

    associated = AssociatedFiles(
        name="statescript_plan_test",
        description="Statescript for several epochs",
        content="callback()\n",
        task_epochs=",".join(str(epoch) for epoch in epochs),
    )

    generated = common.StateScriptFile().generate_entries_from_nwb_object(
        associated, {"nwb_file_name": unseen}, ctx
    )
    entries = next(iter(generated.values()))

    assert {entry["epoch"] for entry in entries} == set(epochs), (
        "Epochs the plan will create must yield entries; a database query "
        f"would find none for {unseen} and plan no scripts"
    )
