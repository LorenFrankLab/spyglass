from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject


def test_imported_spike_sorting_ingestion(imported_spike):
    nwbfile = mock_NWBFile(
        identifier="sorting_bug",
        session_description="Mock NWB file demonstrating Spyglass Spike Sorting insertion bug",
    )
    mock_Subject(nwbfile=nwbfile)

    for i in range(3):
        nwbfile.add_unit(id=i, spike_times=np.random.rand(10))

    ss_table = imported_spike.ImportedSpikeSorting()

    objs_to_insert = ss_table.get_nwb_objects(nwbfile)
    assert (
        len(objs_to_insert) == 1
    ), f"Should return the single UnitTable, but got {len(objs_to_insert)} objects."

    inserts = ss_table.generate_entries_from_nwb_object(
        objs_to_insert[0], base_key={"nwb_file_name": "mock_units_file.nwb"}
    )
    insert_keys = inserts[ss_table]
    assert (
        len(insert_keys) == 1
    ), f"Should generate a single insert key, but got {len(insert_keys)}"
    assert (
        insert_keys[0]["object_id"] == nwbfile.units.object_id
    ), f"Inserted object_id {insert_keys[0]['object_id']} does not match original {nwbfile.units.object_id}"


@pytest.fixture(scope="module")
def two_sessions_with_annotations(imported_spike, verbose_context):
    """Insert two NWB sessions with overlapping unit ids and add per-unit
    annotations to both. Reproduces the data setup in issue #1581."""
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    raw_dir_path = Path(raw_dir)
    raw_file_names = ["issue_1581_a.nwb", "issue_1581_b.nwb"]
    session_info = []

    for raw_file_name in raw_file_names:
        nwbfile = mock_NWBFile(
            identifier=raw_file_name,
            session_description=f"Issue #1581 regression session for {raw_file_name}",
        )
        mock_Subject(nwbfile=nwbfile)
        for unit_id in range(3):
            nwbfile.add_unit(id=unit_id, spike_times=np.array([0.1, 0.2, 0.3]))

        file_path = raw_dir_path / raw_file_name
        file_path.unlink(missing_ok=True)
        with NWBHDF5IO(file_path, mode="w") as io:
            io.write(nwbfile)
        insert_sessions([str(file_path)], raise_err=True)
        session_info.append(
            {
                "nwb_file_name": get_nwb_copy_filename(raw_file_name),
                "raw_path": file_path,
            }
        )

    sorting_table = imported_spike.ImportedSpikeSorting()
    for info in session_info:
        for unit_id in range(3):
            sorting_table.add_annotation(
                key={"nwb_file_name": info["nwb_file_name"]},
                id=unit_id,
                annotations={"score": float(unit_id)},
            )

    yield session_info

    with verbose_context:
        for info in session_info:
            (Nwbfile & {"nwb_file_name": info["nwb_file_name"]}).delete(
                safemode=False
            )
            info["raw_path"].unlink(missing_ok=True)


def test_make_df_from_annotations_restricted_to_session(
    imported_spike, two_sessions_with_annotations
):
    """Regression test for issue #1581: `make_df_from_annotations` must
    restrict the Annotations part-table to the parent's restriction. Without
    the restriction, overlapping unit ids across sessions produce a
    non-unique index in the returned dataframe."""
    session_a, _ = two_sessions_with_annotations
    table = imported_spike.ImportedSpikeSorting & {
        "nwb_file_name": session_a["nwb_file_name"]
    }
    annotation_df = table.make_df_from_annotations()
    assert list(annotation_df.index) == [0, 1, 2]
    assert annotation_df["score"].tolist() == [0.0, 1.0, 2.0]


def test_fetch_nwb_with_annotations_multiple_sessions(
    imported_spike, two_sessions_with_annotations
):
    """Regression test for issue #1581: `fetch_nwb` on `ImportedSpikeSorting`
    must not raise `pandas.errors.InvalidIndexError` when multiple sessions
    have annotations with overlapping unit ids."""
    _, session_b = two_sessions_with_annotations
    result = (
        imported_spike.ImportedSpikeSorting
        & {"nwb_file_name": session_b["nwb_file_name"]}
    ).fetch_nwb()
    assert len(result) == 1
    spikes_df = result[0]["object_id"]
    assert list(spikes_df.index) == [0, 1, 2]
    assert spikes_df["score"].tolist() == [0.0, 1.0, 2.0]
