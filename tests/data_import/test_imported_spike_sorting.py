from pynwb.testing.mock.file import mock_NWBFile, mock_Subject
import numpy as np


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
