from pathlib import Path

import numpy as np
import pytest
from ndx_franklab_novela import (
    DataAcqDevice,
    NwbElectrodeGroup,
    Probe,
    Shank,
    ShanksElectrode,
)
from numpy import array_equal
from pynwb import NWBHDF5IO
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile, mock_Subject

from ..conftest import TEARDOWN


def test_create_from_config(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_copy_name)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(mini_insert, common_ephys, mini_dict, mini_content):
    obj_fetch = common_ephys.Raw().nwb_object(mini_dict).object_id
    obj_raw = mini_content.get_acquisition().object_id
    assert obj_fetch == obj_raw, "Raw.nwb_object did not return expected object"


def test_electrode_populate(common_ephys):
    common_ephys.Electrode.populate()
    assert len(common_ephys.Electrode()) == 128, "Electrode.populate failed"


def test_elec_group_populate(pop_common_electrode_group):
    assert (
        len(pop_common_electrode_group) == 32
    ), "ElectrodeGroup.populate failed"


def test_raw_populate(common_ephys):
    common_ephys.Raw.populate()
    assert len(common_ephys.Raw()) == 1, "Raw.populate failed"


def test_sample_count_populate(common_ephys):
    common_ephys.SampleCount.populate()
    assert len(common_ephys.SampleCount()) == 1, "SampleCount.populate failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_set_lfp_electrodes(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass


def test_duplicate_electrode_ids_error(common_ephys, raw_dir, common):
    """Test that duplicate electrode IDs (probe_electrode) produce clear error.

    This test reproduces issue #1447 where NWB files with locally unique
    electrode IDs (unique within a shank but duplicated across shanks)
    produce unclear IntegrityErrors.
    """
    from spyglass.data_import import insert_sessions
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwbfile = mock_NWBFile(
        identifier="test_duplicate_electrode_ids",
        session_description="Test file demonstrating issue #1447",
    )
    nwbfile.subject = mock_Subject()

    data_acq_device = DataAcqDevice(
        name="test_device",
        system="test_system",
        amplifier="test_amplifier",
        adc_circuit="test_circuit",
    )
    nwbfile.add_device(data_acq_device)

    # Create probe with duplicate electrode IDs across shanks
    probe1_shank1_electrodes = [
        ShanksElectrode(name="1", rel_x=0.0, rel_y=0.0, rel_z=0.0),
        ShanksElectrode(name="2", rel_x=0.0, rel_y=10.0, rel_z=0.0),
    ]
    probe1_shank2_electrodes = [
        ShanksElectrode(name="1", rel_x=0.0, rel_y=0.0, rel_z=0.0),  # Dupe
        ShanksElectrode(name="2", rel_x=0.0, rel_y=10.0, rel_z=0.0),  # Dupe
    ]

    probe1 = Probe(
        name="probe1",
        id=1,
        probe_type="test_duplicate_elec_probe",
        units="um",
        probe_description="probe for testing duplicate electrode ID detection",
        contact_side_numbering=False,
        contact_size=10.0,
        shanks=[
            Shank(name="1", shanks_electrodes=probe1_shank1_electrodes),
            Shank(name="2", shanks_electrodes=probe1_shank2_electrodes),
        ],
    )
    nwbfile.add_device(probe1)

    # Create electrode group
    electrode_group1 = NwbElectrodeGroup(
        name="probe1_group",
        description="electrode group for probe1",
        location="CA1",
        device=probe1,
        targeted_location="CA1",
        targeted_x=0.0,
        targeted_y=0.0,
        targeted_z=0.0,
        units="um",
    )
    nwbfile.add_electrode_group(electrode_group1)

    # Add required custom columns
    for col in [
        "probe_shank",
        "probe_electrode",
        "bad_channel",
        "ref_elect_id",
    ]:
        nwbfile.add_electrode_column(
            name=col, description=f"description for {col}"
        )

    # Add electrodes with duplicate probe_electrode values across shanks
    electrode_counter = 0
    for shank_id in [1, 2]:
        for electrode_id in [1, 2]:  # Same electrode IDs for each shank!
            nwbfile.add_electrode(
                location="CA1",
                group=electrode_group1,
                probe_shank=shank_id,
                probe_electrode=electrode_id,  # This duplicates across shanks
                bad_channel=False,
                ref_elect_id=0,
                x=0.0,
                y=float(electrode_id * 10),
                z=0.0,
            )
            electrode_counter += 1

    # Add electrical series
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(range(electrode_counter)),
        description="all electrodes",
    )
    mock_ElectricalSeries(
        electrodes=electrode_table_region,
        nwbfile=nwbfile,
        data=np.ones((10, electrode_counter)),
    )

    # Add required behavior module
    nwbfile.create_processing_module(
        name="behavior",
        description="behavior module",
    )

    # Write NWB file
    test_nwb_path = Path(raw_dir) / "test_duplicate_electrode_ids.nwb"
    if test_nwb_path.exists():
        test_nwb_path.unlink()

    with NWBHDF5IO(test_nwb_path, "w") as io:
        io.write(nwbfile)

    # Try to insert - this should fail with IntegrityError
    nwb_copy_filename = get_nwb_copy_filename(test_nwb_path.name)

    # Clean up any existing entry
    query = common.Nwbfile & {"nwb_file_name": nwb_copy_filename}
    query.delete(safemode=False)

    # Attempt insertion - expect clear ValueError about duplicate electrode IDs
    with pytest.raises(ValueError) as exc_info:
        insert_sessions(
            str(test_nwb_path), rollback_on_fail=True, raise_err=True
        )

    # Verify we get a clear, informative error message
    error_message = str(exc_info.value)
    assert "Duplicate electrode IDs detected" in error_message
    assert "must be globally unique" in error_message
    assert "Electrode ID 1 appears 2 times" in error_message
    assert "Electrode ID 2 appears 2 times" in error_message

    # Cleanup
    query.delete(safemode=False)
    if test_nwb_path.exists():
        test_nwb_path.unlink()
