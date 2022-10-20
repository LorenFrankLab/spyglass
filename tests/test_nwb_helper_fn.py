import datetime
import unittest

import pynwb

# NOTE: importing this calls spyglass.__init__ whichand spyglass.common.__init__ which both require the
# DataJoint MySQL server to be already set up and running
from spyglass.common import get_electrode_indices


class TestGetElectrodeIndices(unittest.TestCase):
    def setUp(self):
        self.nwbfile = pynwb.NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )
        dev = self.nwbfile.create_device(name="device")
        elec_group = self.nwbfile.create_electrode_group(
            name="electrodes",
            description="description",
            location="location",
            device=dev,
        )
        for i in range(10):
            self.nwbfile.add_electrode(
                id=100 + i,
                x=0.0,
                y=0.0,
                z=0.0,
                imp=-1.0,
                location="location",
                filtering="filtering",
                group=elec_group,
            )

        elecs_region = self.nwbfile.electrodes.create_region(
            name="electrodes", region=[2, 3, 4, 5], description="description"  # indices
        )

        eseries = pynwb.ecephys.ElectricalSeries(
            name="eseries",
            data=[0, 1, 2],
            timestamps=[0.0, 1.0, 2.0],
            electrodes=elecs_region,
        )
        self.nwbfile.add_acquisition(eseries)

    def test_nwbfile(self):
        ret = get_electrode_indices(self.nwbfile, [102, 105])
        assert ret == [2, 5]

    def test_electrical_series(self):
        eseries = self.nwbfile.acquisition["eseries"]
        ret = get_electrode_indices(eseries, [102, 105])
        assert ret == [0, 3]
