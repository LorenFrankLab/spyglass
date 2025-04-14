from typing import List, Union

import datajoint as dj
import kachery_cloud as kcl
import numpy as np
import spikeinterface as si
from sortingview.SpikeSortingView import create_raw_traces_plot
from sortingview.SpikeSortingView.Figure import Figure

from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("figurl_view_spike_sorting_recording")


@schema
class SpikeSortingRecordingView(SpyglassMixin, dj.Computed):
    definition = """
    # Schema for storing figurl views of spike sorting recordings
    -> SpikeSortingRecording
    ---
    figurl: varchar(10000)
    """

    def make(self, key):
        """Populates SpikeSortingRecordingView.

        Fetches the recording from SpikeSortingRecording and extracts traces
        and electrode geometry.
        """
        # Get the SpikeSortingRecording row
        rec = (SpikeSortingRecording & key).fetch1()
        nwb_file_name = rec["nwb_file_name"]
        sort_group_id = rec["sort_group_id"]
        sort_interval_name = rec["sort_interval_name"]

        # Load the SI recording extractor
        logger.info("Loading recording")
        recording: si.BaseRecording = SpikeSortingRecording().load_recording(
            key
        )

        # Raw traces (sample)
        # Extract the first 1 second of traces
        logger.info("Extracting traces")
        traces: np.array = recording.get_traces(
            start_frame=0, end_frame=int(recording.get_sampling_frequency() * 1)
        ).astype(np.float32)
        f1 = create_raw_traces_plot(
            traces=traces,
            start_time_sec=0,
            sampling_frequency=recording.get_sampling_frequency(),
            label="Raw traces (sample)",
        )

        # Electrode geometry
        logger.info("Electrode geometry")
        f2 = create_electrode_geometry(recording)

        label = f"{nwb_file_name}:{sort_group_id}:{sort_interval_name}"
        logger.info(label)

        # Mountain layout
        F = create_mountain_layout(figures=[f1, f2], label=label)

        # Insert row into table
        key2 = dict(key, **{"figurl": F.url()})
        self.insert1(key2)


def create_electrode_geometry(recording: si.BaseRecording):
    """Create a figure for the electrode geometry of a recording."""
    channel_locations = {
        str(channel_id): location.astype(np.float32)
        for location, channel_id in zip(
            recording.get_channel_locations(), recording.get_channel_ids()
        )
    }
    data = {"type": "ElectrodeGeometry", "channelLocations": channel_locations}
    return Figure(data=data, label="Electrode geometry")


def create_mountain_layout(
    figures: List[Figure],
    label: Union[str, None] = None,
    sorting_curation_uri: Union[str, None] = None,
) -> Figure:
    """Create a figure for a mountain layout of multiple figures"""
    if label is None:
        label = "SpikeSortingView"

    data = {
        "type": "MountainLayout",
        "views": [
            {
                "type": fig0.data["type"],
                "label": fig0.label,
                "figureDataSha1": _upload_data_and_return_sha1(
                    fig0.get_serialized_figure_data()
                ),
            }
            for fig0 in figures
        ],
    }
    if sorting_curation_uri is not None:
        data["sortingCurationUri"] = sorting_curation_uri
    return Figure(data=data, label=label)


def _upload_data_and_return_sha1(data):
    data_uri = kcl.store_json(data)
    data_hash = data_uri.split("/")[2]
    return data_hash
