import datajoint as dj
import kachery_client as kc
import spikeinterface as si
from sortingview.SpikeSortingView import SpikeSortingView as SortingViewSpikeSortingView

from ..common.common_spikesorting import SpikeSorting, SpikeSortingRecording
from .prepare_spikesortingview_data import prepare_spikesortingview_data

schema = dj.schema("figurl_view_spike_sorting_recording")


@schema
class SpikeSortingView(dj.Computed):
    definition = """
    # Schema for storing figurl views of spike sortings
    -> SpikeSorting
    ---
    figurl: varchar(10000)
    """

    def make(self, key):
        recording_record = (
            SpikeSortingRecording & {"recording_id": key["recording_id"]}
        ).fetch1()
        sorting_record = (SpikeSorting & {"sorting_id": key["sorting_id"]}).fetch1()

        recording_path = recording_record["recording_path"]
        sorting_path = sorting_record["sorting_path"]

        # Load the SI extractors
        recording: si.BaseRecording = si.load_extractor(recording_path)
        sorting: si.BaseSorting = si.load_extractor(sorting_path)

        with kc.TemporaryDirectory() as tmpdir:
            fname = f"{tmpdir}/spikesortingview.h5"
            print("Preparing spikesortingview data")
            prepare_spikesortingview_data(
                recording=recording,
                sorting=sorting,
                segment_duration_sec=60 * 20,
                snippet_len=(20, 20),
                max_num_snippets_per_segment=100,
                channel_neighborhood_size=7,
                output_file_name=fname,
            )

            print("Creating view object")
            X = SortingViewSpikeSortingView(fname)

            print("Creating summary")
            f1 = X.create_summary()
            # f2 = X.create_units_table(unit_ids=X.unit_ids, unit_metrics=unit_metrics)
            print("Creating autocorrelograms")
            f3 = X.create_autocorrelograms(unit_ids=X.unit_ids)
            print("Creating raster plot")
            f4 = X.create_raster_plot(unit_ids=X.unit_ids)
            print("Creating average waveforms")
            f5 = X.create_average_waveforms(unit_ids=X.unit_ids)
            print("Creating spike amplitudes")
            f6 = X.create_spike_amplitudes(unit_ids=X.unit_ids)
            print("Creating electrode geometry")
            f7 = X.create_electrode_geometry()
            # f8 = X.create_live_cross_correlograms()

            sorting_curation_uri = None

            nwb_file_name = recording_record["nwb_file_name"]
            sort_group_id = sorting_record["sort_group_id"]
            sort_interval_name = sorting_record["sort_interval_name"]
            sorter = sorting_record["sorter"]
            sorter_params_name = sorting_record["sorter_params_name"]
            label = f"{nwb_file_name}:{sort_group_id}:{sort_interval_name}:{sorter}:{sorter_params_name}"
            print(label)

            print("Creating mountain layout")
            mountain_layout = X.create_mountain_layout(
                figures=[f1, f3, f4, f5, f6, f7],
                label=label,
                sorting_curation_uri=sorting_curation_uri,
            )

            print("Making URL")
            url = mountain_layout.url()

            # Insert row into table
            key2 = dict(key, **{"figurl": url})
            self.insert1(key2)
