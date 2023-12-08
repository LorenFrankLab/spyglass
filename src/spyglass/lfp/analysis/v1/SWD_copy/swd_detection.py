import datajoint as dj
import numpy as np
import pandas as pd
import scipy.signal butter, filtfilt, hilbert
from spyglass.common import get_electrode_indices
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.v1 import LFPV1
from spyglass.lfp.analysis.v1.lfp_band import LFPBandSelection, LFPBandV1
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.dj_helper_fn import fetch_nwb

from spyglass_custom.utils.schema_prefix import schema_prefix
from spyglass_custom.bsharma.utils.filters import (
    bandpass_filter,
    compute_metric,
    z_score,
)

schema = dj.schema(schema_prefix + "_swd_detection")


@schema
class SWDSelection(dj.Manual):
    definition = """
    -> LFPV1
    electrode_id: int  # ID of the electrode for SWD detection
    """

    def insert_electrodes(
        self,
        nwb_file,
        target_interval_list_name,
        lfp_electrode_group_name,
        filter_name,
        electrode_ids,
        filter_sampling_rate,
    ):

        for elec_id in electrode_ids:
            key = {
                "nwb_file_name": nwb_file,
                "target_interval_list_name": target_interval_list_name,
                "lfp_electrode_group_name": lfp_electrode_group_name,
                "filter_name": filter_name,
                "filter_sampling_rate": filter_sampling_rate,
                "electrode_id": elec_id,
            }
            self.insert1(
                key,
                skip_duplicates=True,
            )
        insert_list = [
            {
                "nwb_file_name": nwb_file,
                "target_interval_list_name": target_interval_list_name,
                "lfp_electrode_group_name": lfp_electrode_group_name,
                "filter_name": filter_name,
                "filter_sampling_rate": filter_sampling_rate,
                "electrode_id": elec_id,
            }
            for elec_id in electrode_ids
        ]
        self.insert(
            insert_list,
            skip_duplicates=True,
        )



@schema
class SWDParameters(dj.Lookup):
    definition = """
    param_id: int  # unique identifier for a parameter set
    ---
    frequency_bands: blob  # list of tuples for frequency bands
    event_gap: float  # threshold value for event gap
    """

    def insert_defaults(self):
        # Method to insert default parameter sets
        default_params = [
            # Add your default parameter sets here
            (1, [(6, 9), (14, 18), (22, 26), (30, 34), (38, 42)], 100),
            # Add more as needed
        ]
        self.insert(default_params, skip_duplicates=True)


@schema
class SWDDetection(SpyglassMixin, dj.Computed):
    definition = """
    -> SWDSelection
    -> SWDParameters
    ---
    -> AnalysisNwbfile
    detection_results_object_id : varchar(40)  # Identifier for the stored results
    """
    column_names = [
        "chunk_duration",
        "elapsed_time_between_chunks",
        "start_times",
        "end_times",
        "ratios",
        "diffs",
    ]
    
    def _butter_bandpass(lowcut, highcut, fs, order=2):
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


    def _bandpass_filter(data, lowcut, highcut, fs, order=2):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    def _z_score(arr):
        return (arr - arr.mean()) / arr.std()


    # Compute metrics
    def _compute_metric(data, n, metric):
        if metric == "ratio":
            max_val = data[f"LFP_{n}"].max()
            min_val = data[f"LFP_{n}"].min()
            return np.abs(min_val / max_val)
        elif metric == "diff":
            max_val = np.abs(data[f"LFP_{n}"].max())
            min_val = np.abs(data[f"LFP_{n}"].min())
            return max_val - min_val

    def _smoothing(self, lfp_sampling_rate, envelope) -> np.ndarray:
        SIGMA = 1 / (2 * np.pi * 0.5)
        window_len = int(6 * SIGMA * lfp_sampling_rate)
        if window_len % 2 == 0:
            window_len += 1
        window = scipy.signal.gaussian(
            window_len, std=SIGMA * lfp_sampling_rate
        )
        window /= window.sum()
        return np.convolve(envelope, window, mode="same")

    def _processing(self, signal, lfp_sampling_rate, frequency_bands):
        """bandpass filter lfp, then z-score, average and then return the hilbert ammplitude of the filtered waves"""
        filtered_waves = [
            bandpass_filter(signal, low, high, lfp_sampling_rate)
            for low, high in frequency_bands
        ]

        zscored_waves = [z_score(wave) for wave in filtered_waves]
        averaged_wave = np.mean(zscored_waves, axis=0)
        envelope = np.abs(hilbert(averaged_wave))
        return averaged_wave, envelope

    def _threshold_signal(self, signal, std_dev=1):
        """Threshold's the processed waves"""
        return signal >= np.mean(signal) + (std_dev * np.std(signal))

    def _identify_events(
        self,
        signal,
    ):
        """identify potential events and then combine events that are occur eithin the event_gap"""
        chunks = np.diff(np.insert(signal.astype(int), [0, signal.size], [0]))
        chunk_starts = np.where(chunks == 1)[0]
        chunk_ends = np.where(chunks == -1)[0]

        # Combine chunks closer than 400 rows
        combined_starts, combined_ends = [chunk_starts[0]], [chunk_ends[0]]
        for i in range(1, len(chunk_starts)):
            if chunk_starts[i] - chunk_ends[i - 1] < event_gap:
                combined_ends[-1] = chunk_ends[i]
            else:
                combined_starts.append(chunk_starts[i])
                combined_ends.append(chunk_ends[i])

        chunk_starts = combined_starts
        chunk_ends = combined_ends
        return chunk_starts, chunk_ends

    def make(self, key):
        print("Processing key:", key)
        frequency_bands, event_gap = (SWDParameters & key).fetch1(
            "frequency_bands", "event_gap"
        )
        nwb_file_name = key["nwb_file_name"]
        lfp_data = (LFPV1() & key).fetch_nwb()[0]
        lfp_eseries = lfp_data["lfp"]
        sample_rate = lfp_data["lfp_sampling_rate"]
        timestamp_data = lfp_eseries.timestamps[:]  # Extract timestamps

        electrode_ids = (
            [key["electrode_id"]]
            if isinstance(key["electrode_id"], int)
            else key["electrode_id"]
        )
        print("electrode_ids:", electrode_ids)
        lfp_elect_indeces = get_electrode_indices(lfp_eseries, electrode_ids)
        lfp_refs = lfp_eseries.data[:, lfp_elect_indeces]

        all_results = []
        for i, ref_signal in enumerate(lfp_refs.T):
            averaged_wave, envelope = self._processing(
                ref_signal, sample_rate, frequency_bands
            )
            print(f"Processing electrode {electrode_ids[i]} is finished")
            smoothed_signal = self._smoothing(sample_rate, envelope)
            print(
                f"Smoothing signal for electrode {electrode_ids[i]} is finished"
            )
            thresholded_signal = self._threshold_signal(smoothed_signal)
            print(
                f"Thresholding signal for electrode {electrode_ids[i]} is finished"
            )

            print(f"Chunking complete for {electrode_ids[i]}")
            chunk_starts, chunk_ends = self._identify_events(thresholded_signal)
            # Compute metrics
            ratios = np.array(
                [
                    compute_metric(
                        ref_signal[chunk_starts[j] : chunk_ends[j]], "ratio"
                    )
                    for j in range(len(chunk_starts))
                ]
            )
            diffs = np.array(
                [
                    compute_metric(
                        ref_signal[chunk_starts[j] : chunk_ends[j]], "diff"
                    )
                    for j in range(len(chunk_starts))
                ]
            )

            # Extract results
            chunk_duration = np.array(
                [
                    timestamp_data[end] - timestamp_data[start]
                    for start, end in zip(combined_starts, combined_ends)
                ]
            )
            elapsed_time_between_chunks = np.diff(
                np.insert(timestamp_data[combined_starts], 0, 0)
            )
            start_times = timestamp_data[combined_starts]
            end_times = timestamp_data[combined_ends]
            single_result = np.column_stack(
                (
                    chunk_duration,
                    elapsed_time_between_chunks,
                    start_times,
                    end_times,
                    ratios,
                    diffs,
                )
            )
            all_results.append(single_result)
            # import ipdb; ipdb.set_trace()
            detection_results = all_results

            detection_results_array = np.array(detection_results)

            detection_results_array = np.array(all_results)

            if (
                detection_results_array.size > 0
                and detection_results_array.ndim > 1
            ):
                reshaped_results = detection_results_array.reshape(
                    -1, detection_results_array.shape[2]
                )
                detection_results_df = pd.DataFrame(
                    reshaped_results, columns=column_names
                )
            else:
                detection_results_df = pd.DataFrame(
                    columns=column_names
                )  # Empty DataFrame with correct columns

            # Create an analysis NWB file and add the results
            nwb_analysis_file = AnalysisNwbfile()
            key["analysis_file_name"] = nwb_analysis_file.create(nwb_file_name)

            # Add the detection results to the NWB file
            key[
                "detection_results_object_id"
            ] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=key["analysis_file_name"],
                nwb_object=detection_results_df,
                table_name="SWD_detection_results_table",
            )

            # Insert into the database
            self.insert1(key)

    def fetch_dataframe(self):
        """Fetch data in the NWBAnalysisFile as a pandas dataframe"""
        nwb_data = self.fetch_nwb()

        # Define the new column names
        new_column_names = {
            "0": "event_duration",
            "1": "elapsed_time_between_events",
            "2": "start_times",
            "3": "end_times",
            "4": "min_max_peak_ratios",
            "5": "min_max_peak_diffs",
        }

        # Extract and return the 'detection_results' DataFrames with renamed columns
        return [
            data["detection_results"].rename(columns=new_column_names)
            for data in nwb_data
            if "detection_results" in data
        ]

    def fetch1_dataframe(self):
        """Convenience function for returning the marks in a readable format"""
        dataframes = self.fetch_dataframe()
        if dataframes:
            return dataframes[0]  # Return the first dataframe
        else:
            return pd.DataFrame(
                columns=self.column_names
            )  # Return an empty DataFrame if no data is available
