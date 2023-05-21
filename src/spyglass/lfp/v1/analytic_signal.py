import datajoint as dj
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import math
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.lfp.v1 import LFPBand, LFPElectrodeGroup
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile

schema = dj.schema("lfp_v1_analytic_signal")


@schema
class AnalyticSignalParameters(dj.Manual):
    definition = """
    analytic_signal_params_name: varchar(100) # name of the parameter set for detecting analytic signals
    ---
    analytic_signal_params: blob # a dict of analytic-signal detection parameters(the method name, e.g., hilbert transform, and specific parameters if any)
    """

    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict."""
        analytic_signal_params = dict()
        analytic_signal_params["analytic_method_name"] = "hilbert_transform"
        analytic_signal_params[
            "other_params"
        ] = None  # Hilbert transform doesn't contain other parameters; alternative methods may contain additional params.
        self.insert1(["default", analytic_signal_params], skip_duplicates=True)

    def insert1(self, key, skip_duplicates):
        if key[1]["analytic_method_name"] == "hilbert_transform":
            super().insert1(key, skip_duplicates=skip_duplicates)
        else:
            raise KeyError(
                f"analytic_method_name '{key[1]['analytic_method_name']}' is not currently supported."
            )


@schema
class AnalyticSignalSelection(dj.Manual):
    definition = """
    # Table that combines the LFP data, and parameters for generating analytic signals
    -> LFPBand
    -> IntervalList
    -> LFPElectrodeGroup
    -> AnalyticSignalParameters
    ---
    """


@schema
class AnalyticSignal(dj.Computed):
    definition = """
    -> AnalyticSignalSelection
    ---
    -> AnalysisNwbfile # the name of the nwb file with the analytic sinal & time stamps data
    analytic_signal_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        # First, get the filtered lfp data
        filtered_band = (LFPBand() & key).fetch_nwb()[0]["filtered_data"]

        # Next, select data using defined electrode id(s) and defined valid interval.
        electrode_list = (LFPElectrodeGroup.LFPElectrode() & key).fetch("electrode_id")
        electrode_index = np.isin(filtered_band.electrodes.data[:], electrode_list)
        start, end = (IntervalList() & key).fetch1("valid_times")[0]
        timestamps_selected = filtered_band.timestamps[
            np.logical_and(
                filtered_band.timestamps >= start, filtered_band.timestamps <= end
            )
        ]
        filtered_band_selected = filtered_band.data[
            np.logical_and(
                filtered_band.timestamps >= start, filtered_band.timestamps <= end
            )
        ][:, electrode_index]

        # Get analytic signal with the specified method (we're only using Hilbert transform for now, and the algorithm will raise errors if other methods are defined by the user)
        filtered_analytic = np.empty(
            shape=(filtered_band_selected.shape[0], 0), dtype=np.complex_
        )
        params = (AnalyticSignalParameters() & key).fetch1("analytic_signal_params")
        for lfp_electrode_index in range(filtered_band_selected.shape[1]):
            if params["analytic_method_name"] == "hilbert_transform":
                analytic_signal_one_channel = hilbert(
                    np.array(filtered_band_selected)[:, lfp_electrode_index], axis=0
                )
            else:
                raise KeyError(
                    f"analytic_method_name '{params['analytic_method_name']}' is not supported."
                )
            filtered_analytic = np.column_stack(
                (filtered_analytic, analytic_signal_one_channel)
            )

        # Combine times stamps and analytic signal results into a dataframe
        timestamps_selected_df = pd.DataFrame({"time stamps": timestamps_selected})
        analytic_signal_df = pd.DataFrame(
            filtered_analytic,
            columns=[f"electrode {e}" for e in electrode_list],
        )
        analytic_signal_results = pd.concat(
            (timestamps_selected_df, analytic_signal_df), axis=1
        )

        # Create an analysis nwb file to save the results (analytic signal and corresponding time stamps)
        analytic_signal_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analytic_signal_file_name

        # get object id while generating the nwb file
        analytic_signal_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=analytic_signal_results,
        )
        AnalysisNwbfile().add(key["nwb_file_name"], analytic_signal_file_name)
        key["analytic_signal_object_id"] = analytic_signal_object_id
        self.insert1(key)
        print(
            f"\nPopulated analytic signal table for nwb_file_name={key['nwb_file_name']} between {start}ms and {end}ms"
        )

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["analytic_signal"].set_index("time stamps")

    def compute_signal_phase(self, electrode_id):
        analytic_df = self.fetch_nwb()[0]["analytic_signal"]
        # note that the conventional theta phase is 180 degrees shifted from the phase detected by hilbert transform.
        signal_phase = np.angle(analytic_df[f"electrode {electrode_id}"]) + math.pi
        return signal_phase

    def compute_signal_power(self, electrode_id):
        analytic_df = self.fetch_nwb()[0]["analytic_signal"]
        signal_power = np.abs(analytic_df[f"electrode {electrode_id}"]) ** 2
        return signal_power
