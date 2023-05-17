import os
import datajoint as dj
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import math
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.lfp.v1.lfp import LFP,LFPBand
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

@schema
class AnalyticSignalSelection(dj.Manual):
    definition = """
    # Table that combines the LFP data, and parameters for generating analytic signals
    -> LFPBand
    -> IntervalList
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
        filtered_band = (LFPBand() & key).fetch_nwb()[0]['filtered_data']
        
        # Next, select data using defined valid interval
        start = (IntervalList() & key).fetch1('valid_times')[0][0]
        end = (IntervalList() & key).fetch1('valid_times')[0][-1]
        time_stamps_selected = filtered_band.timestamps[np.logical_and(filtered_band.timestamps >= start, filtered_band.timestamps <= end)]
        filtered_band_selected = filtered_band.data[np.logical_and(filtered_band.timestamps >= start, filtered_band.timestamps <= end)]
        
        # Get analytic signal with the specified method (we're only using Hilbert transform for now)
        filtered_analytic = np.empty(shape=(FilteredBand_selected.shape[0],0),dtype=np.complex_)
        for lfp_electrode_id in range(FilteredBand_selected.shape[1]):
            filtered_lfp_one_channel = np.array(FilteredBand_selected)[:,lfp_electrode_id]
            analytic_signal_one_channel = hilbert(filtered_lfp_one_channel, axis=0)
            filtered_analytic = np.column_stack((filtered_analytic,analytic_signal_one_channel))
        
        # Combine times stamps and analytic signal results into a dataframe
        timestamps_selected_df = pd.DataFrame({"time stamps":TimeStamps_selected})
        analytic_signal_df = pd.DataFrame(filtered_analytic,columns=[f"electrode {e}" for e in FilteredBand.electrodes.data[:]])
        analytic_signal_results = pd.concat((timestamps_selected_df,analytic_signal_df),axis=1)
        
        # Create an analysis nwb file to save the results (analytic signal and corresponding time stamps)
        analytic_signal_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key['analysis_file_name'] = analytic_signal_file_name
        
        # get object id while generating the nwb file
        analytic_signal_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=analytic_signal_results,
        )
        AnalysisNwbfile().add(key["nwb_file_name"], analytic_signal_file_name)
        key['analytic_signal_object_id'] = analytic_signal_object_id
        self.insert1(key)
        print(
            f"\nPopulated analytic signal table for nwb_file_name={key['nwb_file_name']} between {start}ms and {end}ms"
        )
        
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, 'analysis_file_abs_path'), *attrs, **kwargs
        )
    
    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['analytic_signal']
    
    def compute_signal_phase(self,electrode_id):
        analytic_df = self.fetch_nwb()[0]['analytic_signal']
        # note that the conventional theta phase is 180 degrees shifted from the phase detected by hilbert transform.
        signal_phase = np.angle(analytic_df[f"electrode {electrode_id}"]) + math.pi
        return signal_phase
        
    def compute_signal_power(self,electrode_id):
        analytic_df = self.fetch_nwb()[0]['analytic_signal']
        signal_power = np.abs(analytic_df[f"electrode {electrode_id}"]) ** 2
        return signal_power
