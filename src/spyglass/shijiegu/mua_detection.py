import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pynwb
import scipy.signal
import ghostipy as gsp
from spyglass.shijiegu.load import load_spike


from spyglass.common import (IntervalPositionInfo, IntervalPositionInfoSelection, IntervalList,
                             ElectrodeGroup, LFP, interval_list_intersect, LFPBand, Electrode)

from spyglass.spikesorting.v0 import (SortGroup,
                                    SortInterval,
                                    SpikeSortingPreprocessingParameters,
                                    SpikeSortingRecording,
                                    SpikeSorterParameters,
                                    SpikeSortingRecordingSelection,
                                    ArtifactDetectionParameters, ArtifactDetectionSelection,
                                    ArtifactRemovedIntervalList, ArtifactDetection,
                                      SpikeSortingSelection, SpikeSorting,)

from ripple_detection.core import (gaussian_smooth,
                                   get_envelope,
                                   get_multiunit_population_firing_rate,
                                   threshold_by_zscore,
                                   merge_overlapping_ranges,
                                   exclude_close_events,
                                   extend_threshold_to_mean)
from itertools import chain

from ripple_detection.detectors import _get_event_stats

from scipy.stats import zscore
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_position,load_maze_spike
from spyglass.shijiegu.helpers import interval_union
from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,RippleTimes,HSETimes,MUA
from spyglass.shijiegu.helpers import interval_union,interpolate_to_new_time
from spyglass.shijiegu.ripple_detection import (loadRippleLFP_OneChannelPerElectrode,
                                                removeDataBeforeTrial1,removeArtifactTime,
                                                multiunit_HSE_detector)

def mua_detection_master(nwb_copy_file_name, epochID):
    # find epoch/session name and position interval name
    key = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'epoch':epochID}).fetch1()
    epoch_name = key['epoch_name']
    position_interval = key['position_interval']

    # Get MUA
    _0,_1,mua_time,mua,_2=load_maze_spike(nwb_copy_file_name,epoch_name)

    mua_smooth = gaussian_smooth(mua, 0.004, 30000) # 4ms smoothing, as in Kay, Karlsson, spiking data are in 30000Hz
    mua_ds = mua_smooth[::10]
    mua_time_ds = mua_time[::10]

    # Remove Data before 1st trial and after last trial and artifact
    # to remove artifact, we use LFP to help, where artifact times are noted already

    position_valid_times = (IntervalList & {'nwb_file_name': nwb_copy_file_name,
                                            'interval_list_name': position_interval}).fetch1('valid_times')

    filtered_lfps, filtered_lfps_t, CA1TetrodeInd, CCTetrodeInd = loadRippleLFP_OneChannelPerElectrode(
            nwb_copy_file_name,epoch_name,position_valid_times)

    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch':int(epoch_name[:2])}).fetch1('choice_reward')
    )
    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O

    position_info = load_position(nwb_copy_file_name,position_interval)
    position_info_upsample = interpolate_to_new_time(position_info, filtered_lfps_t)
    position_info_upsample = removeDataBeforeTrial1(position_info_upsample,trial_1_t,trial_last_t)
    position_info_upsample = removeArtifactTime(position_info_upsample, filtered_lfps)

    position_info_upsample2 = interpolate_to_new_time(position_info_upsample, mua_time_ds)

    hse_times,firing_rate_raw, mua_mean, mua_std = multiunit_HSE_detector(mua_time_ds,mua_ds,
                                                                      np.array(position_info_upsample2.head_speed),
                                                                      3000,speed_threshold=4.0,
                                                                      zscore_threshold=0,use_speed_threshold_for_zscore=True)
    # Insert into HSETimes table
    animal = nwb_copy_file_name[:5]
    savePath=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                        nwb_copy_file_name+'_'+epoch_name+'_hse_times.nc')
    hse_times.to_csv(savePath)

    key = {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': epoch_name}
    key['hse_times'] = savePath
    HSETimes().insert1(key,replace = True)

    # Insert into MUA table for future plotting
    key = {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': epoch_name}
    mua_df = pd.DataFrame(data=firing_rate_raw, index=mua_time_ds, columns = ['mua'])
    mua_df.index.name='time'
    mua_df=xr.Dataset.from_dataframe(mua_df)

    savePath=os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                             nwb_copy_file_name+'_'+epoch_name+'_mua.nc')
    mua_df.to_netcdf(savePath)

    key['mua_trace'] = savePath
    (key['mean'],key['sd']) = (mua_mean,mua_std)
    MUA().insert1(key,replace = True)