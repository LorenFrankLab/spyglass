"Functions that are commonly used"

import itertools
import os
import tempfile
from pathlib import Path
from typing import List

import kachery_client as kc
import numpy as np
import sortingview as sv
import spikeextractors
import spikeextractors as se
import spikeinterface as si
import spikeinterface.toolkit as st
from joblib import Parallel, delayed
from spikeextractors import MdaSortingExtractor, NumpySortingExtractor
from spikesorters.basesorter import BaseSorter
from spikesorters.sorter_tools import recover_recording
from spiketoolkit.postprocessing.postprocessing_tools import \
    divide_recording_into_time_chunks
from tqdm import tqdm

from .common_interval import IntervalList
from .common_lab import LabMember
from .common_nwbfile import AnalysisNwbfile


def store_sorting_nwb(key, *, sorting, sort_interval_list_name, sort_interval, metrics=None, unit_ids=None):
    """store a sorting in a new AnalysisNwbfile
    :param key: key to SpikeSortingRecoring
    :type key: dict
    :param sorting: sorting extractor
    :type sorting: BaseSortingExtractor
    :param sort_interval_list_name: name of interval list for sort
    :type sort_interval_list_name: str
    :param sort_interval: interval for start and end of sort
    :type sort_interval: list
    :param path_suffix: string to append to end of sorting extractor file name
    :type path_suffix: str
    :param metrics: spikesorting metrics, optional, defaults to None
    :type metrics: dict
    :param unit_ids: the ids of the units to save, optional, defaults to None
    :type list
    :returns: analysis_file_name, units_object_id
    :rtype (str, str)
    """
    from .common_spikesorting import SpikeSortingRecording

    sort_interval_valid_times = (IntervalList & \
            {'interval_list_name': sort_interval_list_name}).fetch1('valid_times')

    recording_timestamps = SpikeSortingRecording.get_recording_timestamps(key)
    units = dict()
    units_valid_times = dict()
    units_sort_interval = dict()
    all_unit_ids = sorting.get_unit_ids()
    if unit_ids is None:
        unit_ids = all_unit_ids
    for unit_id in all_unit_ids:
        if unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(
                unit_id=unit_id)
            units[unit_id] = recording_timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

    analysis_file_name = AnalysisNwbfile().create(key['nwb_file_name'])
    units_object_id, _ = AnalysisNwbfile().add_units(analysis_file_name,
                                                    units, units_valid_times,
                                                    units_sort_interval,
                                                    metrics=metrics)

    AnalysisNwbfile().add(key['nwb_file_name'], analysis_file_name)
    return analysis_file_name,  units_object_id


def set_workspace_permission(workspace_name: str, team_members: List[str]):
    """
    Sets permission to sortingview workspace based on google ID

    Parameters
    ----------
    workspace_name: str
        name of workspace
    team_members: List[str]
        list of team members to be given permission to edit the workspace

    Output
    ------
    workspace_uri: str
        URI of workspace
    """
    workspace_uri = kc.get(workspace_name)
    # check if workspace exists
    if not workspace_uri:
        raise ValueError('Workspace with given name does not exist. Create it first before setting permission.')
    workspace = sv.load_workspace(workspace_uri)
    # check if team_members is empty
    if len(team_members)==0:
        raise ValueError('The specified team does not exist or there are no members in the team;\
                            create or change the entry in LabTeam table first')
    for team_member in team_members:
        google_user_id = (LabMember.LabMemberInfo & {'lab_member_name': team_member}).fetch('google_user_name')
        if len(google_user_id)!=1:
            print(f'Google user ID for {team_member} does not exist or more than one ID detected;\
                    permission not given to {team_member}, skipping...')
            continue
        workspace.set_user_permissions(google_user_id[0], {'edit': True})
        print(f'Permissions for {google_user_id[0]} set to: {workspace.get_user_permissions(google_user_id[0])}')
    return workspace_uri

def add_metrics_to_workspace(workspace_uri: str, sorting_id: str=None,
                             user_ids: List[str]=None):
    """Computes nearest neighbor isolation and noise overlap metrics and inserts them
    in the specified sorting of the workspace.

    Parameters
    ----------
    workspace_uri : str
        workspace URI
    sorting_id : str, optional
        sorting id, by default None
    user_ids : list of str, optional
        google ids to confer curation permission, by default None
        
    Returns
    -------
    url : str
        new URL for the view
    external_metrics :  dict
        metrics that were added to workspace
    """

    workspace = sv.load_workspace(workspace_uri)
    
    recording_id = workspace.recording_ids[0]
    recording = workspace.get_recording_extractor(recording_id=recording_id)

    # first cache as binary using old spikeextractors to make it compatible with new si
    tmpdir = tempfile.TemporaryDirectory(dir=os.environ['KACHERY_TEMP_DIR'])
    cached_recording = spikeextractors.CacheRecordingExtractor(recording, 
                                                               save_path=str(Path(tmpdir.name) / 'r.dat'))

    # then load with spikeinterface
    new_recording = si.read_binary(file_paths=str(Path(tmpdir.name) / 'r.dat'),
                                   sampling_frequency=cached_recording.get_sampling_frequency(),
                                   num_chan=cached_recording.get_num_channels(),
                                   dtype=cached_recording.get_dtype(),
                                   time_axis=0,
                                   is_filtered=True)
    new_recording.set_channel_locations(locations=recording.get_channel_locations())
    new_recording = st.preprocessing.whiten(recording=new_recording, seed=0)
    
    if sorting_id is None:
        sorting_id = workspace.sorting_ids[0]
    sorting = workspace.get_sorting_extractor(sorting_id=sorting_id)
    new_sorting = si.core.create_sorting_from_old_extractor(sorting)
    # save sorting to enable parallelized waveform extraction
    new_sorting = new_sorting.save(folder=str(Path(tmpdir.name) / 's'),)

    waveforms = si.extract_waveforms(new_recording, new_sorting, folder=str(Path(tmpdir.name) / 'wf'),
                                     ms_before=1,ms_after=1, max_spikes_per_unit=2000,
                                     overwrite=True,return_scaled=True, n_jobs=5, total_memory='5G')

    isolation = {}
    noise_overlap = {}
    for unit_id in sorting.get_unit_ids():
        isolation[unit_id] = st.qualitymetrics.pca_metrics.nearest_neighbors_isolation(waveforms, this_unit_id=unit_id)
        noise_overlap[unit_id] = st.qualitymetrics.pca_metrics.nearest_neighbors_noise_overlap(waveforms, this_unit_id=unit_id)
    
    # external metrics must be in this format to be added to workspace
    external_metrics = [{'name': 'isolation', 'label': 'isolation', 'tooltip': 'isolation',
                        'data': isolation},
                        {'name': 'noise_overlap', 'label': 'noise_overlap', 'tooltip': 'noise_overlap',
                        'data': noise_overlap}
                        ] 
    workspace.set_unit_metrics_for_sorting(sorting_id=sorting_id, metrics=external_metrics)

    if user_ids is not None:
        for user_id in user_ids:
            workspace.set_user_permissions(user_id, {'edit': True})

    F = workspace.figurl()
    url = F.url(label=workspace.label, channel='franklab2')

    # in the future (once new view works fully):
    # workspace.set_sorting_curation_authorized_users(sorting_id=sorting_id, user_ids=user_ids)
    # url = workspace.experimental_spikesortingview(recording_id=recording_id, sorting_id=sorting_id,
    #                                               label=workspace.label, include_curation=True)

    print(f'URL for sortingview: {url}')

    return url, external_metrics


class ClusterlessThresholder(BaseSorter):
    sorter_name = 'clusterless_thresholder'  # convenience for reporting
    SortingExtractor_Class = NumpySortingExtractor  # convenience to get the extractor
    compatible_with_parallel = {'loky': False,
                                'multiprocessing': False,
                                'threading': False}
    _default_params = dict(
        detect_threshold=100,
        channel_ids=None,
        detect_sign=-1,
        n_shifts=2,
        n_snippets_for_threshold=10,
        snippet_size_sec=1,
        start_frame=None,
        end_frame=None,
        n_jobs=1,
        joblib_backend='loky',
        chunk_size=None,
        chunk_mb=500,
        verbose=False,
    )
    _params_description = {}
    sorter_description = ""
    requires_locations = False
    installation_mesg = ""  # error message when not installed

    @classmethod
    def is_installed(cls):
        return True

    @staticmethod
    def get_sorter_version():
        return 'unknown'

    @classmethod
    def _setup_recording(cls, recording, output_folder):
        pass

    def _run(self, recording, output_folder):

        recording = recover_recording(recording)
        params = self.params.copy()
        detect_threshold = params.pop('detect_threshold')  # in mV

        sorting = detect_spikes(
            recording,
            detect_threshold=detect_threshold,
            **params
        )

        output_folder = Path(output_folder)
        MdaSortingExtractor.write_sorting(
            sorting, str(output_folder / 'firings.mda'))

        samplerate = recording.get_sampling_frequency()
        samplerate_fname = str(output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as file:
            file.write('{}'.format(samplerate))

    @staticmethod
    def get_result_from_folder(output_folder):
        output_folder = Path(output_folder)
        tmpdir = output_folder

        result_fname = str(tmpdir / 'firings.mda')
        samplerate_fname = str(tmpdir / 'samplerate.txt')
        with open(samplerate_fname, 'r') as file:
            samplerate = float(file.read())

        return MdaSortingExtractor(
            file_path=result_fname, sampling_frequency=samplerate)


def detect_spikes(recording, channel_ids=None, detect_threshold=5, detect_sign=-1,
                  n_shifts=2, n_snippets_for_threshold=10, snippet_size_sec=1,
                  start_frame=None, end_frame=None, n_jobs=1, joblib_backend='loky',
                  chunk_size=None, chunk_mb=500, verbose=False):
    '''
    Detects spikes per channel. Spikes are detected as threshold crossings and the threshold is in terms of the median
    average deviation (MAD). The MAD is computed by taking 'n_snippets_for_threshold' snippets of the recordings
    of 'snippet_size_sec' seconds uniformly distributed between 'start_frame' and 'end_frame'.
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    channel_ids: list or None
        List of channels to perform detection. If None all channels are used
    detect_threshold: float
        Threshold in median absolute deviations (MAD) to detect peaks
    n_shifts: int
        Number of shifts to find peak. E.g. if n_shift is 2, a peak is detected (if detect_sign is 'negative') if
        a sample is below the threshold, the two samples before are higher than the sample, and the two samples after
        the sample are higher than the sample.
    n_snippets_for_threshold: int
        Number of snippets to use to compute channel-wise thresholds
    snippet_size_sec: float
        Length of each snippet in seconds
    detect_sign: int
        Sign of the detection: -1 (negative), 1 (positive), 0 (both)
    start_frame: int
        Start frame for detection
    end_frame: int
        End frame end frame for detection
    n_jobs: int
        Number of jobs for parallelization. Default is None (no parallelization)
    joblib_backend: str
        The backend for joblib. Default is 'loky'
    chunk_size: int
        Size of chunks in number of samples. If None, it is automatically calculated
    chunk_mb: int
        Size of chunks in Mb (default 500 Mb)
    verbose: bool
        If True output is verbose
    Returns
    -------
    sorting_detected: SortingExtractor
        The sorting extractor object with the detected spikes. Unit ids are the same as channel ids and units have the
        'channel' property to specify which channel they correspond to. The sorting extractor also has the `spike_rate`
        and `spike_amplitude` properties.
    '''
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = recording.get_num_frames()

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    else:
        assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Not all 'channel_ids' are in the" \
                                                                                  "recording."
    if n_jobs is None:
        n_jobs = 1
    if n_jobs == 0:
        n_jobs = 1

    if start_frame != 0 or end_frame != recording.get_num_frames():
        recording_sub = se.SubRecordingExtractor(
            recording, start_frame=start_frame, end_frame=end_frame)
    else:
        recording_sub = recording

    num_frames = recording_sub.get_num_frames()

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (recording.get_num_channels() * n_bytes)

    if n_jobs > 1:
        chunk_size /= n_jobs

    # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
    chunks = divide_recording_into_time_chunks(
        num_frames=num_frames,
        chunk_size=chunk_size,
        padding_size=0
    )
    n_chunk = len(chunks)

    if verbose:
        print(f"Number of chunks: {len(chunks)} - Number of jobs: {n_jobs}")

    if verbose and n_jobs == 1:
        chunk_iter = tqdm(range(n_chunk), ascii=True,
                          desc="Detecting spikes in chunks")
    else:
        chunk_iter = range(n_chunk)

    if not recording_sub.check_if_dumpable():
        if n_jobs > 1:
            n_jobs = 1
            print("RecordingExtractor is not dumpable and can't be processed in parallel")
        rec_arg = recording_sub
    else:
        if n_jobs > 1:
            rec_arg = recording_sub.dump_to_dict()
        else:
            rec_arg = recording_sub

    all_channel_times = [[] for ii in range(len(channel_ids))]
    all_channel_amps = [[] for ii in range(len(channel_ids))]

    if n_jobs > 1:
        output = Parallel(n_jobs=n_jobs, backend=joblib_backend)(delayed(_detect_and_align_peaks_chunk)
                                                                 (ii, rec_arg, chunks, channel_ids, detect_threshold,
                                                                  detect_sign,
                                                                  n_shifts, verbose)
                                                                 for ii in chunk_iter)
        for ii, (times_ii, amps_ii) in enumerate(output):
            for i, ch in enumerate(channel_ids):
                times = times_ii[i]
                amps = amps_ii[i]
                all_channel_amps[i].append(amps)
                all_channel_times[i].append(times)
    else:
        for ii in chunk_iter:
            times_ii, amps_ii = _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, detect_threshold,
                                                              detect_sign, n_shifts, False)

            for i, ch in enumerate(channel_ids):
                times = times_ii[i]
                amps = amps_ii[i]
                all_channel_amps[i].append(amps)
                all_channel_times[i].append(times)

    if len(chunks) > 1:
        times_list = []
        amp_list = []
        for i_ch in range(len(channel_ids)):
            times_concat = np.concatenate([all_channel_times[i_ch][ch] for ch in range(len(chunks))],
                                          axis=0)
            times_list.append(times_concat)
            amps_concat = np.concatenate([all_channel_amps[i_ch][ch] for ch in range(len(chunks))],
                                         axis=0)
            amp_list.append(amps_concat)
    else:
        times_list = [times[0] for times in all_channel_times]
        amp_list = [amps[0] for amps in all_channel_amps]

    labels_list = [[ch] * len(times)
                   for (ch, times) in zip(channel_ids, times_list)]

    # create sorting extractor
    sorting = se.NumpySortingExtractor()
    labels_flat = np.array(list(itertools.chain(*labels_list)))
    times_flat = np.array(list(itertools.chain(*times_list)))
    sorting.set_times_labels(times=times_flat, labels=labels_flat)
    sorting.set_sampling_frequency(recording.get_sampling_frequency())

    duration = (end_frame - start_frame) / recording.get_sampling_frequency()

    for i_u, u in enumerate(sorting.get_unit_ids()):
        sorting.set_unit_property(u, 'channel', u)
        amps = amp_list[i_u]
        if len(amps) > 0:
            sorting.set_unit_property(
                u, 'spike_amplitude', np.median(amp_list[i_u]))
        else:
            sorting.set_unit_property(u, 'spike_amplitude', 0)
        sorting.set_unit_property(u, 'spike_rate', len(
            sorting.get_unit_spike_train(u)) / duration)

    return sorting


def _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, thresholds, detect_sign, n_shifts,
                                  verbose):
    chunk = chunks[ii]

    if verbose:
        print(f"Chunk {ii + 1}: detecting spikes")
    if isinstance(rec_arg, dict):
        recording = se.load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg

    traces = recording.get_traces(start_frame=chunk['istart'],
                                  end_frame=chunk['iend'])

    if detect_sign == -1:
        traces = -traces
    elif detect_sign == 0:
        traces = np.abs(traces)

    sig_center = traces[:, n_shifts:-n_shifts]
    peak_mask = sig_center > thresholds
    for i in range(n_shifts):
        peak_mask &= sig_center > traces[:, i:i + sig_center.shape[1]]
        peak_mask &= sig_center >= traces[:, n_shifts +
                                          i + 1:n_shifts + i + 1 + sig_center.shape[1]]

    # find peaks
    peak_chan_ind, peak_sample_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += n_shifts

    sp_times = []
    sp_amplitudes = []

    for ch in range(len(channel_ids)):
        peak_times = peak_sample_ind[np.where(peak_chan_ind == ch)]
        sp_times.append(peak_sample_ind[np.where(
            peak_chan_ind == ch)] + chunk['istart'])
        sp_amplitudes.append(traces[ch, peak_times])

    return sp_times, sp_amplitudes
