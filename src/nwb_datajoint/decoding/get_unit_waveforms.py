import spikeextractors as se
import numpy as np

def _extract_snippet_from_traces(
    traces,
    start_frame,
    end_frame,
    channel_indices
):
    if (0 <= start_frame) and (end_frame <= traces.shape[1]):
        x = traces[:, start_frame:end_frame]
    else:
        # handle edge cases
        x = np.zeros((traces.shape[0], end_frame - start_frame), dtype=traces.dtype)
        i1 = int(max(0, start_frame))
        i2 = int(min(traces.shape[1], end_frame))
        x[:, (i1 - start_frame):(i2 - start_frame)] = traces[:, i1:i2]
    if channel_indices is not None:
        x = x[channel_indices, :]
    return x

def _get_unit_waveforms_for_chunk(
    recording,
    sorting,
    frame_offset,
    unit_ids,
    snippet_len,
    channel_ids_by_unit
):
    # chunks are chosen small enough so that all traces can be loaded into memory
    print('Retrieving traces for chunk')
    traces = recording.get_traces()

    print('Collecting waveforms for chunk')
    unit_waveforms = []
    for unit_id in unit_ids:
        times0 = sorting.get_unit_spike_train(unit_id=unit_id)
        if channel_ids_by_unit is not None:
            channel_ids = channel_ids_by_unit[unit_id]
            all_channel_ids = recording.get_channel_ids()
            channel_indices = [
                np.array(all_channel_ids).tolist().index(ch_id)
                for ch_id in channel_ids
            ]
            len_channel_indices = len(channel_indices)
        else:
            channel_indices = None
            len_channel_indices = traces.shape[0]
        # num_channels x len_of_one_snippet
        snippets = [
            _extract_snippet_from_traces(
                traces,
                start_frame=frame_offset + int(t) - snippet_len[0],
                end_frame=frame_offset + int(t) + snippet_len[1],
                channel_indices=channel_indices
            )
            for t in times0
        ]
        if len(snippets) > 0:
            unit_waveforms.append(
                # len(times0) x num_channels_in_nbhd[unit_id] x len_of_one_snippet
                np.stack(snippets)
            )
        else:
            unit_waveforms.append(
                np.zeros((0, len_channel_indices, snippet_len[0] + snippet_len[1]), dtype=traces.dtype)
            )
    return unit_waveforms

def _divide_recording_into_time_chunks(num_frames, chunk_size, padding_size):
    chunks = []
    ii = 0
    while ii < num_frames:
        ii2 = int(min(ii + chunk_size, num_frames))
        chunks.append(dict(
            istart=ii,
            iend=ii2,
            istart_with_padding=int(max(0, ii - padding_size)),
            iend_with_padding=int(min(num_frames, ii2 + padding_size))
        ))
        ii = ii2
    return chunks

def get_unit_waveforms(
    recording,
    sorting,
    unit_ids,
    channel_ids_by_unit,
    snippet_len
):
    if not isinstance(snippet_len, list) and not isinstance(snippet_len, tuple):
        b = int(snippet_len / 2)
        a = int(snippet_len) - b
        snippet_len = [a, b]

    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames()
    num_bytes_per_chunk = 1000 * 1000 * 1000 # ? how to choose this
    num_bytes_per_frame = num_channels * 2
    chunk_size = num_bytes_per_chunk / num_bytes_per_frame
    padding_size = 100 + snippet_len[0] + snippet_len[1] # a bit excess padding
    chunks = _divide_recording_into_time_chunks(
        num_frames=num_frames,
        chunk_size=chunk_size,
        padding_size=padding_size
    )
    all_unit_waveforms = [[] for ii in range(len(unit_ids))]
    for ii, chunk in enumerate(chunks):
        # chunk: {istart, iend, istart_with_padding, iend_with_padding} # include padding
        print(f'Processing chunk {ii + 1} of {len(chunks)}; chunk-range: {chunk["istart_with_padding"]} {chunk["iend_with_padding"]}; num-frames: {num_frames}')
        recording_chunk = se.SubRecordingExtractor(
            parent_recording=recording,
            start_frame=chunk['istart_with_padding'],
            end_frame=chunk['iend_with_padding']
        )
        # note that the efficiency of this operation may need improvement (really depends on sorting extractor implementation)
        sorting_chunk = se.SubSortingExtractor(
            parent_sorting=sorting,
            start_frame=chunk['istart'],
            end_frame=chunk['iend']
        )
        print(f'Getting unit waveforms for chunk {ii + 1} of {len(chunks)}')
        # num_events_in_chunk x num_channels_in_nbhd[unit_id] x len_of_one_snippet
        unit_waveforms = _get_unit_waveforms_for_chunk(
            recording=recording_chunk,
            sorting=sorting_chunk,
            frame_offset=chunk['istart'] - chunk['istart_with_padding'], # just the padding size (except 0 for first chunk)
            unit_ids=unit_ids,
            snippet_len=snippet_len,
            channel_ids_by_unit=channel_ids_by_unit
        )
        for i_unit, x in enumerate(unit_waveforms):
            all_unit_waveforms[i_unit].append(x)
    
    # concatenate the results over the chunks
    unit_waveforms = [
        # tot_num_events_for_unit x num_channels_in_nbhd[unit_id] x len_of_one_snippet
        np.concatenate(all_unit_waveforms[i_unit], axis=0)
        for i_unit in range(len(unit_ids))
    ]
    return unit_waveforms