import numpy as np
import spikeinterface as si
from typing import List, Dict, Union

# Note this is completely untested
def apply_merge_groups_to_sorting(sorting: si.BaseSorting, merge_groups: List[List[int]]):
    # return a new sorting where the units are merged according to merge_groups
    # merge_groups is a list of lists of unit_ids.
    # for example: merge_groups = [[1, 2], [5, 8, 4]]]

    # A sorting segment in the new sorting
    class NewSortingSegment(si.BaseSortingSegment):
        def __init__(self):
            # Store all the unit spike trains in RAM
            self._unit_spike_trains: Dict[int, np.array] = {}
        
        def add_unit(self, unit_id: int, spike_times: np.array):
            # Add a unit spike train
            self._unit_spike_trains[unit_id] = spike_times

        def get_unit_spike_train(self,
            unit_id,
            start_frame: Union[int, None] = None,
            end_frame: Union[int, None] = None,
        ) -> np.ndarray:
            # Get a unit spike train
            spike_times = self._unit_spike_trains[unit_id]
            if start_frame is not None:
                spike_times = spike_times[spike_times >= start_frame]
            if end_frame is not None:
                spike_times = spike_times[spike_times < end_frame]
            return spike_times

    # Here's the new sorting we are going to return
    new_sorting = si.BaseSorting(
        sampling_frequency=sorting.get_sampling_frequency(),
        unit_ids=sorting.get_unit_ids()
    )
    # Loop through the sorting segments in the original sorting
    # and add merged versions to the new sorting
    for sorting_segment in sorting._sorting_segments: # Note: sorting_segments should be exposed in spikeinterface
        # initialize the new sorting segment
        new_sorting_segment = NewSortingSegment()
        # keep track of which unit_ids are part of the merge groups
        used_unit_ids = []
        # first loop through the merge groups
        for merge_group in merge_groups:
            # the representative unit_id is the min in the group
            representative_unit_id = min(merge_group)
            # we are going to take the union of all the spike trains for the merge group
            spike_trains_to_concatenate = []
            for unit_id in merge_group:
                spike_trains_to_concatenate.append(
                    sorting_segment.get_unit_spike_train(unit_id)
                )
                # append this unit_id to the used unit_ids
                used_unit_ids.append(unit_id)
            # concatenate the spike trains for this merge group
            spike_train = np.concatenate(spike_trains_to_concatenate)
            # sort the concatenated spike train (chronological)
            spike_train = np.sort(spike_train)
            # add the unit to the new sorting segment
            new_sorting_segment.add_unit(representative_unit_id, spike_train)
        # Now we'll take care of all of the unit_ids that are not part of a merge group
        for unit_id in sorting.get_unit_ids():
            if unit_id not in used_unit_ids:
                new_sorting_segment.add_unit(
                    unit_id,
                    sorting_segment.get_unit_spike_train(unit_id)
                )
        # add the new sorting segment to the new sorting
        new_sorting.add_sorting_segment(new_sorting_segment)
    # finally, return the new sorting
    return new_sorting