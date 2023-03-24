from typing import Dict, List, Union

import numpy as np
import spikeinterface as si


class MergedSortingExtractor(si.BaseSorting):
    extractor_name = "MergedSortingExtractor"
    installed = True
    installation_mesg = "Always installed"
    is_writable = False

    def __init__(
        self, *, parent_sorting: si.BaseSorting, merge_groups: List[List[int]]
    ):
        # Loop through the sorting segments in the original sorting
        # and add merged versions to the new sorting
        sorting_segment_list = []
        final_unit_ids = []  # TODO: not sure what final_unit_ids is used for
        # Note: sorting_segments should be exposed in spikeinterface
        for sorting_segment in parent_sorting._sorting_segments:
            # initialize the new sorting segment
            new_sorting_segment = MergedSortingSegment()
            # keep track of which unit_ids are part of the merge groups
            used_unit_ids = []
            # first loop through the merge groups
            for merge_group in merge_groups:
                # the representative unit_id is the min in the group
                representative_unit_id = min(merge_group)
                if representative_unit_id not in final_unit_ids:
                    final_unit_ids.append(representative_unit_id)
                # we are going to take the union of all the spike trains for the merge group
                spike_trains_to_concatenate = []
                for unit_id in merge_group:
                    spike_trains_to_concatenate.append(
                        sorting_segment.get_unit_spike_train(
                            unit_id, start_frame=None, end_frame=None
                        )
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
            for unit_id in parent_sorting.get_unit_ids():
                if unit_id not in used_unit_ids:
                    new_sorting_segment.add_unit(
                        unit_id,
                        sorting_segment.get_unit_spike_train(
                            unit_id, start_frame=None, end_frame=None
                        ),
                    )
                    if unit_id not in final_unit_ids:
                        final_unit_ids.append(unit_id)

            # add the new sorting segment to the new sorting
            sorting_segment_list.append(new_sorting_segment)

        # Add the segments to this sorting
        final_unit_ids.sort()  # TODO: not sure what final_unit_ids is used for
        si.BaseSorting.__init__(
            self,
            sampling_frequency=parent_sorting.get_sampling_frequency(),
            unit_ids=final_unit_ids,
        )
        self._kwargs = {
            "parent_sorting": parent_sorting.to_dict(
                include_annotations=True, include_properties=True
            ),
            "merge_groups": merge_groups,
        }
        for new_sorting_segment in sorting_segment_list:
            self.add_sorting_segment(new_sorting_segment)
        print(self)


class MergedSortingSegment(si.BaseSortingSegment):
    def __init__(self):
        si.BaseSortingSegment.__init__(self)
        # Store all the unit spike trains in RAM
        self._unit_spike_trains: Dict[int, np.array] = {}

    def add_unit(self, unit_id: int, spike_times: np.array):
        # Add a unit spike train
        self._unit_spike_trains[unit_id] = spike_times

    def get_unit_spike_train(
        self,
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
