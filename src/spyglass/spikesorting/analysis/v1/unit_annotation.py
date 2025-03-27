from typing import Optional, Union

import datajoint as dj
import numpy as np

from spyglass.spikesorting.analysis.v1.group import _get_spike_obj_name
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_unit_annotation_v1")


@schema
class UnitAnnotation(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
    unit_id: int
    """

    class Annotation(SpyglassMixin, dj.Part):
        definition = """
        -> master
        annotation: varchar(128) # the kind of annotation (e.g. a table name, "cell_type", "firing_rate", etc.)
        ---
        label = NULL: varchar(128) # text labels from analysis
        quantification = NULL: float # quantification label from analysis
        """

        def fetch_unit_spikes(self, return_unit_ids=False):
            """Fetch the spike times for a restricted set of units

            Parameters
            ----------
            return_unit_ids : bool, optional
                whether to return unit ids with spike times, by default False

            Returns
            -------
            list of np.ndarray
                list of spike times for each unit in the group,
                if return_unit_ids is False
            tuple of list of np.ndarray, list of str
                list of spike times for each unit in the group and the unit ids,
                if return_unit_ids is True
            """
            return (UnitAnnotation & self).fetch_unit_spikes(return_unit_ids)

    def add_annotation(self, key, **kwargs):
        """Add an annotation to a unit. Creates the unit if it does not exist.

        Parameters
        ----------
        key : dict
            dictionary with key for Annotation

        Raises
        ------
        ValueError
            if unit_id is not valid for the sorting
        """
        # validate new units
        unit_key = {
            k: v
            for k, v in key.items()
            if k in ["spikesorting_merge_id", "unit_id"]
        }
        if not self & unit_key:
            nwb_file = (
                SpikeSortingOutput & {"merge_id": key["spikesorting_merge_id"]}
            ).fetch_nwb()[0]
            nwb_field_name = _get_spike_obj_name(nwb_file)
            spikes = nwb_file[nwb_field_name]["spike_times"].to_list()
            if key["unit_id"] > len(spikes) and not self._test_mode:
                raise ValueError(
                    f"unit_id {key['unit_id']} is greater than ",
                    f"the number of units in {key['spikesorting_merge_id']}",
                )
            self.insert1(unit_key)
        # add annotation
        self.Annotation().insert1(key, **kwargs)

    def fetch_unit_spikes(
        self, return_unit_ids=False
    ) -> Union[list[np.ndarray], Optional[list[dict]]]:
        """Fetch the spike times for a restricted set of units

        Parameters
        ----------
        return_unit_ids : bool, optional
            whether to return unit ids with spike times, by default False

        Returns
        -------
        list of np.ndarray
            list of spike times for each unit in the group,
            if return_unit_ids is False
        tuple of list of np.ndarray, list of str
            list of spike times for each unit in the group and the unit ids,
            if return_unit_ids is True
        """
        if len(self) == len(UnitAnnotation()):
            logger.warning(
                "fetching all unit spikes if this is unintended, please call as"
                + ": (UnitAnnotation & key).fetch_unit_spikes()"
            )
        # get the set of nwb files to load
        merge_keys = [
            {"merge_id": merge_id}
            for merge_id in list(set(self.fetch("spikesorting_merge_id")))
        ]
        nwb_file_list, merge_ids = (SpikeSortingOutput & merge_keys).fetch_nwb(
            return_merge_ids=True
        )

        spikes = []
        unit_ids = []
        for nwb_file, merge_id in zip(nwb_file_list, merge_ids):
            nwb_field_name = _get_spike_obj_name(nwb_file)
            sorting_spike_times = nwb_file[nwb_field_name][
                "spike_times"
            ].to_list()
            include_unit = np.unique(
                (self & {"spikesorting_merge_id": merge_id}).fetch("unit_id")
            )
            spikes.extend(
                [sorting_spike_times[unit_id] for unit_id in include_unit]
            )
            unit_ids.extend(
                [
                    {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                    for unit_id in include_unit
                ]
            )

        if return_unit_ids:
            return spikes, unit_ids
        return spikes
