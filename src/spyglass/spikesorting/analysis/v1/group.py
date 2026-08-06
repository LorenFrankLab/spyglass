from itertools import compress
from typing import Optional, Union

import datajoint as dj
import numpy as np

from spyglass.common import Session  # noqa: F401
from spyglass.settings import test_mode
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin, SpyglassMixinPart
from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

schema = dj.schema("spikesorting_group_v1")

# Operators accepted in UnitSelectionParams.unit_criteria, each applied
# elementwise to one column of the units table
UNIT_CRITERIA_OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    "between": lambda vals, target: (vals >= target[0]) & (vals <= target[1]),
    "outside": lambda vals, target: (vals < target[0]) | (vals > target[1]),
    "isin": lambda vals, target: np.isin(vals, np.atleast_1d(target)),
    "notin": lambda vals, target: ~np.isin(vals, np.atleast_1d(target)),
}
# Curation label columns, v0: "label", v1: "curation_label"
CURATION_LABEL_COLUMNS = ("label", "curation_label")


@schema
class UnitSelectionParams(SpyglassMixin, dj.Manual):
    """Unit selection parameters for sorted spikes

    Attributes
    ----------
    unit_filter_params_name : str
        name of the unit selection parameters
    include_labels : List[str], optional
        list of labels to include, by default None
    exclude_labels : List[str], optional
        list of labels to exclude, by default None
    label_columns : List[str], optional
        additional units table columns (e.g. ["brain_region"]) whose values are
        cast to strings and appended to the unit's labels, by default None
    unit_criteria : dict, optional
        criteria on units table columns the unit must satisfy, by default None.
        See `SortedSpikesGroup.filter_units_by_criteria`
    """

    definition = """
    unit_filter_params_name: varchar(32)
    ---
    include_labels = Null: longblob
    exclude_labels = Null: longblob
    label_columns = Null: longblob # extra units columns to treat as labels
    unit_criteria = Null: longblob # column -> criterion the unit must satisfy
    """
    # NOTE: pk reduced from 128 to 32 to avoid long primary key error
    contents = [
        ["all_units", [], [], [], {}],
        ["exclude_noise", [], ["noise", "mua"], [], {}],
        ["default_exclusion", [], ["noise", "mua"], [], {}],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default unit selection parameters"""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SortedSpikesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    -> UnitSelectionParams
    sorted_spikes_group_name: varchar(80)
    """

    class Units(SpyglassMixinPart):
        definition = """
        -> master
        -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        nwb_file_name: str,
        unit_filter_params_name: str = "all_units",
        keys: list[dict] = [],
    ):
        """Create a new group of sorted spikes"""
        group_key = {
            "sorted_spikes_group_name": group_name,
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": unit_filter_params_name,
        }
        if self & group_key:
            if test_mode:
                return
            raise ValueError(
                f"Group {nwb_file_name}: {group_name} already exists "
                + "please delete the group before creating a new one",
            )

        parts_insert = [{**key, **group_key} for key in keys]

        self.insert1(
            group_key,
            skip_duplicates=True,
        )
        self.Units.insert(parts_insert, skip_duplicates=True)

    @staticmethod
    def filter_units(
        labels: list[list[str]],
        include_labels: list[str],
        exclude_labels: list[str],
    ) -> np.ndarray:
        """
        Filter units based on labels

        labels: list of list of strings
            list of labels for each unit
        include_labels: list of strings
            if provided, only units with any of these labels will be included
        exclude_labels: list of strings
            if provided, units with any of these labels will be excluded
        """
        include_labels = np.unique(include_labels)
        exclude_labels = np.unique(exclude_labels)

        if include_labels.size == 0 and exclude_labels.size == 0:
            # if no labels are provided, include all units
            return np.ones(len(labels), dtype=bool)

        include_mask = np.zeros(len(labels), dtype=bool)
        for ind, unit_labels in enumerate(labels):
            if isinstance(unit_labels, str):
                unit_labels = [unit_labels]
            if (
                include_labels.size > 0
                and np.all(~np.isin(unit_labels, include_labels))
            ) or np.any(np.isin(unit_labels, exclude_labels)):
                # if the unit does not have any of the include labels
                # or has any of the exclude labels, skip
                continue
            include_mask[ind] = True
        return include_mask

    @staticmethod
    def filter_units_by_criteria(units_df, unit_criteria: dict) -> np.ndarray:
        """Filter units on arbitrary columns of the units table

        Parameters
        ----------
        units_df : pd.DataFrame
            units table of one sorting, one row per unit
        unit_criteria : dict
            column name to {operator: value}, or to a bare value or list of
            values as shorthand for {"isin": value}. A unit is included only if
            it satisfies every criterion. Operators are ">", ">=", "<", "<=",
            "==", "!=", "between" and "outside" (an inclusive [low, high]
            pair), "isin" and "notin".

        Returns
        -------
        np.ndarray
            boolean mask, True for each unit satisfying all criteria

        Notes
        -----
        Units whose value is NaN fail every numeric comparison. A criterion
        naming a column the sorting does not have is skipped with a warning;
        the remaining criteria still apply. Each sorting in a group has its
        own units table, and those tables may not share the same columns
        (e.g. if they were curated differently), so a criterion may apply to
        only some of them.
        """
        include_mask = np.ones(len(units_df), dtype=bool)

        for column, criterion in (unit_criteria or {}).items():
            if column not in units_df:
                logger.warning(
                    f"Unit criteria column '{column}' not in units table. "
                    + "Skipping this criterion."
                )
                continue

            if not isinstance(criterion, dict):  # shorthand for membership
                criterion = {"isin": criterion}
            values = np.asarray(units_df[column].to_list())

            for operator, value in criterion.items():
                if operator not in UNIT_CRITERIA_OPERATORS:
                    raise ValueError(
                        f"Invalid unit criteria operator '{operator}' for "
                        + f"column '{column}'. Expected one of "
                        + f"{list(UNIT_CRITERIA_OPERATORS)}"
                    )
                include_mask &= UNIT_CRITERIA_OPERATORS[operator](values, value)

        return include_mask

    @classmethod
    def fetch_spike_data(
        cls,
        key: dict,
        time_slice: Union[list[float], slice] = None,
        return_unit_ids: bool = False,
    ) -> Union[list[np.ndarray], Optional[list[dict]]]:
        """fetch spike times for units in the group

        Parameters
        ----------
        key : dict
            dictionary containing the group key
        time_slice : list of float or slice, optional
            if provided, filter for spikes occurring in the interval
            [start, stop], by default None
        return_unit_ids : bool, optional
            if True, return the unit_ids along with the spike times, by default
            False. Unit ids defined as a list of dictionaries with keys
            'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        list of np.ndarray
            list of spike times for each unit in the group
        """
        key = cls.get_fully_defined_key(key)

        # get merge_ids for SpikeSortingOutput
        merge_ids = (
            (
                SortedSpikesGroup.Units
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "sorted_spikes_group_name": key["sorted_spikes_group_name"],
                }
            )
        ).fetch("spikesorting_merge_id")

        # get the filtering parameters. label_columns and unit_criteria are
        # fetched by `get` so that this still works against a table that has
        # not yet been altered to add them
        filter_params = (UnitSelectionParams & key).fetch1()
        include_labels = filter_params["include_labels"]
        exclude_labels = filter_params["exclude_labels"]
        label_columns = filter_params.get("label_columns")
        unit_criteria = filter_params.get("unit_criteria")

        # get the spike times for each merge_id
        spike_times = []
        unit_ids = []
        merge_keys = [dict(merge_id=merge_id) for merge_id in merge_ids]
        nwb_file_list, merge_ids = (SpikeSortingOutput & merge_keys).fetch_nwb(
            return_merge_ids=True
        )
        for nwb_file, merge_id in zip(nwb_file_list, merge_ids):
            nwb_field_name = _get_spike_obj_name(nwb_file, allow_empty=True)

            if nwb_field_name is None:
                logger.warning(f"No spike object found for {merge_id}")
                # case where no units found or curation removed all units
                continue

            units_df = nwb_file[nwb_field_name]
            sorting_spike_times = units_df["spike_times"].to_list()
            file_unit_ids = [
                {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                for unit_id in range(len(sorting_spike_times))
            ]

            include_unit = np.ones(len(sorting_spike_times), dtype=bool)

            # filter the spike times based on the labels if present
            unit_labels = _compile_unit_labels(units_df, label_columns)
            if unit_labels is not None and not test_mode:
                include_unit &= SortedSpikesGroup.filter_units(
                    unit_labels, include_labels, exclude_labels
                )

            # filter on arbitrary criteria over the units table columns
            include_unit &= SortedSpikesGroup.filter_units_by_criteria(
                units_df, unit_criteria
            )

            if not include_unit.all():
                sorting_spike_times = list(
                    compress(sorting_spike_times, include_unit)
                )
                file_unit_ids = list(compress(file_unit_ids, include_unit))

            # filter the spike times based on the time slice if provided
            if time_slice is not None:
                if isinstance(time_slice, (list, tuple)):
                    time_slice = slice(*time_slice)
                sorting_spike_times = [
                    times[
                        np.logical_and(
                            times >= time_slice.start, times <= time_slice.stop
                        )
                    ]
                    for times in sorting_spike_times
                ]

            # append the approved spike times to the list
            spike_times.extend(sorting_spike_times)
            unit_ids.extend(file_unit_ids)

        if return_unit_ids:
            return spike_times, unit_ids

        return spike_times

    @classmethod
    def get_spike_indicator(
        cls,
        key: dict,
        time: np.ndarray,
        return_unit_ids: bool = False,
    ) -> np.ndarray:
        """Get spike indicator matrix for the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix
        return_unit_ids : bool, optional
            if True, return the unit ids along with the spike indicator matrix,
            by default False. Unit ids defined as a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        list of dict, optional
            if return_unit_ids is True, returns a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number' for each unit
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times, unit_ids = cls.fetch_spike_data(key, return_unit_ids=True)

        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[np.logical_and(times >= min_time, times <= max_time)]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]
        if return_unit_ids:
            return spike_indicator, unit_ids
        return spike_indicator

    @classmethod
    def get_firing_rate(
        cls,
        key: dict,
        time: np.ndarray,
        multiunit: bool = False,
        smoothing_sigma: float = 0.015,
        return_unit_ids: bool = False,
    ) -> np.ndarray:
        """Get time-dependent firing rate for units in the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the firing rate
        multiunit : bool, optional
            if True, return the multiunit firing rate for units in the group,
            by default False
        smoothing_sigma : float, optional
            standard deviation of gaussian filter to smooth firing rates in
            seconds, by default 0.015
        return_unit_ids : bool, optional
            if True, return the unit ids along with the firing rate, by default
            False. Unit ids defined as a list of dictionaries with keys
            'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        np.ndarray
            time-dependent firing rate with shape (len(time), n_units)
        list of dict, optional
            if return_unit_ids is True, returns a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number' for each unit
        """
        spike_indicator, unit_ids = cls.get_spike_indicator(
            key, time, return_unit_ids=True
        )
        firing_rate = firing_rate_from_spike_indicator(
            spike_indicator=spike_indicator,
            time=time,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )
        if return_unit_ids:
            return firing_rate, unit_ids
        return firing_rate


def _compile_unit_labels(units_df, label_columns=None):
    """Labels of each unit: the curation labels plus any label_columns values

    Returns a list of list of str, or None if the sorting has no label columns.
    """
    columns = [c for c in CURATION_LABEL_COLUMNS if c in units_df]
    for column in label_columns or []:
        if column not in units_df:
            logger.warning(f"Label column '{column}' not in units table")
        else:
            columns.append(column)

    if not columns:
        return None

    unit_labels = [[] for _ in range(len(units_df))]
    for column in columns:
        for labels, value in zip(unit_labels, units_df[column].to_list()):
            labels.extend(
                [str(v) for v in value]
                if isinstance(value, (list, tuple, np.ndarray))
                else [str(value)]
            )

    return unit_labels


def _get_spike_obj_name(nwb_file, allow_empty=False):
    nwb_field_name = (
        "object_id"
        if "object_id" in nwb_file
        else "units" if "units" in nwb_file else None
    )
    if nwb_field_name is None and not allow_empty:
        raise ValueError("NWB file does not have 'object_id' or 'units' field")
    return nwb_field_name
