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
    "!=": lambda vals, target: np.not_equal(vals, target) & _not_missing(vals),
    "between": lambda vals, target: (vals >= target[0]) & (vals <= target[1]),
    "outside": lambda vals, target: (vals < target[0]) | (vals > target[1]),
    "isin": lambda vals, target: _isin(vals, target),
    "notin": lambda vals, target: ~_isin(vals, target) & _not_missing(vals),
}
# Only "isin" and "notin" are list-aware, via _isin. The rest compare a
# unit's value as a whole, so on a column holding a list per unit (e.g. the
# curation labels) numpy compares the list object itself to the target: a
# list never equals a string, making "==" all-False and "!=" all-True
# regardless of the labels. filter_units_by_criteria raises rather than
# return such a mask
SCALAR_ONLY_OPERATORS = frozenset(UNIT_CRITERIA_OPERATORS) - {"isin", "notin"}
# Operators taking a [low, high] pair rather than one value
RANGE_OPERATORS = ("between", "outside")
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
    unit_criteria : dict, optional
        criteria on units table columns the unit must satisfy, by default None.
        See `SortedSpikesGroup.filter_units_by_criteria`
    """

    definition = """
    unit_filter_params_name: varchar(32)
    ---
    include_labels = Null: longblob
    exclude_labels = Null: longblob
    unit_criteria = Null: longblob # column -> criterion the unit must satisfy
    """
    # NOTE: pk reduced from 128 to 32 to avoid long primary key error
    contents = [
        {
            "unit_filter_params_name": "all_units",
            "include_labels": [],
            "exclude_labels": [],
        },
        {
            "unit_filter_params_name": "exclude_noise",
            "include_labels": [],
            "exclude_labels": ["noise", "mua"],
        },
        {
            "unit_filter_params_name": "default_exclusion",
            "include_labels": [],
            "exclude_labels": ["noise", "mua"],
        },
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
    def filter_units_by_criteria(
        units_df,
        unit_criteria: Optional[dict] = None,
        strict: bool = True,
    ) -> np.ndarray:
        """Filter units on arbitrary columns of the units table

        Parameters
        ----------
        units_df : pd.DataFrame
            units table of one sorting, one row per unit
        unit_criteria : dict, optional
            column name to {operator: value}, or to a bare value or list of
            values as shorthand for {"isin": value}. A unit is included only if
            it satisfies every criterion. Operators are ">", ">=", "<", "<=",
            "==", "!=", "between" (matches the inclusive [low, high] pair) and
            "outside" (its exact complement), "isin" and "notin". By default
            None, which includes every unit.
        strict : bool, optional
            by default True, raise if a criterion names a column this units
            table does not have. If False, skip that criterion with a warning
            and apply the rest, which lets **every** unit of this sorting pass
            it. `fetch_spike_data` passes False so that it can report every
            sorting in the group missing the column, then raises itself.

        Returns
        -------
        np.ndarray
            boolean mask of shape (n_units,), True for each unit satisfying
            all criteria

        Raises
        ------
        ValueError
            if a criterion holds no operator at all, if an operator is not one
            of those listed above, if "between" or "outside" is given anything
            but a [low, high] pair, if any operator but "isin" or "notin" is
            applied to a column holding a list per unit, or, when strict, if a
            criterion names a column not in units_df

        Notes
        -----
        Units missing a value (NaN, or potentially None from an imported
        units table) fail every criterion on that column, negated ones
        ("!=", "notin") included: a metric that was never computed is no
        evidence that the unit is good. "isin" and "notin" also work on columns
        holding a list per unit (e.g. the curation labels), matching if any
        item of the list is in the target. An empty list is a value, not a
        missing one, so a unit carrying no labels passes "notin".

        Each sorting in a group has its own units table, and those tables may
        not share the same columns (e.g. if they were curated differently), so
        a criterion may apply to only some of them. Passing a criteria column
        that is not in the units table raises an error. Pass strict=False to
        skip criteria on missing columns (passing all units for that criterion)
        with a warning instead of erroring.
        """
        include_mask = np.ones(len(units_df), dtype=bool)

        for column, criterion in (unit_criteria or {}).items():
            if column not in units_df:
                if strict:
                    raise ValueError(
                        f"Unit criteria column '{column}' not in units table. "
                        + f"Columns are {list(units_df.columns)}. Pass "
                        + "strict=False to skip criteria on missing columns."
                    )
                logger.warning(
                    f"Unit criteria column '{column}' not in units table. "
                    + "Skipping this criterion: every unit of this sorting "
                    + "passes it unfiltered."
                )
                continue

            if not isinstance(criterion, dict):  # shorthand for membership
                criterion = {"isin": criterion}
            if not criterion:
                raise ValueError(
                    f"Empty criterion for column '{column}': no operator to "
                    + "apply. Drop the column from the criteria to include "
                    + "every unit."
                )
            values = units_df[column].to_numpy()

            for operator, value in criterion.items():
                if operator not in UNIT_CRITERIA_OPERATORS:
                    raise ValueError(
                        f"Invalid unit criteria operator '{operator}' for "
                        + f"column '{column}'. Expected one of "
                        + f"{list(UNIT_CRITERIA_OPERATORS)}"
                    )
                if operator in RANGE_OPERATORS and (
                    not isinstance(value, (list, tuple, np.ndarray))
                    or len(value) != 2
                ):
                    raise ValueError(
                        f"Unit criteria operator '{operator}' for column "
                        + f"'{column}' takes a [low, high] pair, got "
                        + f"{value!r}."
                    )
                if operator in SCALAR_ONLY_OPERATORS and _is_list_valued(
                    values
                ):
                    raise ValueError(
                        f"Unit criteria operator '{operator}' cannot filter "
                        + f"column '{column}', which holds a list per unit. "
                        + "Comparing a list to the criterion would mask every "
                        + "unit in or out at once. Use 'isin' or 'notin', "
                        + "which match any item of the list."
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

        # get the filtering parameters. unit_criteria is fetched by `get` so
        # that this still works against a table that has not yet been altered
        # to add it
        filter_params = (UnitSelectionParams & key).fetch1()
        include_labels = filter_params["include_labels"]
        exclude_labels = filter_params["exclude_labels"]
        unit_criteria = filter_params.get("unit_criteria")

        # where each criteria column was, and was not, applied, as criteria
        # column -> merge_id of every sorting whose units table has
        # (applied_to) or lacks (skipped_by) it. Both are needed to tell a
        # column missing everywhere (potentially a typo) from one missing
        # here and there (a result mixing gated and un-gated units), so they
        # are checked once every sorting has been seen
        criteria_columns = set(unit_criteria or {})
        applied_to = {column: [] for column in criteria_columns}
        skipped_by = {column: [] for column in criteria_columns}

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

            # filter the spike times based on the curation labels if present
            group_col = next(
                (c for c in units_df.columns if c in CURATION_LABEL_COLUMNS),
                None,
            )
            if group_col is not None and not test_mode:
                include_unit &= SortedSpikesGroup.filter_units(
                    units_df[group_col].to_list(),
                    include_labels,
                    exclude_labels,
                )

            # filter on arbitrary criteria over the units table columns
            for column in criteria_columns:
                seen = applied_to if column in units_df else skipped_by
                seen[column].append(merge_id)
            include_unit &= SortedSpikesGroup.filter_units_by_criteria(
                units_df, unit_criteria, strict=False
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

        if any(skipped_by.values()):
            raise _skipped_criteria_error(skipped_by, applied_to)

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


def _skipped_criteria_error(skipped_by: dict, applied_to: dict) -> ValueError:
    """Error naming every sorting a unit criterion could not be applied to

    Parameters
    ----------
    skipped_by : dict
        criteria column -> merge_id of every sorting whose units table lacks it
    applied_to : dict
        criteria column -> merge_id of every sorting whose units table has it
    """
    reasons = []
    for column, skipped in skipped_by.items():
        if not skipped:
            continue
        applied = applied_to[column]
        if not applied:
            reasons.append(
                f"'{column}': no sorting in this group has this column, so "
                + "the criterion filtered nothing. Check for a typo."
            )
            continue
        reasons.append(
            f"'{column}': missing from {len(skipped)} of "
            + f"{len(skipped) + len(applied)} units tables "
            + f"({', '.join(str(merge_id) for merge_id in skipped)}), whose "
            + "units would be returned un-gated alongside the gated ones."
        )
    return ValueError(
        "Unit criteria could not be applied to every sorting in this group:\n"
        + "\n".join(f"  - {reason}" for reason in reasons)
        + "\nCorrect the column name in the group's UnitSelectionParams, drop "
        + "the criterion, or drop each sorting whose units table lacks the "
        + "column."
    )


def _not_missing(values):
    """False where a unit has no value for the column a criterion gates on

    We need this check because a missing value satisfies the negated
    operators ("!=", "notin") on its own, which fails open: a criterion
    meant to drop bad units would instead admit the units missing the
    metric it gates on.
    """
    if values.dtype.kind == "f":
        return ~np.isnan(values)

    if values.dtype == object:  # np.isnan takes neither a list nor a string
        return np.array(
            [
                val is not None
                and not (isinstance(val, float) and np.isnan(val))
                for val in values
            ]
        )

    # Only a float column, or an object column (what pandas falls back to for
    # a column of mixed types), can hold a missing value. No other dtype can,
    # so every one of their units passes.
    return np.ones(len(values), dtype=bool)


def _isin(values, target):
    """True where a unit's value, or any item of it, is in target

    Columns of the units table may hold a list per unit (e.g. the curation
    labels), which np.isin cannot compare against target directly.
    """
    target = np.atleast_1d(target)

    if _is_list_valued(values):
        return np.array(
            [np.isin(np.atleast_1d(value), target).any() for value in values]
        )

    return np.isin(values, target)


def _is_list_valued(values):
    """True where a column holds a list per unit, rather than one value"""
    return values.dtype == object and any(
        isinstance(value, (list, tuple, np.ndarray)) for value in values
    )


def _get_spike_obj_name(nwb_file, allow_empty=False):
    nwb_field_name = (
        "object_id"
        if "object_id" in nwb_file
        else "units" if "units" in nwb_file else None
    )
    if nwb_field_name is None and not allow_empty:
        raise ValueError("NWB file does not have 'object_id' or 'units' field")
    return nwb_field_name
