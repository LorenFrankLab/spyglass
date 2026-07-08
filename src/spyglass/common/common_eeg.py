"""Ingestion for chronic-EEG (and co-recorded EMG) signals stored as a raw
``ElectricalSeries`` in ``nwbfile.acquisition``.

Unlike the LFP pipeline, EEG telemetry is already low-pass filtered/downsampled
at acquisition, so it is referenced as-is rather than re-filtered. The trace
stays in the NWB file; ``ImportedEEG`` indexes it by ``object_id`` and records a
valid-times ``IntervalList`` so it can be time-restricted against the rest of
Spyglass. Per-channel metadata is *not* duplicated: the ``.Electrode`` part maps
each series column to the ``common_ephys.Electrode`` row already ingested from
the file's standard ``electrodes`` table (which carries the EEG/EMG group split).

This targets the probe-less skull-screw / array montages used in chronic-EEG
setups (e.g. ``DANDI:001888``), where the electrodes group onto a plain
``Device`` rather than a ``Probe``.
"""

import datajoint as dj
import numpy as np
import pynwb

from spyglass.common.common_ephys import (  # noqa: F401
    Electrode as CommonElectrode,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_nwb_file,
    get_valid_intervals,
)

schema = dj.schema("common_eeg")

# An acquisition ElectricalSeries is treated as EEG when its channels sit in an
# ElectrodeGroup whose name carries one of these tokens. Selection keys on the
# *group* name, not the series name: real files name the series opaquely (e.g.
# "ElectricalSeries_BL2") while the montage lives in "EEGArray"/"EMGArray".
_EEG_GROUP_TOKENS = ("eeg", "emg")


@schema
class ImportedEEG(SpyglassIngestion, dj.Imported):
    definition = """
    # Reference to a raw EEG/telemetry ElectricalSeries in nwbfile.acquisition.
    # The trace stays in the NWB file; this row indexes it (no filtering applied,
    # unlike the LFP pipeline) so it can be time-restricted against Spyglass.
    -> Session
    eeg_object_id: varchar(40)          # NWB ElectricalSeries object_id
    ---
    -> IntervalList                     # the series' valid (recorded) times
    eeg_sampling_rate: float            # samples/sec
    name: varchar(80)
    description: varchar(2000)
    num_samples: bigint
    unit: varchar(16)
    """

    # FKs -> Session (not Nwbfile), so point fetch_nwb() at the file explicitly,
    # as Raw does.
    _nwb_table = Nwbfile
    _source_nwb_object_type = pynwb.ecephys.ElectricalSeries

    class Electrode(SpyglassIngestion, dj.Part):
        definition = """
        # One column of the series -> the common_ephys.Electrode it records.
        # Reuses the already-ingested electrode rows, so the EEG/EMG group split
        # and brain region ride along without duplicated metadata.
        -> master
        region_index: int               # 0-based column within the series
        ---
        -> CommonElectrode
        """

    def make(self, key):
        """Standard Imported entry point.

        ``populate_all_common`` drives ingestion through ``insert_from_nwbfile``;
        ``dj.Imported`` tables are also expected to support ``populate()``.
        Delegate to the same path (as ``Raw`` does) so both work.
        """
        self.insert_from_nwbfile(key["nwb_file_name"])

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Acquisition ElectricalSeries whose channels are EEG/EMG electrodes.

        Selection keys on the referenced ``ElectrodeGroup`` name (matching
        ``eeg``/``emg``, case-insensitive), not the series name: in real files
        the series may be named opaquely (e.g. ``ElectricalSeries_BL2``) while
        the montage lives in ``EEGArray``/``EMGArray`` groups. Probe-ephys and
        analog/aux series (non-EEG group names) are excluded, so wiring this into
        ``populate_all_common`` does not claim a stray acquisition series -- but
        it does require the producer to name EEG/EMG groups accordingly.
        """
        return [
            obj
            for obj in nwb_file.acquisition.values()
            if isinstance(obj, pynwb.ecephys.ElectricalSeries)
            and self._references_eeg_group(obj)
        ]

    @staticmethod
    def _references_eeg_group(series):
        """Whether any electrode the series records is in an EEG/EMG group."""
        region = series.electrodes
        group_names = region.table["group_name"].data
        return any(
            token in str(group_names[int(position)]).lower()
            for position in region.data
            for token in _EEG_GROUP_TOKENS
        )

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Emit the ``IntervalList`` + master + ``.Electrode`` rows for one series.

        The master stores the object id + trace metadata and references an
        ``IntervalList`` of the series' valid (recorded) times (as ``Raw`` does).
        Each series column becomes an ``.Electrode`` row mapping the 0-based
        ``region_index`` to the ``common_ephys.Electrode`` it records, translated
        from the ``.electrodes`` region's positional index to the electrode's
        ``(group_name, id)``.
        """
        base_key = (base_key or {}).copy()
        series = nwb_obj
        object_id = series.object_id
        # Read series.timestamps directly (not get_timestamps()) to match
        # _valid_times, and use `is not None` so an explicit rate of 0.0 is
        # preserved rather than silently re-estimated (mirrors Raw._rate_fallback).
        rate = (
            series.rate
            if series.rate is not None
            else estimate_sampling_rate(
                np.asarray(series.timestamps[: int(1e6)])
            )
        )
        interval_list_name = f"{series.name} valid times"

        interval_entry = dict(
            base_key,
            interval_list_name=interval_list_name,
            valid_times=self._valid_times(series, rate),
            pipeline="imported_eeg",
        )
        master_row = dict(
            base_key,
            eeg_object_id=object_id,
            interval_list_name=interval_list_name,
            eeg_sampling_rate=rate,
            name=series.name,
            description=getattr(series, "description", None),
            num_samples=int(series.data.shape[0]),
            unit=series.unit,
        )

        # An ElectricalSeries always carries an .electrodes region (required by
        # NWB), so map each series column to the common_ephys.Electrode it records.
        region = series.electrodes
        table = region.table
        electrode_ids = np.asarray(table.id.data)
        group_names = table["group_name"].data

        # The region should list one electrode per trace column; a mismatch means
        # some columns can't be mapped (or extra rows go unused), so region_index
        # would no longer track the trace column. Warn rather than silently
        # mis-map (mirrors the photometry response-series ingestion).
        n_columns = series.data.shape[1] if series.data.ndim == 2 else 1
        if len(region.data) != n_columns:
            logger.warning(
                f"ImportedEEG {series.name!r}: .electrodes region maps "
                f"{len(region.data)} channel(s) but the trace has {n_columns} "
                "column(s); the region_index -> Electrode mapping may be wrong."
            )

        elec_rows = []
        for region_index, positional in enumerate(region.data):
            position = int(positional)
            # A negative index would silently wrap to the wrong electrode; an
            # out-of-range one would raise a cryptic IndexError. Fail loud and
            # named for both.
            if not 0 <= position < len(electrode_ids):
                raise ValueError(
                    f"ImportedEEG {series.name!r}: region position "
                    f"{positional} is out of range for the "
                    f"{len(electrode_ids)}-row electrodes table."
                )
            elec_rows.append(
                dict(
                    base_key,
                    eeg_object_id=object_id,
                    region_index=region_index,
                    electrode_group_name=str(group_names[position]),
                    electrode_id=int(electrode_ids[position]),
                )
            )

        # IntervalList before the master (the master FKs it); .Electrode last and
        # always present (possibly empty) so the multi-object insert loop can
        # extend every key across series.
        return {
            IntervalList: [interval_entry],
            self: [master_row],
            self.Electrode: elec_rows,
        }

    def _valid_times(self, series, rate):
        """The series' recorded-time intervals, shape ``(n_intervals, 2)``.

        A rate-based series is uniformly sampled, so it is one contiguous
        ``[start, end]`` interval computed in O(1) from ``starting_time``/``rate``.
        An explicit-``timestamps`` series may have acquisition gaps (wireless EEG
        telemetry drops packets), so its valid times come from
        ``get_valid_intervals`` -- gaps are excluded rather than spanned (as
        ``Raw`` does).
        """
        if series.timestamps is not None:
            return get_valid_intervals(
                timestamps=np.asarray(series.timestamps),
                sampling_rate=rate,
                gap_proportion=1.75,
                min_valid_len=0,
                warn=not self._test_mode,
            )
        start = series.starting_time or 0.0
        end = start + (int(series.data.shape[0]) - 1) / rate
        return np.array([[start, end]])

    def nwb_object(self, key):
        """Return the ``ElectricalSeries`` NWB object for one row.

        Uses the full key (``eeg_object_id``): a file may hold several acquisition
        series, so a ``nwb_file_name``-only fetch would be ambiguous.
        """
        object_id = (self & key).fetch1("eeg_object_id")
        nwbf = get_nwb_file(Nwbfile.get_abs_path(key["nwb_file_name"]))
        return nwbf.objects[object_id]
