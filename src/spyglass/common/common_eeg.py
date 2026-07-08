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
from spyglass.utils.nwb_helper_fn import estimate_sampling_rate, get_nwb_file

schema = dj.schema("common_eeg")


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
        # Reuses the already-ingested electrode rows, so the EEG/EMG group split,
        # region, and hemisphere ride along without duplicated metadata.
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
        """Acquisition ElectricalSeries that ``Raw`` does not claim.

        There is no NWB-native "this is EEG" marker, so selection is by exclusion:
        an ``acquisition`` ``ElectricalSeries`` whose (sanitized) name is not in
        ``Raw``'s wideband-ephys allowlist. This is safe on standard Frank-lab
        files (whose only acquisition series is the Raw-named wideband) and picks
        up dedicated EEG telemetry series. It remains a heuristic: a file with a
        second genuine raw series would be miscaught -- prefer a naming convention
        or ingestion config when the producer can provide one.
        """
        from spyglass.common.common_ephys import Raw

        raw_names = {
            self.sanitize_nwb_object_name(n)
            for n in Raw._source_nwb_object_name
        }
        return [
            obj
            for obj in nwb_file.acquisition.values()
            if isinstance(obj, pynwb.ecephys.ElectricalSeries)
            and self.sanitize_nwb_object_name(obj.name) not in raw_names
        ]

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Emit the ``IntervalList`` + master + ``.Electrode`` rows for one series.

        The master stores the object id + trace metadata and references an
        ``IntervalList`` of the series' valid (recorded) times (as ``Raw`` does).
        Each series column becomes an ``.Electrode`` row mapping the 0-based
        ``region_index`` to the ``common_ephys.Electrode`` it records, translated
        from the ``.electrodes`` region's positional index to the electrode's
        ``(group_name, id)``. A series without an ``.electrodes`` region inserts
        the master row only, with a warning.
        """
        base_key = (base_key or {}).copy()
        series = nwb_obj
        object_id = series.object_id
        rate = series.rate or estimate_sampling_rate(
            np.asarray(series.get_timestamps()[: int(1e6)])
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

        elec_rows = []
        region = getattr(series, "electrodes", None)
        if region is None:
            logger.warning(
                f"ImportedEEG {series.name!r}: no .electrodes region; storing the "
                "signal reference without per-channel Electrode links."
            )
        else:
            table = region.table
            electrode_ids = np.asarray(table.id.data)
            group_names = table["group_name"].data
            for region_index, positional in enumerate(region.data):
                position = int(positional)
                # A negative index would silently wrap; an out-of-range one would
                # mis-map. Fail loud and named instead.
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

    @staticmethod
    def _valid_times(series, rate):
        """The series' recorded-time span as a single contiguous ``[start, end]``
        interval (EEG acquisition is continuous).

        A rate-based series computes its endpoints in O(1) from
        ``starting_time``/``rate``; an explicit-``timestamps`` series uses its
        first/last timestamp.
        """
        if series.timestamps is not None:
            timestamps = np.asarray(series.timestamps)
            return np.array([[timestamps[0], timestamps[-1]]])
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
