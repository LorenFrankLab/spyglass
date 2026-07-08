"""Ingestion of ``ndx-fiber-photometry`` metadata into queryable DataJoint tables.

Six reusable device/indicator tables plus a per-fiber ``FiberPhotometryConfig``
capture the experimental setup (which fiber, indicator, excitation source,
detector, filters, wavelengths, insertion). Recorded fluorescence traces stay in
the NWB file: ``FiberPhotometryResponseSeries`` stores each series' NWB
``object_id`` (not the array) and its ``fetch1_dataframe()`` retrieves the trace
on demand.

The tables are photometry-reference-scoped: each device table's
``get_nwb_objects()`` collects only the objects a ``FiberPhotometryTable``
references (via :mod:`spyglass.common._photometry_nwb`), so a non-photometry file
is a clean no-op and device subtypes are handled without matching on class name.
Ingestion never imports ``ndx-fiber-photometry``; NWB types are matched by
class-name string and the object references resolve from the file-embedded spec.
"""

import datajoint as dj
import numpy as np

from spyglass.common._photometry_nwb import (
    class_discriminator,
    model_attr,
    model_range,
    populated_attrs,
    referenced_devices,
    response_series,
)
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassIngestion
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("common_photometry")

# Both namespaces are read during ingestion (the ndx-fiber-photometry table and
# the ndx-ophys-devices object references), so a file below either minimum is
# gated with a warning by the mixin's post-get_nwb_objects version check.
_EXT_REQ = {"ndx-fiber-photometry": "0.2.3", "ndx-ophys-devices": "0.3.1"}


class _PhotometryDevice(SpyglassIngestion):
    """Shared behavior for the reusable, reference-scoped device catalog tables.

    Subclasses set ``_ref_columns`` (the ``FiberPhotometryTable`` column name(s)
    whose referenced objects this table ingests). ``get_nwb_objects`` collects
    exactly those objects; a file with no ``FiberPhotometry`` container yields an
    empty list — a clean no-op.
    """

    _expected_duplicates = True  # a shared catalog, reused across sessions
    _extension_requirements = _EXT_REQ
    _ref_columns = ()  # set per subclass

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        return referenced_devices(nwb_file, self._ref_columns)


@schema
class Indicator(_PhotometryDevice, dj.Manual):
    definition = """
    # Fluorescent indicator (reagent) referenced by a FiberPhotometryTable
    indicator_name: varchar(80)
    ---
    label: varchar(255)                 # standard notation, e.g. 'dLight3.8'
    description=null: varchar(2000)
    manufacturer=null: varchar(255)
    """

    _ref_columns = ("indicator",)

    table_key_to_obj_attr = {
        "self": dict(
            indicator_name="name",
            label="label",
            description="description",
            manufacturer="manufacturer",
        )
    }


@schema
class ExcitationSource(_PhotometryDevice, dj.Manual):
    definition = """
    # Excitation source referenced by a FiberPhotometryTable (reusable spec)
    excitation_source_name: varchar(80)
    ---
    source_class: enum('continuous', 'pulsed')  # from the referenced NWB subtype
    source_type=null: varchar(255)      # model spec, e.g. 'LED'
    excitation_mode=null: varchar(255)  # e.g. 'one-photon'
    wavelength_min_nm=null: float       # model wavelength_range_in_nm[0]
    wavelength_max_nm=null: float       # model wavelength_range_in_nm[1]
    manufacturer=null: varchar(255)
    model_number=null: varchar(255)
    model_description=null: varchar(2000)
    """

    _ref_columns = ("excitation_source",)

    table_key_to_obj_attr = {
        "self": dict(
            excitation_source_name="name",
            source_class=class_discriminator(
                {"PulsedExcitationSource": "pulsed"}, "continuous"
            ),
            source_type=model_attr("source_type"),
            excitation_mode=model_attr("excitation_mode"),
            wavelength_min_nm=model_range("wavelength_range_in_nm", 0),
            wavelength_max_nm=model_range("wavelength_range_in_nm", 1),
            manufacturer=model_attr("manufacturer"),
            model_number=model_attr("model_number"),
            model_description=model_attr("description"),
        )
    }


@schema
class Photodetector(_PhotometryDevice, dj.Manual):
    definition = """
    # Photodetector referenced by a FiberPhotometryTable (reusable spec)
    photodetector_name: varchar(80)
    ---
    detector_type=null: varchar(255)    # e.g. 'PMT', 'Photodiode'
    gain=null: float
    gain_unit=null: varchar(255)
    wavelength_min_nm=null: float       # model wavelength_range_in_nm[0]
    wavelength_max_nm=null: float       # model wavelength_range_in_nm[1]
    manufacturer=null: varchar(255)
    model_number=null: varchar(255)
    model_description=null: varchar(2000)
    """

    _ref_columns = ("photodetector",)

    table_key_to_obj_attr = {
        "self": dict(
            photodetector_name="name",
            detector_type=model_attr("detector_type"),
            gain=model_attr("gain"),
            gain_unit=model_attr("gain_unit"),
            wavelength_min_nm=model_range("wavelength_range_in_nm", 0),
            wavelength_max_nm=model_range("wavelength_range_in_nm", 1),
            manufacturer=model_attr("manufacturer"),
            model_number=model_attr("model_number"),
            model_description=model_attr("description"),
        )
    }


@schema
class DichroicMirror(_PhotometryDevice, dj.Manual):
    definition = """
    # Dichroic mirror referenced by a FiberPhotometryTable (reusable spec)
    dichroic_mirror_name: varchar(80)
    ---
    cut_on_wavelength_in_nm=null: float
    cut_off_wavelength_in_nm=null: float
    reflection_band_min_nm=null: float   # model reflection_band_in_nm[0]
    reflection_band_max_nm=null: float   # model reflection_band_in_nm[1]
    transmission_band_min_nm=null: float # model transmission_band_in_nm[0]
    transmission_band_max_nm=null: float # model transmission_band_in_nm[1]
    angle_of_incidence_in_degrees=null: float
    manufacturer=null: varchar(255)
    model_number=null: varchar(255)
    model_description=null: varchar(2000)
    """

    _ref_columns = ("dichroic_mirror",)

    table_key_to_obj_attr = {
        "self": dict(
            dichroic_mirror_name="name",
            cut_on_wavelength_in_nm=model_attr("cut_on_wavelength_in_nm"),
            cut_off_wavelength_in_nm=model_attr("cut_off_wavelength_in_nm"),
            reflection_band_min_nm=model_range("reflection_band_in_nm", 0),
            reflection_band_max_nm=model_range("reflection_band_in_nm", 1),
            transmission_band_min_nm=model_range("transmission_band_in_nm", 0),
            transmission_band_max_nm=model_range("transmission_band_in_nm", 1),
            angle_of_incidence_in_degrees=model_attr(
                "angle_of_incidence_in_degrees"
            ),
            manufacturer=model_attr("manufacturer"),
            model_number=model_attr("model_number"),
            model_description=model_attr("description"),
        )
    }


@schema
class OpticalFilter(_PhotometryDevice, dj.Manual):
    definition = """
    # Optical filter (base/band/edge) referenced by a FiberPhotometryTable
    optical_filter_name: varchar(80)
    ---
    filter_class: enum('base', 'band', 'edge')  # from the referenced NWB subtype
    filter_type=null: varchar(255)       # e.g. 'Bandpass', 'Longpass'
    center_wavelength_in_nm=null: float  # band
    bandwidth_in_nm=null: float          # band
    cut_wavelength_in_nm=null: float     # edge
    slope_in_percent_cut_wavelength=null: float          # edge
    slope_starting_transmission_in_percent=null: float   # edge
    slope_ending_transmission_in_percent=null: float     # edge
    manufacturer=null: varchar(255)
    model_number=null: varchar(255)
    model_description=null: varchar(2000)
    """

    # Both filter reference columns feed the one catalog table.
    _ref_columns = ("emission_filter", "excitation_filter")

    table_key_to_obj_attr = {
        "self": dict(
            optical_filter_name="name",
            filter_class=class_discriminator(
                {"BandOpticalFilter": "band", "EdgeOpticalFilter": "edge"},
                "base",
            ),
            filter_type=model_attr("filter_type"),
            center_wavelength_in_nm=model_attr("center_wavelength_in_nm"),
            bandwidth_in_nm=model_attr("bandwidth_in_nm"),
            cut_wavelength_in_nm=model_attr("cut_wavelength_in_nm"),
            slope_in_percent_cut_wavelength=model_attr(
                "slope_in_percent_cut_wavelength"
            ),
            slope_starting_transmission_in_percent=model_attr(
                "slope_starting_transmission_in_percent"
            ),
            slope_ending_transmission_in_percent=model_attr(
                "slope_ending_transmission_in_percent"
            ),
            manufacturer=model_attr("manufacturer"),
            model_number=model_attr("model_number"),
            model_description=model_attr("description"),
        )
    }


@schema
class OpticalFiber(_PhotometryDevice, dj.Manual):
    definition = """
    # Optical fiber referenced by a FiberPhotometryTable (reusable model spec)
    optical_fiber_name: varchar(80)
    ---
    numerical_aperture=null: float
    core_diameter_in_um=null: float
    active_length_in_mm=null: float
    ferrule_name=null: varchar(255)
    ferrule_model=null: varchar(255)
    ferrule_diameter_in_mm=null: float
    manufacturer=null: varchar(255)
    model_number=null: varchar(255)
    model_description=null: varchar(2000)
    """

    _ref_columns = ("optical_fiber",)

    table_key_to_obj_attr = {
        "self": dict(
            optical_fiber_name="name",
            numerical_aperture=model_attr("numerical_aperture"),
            core_diameter_in_um=model_attr("core_diameter_in_um"),
            active_length_in_mm=model_attr("active_length_in_mm"),
            ferrule_name=model_attr("ferrule_name"),
            ferrule_model=model_attr("ferrule_model"),
            ferrule_diameter_in_mm=model_attr("ferrule_diameter_in_mm"),
            manufacturer=model_attr("manufacturer"),
            model_number=model_attr("model_number"),
            model_description=model_attr("description"),
        )
    }


# --- FiberPhotometryConfig ---------------------------------------------------

# FiberPhotometryTable columns this table models; anything else is warned about.
_MODELED_CONFIG_COLUMNS = frozenset(
    {
        "location",
        "excitation_wavelength_in_nm",
        "emission_wavelength_in_nm",
        "indicator",
        "optical_fiber",
        "excitation_source",
        "photodetector",
        "dichroic_mirror",
        "emission_filter",
        "excitation_filter",
        "notes",
        "coordinates",
    }
)

# Config columns fed from the referenced OpticalFiber's FiberInsertion (session
# specific, all nullable). Maps DataJoint column -> FiberInsertion attribute.
_INSERTION_MAP = {
    "hemisphere": "hemisphere",
    "ap_location": "insertion_position_ap_in_mm",
    "ml_location": "insertion_position_ml_in_mm",
    "dv_location": "insertion_position_dv_in_mm",
    "insertion_depth": "depth_in_mm",
    "position_reference": "position_reference",
    "pitch": "insertion_angle_pitch_in_deg",
    "roll": "insertion_angle_roll_in_deg",
    "yaw": "insertion_angle_yaw_in_deg",
}

# Populated-but-unmodeled object attributes surfaced via a warning (metadata
# deferred to a follow-up; not stored yet).
_UNMODELED_ATTRS = {
    "excitation_source": (
        "power_in_W",
        "intensity_in_W_per_m2",
        "exposure_time_in_s",
        "pulse_rate_in_Hz",
        "peak_power_in_W",
        "peak_pulse_energy_in_J",
    ),
    "indicator": ("viral_vector_injection",),
}


@schema
class FiberPhotometryConfig(SpyglassIngestion, dj.Manual):
    definition = """
    # Per-fiber fiber-photometry configuration (one FiberPhotometryTable row)
    -> Session
    fiber_photometry_name: varchar(64)  # FiberPhotometry lab-meta container name
    fiber_id: int                       # FiberPhotometryTable row `id` (may be non-consecutive)
    ---
    -> Indicator
    -> ExcitationSource
    -> Photodetector
    -> [nullable] DichroicMirror
    -> [nullable] OpticalFilter.proj(emission_filter_name='optical_filter_name')
    -> [nullable] OpticalFilter.proj(excitation_filter_name='optical_filter_name')
    -> OpticalFiber
    location: varchar(255)              # the FiberPhotometryTable row site, e.g. 'DLS'
    optical_fiber_description=null: varchar(255)  # per-channel fiber description
    hemisphere=null: enum('left', 'right')
    ap_location=null: float
    ml_location=null: float
    dv_location=null: float
    insertion_depth=null: float
    position_reference=null: varchar(255)
    pitch=null: float
    roll=null: float
    yaw=null: float
    excitation_wavelength_in_nm: float
    emission_wavelength_in_nm: float
    notes=null: varchar(2000)
    coordinates=null: blob             # optional 3-vector for multi-fiber arrays
    """

    # Session-specific (like Raw / VirusInjection): idempotency is file-level, not
    # per-row, so leave _expected_duplicates False.
    _source_nwb_object_type = "FiberPhotometryTable"
    _extension_requirements = _EXT_REQ

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Emit one entry per FiberPhotometryTable row.

        Overrides the declarative path: needs the container name, per-row device
        FK resolution, optional refs, nested fiber-insertion fields, and the
        unmodeled-metadata warnings.
        """
        base_key = (base_key or {}).copy()
        table = nwb_obj
        container = getattr(table, "parent", None)
        fiber_photometry_name = (
            getattr(container, "name", None) or "fiber_photometry"
        )

        df = table.to_dataframe()
        self._warn_unmodeled_columns(set(df.columns))

        entries = []
        warned_attrs = set()
        # Read values by column name (df.index gives the row `id`); reading off
        # itertuples() namedtuples would risk pandas mangling a column name and
        # silently dropping a modeled column.
        for row_id, record in zip(df.index, df.to_dict("records")):
            fiber = record["optical_fiber"]
            insertion = getattr(fiber, "fiber_insertion", None)
            entry = dict(
                base_key,
                fiber_photometry_name=fiber_photometry_name,
                fiber_id=int(row_id),
                indicator_name=record["indicator"].name,
                excitation_source_name=record["excitation_source"].name,
                photodetector_name=record["photodetector"].name,
                optical_fiber_name=fiber.name,
                dichroic_mirror_name=_ref_name(record, "dichroic_mirror"),
                emission_filter_name=_ref_name(record, "emission_filter"),
                excitation_filter_name=_ref_name(record, "excitation_filter"),
                location=record["location"],
                optical_fiber_description=getattr(fiber, "description", None),
                excitation_wavelength_in_nm=float(
                    record["excitation_wavelength_in_nm"]
                ),
                emission_wavelength_in_nm=float(
                    record["emission_wavelength_in_nm"]
                ),
                notes=record.get("notes"),
                coordinates=record.get("coordinates"),
            )
            for col, attr in _INSERTION_MAP.items():
                entry[col] = (
                    getattr(insertion, attr, None)
                    if insertion is not None
                    else None
                )
            self._collect_unmodeled_attrs(record, warned_attrs)
            entries.append(entry)

        if warned_attrs:
            logger.warning(
                f"FiberPhotometryConfig: populated but unmodeled attribute(s) "
                f"{sorted(warned_attrs)} were not ingested (deferred metadata)."
            )
        return {self: entries}

    def _warn_unmodeled_columns(self, colnames):
        unmodeled = sorted(set(colnames) - _MODELED_CONFIG_COLUMNS)
        if unmodeled:
            logger.warning(
                f"FiberPhotometryConfig: ignoring unmodeled FiberPhotometryTable "
                f"column(s) {unmodeled} (not stored)."
            )

    @staticmethod
    def _collect_unmodeled_attrs(record, warned_attrs):
        for column, attrs in _UNMODELED_ATTRS.items():
            obj = record.get(column)
            if obj is None:
                continue
            for name in populated_attrs(obj, attrs):
                warned_attrs.add(f"{column}.{name}")


def _ref_name(record, column):
    """The ``name`` of an optional per-row object reference (absent column or
    null value both yield ``None``)."""
    obj = record.get(column)
    return getattr(obj, "name", None)


# --- FiberPhotometryResponseSeries -------------------------------------------


@schema
class FiberPhotometryResponseSeries(SpyglassIngestion, dj.Imported):
    definition = """
    # Reference to a FiberPhotometryResponseSeries; the trace stays in the NWB file
    -> Session
    response_series_object_id: varchar(40)  # NWB object id, for fetch_nwb()
    ---
    -> IntervalList                         # the series' valid (recorded) times
    name: varchar(80)
    description: varchar(2000)
    comments=null: varchar(2000)            # TimeSeries.comments (optional)
    num_samples: bigint                     # length of the recorded trace
    unit: varchar(16)
    """

    # The table FKs -> Session (not -> Nwbfile), so fetch_nwb() cannot resolve the
    # source file from the definition; point it at Nwbfile explicitly (as Raw does).
    _nwb_table = Nwbfile
    # Gate on both namespaces (like FiberPhotometryConfig): the .Fiber rows FK into
    # FiberPhotometryConfig, so if the ndx-ophys-devices schema is too old the config
    # rows are skipped and a fiber-photometry-only gate here would leave the .Fiber
    # insert with an unresolvable FK. Skip the whole file together instead.
    _extension_requirements = _EXT_REQ

    class Fiber(SpyglassIngestion, dj.Part):
        definition = """
        # One column of the response series -> the FiberPhotometryConfig row it records
        -> master
        region_index: int                   # 0-based position within the table region
        ---
        -> FiberPhotometryConfig
        """

    def make(self, key):
        """Ingest the file's response series (standard Imported entry point).

        ``populate_all_common`` drives ingestion through ``insert_from_nwbfile``,
        but ``dj.Imported`` tables are also expected to support ``populate()``;
        delegate to the same path (as ``Raw`` does) so both work.
        """
        self.insert_from_nwbfile(key["nwb_file_name"])

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """The FiberPhotometryResponseSeries objects to ingest.

        Reference-scoped like the device tables: a file with no
        ``FiberPhotometry`` container yields ``[]`` (a clean no-op), so a
        non-photometry file never ingests a stray response series.
        """
        return response_series(nwb_file)

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Emit the ``IntervalList`` + master + ``.Fiber`` rows for one series.

        Custom override (not the declarative path): the master row stores the
        object id + trace metadata and references an ``IntervalList`` of the
        series' valid (recorded) times (as ``Raw`` does), so the trace can be
        time-restricted against the rest of Spyglass. Each
        ``fiber_photometry_table_region`` entry becomes a ``.Fiber`` row; the
        region stores **positional** row indices into the ``FiberPhotometryTable``,
        translated to the table row ``id`` (the config ``fiber_id``, which may be
        non-consecutive). A series without a region inserts the master row only,
        with no ``.Fiber`` rows and a warning — still retrievable via
        ``fetch_nwb()``.
        """
        base_key = (base_key or {}).copy()
        series = nwb_obj
        object_id = series.object_id
        interval_list_name = f"{series.name} valid times"
        interval_entry = dict(
            base_key,
            interval_list_name=interval_list_name,
            valid_times=self._valid_times(series),
            pipeline="fiber_photometry",
        )
        master_row = dict(
            base_key,
            response_series_object_id=object_id,
            interval_list_name=interval_list_name,
            name=series.name,
            description=getattr(series, "description", None),
            comments=getattr(series, "comments", None),
            num_samples=int(series.data.shape[0]),
            unit=series.unit,
        )

        fiber_rows = []
        region = getattr(series, "fiber_photometry_table_region", None)
        if region is None:
            logger.warning(
                f"FiberPhotometryResponseSeries {series.name!r}: no "
                "fiber_photometry_table_region; storing the signal reference "
                "without a per-column fiber mapping (no .Fiber rows)."
            )
        else:
            table = region.table
            container = getattr(table, "parent", None)
            # Mirror FiberPhotometryConfig's container-name derivation so the
            # .Fiber -> config FK resolves (one container per file is the norm).
            fiber_photometry_name = (
                getattr(container, "name", None) or "fiber_photometry"
            )
            region_positions = list(region.data)
            # The region should reference one table row per trace column; a
            # mismatch means some columns can't be mapped to a fiber (or extra
            # rows go unused). Warn rather than guess.
            n_columns = series.data.shape[1] if series.data.ndim == 2 else 1
            if len(region_positions) != n_columns:
                logger.warning(
                    f"FiberPhotometryResponseSeries {series.name!r}: region "
                    f"maps {len(region_positions)} fiber(s) but the trace has "
                    f"{n_columns} column(s); some columns will be unlabeled."
                )
            # region.data holds positional indices; the DynamicTable row ids
            # (which may be non-consecutive) map a position to the config
            # fiber_id. Index ``table.id`` directly rather than materializing the
            # whole table via to_dataframe() just for the id ordering.
            row_ids = table.id
            for region_index, positional in enumerate(region_positions):
                position = int(positional)
                # A negative index would silently wrap to the wrong row; an
                # out-of-range one would mis-map. Fail loud and named instead.
                if not 0 <= position < len(row_ids):
                    raise ValueError(
                        f"FiberPhotometryResponseSeries {series.name!r}: region "
                        f"position {positional} is out of range for the "
                        f"{len(row_ids)}-row FiberPhotometryTable."
                    )
                fiber_rows.append(
                    dict(
                        base_key,
                        response_series_object_id=object_id,
                        region_index=region_index,
                        fiber_photometry_name=fiber_photometry_name,
                        fiber_id=int(row_ids[position]),
                    )
                )

        # IntervalList before the master (the master FKs it); .Fiber last and
        # always present (possibly empty) so the multi-object insert loop can
        # extend every key across series.
        return {
            IntervalList: [interval_entry],
            self: [master_row],
            self.Fiber: fiber_rows,
        }

    @staticmethod
    def _valid_times(series):
        """The series' recorded-time span as a single contiguous ``[start, end]``
        interval (photometry acquisition is continuous).

        A rate-based series computes its endpoints in O(1) from
        ``starting_time``/``rate`` (matching ``get_timestamps()[0]``/``[-1]``)
        rather than materializing the full time axis; an explicit-``timestamps``
        series uses its first/last timestamp.
        """
        if series.timestamps is not None:
            timestamps = np.asarray(series.timestamps)
            return np.array([[timestamps[0], timestamps[-1]]])
        start = series.starting_time
        end = start + (int(series.data.shape[0]) - 1) / series.rate
        return np.array([[start, end]])

    def nwb_object(self, key):
        """Return the ``FiberPhotometryResponseSeries`` NWB object for one row.

        Uses the full key (``response_series_object_id``): a photometry file has
        many response series, so a ``nwb_file_name``-only fetch would be
        ambiguous.
        """
        nwb_file_name = key["nwb_file_name"]
        object_id = (self & key).fetch1("response_series_object_id")
        nwbf = get_nwb_file(Nwbfile.get_abs_path(nwb_file_name))
        return nwbf.objects[object_id]

    def fetch1_dataframe(self):
        """Return the recorded trace as a time-indexed ``pandas.DataFrame``.

        The time axis comes from explicit ``timestamps`` when present, else from
        the series' ``rate``/``starting_time`` via ``get_timestamps()``. Columns
        are labeled from the ``.Fiber`` -> config mapping; a series ingested
        without a region (empty ``.Fiber``) falls back to ``f"{name}_col{i}"``.
        """
        import pandas as pd

        key = self.fetch1("KEY")  # enforce exactly one row
        record = (self & key).fetch_nwb()[0]
        series = record["response_series"]

        data = np.asarray(series.data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_cols = data.shape[1]

        # get_timestamps() computes starting_time + arange(n)/rate for a
        # rate-based series, but on pynwb 3.1.3 it raises on an explicit-
        # timestamps series (array-truthiness bug), so read series.timestamps
        # directly there.
        if series.timestamps is not None:
            index = np.asarray(series.timestamps)
        else:
            index = np.asarray(series.get_timestamps())

        columns = self._column_labels(key, series, n_cols)
        return pd.DataFrame(
            data, index=pd.Index(index, name="time"), columns=columns
        )

    def _column_labels(self, key, series, n_cols):
        """Deterministic per-column labels from the ``.Fiber`` -> config rows.

        ``f"{location or optical_fiber_name}_{excitation_wavelength}nm"``,
        disambiguated by ``fiber_id`` if two columns still collide. Columns with
        no ``.Fiber`` mapping (e.g. a series ingested without a region) fall back
        to ``f"{series.name}_col{i}"``.
        """
        from collections import Counter

        joined = (self.Fiber & key) * FiberPhotometryConfig
        rows = joined.fetch(
            "region_index",
            "location",
            "optical_fiber_name",
            "excitation_wavelength_in_nm",
            "fiber_id",
            as_dict=True,
        )
        base_by_index = {}
        for row in rows:
            site = row["location"] or row["optical_fiber_name"]
            base = f"{site}_{int(row['excitation_wavelength_in_nm'])}nm"
            base_by_index[row["region_index"]] = (base, row["fiber_id"])

        base_counts = Counter(base for base, _ in base_by_index.values())
        labels = []
        for i in range(n_cols):
            if i in base_by_index:
                base, fiber_id = base_by_index[i]
                labels.append(
                    base if base_counts[base] == 1 else f"{base}_id{fiber_id}"
                )
            else:
                labels.append(f"{series.name}_col{i}")
        return labels
