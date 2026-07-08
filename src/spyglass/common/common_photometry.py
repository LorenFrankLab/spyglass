"""Ingestion of ``ndx-fiber-photometry`` metadata into queryable DataJoint tables.

Six reusable device/indicator tables plus a per-fiber ``FiberPhotometryConfig``
capture the experimental setup (which fiber, indicator, excitation source,
detector, filters, wavelengths, insertion). Recorded fluorescence traces stay in
the NWB file; a signal-reference table and retrieval land in a follow-up.

The tables are photometry-reference-scoped: each device table's
``get_nwb_objects()`` collects only the objects a ``FiberPhotometryTable``
references (via :mod:`spyglass.common._photometry_nwb`), so a non-photometry file
is a clean no-op and device subtypes are handled without matching on class name.
Ingestion never imports ``ndx-fiber-photometry``; NWB types are matched by
class-name string and the object references resolve from the file-embedded spec.
"""

import datajoint as dj

from spyglass.common._photometry_nwb import (
    class_discriminator,
    model_attr,
    model_range,
    populated_attrs,
    referenced_devices,
)
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassIngestion

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

# Populated-but-unmodeled object attributes to surface via a warning (deferred
# metadata; see design doc "Schema coverage").
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
        for row in df.itertuples():
            fiber = row.optical_fiber
            insertion = getattr(fiber, "fiber_insertion", None)
            entry = dict(
                base_key,
                fiber_photometry_name=fiber_photometry_name,
                fiber_id=int(row.Index),
                indicator_name=row.indicator.name,
                excitation_source_name=row.excitation_source.name,
                photodetector_name=row.photodetector.name,
                optical_fiber_name=fiber.name,
                dichroic_mirror_name=_ref_name(row, "dichroic_mirror"),
                emission_filter_name=_ref_name(row, "emission_filter"),
                excitation_filter_name=_ref_name(row, "excitation_filter"),
                location=row.location,
                optical_fiber_description=getattr(fiber, "description", None),
                excitation_wavelength_in_nm=float(
                    row.excitation_wavelength_in_nm
                ),
                emission_wavelength_in_nm=float(row.emission_wavelength_in_nm),
                notes=getattr(row, "notes", None),
                coordinates=_optional_value(row, "coordinates"),
            )
            for col, attr in _INSERTION_MAP.items():
                entry[col] = (
                    getattr(insertion, attr, None)
                    if insertion is not None
                    else None
                )
            self._collect_unmodeled_attrs(row, warned_attrs)
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
    def _collect_unmodeled_attrs(row, warned_attrs):
        for column, attrs in _UNMODELED_ATTRS.items():
            obj = getattr(row, column, None)
            if obj is None:
                continue
            for name in populated_attrs(obj, attrs):
                warned_attrs.add(f"{column}.{name}")


def _ref_name(row, column):
    """The ``name`` of an optional per-row object reference, or ``None``."""
    obj = getattr(row, column, None)
    return getattr(obj, "name", None)


def _optional_value(row, column):
    """A per-row optional scalar/array value, or ``None`` if absent."""
    return getattr(row, column, None)
