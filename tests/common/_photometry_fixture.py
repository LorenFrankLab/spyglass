"""Builders for synthetic fiber-photometry NWB fixtures (NWB core 2.9.0).

These build ``ndx-fiber-photometry`` / ``ndx-ophys-devices`` objects onto a
minimal NWBFile so the photometry ingestion path can be exercised end to end.
Writing with the current pynwb (3.1.x) embeds NWB ``core`` 2.9.0, the supported
floor.

Every device name takes a ``suffix`` so each fixture shape owns an independent
device catalog: the reusable device tables (``_expected_duplicates=True``) are not
deleted with a session, so two shapes sharing a name with different specs would
otherwise trip divergence validation. The cross-session test deliberately passes
the *same* suffix to two files to exercise catalog reuse.

The ``ndx_*`` packages are imported *inside* the build functions, never at module
scope, so importing this module does not pull ``ndx_fiber_photometry`` into
``sys.modules`` — the package-absent import-safety guarantee relies on that.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

# Base device names; a per-fixture suffix is appended (see module docstring).
INDICATOR_NAME = "dLight38"
EXC_SOURCE_PULSED = "ExcSrc_pulsed"
EXC_SOURCE_CONT = "ExcSrc_cont"
PHOTODETECTOR_NAME = "Det01"
DICHROIC_NAME = "Dichroic01"
BAND_FILTER_NAME = "BandFilter525"
EDGE_FILTER_NAME = "EdgeFilter490"
BASE_FILTER_NAME = "BaseFilter"
FIBER_MODEL_NAME = "DoricFlatFiber400um"


def _new_nwb(identifier: str) -> NWBFile:
    return NWBFile(
        session_description="fiber photometry fixture",
        identifier=identifier,
        session_start_time=datetime(
            2025, 10, 2, 11, 48, tzinfo=ZoneInfo("America/Los_Angeles")
        ),
        subject=Subject(subject_id="400", species="Rattus norvegicus"),
        experimenter=["Test, Experimenter"],
        lab="Test Lab",
        institution="Test Institution",
    )


def _fiber_insertion(od, complete: bool):
    """A FiberInsertion; ``complete`` fills the angle fields, else leaves them null."""
    kwargs = dict(
        name="fiber_insertion",
        hemisphere="left",
        insertion_position_ap_in_mm=0.6,
        insertion_position_ml_in_mm=3.8,
        insertion_position_dv_in_mm=-3.8,
        depth_in_mm=3.8,
        position_reference="bregma",
    )
    if complete:
        kwargs.update(
            insertion_angle_pitch_in_deg=1.0,
            insertion_angle_roll_in_deg=2.0,
            insertion_angle_yaw_in_deg=3.0,
        )
    return od.FiberInsertion(**kwargs)


def _fiber_model(od, kind, suffix):
    """OpticalFiberModel: 'full' (all specs), 'sparse' (only NA), or None."""
    if kind is None:
        return None
    if kind == "sparse":
        return od.OpticalFiberModel(
            name=FIBER_MODEL_NAME + suffix,
            manufacturer="Doric",
            numerical_aperture=0.48,
        )
    return od.OpticalFiberModel(
        name=FIBER_MODEL_NAME + suffix,
        manufacturer="Doric",
        model_number="FF-400",
        description="flat-tip 400um fiber",
        numerical_aperture=0.48,
        core_diameter_in_um=400.0,
        active_length_in_mm=1.5,
        ferrule_name="LC ferrule",
        ferrule_model="FER-LC",
        ferrule_diameter_in_mm=1.25,
    )


def _excitation_source_model(od, suffix):
    return od.ExcitationSourceModel(
        name="LEDmodel" + suffix,
        manufacturer="Doric",
        source_type="LED",
        excitation_mode="one-photon",
        wavelength_range_in_nm=[470.0, 490.0],
    )


def _photodetector_model(od, suffix):
    return od.PhotodetectorModel(
        name="PMTmodel" + suffix,
        manufacturer="Doric",
        detector_type="PMT",
        gain=1.5,
        gain_unit="V/A",
        wavelength_range_in_nm=[400.0, 700.0],
    )


def _fiber_photometry(fp, table, indicator):
    return fp.FiberPhotometry(
        name="fiber_photometry",
        fiber_photometry_table=table,
        fiber_photometry_indicators=fp.FiberPhotometryIndicators(
            indicators=[indicator]
        ),
    )


def build_full(nwb: NWBFile, suffix: str = "_full") -> NWBFile:
    """Comprehensive fixture: 2 fibers (shared model), pulsed + continuous
    sources, band/edge/base filters, a dichroic, notes + coordinates columns, and
    a response series with a single-row region."""
    import ndx_fiber_photometry as fp
    import ndx_ophys_devices as od
    from hdmf.common import DynamicTableRegion

    fiber_model = _fiber_model(od, "full", suffix)
    exc_model = _excitation_source_model(od, suffix)
    det_model = _photodetector_model(od, suffix)
    dichroic_model = od.DichroicMirrorModel(
        name="DichroicModel" + suffix,
        manufacturer="Chroma",
        cut_on_wavelength_in_nm=505.0,
        cut_off_wavelength_in_nm=750.0,
        reflection_band_in_nm=[450.0, 500.0],
        transmission_band_in_nm=[510.0, 700.0],
        angle_of_incidence_in_degrees=45.0,
    )
    band_model = od.BandOpticalFilterModel(
        name="BandModel" + suffix,
        manufacturer="Semrock",
        filter_type="Bandpass",
        center_wavelength_in_nm=525.0,
        bandwidth_in_nm=50.0,
    )
    edge_model = od.EdgeOpticalFilterModel(
        name="EdgeModel" + suffix,
        manufacturer="Semrock",
        filter_type="Longpass",
        cut_wavelength_in_nm=490.0,
        slope_in_percent_cut_wavelength=1.0,
        slope_starting_transmission_in_percent=10.0,
        slope_ending_transmission_in_percent=80.0,
    )
    base_model = od.OpticalFilterModel(
        name="BaseFilterModel" + suffix,
        manufacturer="Semrock",
        filter_type="Bandpass",
    )
    for m in (
        fiber_model,
        exc_model,
        det_model,
        dichroic_model,
        band_model,
        edge_model,
        base_model,
    ):
        nwb.add_device_model(m)

    fiber_dls = od.OpticalFiber(
        name="OpticalFiber_DLS" + suffix,
        model=fiber_model,
        description="400um fiber in DLS",
        fiber_insertion=_fiber_insertion(od, complete=True),
    )
    fiber_dms = od.OpticalFiber(
        name="OpticalFiber_DMS" + suffix,
        model=fiber_model,  # shared model
        description="400um fiber in DMS",
        fiber_insertion=_fiber_insertion(od, complete=False),
    )
    exc_pulsed = od.PulsedExcitationSource(
        name=EXC_SOURCE_PULSED + suffix,
        model=exc_model,
        description="470nm pulsed",
        pulse_rate_in_Hz=100.0,
    )
    exc_cont = od.ExcitationSource(
        name=EXC_SOURCE_CONT + suffix,
        model=exc_model,
        description="490nm continuous",
    )
    det = od.Photodetector(
        name=PHOTODETECTOR_NAME + suffix, model=det_model, description="DLS"
    )
    dichroic = od.DichroicMirror(
        name=DICHROIC_NAME + suffix, model=dichroic_model
    )
    band_filter = od.BandOpticalFilter(
        name=BAND_FILTER_NAME + suffix, model=band_model
    )
    edge_filter = od.EdgeOpticalFilter(
        name=EDGE_FILTER_NAME + suffix, model=edge_model
    )
    base_filter = od.OpticalFilter(
        name=BASE_FILTER_NAME + suffix, model=base_model
    )
    for d in (
        fiber_dls,
        fiber_dms,
        exc_pulsed,
        exc_cont,
        det,
        dichroic,
        band_filter,
        edge_filter,
        base_filter,
    ):
        nwb.add_device(d)

    indicator = od.Indicator(
        name=INDICATOR_NAME + suffix,
        label="dLight3.8",
        description="dopamine indicator",
    )

    table = fp.FiberPhotometryTable(
        name="fiber_photometry_table", description="per-fiber config"
    )
    table.add_row(
        location="DLS",
        excitation_wavelength_in_nm=470.0,
        emission_wavelength_in_nm=525.0,
        indicator=indicator,
        optical_fiber=fiber_dls,
        excitation_source=exc_pulsed,
        photodetector=det,
        dichroic_mirror=dichroic,
        emission_filter=band_filter,
        excitation_filter=edge_filter,
        notes="row for DLS",
        coordinates=np.array([0.6, 3.8, -3.8]),
    )
    table.add_row(
        location="DMS",
        excitation_wavelength_in_nm=490.0,
        emission_wavelength_in_nm=525.0,
        indicator=indicator,
        optical_fiber=fiber_dms,
        excitation_source=exc_cont,
        photodetector=det,
        dichroic_mirror=dichroic,
        emission_filter=base_filter,
        excitation_filter=edge_filter,
        notes="row for DMS",
        coordinates=np.array([0.6, 2.0, -3.8]),
    )
    nwb.add_lab_meta_data(_fiber_photometry(fp, table, indicator))

    region = DynamicTableRegion(
        name="fiber_photometry_table_region",
        data=[0],
        description="row 0",
        table=table,
    )
    nwb.add_acquisition(
        fp.FiberPhotometryResponseSeries(
            name="FPResponseSeries_DLS_470nm",
            data=np.arange(1000, dtype="float64"),
            unit="V",
            rate=6024.096,
            starting_time=0.083,
            fiber_photometry_table_region=region,
        )
    )
    return nwb


def build_minimal(
    nwb: NWBFile,
    suffix: str = "_min",
    *,
    fiber_model_kind="full",
    complete_insertion=True,
    unmodeled_column=False,
    excitation_power=False,
    row_id=None,
) -> NWBFile:
    """A single-fiber photometry file, tunable for the null/gate/warn cases.

    ``fiber_model_kind``: 'full' | 'sparse' | None (model-less fiber).
    ``unmodeled_column``: add a ``commanded_voltage_series`` column (warn path).
    ``excitation_power``: populate ``ExcitationSource.power_in_W`` (warn path).
    ``row_id``: explicit (possibly non-consecutive) FiberPhotometryTable row id.
    """
    import ndx_fiber_photometry as fp
    import ndx_ophys_devices as od
    from hdmf.common import DynamicTableRegion

    fiber_model = _fiber_model(od, fiber_model_kind, suffix)
    exc_model = od.ExcitationSourceModel(
        name="LEDmodel" + suffix,
        manufacturer="Doric",
        source_type="LED",
        excitation_mode="one-photon",
    )
    det_model = od.PhotodetectorModel(
        name="PMTmodel" + suffix, manufacturer="Doric", detector_type="PMT"
    )
    for m in (fiber_model, exc_model, det_model):
        if m is not None:
            nwb.add_device_model(m)

    fiber = od.OpticalFiber(
        name="OpticalFiber_DLS" + suffix,
        model=fiber_model,
        description="400um fiber in DLS",
        fiber_insertion=_fiber_insertion(od, complete=complete_insertion),
    )
    exc_kwargs = dict(
        name=EXC_SOURCE_CONT + suffix, model=exc_model, description="490nm"
    )
    if excitation_power:
        exc_kwargs["power_in_W"] = 0.001
    exc = od.ExcitationSource(**exc_kwargs)
    det = od.Photodetector(
        name=PHOTODETECTOR_NAME + suffix, model=det_model, description="DLS"
    )
    for d in (fiber, exc, det):
        nwb.add_device(d)

    indicator = od.Indicator(
        name=INDICATOR_NAME + suffix,
        label="dLight3.8",
        description="dopamine indicator",
    )

    table = fp.FiberPhotometryTable(
        name="fiber_photometry_table", description="per-fiber config"
    )
    row = dict(
        location="DLS",
        excitation_wavelength_in_nm=490.0,
        emission_wavelength_in_nm=525.0,
        indicator=indicator,
        optical_fiber=fiber,
        excitation_source=exc,
        photodetector=det,
    )
    if unmodeled_column:
        cmd = fp.CommandedVoltageSeries(
            name="commanded_voltage",
            data=np.zeros(10, dtype="float64"),
            unit="volts",
            rate=100.0,
            starting_time=0.0,
        )
        nwb.add_acquisition(cmd)
        row["commanded_voltage_series"] = cmd
    if row_id is not None:
        row["id"] = row_id
    table.add_row(**row)
    nwb.add_lab_meta_data(_fiber_photometry(fp, table, indicator))

    region = DynamicTableRegion(
        name="fiber_photometry_table_region",
        data=[0],
        description="row 0",
        table=table,
    )
    nwb.add_acquisition(
        fp.FiberPhotometryResponseSeries(
            name="FPResponseSeries_DLS_490nm",
            data=np.arange(500, dtype="float64"),
            unit="V",
            rate=6024.096,
            starting_time=0.0,
            fiber_photometry_table_region=region,
        )
    )
    return nwb


def build_pure_devices(nwb: NWBFile, suffix: str = "_pure") -> NWBFile:
    """ndx-ophys-devices ``OpticalFiber`` + ``Photodetector`` but NO
    ``FiberPhotometry`` container — a non-photometry file the device tables must
    no-op on."""
    import ndx_ophys_devices as od

    fiber_model = _fiber_model(od, "full", suffix)
    nwb.add_device_model(fiber_model)
    fiber = od.OpticalFiber(
        name="OpticalFiber_opto" + suffix,
        model=fiber_model,
        description="CA1",
        fiber_insertion=_fiber_insertion(od, complete=True),
    )
    nwb.add_device(fiber)
    det_model = od.PhotodetectorModel(
        name="PMTmodel_opto" + suffix, manufacturer="Doric", detector_type="PMT"
    )
    nwb.add_device_model(det_model)
    nwb.add_device(od.Photodetector(name="Det_opto" + suffix, model=det_model))
    return nwb


def build_mixed_modality(nwb: NWBFile, suffix: str = "_mixed") -> NWBFile:
    """A photometry fiber and a *separate* non-photometry fiber that share one
    ``OpticalFiberModel`` — the gate must keep the shared model."""
    import ndx_fiber_photometry as fp
    import ndx_ophys_devices as od

    shared_model = _fiber_model(od, "full", suffix)  # both fibers use this
    exc_model = od.ExcitationSourceModel(
        name="LEDmodel" + suffix,
        manufacturer="Doric",
        source_type="LED",
        excitation_mode="one-photon",
    )
    det_model = od.PhotodetectorModel(
        name="PMTmodel" + suffix, manufacturer="Doric", detector_type="PMT"
    )
    for m in (shared_model, exc_model, det_model):
        nwb.add_device_model(m)

    photo_fiber = od.OpticalFiber(
        name="OpticalFiber_photo" + suffix,
        model=shared_model,
        description="photometry fiber",
        fiber_insertion=_fiber_insertion(od, complete=False),
    )
    opto_fiber = od.OpticalFiber(
        name="OpticalFiber_opto" + suffix,
        model=shared_model,  # SHARED with the photometry fiber
        description="optogenetics fiber",
        fiber_insertion=_fiber_insertion(od, complete=True),
    )
    exc = od.ExcitationSource(
        name=EXC_SOURCE_CONT + suffix, model=exc_model, description="490"
    )
    det = od.Photodetector(name=PHOTODETECTOR_NAME + suffix, model=det_model)
    for d in (photo_fiber, opto_fiber, exc, det):
        nwb.add_device(d)

    indicator = od.Indicator(name=INDICATOR_NAME + suffix, label="dLight3.8")
    table = fp.FiberPhotometryTable(
        name="fiber_photometry_table", description="per-fiber config"
    )
    table.add_row(
        location="DLS",
        excitation_wavelength_in_nm=490.0,
        emission_wavelength_in_nm=525.0,
        indicator=indicator,
        optical_fiber=photo_fiber,  # only the photometry fiber is referenced
        excitation_source=exc,
        photodetector=det,
    )
    nwb.add_lab_meta_data(_fiber_photometry(fp, table, indicator))
    return nwb


def build_two_containers(nwb: NWBFile, suffix: str = "_2c") -> NWBFile:
    """Two ``FiberPhotometry`` lab-meta containers, each with a one-row table
    whose row ``id`` is 0 — only the config PK's ``fiber_photometry_name``
    disambiguates the two ``fiber_id=0`` rows."""
    import ndx_fiber_photometry as fp
    import ndx_ophys_devices as od

    fiber_model = _fiber_model(od, "full", suffix)
    exc_model = od.ExcitationSourceModel(
        name="LEDmodel" + suffix,
        manufacturer="Doric",
        source_type="LED",
        excitation_mode="one-photon",
    )
    det_model = od.PhotodetectorModel(
        name="PMTmodel" + suffix, manufacturer="Doric", detector_type="PMT"
    )
    for m in (fiber_model, exc_model, det_model):
        nwb.add_device_model(m)
    exc = od.ExcitationSource(name=EXC_SOURCE_CONT + suffix, model=exc_model)
    det = od.Photodetector(name=PHOTODETECTOR_NAME + suffix, model=det_model)
    for d in (exc, det):
        nwb.add_device(d)

    for site in ("A", "B"):
        indicator = od.Indicator(
            name=f"{INDICATOR_NAME}{suffix}_{site}", label="dLight3.8"
        )
        fiber = od.OpticalFiber(
            name=f"OpticalFiber_{site}{suffix}",
            model=fiber_model,
            description=site,
            fiber_insertion=_fiber_insertion(od, complete=False),
        )
        nwb.add_device(fiber)
        table = fp.FiberPhotometryTable(
            name=f"fpt_{site}", description="per-fiber config"
        )
        table.add_row(  # row id defaults to 0 in each independent table
            location=site,
            excitation_wavelength_in_nm=490.0,
            emission_wavelength_in_nm=525.0,
            indicator=indicator,
            optical_fiber=fiber,
            excitation_source=exc,
            photodetector=det,
        )
        nwb.add_lab_meta_data(
            fp.FiberPhotometry(
                name=f"fiber_photometry_{site}",
                fiber_photometry_table=table,
                fiber_photometry_indicators=fp.FiberPhotometryIndicators(
                    indicators=[indicator]
                ),
            )
        )
    return nwb


def write(path, builder, identifier=None) -> str:
    """Build via ``builder(nwb)`` and write to ``path``; returns the file name."""
    path = Path(path)
    nwb = _new_nwb(identifier or path.stem)
    builder(nwb)
    with NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    return path.name
