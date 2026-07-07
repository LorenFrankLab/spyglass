"""VERIFIED DESIGN-SPIKE PROTOTYPE — minimal ndx-fiber-photometry NWB fixture.

Reference artifact for the fiber-photometry ingestion spec
(2026-07-07-fiber-photometry-ingestion-design.md). NOT production code — the
implementation plan turns this into a proper `tests/` fixture builder.

Verified 2026-07-07 to write + read round-trip on pynwb 3.1.3 (Spyglass's
dependency floor), embedding NWB core 2.9.0 — i.e. no pynwb/hdmf bump needed.
The file also reads with `ndx_fiber_photometry` UNINSTALLED (class-name match +
object-ref resolution intact), which is the no-import ingestion contract.

Env used for verification:
    uv venv --python 3.12 env
    uv pip install --python ./env/bin/python "pynwb==3.1.3" "ndx-fiber-photometry==0.2.3"
    # -> pynwb 3.1.3, hdmf 4.3.1, ndx-ophys-devices 0.3.1

Key gotchas discovered:
  - DeviceModel / NWBFile.add_device_model exist in pynwb 3.1.3 (core 2.9.0),
    so the ndx-ophys-devices model/instance pattern does NOT require core 2.10.0.
  - DynamicTableRegion is imported from hdmf.common.
  - FiberPhotometryTable.add_row takes the object-reference columns directly
    (indicator=<Indicator>, optical_fiber=<OpticalFiber>, ...).
"""
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
import ndx_ophys_devices as od
import ndx_fiber_photometry as fp
from hdmf.common import DynamicTableRegion


def build(path="fp_core29.nwb"):
    nwb = NWBFile(
        session_description="fp fixture",
        identifier="fp-fixture",
        session_start_time=datetime(
            2025, 10, 2, 11, 48, tzinfo=ZoneInfo("America/Los_Angeles")
        ),
        subject=Subject(subject_id="400", species="Rattus norvegicus"),
    )

    # device models + instances (ndx-ophys-devices model/instance pattern)
    of_model = od.OpticalFiberModel(
        name="Fiber400", manufacturer="Doric",
        numerical_aperture=0.48, core_diameter_in_um=400.0,
    )
    es_model = od.ExcitationSourceModel(
        name="LED490", manufacturer="Doric",
        source_type="LED", excitation_mode="one-photon",
    )
    pd_model = od.PhotodetectorModel(
        name="PMT1", manufacturer="Doric", detector_type="Photodiode",
    )
    for m in (of_model, es_model, pd_model):
        nwb.add_device_model(m)

    fiber = od.OpticalFiber(
        name="OpticalFiber_DLS", model=of_model, description="400um fiber in DLS",
        fiber_insertion=od.FiberInsertion(
            name="fiber_insertion", hemisphere="left",
            insertion_position_ap_in_mm=0.6,
            insertion_position_ml_in_mm=3.8,
            insertion_position_dv_in_mm=-3.8,
            position_reference="bregma",
        ),
    )
    exc = od.ExcitationSource(name="ExcSrc01", model=es_model, description="490nm")
    det = od.Photodetector(name="Det01", model=pd_model, description="DLS")
    for d in (fiber, exc, det):
        nwb.add_device(d)

    indicator = od.Indicator(
        name="dLight38", label="dLight3.8", description="dopamine indicator"
    )

    table = fp.FiberPhotometryTable(
        name="fiber_photometry_table", description="per-fiber config"
    )
    table.add_row(
        location="DLS", excitation_wavelength_in_nm=490.0,
        emission_wavelength_in_nm=525.0, indicator=indicator,
        optical_fiber=fiber, excitation_source=exc, photodetector=det,
    )

    nwb.add_lab_meta_data(
        fp.FiberPhotometry(
            name="fiber_photometry",
            fiber_photometry_table=table,
            fiber_photometry_indicators=fp.FiberPhotometryIndicators(
                indicators=[indicator]
            ),
        )
    )

    region = DynamicTableRegion(
        name="fiber_photometry_table_region", data=[0],
        description="row 0", table=table,
    )
    nwb.add_acquisition(
        fp.FiberPhotometryResponseSeries(
            name="FPResponseSeries_DLS_490nm",
            data=np.arange(1000, dtype="float64"), unit="V",
            rate=6024.096, starting_time=0.083,
            fiber_photometry_table_region=region,
        )
    )

    with NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    return path


if __name__ == "__main__":
    build()
