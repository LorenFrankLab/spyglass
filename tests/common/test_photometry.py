"""Validation slice for fiber-photometry metadata ingestion.

Covers device + config ingestion, the reference-scoped no-op, extension-version
gating, re-ingestion / cross-session catalog reuse, null fiber metadata, the
optogenetics fiber-table gate (including the shared-model over-prune guard),
subtype/optional round-trip, unmodeled-metadata warnings, and the
package-absent import-safety guarantee.
"""

import numpy as np
import pytest

from tests.common import _photometry_fixture as fx


class _WarnRecorder:
    """Stand-in ``logger`` that records ``warning`` messages and no-ops the rest."""

    def __init__(self):
        self.messages = []

    def warning(self, msg, *args, **kwargs):
        self.messages.append(str(msg))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_fixture_embeds_core_2_9_0(raw_dir, common):
    """The synthetic fixture must embed NWB core 2.9.0 (the supported floor)."""
    from spyglass.utils.nwb_hash import get_file_namespaces

    path = raw_dir / "mock_photometry_coreversion.nwb"
    fx.write(path, fx.build_minimal, identifier="coreversion")
    namespaces = get_file_namespaces(str(path))
    assert namespaces.get("core") == "2.9.0"
    assert "ndx-fiber-photometry" in namespaces


def test_device_catalog_no_blob_columns(common):
    """The `_expected_duplicates=True` device tables must hold no blob/json
    column: re-ingest divergence validation compares with `!=`, which raises
    "truth value ambiguous" on an array. Locks the duplicate-safety invariant.
    """
    device_tables = [
        common.Indicator,
        common.ExcitationSource,
        common.Photodetector,
        common.DichroicMirror,
        common.OpticalFilter,
        common.OpticalFiber,
    ]
    for table in device_tables:
        assert table._expected_duplicates is True
        for name, attr in table().heading.attributes.items():
            sql_type = attr.type.lower()
            assert "blob" not in sql_type and "json" not in sql_type, (
                f"{table.__name__}.{name} is {attr.type}; array/blob columns "
                "are unsafe in an `_expected_duplicates` table"
            )


@pytest.mark.slow
def test_device_and_config_ingest(photometry_full, common):
    key = photometry_full

    # both config rows present
    config = common.FiberPhotometryConfig & key
    assert len(config) == 2

    # device catalog rows exist
    assert len(common.Indicator & {"indicator_name": "dLight38_full"}) == 1
    assert (
        len(
            common.OpticalFiber
            & {"optical_fiber_name": "OpticalFiber_DLS_full"}
        )
        == 1
    )
    assert len(common.Photodetector & {"photodetector_name": "Det01_full"}) == 1

    # DLS row: device FKs resolve and session-local fiber fields are stored
    dls = (config & {"fiber_id": 0}).fetch1()
    assert dls["indicator_name"] == "dLight38_full"
    assert dls["excitation_source_name"] == "ExcSrc_pulsed_full"
    assert dls["photodetector_name"] == "Det01_full"
    assert dls["optical_fiber_name"] == "OpticalFiber_DLS_full"
    assert dls["optical_fiber_description"] == "400um fiber in DLS"
    assert dls["insertion_depth"] == pytest.approx(3.8, abs=1e-4)
    assert dls["position_reference"] == "bregma"
    assert dls["ap_location"] == pytest.approx(0.6, abs=1e-4)
    assert dls["pitch"] == pytest.approx(1.0, abs=1e-4)  # complete insertion

    # DMS row: incomplete insertion -> pitch/roll/yaw stored null
    dms = (config & {"fiber_id": 1}).fetch1()
    assert dms["optical_fiber_name"] == "OpticalFiber_DMS_full"
    assert dms["pitch"] is None
    assert dms["roll"] is None
    assert dms["yaw"] is None
    assert dms["ap_location"] == pytest.approx(
        0.6, abs=1e-4
    )  # position still set


@pytest.mark.slow
def test_location_from_row(photometry_full, common):
    """``location`` is the FiberPhotometryTable row's site, not the fiber desc."""
    config = common.FiberPhotometryConfig & photometry_full
    assert (config & {"fiber_id": 0}).fetch1("location") == "DLS"
    assert (config & {"fiber_id": 1}).fetch1("location") == "DMS"


@pytest.mark.slow
def test_non_consecutive_fiber_id(insert_photometry, common):
    """``fiber_id`` is the FiberPhotometryTable row ``id`` (which may be
    non-consecutive), not a positional counter — the invariant the response-series
    region->config mapping relies on."""
    key, result = insert_photometry(
        "mock_photometry_nonconsec.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_nonconsec", row_id=7),
    )
    assert not result
    assert (common.FiberPhotometryConfig & key).fetch1("fiber_id") == 7


@pytest.mark.slow
def test_multi_container_pk_disambiguation(insert_photometry, common):
    """Two FiberPhotometry containers, each with a row `id` 0, ingest as two
    distinct config rows — the PK's `fiber_photometry_name` disambiguates them
    (without it the two `fiber_id=0` rows would collide)."""
    key, result = insert_photometry(
        "mock_photometry_2c.nwb", fx.build_two_containers
    )
    assert not result
    config = common.FiberPhotometryConfig & key
    assert len(config) == 2
    assert set(config.fetch("fiber_photometry_name")) == {
        "fiber_photometry_A",
        "fiber_photometry_B",
    }
    assert set(config.fetch("fiber_id")) == {0}  # both are row id 0


@pytest.mark.slow
def test_subtype_and_optional_roundtrip(photometry_full, common):
    # excitation source subtype -> source_class
    src = common.ExcitationSource
    assert (src & {"excitation_source_name": "ExcSrc_pulsed_full"}).fetch1(
        "source_class"
    ) == "pulsed"
    cont = (src & {"excitation_source_name": "ExcSrc_cont_full"}).fetch1()
    assert cont["source_class"] == "continuous"
    assert cont["wavelength_min_nm"] == pytest.approx(470.0, abs=1e-3)
    assert cont["wavelength_max_nm"] == pytest.approx(490.0, abs=1e-3)

    # filter subtypes -> filter_class + model specs
    filt = common.OpticalFilter
    band = (filt & {"optical_filter_name": "BandFilter525_full"}).fetch1()
    assert band["filter_class"] == "band"
    assert band["center_wavelength_in_nm"] == pytest.approx(525.0, abs=1e-3)
    assert band["bandwidth_in_nm"] == pytest.approx(50.0, abs=1e-3)
    edge = (filt & {"optical_filter_name": "EdgeFilter490_full"}).fetch1()
    assert edge["filter_class"] == "edge"
    assert edge["cut_wavelength_in_nm"] == pytest.approx(490.0, abs=1e-3)
    assert edge["slope_ending_transmission_in_percent"] == pytest.approx(
        80.0, abs=1e-3
    )
    assert (filt & {"optical_filter_name": "BaseFilter_full"}).fetch1(
        "filter_class"
    ) == "base"

    # dichroic mirror model specs (incl. [2]-vector -> min/max pairs)
    dm = (
        common.DichroicMirror & {"dichroic_mirror_name": "Dichroic01_full"}
    ).fetch1()
    assert dm["reflection_band_min_nm"] == pytest.approx(450.0, abs=1e-3)
    assert dm["reflection_band_max_nm"] == pytest.approx(500.0, abs=1e-3)
    assert dm["transmission_band_min_nm"] == pytest.approx(510.0, abs=1e-3)
    assert dm["angle_of_incidence_in_degrees"] == pytest.approx(45.0, abs=1e-3)

    # optical fiber model specs
    fiber = (
        common.OpticalFiber & {"optical_fiber_name": "OpticalFiber_DLS_full"}
    ).fetch1()
    assert fiber["numerical_aperture"] == pytest.approx(0.48, abs=1e-4)
    assert fiber["core_diameter_in_um"] == pytest.approx(400.0, abs=1e-2)
    assert fiber["ferrule_name"] == "LC ferrule"
    assert fiber["ferrule_model"] == "FER-LC"
    # core DeviceModel identity fields (shared by all 5 model-backed tables)
    assert fiber["manufacturer"] == "Doric"
    assert fiber["model_number"] == "FF-400"
    assert fiber["model_description"] == "flat-tip 400um fiber"

    # photodetector model specs
    det = (common.Photodetector & {"photodetector_name": "Det01_full"}).fetch1()
    assert det["detector_type"] == "PMT"
    assert det["gain"] == pytest.approx(1.5, abs=1e-4)
    assert det["gain_unit"] == "V/A"

    # config optional refs + scalars
    dls = (
        common.FiberPhotometryConfig & photometry_full & {"fiber_id": 0}
    ).fetch1()
    assert dls["dichroic_mirror_name"] == "Dichroic01_full"
    assert dls["emission_filter_name"] == "BandFilter525_full"
    assert dls["excitation_filter_name"] == "EdgeFilter490_full"
    assert dls["notes"] == "row for DLS"
    assert np.allclose(dls["coordinates"], [0.6, 3.8, -3.8])


@pytest.mark.slow
def test_ref_scoped_no_op(insert_photometry, common):
    """A file with ndx-ophys-devices objects but no FiberPhotometry container
    produces no photometry rows and no exception."""
    key, result = insert_photometry(
        "mock_photometry_pure.nwb", fx.build_pure_devices
    )

    assert not result  # no InsertError keys
    assert len(common.FiberPhotometryConfig & key) == 0
    # the generic OpticalFiber was NOT ingested into the photometry catalog
    assert (
        len(
            common.OpticalFiber
            & {"optical_fiber_name": "OpticalFiber_opto_pure"}
        )
        == 0
    )
    assert len(common.Indicator & {"indicator_name": "dLight38_pure"}) == 0


@pytest.mark.slow
def test_below_min_version_warns(insert_photometry, common, monkeypatch):
    """Photometry objects present but ndx-fiber-photometry below min -> warn, no rows."""
    from spyglass.utils.mixins import ingestion

    rec = _WarnRecorder()
    monkeypatch.setattr(ingestion, "logger", rec)
    monkeypatch.setattr(
        ingestion,
        "get_file_namespaces",
        lambda path: {
            "ndx-fiber-photometry": "0.1.0",  # below the 0.2.3 minimum
            "ndx-ophys-devices": "0.3.1",
        },
    )

    key, _ = insert_photometry(
        "mock_photometry_belowmin.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_belowmin"),
    )

    assert len(common.FiberPhotometryConfig & key) == 0
    assert (
        len(
            common.OpticalFiber
            & {"optical_fiber_name": "OpticalFiber_DLS_belowmin"}
        )
        == 0
    )
    assert any("ndx-fiber-photometry" in m for m in rec.messages)


@pytest.mark.slow
def test_device_reingest_and_cross_session(insert_photometry, common):
    """A second session reusing a device name with the same reusable spec ingests
    cleanly (no DuplicateError); blob-free vector columns don't raise."""
    build = lambda nwb: fx.build_full(nwb, suffix="_xsession")  # noqa: E731

    key_a, result_a = insert_photometry("mock_photometry_xs_a.nwb", build)
    key_b, result_b = insert_photometry("mock_photometry_xs_b.nwb", build)

    assert not result_a and not result_b
    # one shared catalog row, two sessions of config
    assert len(common.Indicator & {"indicator_name": "dLight38_xsession"}) == 1
    assert len(common.FiberPhotometryConfig & key_a) == 2
    assert len(common.FiberPhotometryConfig & key_b) == 2


@pytest.mark.slow
def test_same_file_config_reingest_raises(insert_photometry, common):
    """The session-specific config table is not `_expected_duplicates`, so a
    naive re-ingest of the same file raises `DuplicateError` (idempotency is
    file-level, via the `reinsert` flow, not per-row skipping)."""
    import datajoint as dj

    key, _ = insert_photometry(
        "mock_photometry_reingest.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_reingest"),
    )
    assert len(common.FiberPhotometryConfig & key) == 1
    with pytest.raises(dj.errors.DuplicateError):
        common.FiberPhotometryConfig().insert_from_nwbfile(key["nwb_file_name"])


@pytest.mark.slow
def test_null_fiber_metadata(insert_photometry, common):
    """A model-less fiber ingests with null model cols (no AttributeError); null
    insertion angles are stored null; the required location is still present."""
    key, result = insert_photometry(
        "mock_photometry_null.nwb",
        lambda nwb: fx.build_minimal(
            nwb,
            suffix="_null",
            fiber_model_kind=None,
            complete_insertion=False,
        ),
    )
    assert not result

    fiber = (
        common.OpticalFiber & {"optical_fiber_name": "OpticalFiber_DLS_null"}
    ).fetch1()
    assert fiber["numerical_aperture"] is None
    assert fiber["ferrule_name"] is None
    assert fiber["model_number"] is None

    cfg = (common.FiberPhotometryConfig & key).fetch1()
    assert cfg["location"] == "DLS"
    assert cfg["pitch"] is None and cfg["roll"] is None and cfg["yaw"] is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "suffix,builder_kwargs",
    [
        (
            "_gatesample",
            dict(fiber_model_kind="sparse", complete_insertion=False),
        ),
        ("_gatenull", dict(fiber_model_kind=None, complete_insertion=False)),
        (
            "_gatesparse",
            dict(fiber_model_kind="sparse", complete_insertion=True),
        ),
    ],
    ids=["sample-like", "model-less", "sparse-model+complete-insertion"],
)
def test_populate_all_common_gate(
    insert_photometry, common, suffix, builder_kwargs
):
    """populate_all_common on photometry files whose fibers would otherwise error
    the optogenetics tables returns no InsertError; with rollback_on_fail=True the
    Nwbfile and photometry rows survive."""
    key, result = insert_photometry(
        f"mock_photometry_gate{suffix}.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix=suffix, **builder_kwargs),
        rollback_on_fail=True,
        raise_err=False,
    )

    assert not result, f"unexpected InsertError keys: {result}"
    assert len(common.Nwbfile & key) == 1  # not super_deleted
    assert len(common.FiberPhotometryConfig & key) == 1


@pytest.mark.slow
def test_opto_gate_noop_on_non_photometry(insert_photometry, common):
    """A pure-optogenetics file (no FiberPhotometry container) still ingests its
    fibers into the gated optogenetics tables (gate is a no-op)."""
    key, result = insert_photometry(
        "mock_opto_gatenoop.nwb", fx.build_pure_devices
    )
    assert not result
    assert (
        len(
            common.OpticalFiberDevice
            & {"fiber_name": "DoricFlatFiber400um_pure"}
        )
        == 1
    )
    assert len(common.OpticalFiberImplant & key) == 1
    # and the photometry tables ingested nothing
    assert len(common.FiberPhotometryConfig & key) == 0


@pytest.mark.slow
def test_opto_gate_mixed_modality_shared_model(insert_photometry, common):
    """A shared OpticalFiberModel used by both a photometry and a non-photometry
    fiber is kept; the optogenetics fiber ingests with a resolvable FK."""
    key, result = insert_photometry(
        "mock_photometry_mixed.nwb", fx.build_mixed_modality
    )
    assert not result, f"unexpected InsertError keys: {result}"

    # shared model kept for the surviving optogenetics fiber
    assert (
        len(
            common.OpticalFiberDevice
            & {"fiber_name": "DoricFlatFiber400um_mixed"}
        )
        == 1
    )
    # optogenetics fiber ingested with resolvable -> OpticalFiberDevice FK
    implant = common.OpticalFiberImplant & key
    assert len(implant) == 1
    assert implant.fetch1("fiber_name") == "DoricFlatFiber400um_mixed"
    # photometry fiber ingested into the photometry tables
    assert len(common.FiberPhotometryConfig & key) == 1
    assert (
        len(
            common.OpticalFiber
            & {"optical_fiber_name": "OpticalFiber_photo_mixed"}
        )
        == 1
    )


@pytest.mark.slow
def test_unmodeled_column_warns(insert_photometry, common, monkeypatch):
    """A populated commanded_voltage_series column is warned about, not dropped
    silently; ingestion still succeeds."""
    from spyglass.common import common_photometry

    rec = _WarnRecorder()
    monkeypatch.setattr(common_photometry, "logger", rec)

    key, result = insert_photometry(
        "mock_photometry_unmodeled_col.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_unmodcol", unmodeled_column=True
        ),
    )
    assert not result
    assert len(common.FiberPhotometryConfig & key) == 1
    # warned exactly once (batched per ingest), naming the column
    assert (
        len([m for m in rec.messages if "commanded_voltage_series" in m]) == 1
    )


@pytest.mark.slow
def test_unmodeled_attr_warns(insert_photometry, common, monkeypatch):
    """A populated ExcitationSource.power_in_W is warned about; ingest succeeds."""
    from spyglass.common import common_photometry

    rec = _WarnRecorder()
    monkeypatch.setattr(common_photometry, "logger", rec)

    key, result = insert_photometry(
        "mock_photometry_unmodeled_attr.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_unmodattr", excitation_power=True
        ),
    )
    assert not result
    assert len(common.FiberPhotometryConfig & key) == 1
    # warned exactly once (batched per ingest), naming the attribute
    assert len([m for m in rec.messages if "power_in_W" in m]) == 1


# Runs in a fresh subprocess where ndx_fiber_photometry is blocked from the very
# start, so pynwb must reconstruct the typed objects from the file-embedded spec
# (the genuine stock-install case — no cached type-map registration). Imports the
# REAL ingestion module (spyglass.common -> common_photometry) and runs the
# ingest path under the block, so an accidental extension import in the ingestion
# code — at module load OR at runtime — raises ImportError and fails the child.
# DB credentials come in via env vars (the child connects to the same test
# container to activate schemas and read the file).
_IMPORT_SAFETY_SUBPROCESS = """
import os, sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "ndx_fiber_photometry":
            raise ImportError(name + " blocked (import-safety test)")
        return None

sys.meta_path.insert(0, _Block())

import datajoint as dj
dj.config["database.host"] = os.environ["_DJ_HOST"]
dj.config["database.port"] = int(os.environ["_DJ_PORT"])
dj.config["database.user"] = os.environ["_DJ_USER"]
dj.config["database.password"] = os.environ["_DJ_PASS"]

import spyglass.common as sgc  # imports common_photometry under the block

name = os.environ["_NWB_NAME"]
# dry_run exercises read + class-name match + version gate, without inserting
entries = sgc.FiberPhotometryConfig().insert_from_nwbfile(name, dry_run=True)
assert entries, "no config entries produced from the file-embedded spec"
sgc.OpticalFiber().insert_from_nwbfile(name, dry_run=True)
assert "ndx_fiber_photometry" not in sys.modules, sorted(
    m for m in sys.modules if "ndx" in m
)
print("IMPORT_SAFETY_OK")
"""


@pytest.mark.slow
def test_package_absent_import_safety(insert_photometry, common):
    """Ingestion must not import ndx-fiber-photometry: NWB types are matched by
    class-name string and gated on the file-embedded namespace version.

    Build the fixture (which imports the extension), then import the real
    ingestion module and run its ingest path in a fresh subprocess where the
    extension's import is blocked from the start — so any import attempt (module
    load or runtime) raises ``ImportError`` and fails the child.
    """
    import os
    import subprocess
    import sys

    import datajoint as dj

    key, _ = insert_photometry(
        "mock_photometry_absent.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_absent"),
    )
    assert len(common.FiberPhotometryConfig & key) == 1

    env = {
        **os.environ,
        "_DJ_HOST": str(dj.config["database.host"]),
        "_DJ_PORT": str(dj.config["database.port"]),
        "_DJ_USER": str(dj.config["database.user"]),
        "_DJ_PASS": str(dj.config["database.password"]),
        "_NWB_NAME": key["nwb_file_name"],
    }
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SAFETY_SUBPROCESS],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_SAFETY_OK" in result.stdout
