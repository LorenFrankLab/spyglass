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
    """The synthetic fixture must embed NWB core 2.9.0 (the supported floor) and
    ndx-* namespaces at/above the ingest gate minimums — otherwise the whole
    suite would silently no-op (gate skip) instead of exercising ingestion."""
    from packaging.version import Version

    from spyglass.utils.nwb_hash import get_file_namespaces

    path = raw_dir / "mock_photometry_coreversion.nwb"
    fx.write(path, fx.build_minimal, identifier="coreversion")
    namespaces = get_file_namespaces(str(path))
    assert namespaces.get("core") == "2.9.0"
    assert Version(namespaces["ndx-fiber-photometry"]) >= Version("0.2.3")
    assert Version(namespaces["ndx-ophys-devices"]) >= Version("0.3.1")


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
    """The row's site is normalized into BrainRegion (not the fiber desc)."""
    config = (
        common.FiberPhotometryConfig & photometry_full
    ) * common.BrainRegion
    assert (config & {"fiber_id": 0}).fetch1("region_name") == "DLS"
    assert (config & {"fiber_id": 1}).fetch1("region_name") == "DMS"


@pytest.mark.slow
def test_same_location_shares_one_brain_region(insert_photometry, common):
    """Two fibers at the same site normalize to a single, shared BrainRegion row
    (the fetch_add dedup invariant), not one region row per config row. A site
    unique to this test proves ingestion creates the row (rather than reusing one
    another fixture pre-seeded) and that a second config row reuses it."""
    site = "PhotomDedupSite"
    assert len(common.BrainRegion & {"region_name": site}) == 0

    key, result = insert_photometry(
        "mock_photometry_dedup.nwb",
        lambda nwb: fx.build_colliding_columns(nwb, location=site),
    )
    assert not result

    # exactly one BrainRegion row was created for the shared site
    assert len(common.BrainRegion & {"region_name": site}) == 1
    # ...and both config rows resolve to that same region_id
    region_ids = (common.FiberPhotometryConfig & key).fetch("region_id")
    assert len(region_ids) == 2
    assert len(set(region_ids)) == 1


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
    assert cfg["pitch"] is None and cfg["roll"] is None and cfg["yaw"] is None
    region = ((common.FiberPhotometryConfig & key) * common.BrainRegion).fetch1(
        "region_name"
    )
    assert region == "DLS"


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


# --- signal reference: FiberPhotometryResponseSeries ------------------------


@pytest.mark.slow
def test_response_series_ingest(photometry_full, common):
    """One master row per response series; the object id, num_samples and unit
    round-trip, and the ``.Fiber`` part maps the region's positional index to the
    right config ``fiber_id`` (not the region data value)."""
    resp = common.FiberPhotometryResponseSeries & photometry_full
    assert len(resp) == 1
    row = resp.fetch1()
    assert row["name"] == "FPResponseSeries_DLS_470nm"
    assert row["num_samples"] == 1000
    assert row["unit"] == "V"

    # the stored object id resolves to that very series
    series = common.FiberPhotometryResponseSeries().nwb_object(row)
    assert series.name == "FPResponseSeries_DLS_470nm"
    assert series.object_id == row["response_series_object_id"]

    # .Fiber: region positional index 0 -> config fiber_id 0 (DLS)
    fibers = common.FiberPhotometryResponseSeries.Fiber & photometry_full
    assert len(fibers) == 1
    fiber = fibers.fetch1()
    assert fiber["region_index"] == 0
    assert fiber["fiber_id"] == 0
    assert fiber["fiber_photometry_name"] == "fiber_photometry"


@pytest.mark.slow
def test_fetch1_dataframe_roundtrip(photometry_full, common):
    """``fetch1_dataframe`` returns a time-indexed frame whose length, time axis
    and values match the NWB trace, with a column labeled by location+wavelength.
    """
    df = (
        common.FiberPhotometryResponseSeries & photometry_full
    ).fetch1_dataframe()

    assert len(df) == 1000
    assert df.index.name == "time"
    # time axis from starting_time (0.083) + arange(n)/rate (6024.096)
    assert df.index[0] == pytest.approx(0.083, abs=1e-9)
    assert df.index[1] == pytest.approx(0.083 + 1 / 6024.096, abs=1e-12)
    assert list(df.columns) == ["DLS_470nm"]
    assert np.allclose(df["DLS_470nm"].to_numpy(), np.arange(1000))


@pytest.mark.slow
def test_nwb_table_set(photometry_full, common):
    """Regression guard: the master FKs ``-> Session`` (not ``-> Nwbfile``), so
    without ``_nwb_table = Nwbfile`` the fetch mixin raises ``NotImplementedError``.
    """
    assert common.FiberPhotometryResponseSeries._nwb_table is common.Nwbfile
    records = (
        common.FiberPhotometryResponseSeries & photometry_full
    ).fetch_nwb()
    assert len(records) == 1
    # the object_id column resolves to the NWB object under key 'response_series'
    assert records[0]["response_series"].name == "FPResponseSeries_DLS_470nm"


@pytest.mark.slow
def test_optional_region_none(insert_photometry, common, monkeypatch):
    """A series with no ``fiber_photometry_table_region`` inserts the master row
    with **no** ``.Fiber`` rows and a warning (not a skip/raise);
    ``fetch1_dataframe`` falls back to generic ``f"{name}_col{i}"`` labels."""
    from spyglass.common import common_photometry

    rec = _WarnRecorder()
    monkeypatch.setattr(common_photometry, "logger", rec)

    key, result = insert_photometry(
        "mock_photometry_noregion.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_noregion", response_region=False
        ),
    )
    assert not result

    resp = common.FiberPhotometryResponseSeries & key
    assert len(resp) == 1
    assert len(common.FiberPhotometryResponseSeries.Fiber & key) == 0
    assert any("fiber_photometry_table_region" in m for m in rec.messages)

    df = resp.fetch1_dataframe()
    assert len(df) == 500
    assert list(df.columns) == ["FPResponseSeries_DLS_490nm_col0"]


@pytest.mark.slow
def test_multiple_series_per_file(insert_photometry, common):
    """A file with several response series: each master row's ``nwb_object`` /
    ``fetch1_dataframe`` resolves its **own** series (guards a ``Raw``-style
    ``nwb_file_name``-only fetch, which would ``fetch1()`` multiple rows)."""
    key, result = insert_photometry(
        "mock_photometry_mseries.nwb",
        lambda nwb: fx.build_multi_series(nwb, suffix="_mseries"),
    )
    assert not result

    resp = common.FiberPhotometryResponseSeries & key
    assert len(resp) == 3
    by_name = {r["name"]: r for r in resp.fetch(as_dict=True)}
    assert set(by_name) == {
        "FPResponseSeries_DLS_470nm",
        "FPResponseSeries_DMS_490nm",
        "FPResponseSeries_both_2d",
    }
    assert by_name["FPResponseSeries_DLS_470nm"]["num_samples"] == 1000
    assert by_name["FPResponseSeries_DMS_490nm"]["num_samples"] == 500
    assert by_name["FPResponseSeries_both_2d"]["num_samples"] == 300

    tbl = common.FiberPhotometryResponseSeries()
    for name, row in by_name.items():
        assert tbl.nwb_object(row).name == name  # correct object per row
        df = (resp & {"name": name}).fetch1_dataframe()
        assert len(df) == row["num_samples"]


@pytest.mark.slow
def test_multi_row_region(insert_photometry, common):
    """A 2-D ``[time, n_fibers]`` series with a multi-row region -> one ``.Fiber``
    row per referenced config row, each data column labeled per fiber."""
    key, result = insert_photometry(
        "mock_photometry_mrow.nwb",
        lambda nwb: fx.build_multi_series(nwb, suffix="_mrow"),
    )
    assert not result

    resp = (
        common.FiberPhotometryResponseSeries
        & key
        & {"name": "FPResponseSeries_both_2d"}
    )
    fibers = common.FiberPhotometryResponseSeries.Fiber & resp.fetch1("KEY")
    assert len(fibers) == 2
    got = {f["region_index"]: f["fiber_id"] for f in fibers.fetch(as_dict=True)}
    assert got == {0: 0, 1: 1}  # positional 0->id 0, positional 1->id 1

    df = resp.fetch1_dataframe()
    assert df.shape == (300, 2)
    assert list(df.columns) == ["DLS_470nm", "DMS_490nm"]
    expected = np.arange(600, dtype="float64").reshape(300, 2)
    assert np.allclose(df["DLS_470nm"].to_numpy(), expected[:, 0])
    assert np.allclose(df["DMS_490nm"].to_numpy(), expected[:, 1])


@pytest.mark.slow
def test_response_series_nonconsecutive_fiber_id(insert_photometry, common):
    """The ``.Fiber`` row maps the region's positional index to the table row
    ``id``, not to the position itself: a 1-row table with ``id`` 7 and region
    ``[0]`` must yield ``fiber_id`` 7 — a guard that would fail an implementation
    that used ``fiber_id = positional``."""
    key, result = insert_photometry(
        "mock_photometry_ncfiber.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_ncfiber", row_id=7),
    )
    assert not result
    fiber = (common.FiberPhotometryResponseSeries.Fiber & key).fetch1()
    assert fiber["region_index"] == 0
    assert fiber["fiber_id"] == 7


@pytest.mark.slow
def test_fetch1_dataframe_timestamps(insert_photometry, common):
    """A series with explicit (irregular) ``timestamps`` and no ``rate`` yields a
    dataframe indexed by those timestamps, not a rate-derived axis."""
    ts = [0.0, 0.5, 1.5, 3.0]
    key, result = insert_photometry(
        "mock_photometry_ts.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_ts", timestamps=ts),
    )
    assert not result
    df = (common.FiberPhotometryResponseSeries & key).fetch1_dataframe()
    assert len(df) == len(ts)
    assert np.allclose(df.index.to_numpy(), ts)


def _first_response_series(nwb):
    from spyglass.common._photometry_nwb import response_series

    return response_series(nwb)[0]


def test_region_absent_emits_empty_fiber_key(common):
    """A region-absent series still returns the ``.Fiber`` part key (empty), so
    the mixin's multi-object insert loop can extend it across series in one file
    regardless of order — otherwise a later region-present series would
    ``KeyError``. Exercises the override directly (no ingest)."""
    tbl = common.FiberPhotometryResponseSeries()
    nwb = fx.build_minimal(
        fx._new_nwb("noreg"), suffix="_noreg", response_region=False
    )
    entries = tbl.generate_entries_from_nwb_object(
        _first_response_series(nwb), base_key={"nwb_file_name": "x"}
    )
    assert tbl.Fiber in entries and entries[tbl.Fiber] == []
    assert len(entries[tbl]) == 1  # master row still emitted


def test_region_index_out_of_range_raises(common):
    """A region position with no matching table row raises a named ``ValueError``
    rather than silently mis-mapping (a negative index would otherwise wrap to the
    wrong row)."""
    tbl = common.FiberPhotometryResponseSeries()
    nwb = fx.build_bad_region(
        fx._new_nwb("oor"), suffix="_oor", kind="out_of_range"
    )
    with pytest.raises(ValueError, match="out of range"):
        tbl.generate_entries_from_nwb_object(
            _first_response_series(nwb), base_key={"nwb_file_name": "x"}
        )


def test_region_width_mismatch_warns(common, monkeypatch):
    """A 2-D trace whose region lists fewer fibers than columns warns (some
    columns cannot be labeled) but still maps the referenced fiber."""
    from spyglass.common import common_photometry

    rec = _WarnRecorder()
    monkeypatch.setattr(common_photometry, "logger", rec)
    tbl = common.FiberPhotometryResponseSeries()
    nwb = fx.build_bad_region(
        fx._new_nwb("wm"), suffix="_wm", kind="width_mismatch"
    )
    entries = tbl.generate_entries_from_nwb_object(
        _first_response_series(nwb), base_key={"nwb_file_name": "x"}
    )
    assert len(entries[tbl.Fiber]) == 1  # only the one referenced fiber
    assert any("column" in m for m in rec.messages)


@pytest.mark.slow
def test_response_series_ophys_below_min_skips(
    insert_photometry, common, monkeypatch
):
    """When ndx-ophys-devices is below the minimum, config/device ingestion is
    skipped; the response series must skip **too** (it shares the config gate),
    rather than inserting .Fiber rows whose FK into the now-empty config fails.
    """
    from spyglass.utils.mixins import ingestion

    rec = _WarnRecorder()
    monkeypatch.setattr(ingestion, "logger", rec)
    monkeypatch.setattr(
        ingestion,
        "get_file_namespaces",
        lambda path: {
            "ndx-fiber-photometry": "0.2.3",  # at/above its minimum
            "ndx-ophys-devices": "0.2.0",  # below the 0.3.1 minimum
        },
    )
    key, result = insert_photometry(
        "mock_photometry_ophysbelow.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_ophysbelow"),
        raise_err=False,
    )
    assert not result  # no InsertError from a dangling .Fiber FK
    assert len(common.FiberPhotometryConfig & key) == 0
    assert len(common.FiberPhotometryResponseSeries & key) == 0
    assert len(common.FiberPhotometryResponseSeries.Fiber & key) == 0
    # the skip is diagnostic, not silent
    assert any("ndx-ophys-devices" in m for m in rec.messages)


@pytest.mark.slow
def test_make_repopulates(insert_photometry, common):
    """The ``dj.Imported`` entry point works: ``make()`` delegates to
    ``insert_from_nwbfile`` (so ``populate()`` is not a ``NotImplementedError``
    footgun)."""
    key, result = insert_photometry(
        "mock_photometry_make.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_make"),
    )
    assert not result
    tbl = common.FiberPhotometryResponseSeries
    assert len(tbl & key) == 1
    (tbl & key).delete(safemode=False)  # cascades to .Fiber
    assert len(tbl & key) == 0
    tbl().make(key)  # the make() shim
    assert len(tbl & key) == 1 and len(tbl.Fiber & key) == 1


@pytest.mark.slow
def test_column_label_collision_disambiguation(insert_photometry, common):
    """Two columns of one series whose config rows share location+wavelength get
    the same base label; ``fetch1_dataframe`` disambiguates them by ``fiber_id``
    (unique, usable column names — pandas silently mangles duplicate labels)."""
    key, result = insert_photometry(
        "mock_photometry_collide.nwb", fx.build_colliding_columns
    )
    assert not result
    df = (common.FiberPhotometryResponseSeries & key).fetch1_dataframe()
    assert df.shape == (100, 2)
    assert list(df.columns) == ["DLS_470nm_id0", "DLS_470nm_id1"]
    assert len(set(df.columns)) == 2  # unique


@pytest.mark.slow
def test_multi_container_response_fiber_derivation(insert_photometry, common):
    """With two FiberPhotometry containers, each response series' ``.Fiber`` FK
    resolves to its **own** container's config (the region-side container-name
    derivation picks the right one of two, not always ``"fiber_photometry"``).
    """
    key, result = insert_photometry(
        "mock_photometry_2c_resp.nwb", fx.build_two_containers
    )
    assert not result
    resp = common.FiberPhotometryResponseSeries & key
    assert len(resp) == 2
    # each series' .Fiber row carries the container name matching its series
    for name, container in (
        ("FPResponseSeries_A", "fiber_photometry_A"),
        ("FPResponseSeries_B", "fiber_photometry_B"),
    ):
        series_key = (resp & {"name": name}).fetch1("KEY")
        fiber = (
            common.FiberPhotometryResponseSeries.Fiber & series_key
        ).fetch1()
        assert fiber["fiber_photometry_name"] == container
        assert fiber["fiber_id"] == 0


@pytest.mark.slow
def test_valid_times_interval_list(insert_photometry, common):
    """Each response series links to an ``IntervalList`` of its valid (recorded)
    times, so the trace can be time-restricted against the rest of Spyglass (as
    ``Raw`` does). A rate-based series spans ``[starting_time, start+(n-1)/rate]``.
    """
    key, result = insert_photometry(
        "mock_photometry_intervals.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_intervals"),
    )
    assert not result
    resp = (common.FiberPhotometryResponseSeries & key).fetch1()
    interval_name = resp["interval_list_name"]
    assert interval_name == "FPResponseSeries_DLS_490nm valid times"

    interval = (
        common.IntervalList & key & {"interval_list_name": interval_name}
    ).fetch1()
    assert interval["pipeline"] == "fiber_photometry"
    valid_times = np.asarray(interval["valid_times"])
    assert valid_times.shape == (
        1,
        2,
    )  # build_minimal: 500 samples @ 6024.096 Hz
    assert valid_times[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert valid_times[0, 1] == pytest.approx(499 / 6024.096, abs=1e-9)


@pytest.mark.slow
def test_valid_times_from_timestamps(insert_photometry, common):
    """A timestamps-based series' valid interval spans its first/last timestamp."""
    ts = [0.0, 0.5, 1.5, 3.0]
    key, result = insert_photometry(
        "mock_photometry_ts_interval.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_tsiv", timestamps=ts),
    )
    assert not result
    resp = (common.FiberPhotometryResponseSeries & key).fetch1()
    interval = (
        common.IntervalList
        & key
        & {"interval_list_name": resp["interval_list_name"]}
    ).fetch1()
    assert np.asarray(interval["valid_times"]).tolist() == [[0.0, 3.0]]


# --- injection metadata (link to the shared VirusInjection) ------------------


@pytest.mark.slow
def test_injection_populates_shared_tables(insert_photometry, common):
    """A photometry file carrying a viral injection populates the shared,
    session-scoped ``VirusInjection`` (site fields) and its parent ``Virus``
    (construct) — the single-source-of-truth outcome, no new photometry table.
    """
    key, result = insert_photometry(
        "mock_photometry_inj.nwb",
        lambda nwb: fx.build_minimal(nwb, suffix="_inj", injection="complete"),
    )
    assert not result
    vi = (common.VirusInjection & key).fetch1()
    assert vi["location"] == "NAcc"
    assert vi["hemisphere"] == "left"
    assert vi["titer"] == pytest.approx(1.5e13, rel=1e-3)
    assert vi["volume"] == pytest.approx(0.4, abs=1e-4)
    virus = (common.Virus & {"virus_name": vi["virus_name"]}).fetch1()
    assert virus["construct_name"] == "AAV-dLight3.8"


@pytest.mark.slow
def test_config_injection_link(insert_photometry, common):
    """A config row whose indicator has an injection carries the FK; the
    ``fetch_injection()`` accessor returns the joined injection+virus row (and
    does **not** silently drop it via the config/injection column-name collision
    a bare natural join would hit)."""
    key, result = insert_photometry(
        "mock_photometry_injlink.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_injlink", injection="complete"
        ),
    )
    assert not result
    cfg = common.FiberPhotometryConfig & key
    assert cfg.fetch1("injection_object_id")  # non-null FK set

    rows = cfg.fetch_injection(as_dict=True)
    assert len(rows) == 1  # the collision-free accessor keeps the row
    row = rows[0]
    assert (
        row["location"] == "NAcc"
    )  # injection site (VirusInjection, free text)
    assert row["titer"] == pytest.approx(1.5e13, rel=1e-3)
    assert row["construct_name"] == "AAV-dLight3.8"


@pytest.mark.slow
def test_no_injection_frank_shape(insert_photometry, common):
    """A file with no injection (the real Frank shape) ingests cleanly; every
    config row's ``injection_object_id`` is null and ``fetch_injection()`` is
    empty."""
    key, result = insert_photometry(
        "mock_photometry_noinj.nwb",
        lambda nwb: fx.build_full(nwb, suffix="_noinj"),
    )
    assert not result
    cfg = common.FiberPhotometryConfig & key
    assert len(cfg) == 2
    assert all(v is None for v in cfg.fetch("injection_object_id"))
    assert len(cfg.fetch_injection(as_dict=True)) == 0


@pytest.mark.slow
def test_sparse_injection_no_dangling_fk(insert_photometry, common):
    """An injection missing a ``VirusInjection`` NOT-NULL field (optional in ndx)
    is dropped by ``VirusInjection``; the config link is left null — no dangling
    FK, no ``InsertError``."""
    key, result = insert_photometry(
        "mock_photometry_sparseinj.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_sparseinj", injection="sparse_injection"
        ),
    )
    assert not result
    assert len(common.VirusInjection & key) == 0  # dropped
    cfg = common.FiberPhotometryConfig & key
    assert len(cfg) == 1
    assert cfg.fetch1("injection_object_id") is None


@pytest.mark.slow
def test_sparse_parent_virus_photometry_survives(insert_photometry, common):
    """A complete injection with a sparse parent ``ViralVector`` (no
    ``description``): ``Virus`` drops the parent, so ``VirusInjection``'s
    ``-> Virus`` FK fails (a logged, pre-existing opto ``InsertError``) — but with
    ``rollback_on_fail=False`` the photometry rows survive and the config link is
    null. Pins the document-and-defer behavior; the fixture must override
    ``raise_err`` (it defaults ``True``, which would propagate the error)."""
    key, result = insert_photometry(
        "mock_photometry_sparsevirus.nwb",
        lambda nwb: fx.build_minimal(
            nwb, suffix="_sparsevirus", injection="sparse_virus"
        ),
        raise_err=False,
        rollback_on_fail=False,
    )
    assert result  # the VirusInjection -> Virus FK failure was recorded
    assert len(common.Virus & {"virus_name": "ViralVector_sparsevirus"}) == 0
    assert len(common.VirusInjection & key) == 0
    # photometry survives, link null
    cfg = common.FiberPhotometryConfig & key
    assert len(cfg) == 1
    assert cfg.fetch1("injection_object_id") is None
    assert len(common.FiberPhotometryResponseSeries & key) == 1


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
# also exercise the response-series runtime path (object discovery, region ->
# fiber_id translation) under the block, so an extension import there is caught
resp = sgc.FiberPhotometryResponseSeries().insert_from_nwbfile(name, dry_run=True)
assert resp, "no response-series entries produced from the file-embedded spec"
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
