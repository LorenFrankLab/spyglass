"""Unit tests for the DB-free fiber-photometry NWB helpers.

These build in-memory NWB objects only (no write, no ingest). They take the
``common`` fixture solely because importing ``spyglass.common._photometry_nwb``
initializes the ``spyglass.common`` package; the helper functions themselves are
database-free.
"""

from tests.common import _photometry_fixture as fx


def _hp(common):
    from spyglass.common import _photometry_nwb as hp

    return hp


def test_referenced_devices_by_column(common):
    hp = _hp(common)
    nwb = fx.build_full(fx._new_nwb("h1"))

    assert [o.name for o in hp.referenced_devices(nwb, "indicator")] == [
        "dLight38_full"
    ]
    fibers = hp.referenced_devices(nwb, "optical_fiber")
    assert sorted(o.name for o in fibers) == [
        "OpticalFiber_DLS_full",
        "OpticalFiber_DMS_full",
    ]
    # union of two columns, deduped by name (EdgeFilter shared across both rows)
    filters = hp.referenced_devices(
        nwb, ["emission_filter", "excitation_filter"]
    )
    assert sorted(o.name for o in filters) == [
        "BandFilter525_full",
        "BaseFilter_full",
        "EdgeFilter490_full",
    ]


def test_no_op_without_container(common):
    hp = _hp(common)
    nwb = fx.build_pure_devices(fx._new_nwb("h2"))

    assert hp.is_photometry_file(nwb) is False
    assert hp.referenced_devices(nwb, "optical_fiber") == []
    assert hp.photometry_fiber_names(nwb) == set()
    # generic ndx-ophys-devices OpticalFiber instances are still discoverable
    assert [f.name for f in hp.optical_fiber_instances(nwb)] == [
        "OpticalFiber_opto_pure"
    ]


def test_missing_column_returns_empty_no_raise(common):
    hp = _hp(common)
    nwb = fx.build_full(fx._new_nwb("h3"))
    # a column absent from the table is skipped, not an error
    assert hp.referenced_devices(nwb, "does_not_exist") == []


def test_model_folders_null_safe(common):
    hp = _hp(common)
    modelless = fx.build_minimal(fx._new_nwb("h4"), fiber_model_kind=None)
    fiber = hp.referenced_devices(modelless, "optical_fiber")[0]
    assert getattr(fiber, "model", None) is None
    assert hp.model_attr("numerical_aperture")(fiber) is None
    assert hp.model_range("wavelength_range_in_nm", 0)(fiber) is None

    full = fx.build_full(fx._new_nwb("h5"))
    exc = {e.name: e for e in hp.referenced_devices(full, "excitation_source")}
    assert (
        hp.model_range("wavelength_range_in_nm", 0)(exc["ExcSrc_cont_full"])
        == 470.0
    )
    assert (
        hp.model_range("wavelength_range_in_nm", 1)(exc["ExcSrc_cont_full"])
        == 490.0
    )


def test_class_discriminator(common):
    hp = _hp(common)
    full = fx.build_full(fx._new_nwb("h6"))

    src = hp.class_discriminator(
        {"PulsedExcitationSource": "pulsed"}, "continuous"
    )
    by_name = {
        e.name: src(e) for e in hp.referenced_devices(full, "excitation_source")
    }
    assert by_name == {
        "ExcSrc_pulsed_full": "pulsed",
        "ExcSrc_cont_full": "continuous",
    }

    fc = hp.class_discriminator(
        {"BandOpticalFilter": "band", "EdgeOpticalFilter": "edge"}, "base"
    )
    by_name = {
        f.name: fc(f)
        for f in hp.referenced_devices(
            full, ["emission_filter", "excitation_filter"]
        )
    }
    assert by_name == {
        "BandFilter525_full": "band",
        "EdgeFilter490_full": "edge",
        "BaseFilter_full": "base",
    }


def test_gate_remaining_fibers_shared_model(common):
    hp = _hp(common)
    nwb = fx.build_mixed_modality(fx._new_nwb("h7"))

    photo = hp.photometry_fiber_names(nwb)
    assert photo == {"OpticalFiber_photo_mixed"}
    remaining = [
        f for f in hp.optical_fiber_instances(nwb) if f.name not in photo
    ]
    assert [f.name for f in remaining] == ["OpticalFiber_opto_mixed"]
    # the shared model is needed by a remaining (non-photometry) fiber -> keep
    needed = {f.model.name for f in remaining if f.model is not None}
    assert needed == {"DoricFlatFiber400um_mixed"}


def test_populated_attrs(common):
    hp = _hp(common)
    nwb = fx.build_minimal(fx._new_nwb("h8"), excitation_power=True)
    exc = hp.referenced_devices(nwb, "excitation_source")[0]
    assert hp.populated_attrs(exc, ["power_in_W", "exposure_time_in_s"]) == [
        "power_in_W"
    ]
    assert hp.populated_attrs(exc, ["exposure_time_in_s"]) == []


def test_response_series_scoped(common):
    hp = _hp(common)
    # a photometry file: the response series is discoverable
    full = fx.build_full(fx._new_nwb("h9"))
    names = [s.name for s in hp.response_series(full)]
    assert names == ["FPResponseSeries_DLS_470nm"]

    # several series in one file are all returned
    multi = fx.build_multi_series(fx._new_nwb("h10"))
    assert len(hp.response_series(multi)) == 3

    # a non-photometry file (no FiberPhotometry container) is a clean no-op
    pure = fx.build_pure_devices(fx._new_nwb("h11"))
    assert hp.response_series(pure) == []
