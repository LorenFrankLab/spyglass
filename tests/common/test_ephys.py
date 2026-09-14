from unittest.mock import Mock

import numpy as np
import pytest
from numpy import array_equal

from ..conftest import TEARDOWN


def test_create_from_config(common_ephys, mini_copy_name):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_copy_name)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(common_ephys, mini_dict, mini_content):
    obj_fetch = common_ephys.Raw().nwb_object(mini_dict).object_id
    obj_raw = mini_content.get_acquisition().object_id
    assert obj_fetch == obj_raw, "Raw.nwb_object did not return expected object"


def test_electrode_populate(common_ephys):
    common_ephys.Electrode.populate()
    assert len(common_ephys.Electrode()) == 128, "Electrode.populate failed"


def test_elec_group_populate(pop_common_electrode_group):
    assert (
        len(pop_common_electrode_group) == 32
    ), "ElectrodeGroup.populate failed"


def test_raw_populate(common_ephys):
    common_ephys.Raw.populate()
    assert len(common_ephys.Raw()) == 1, "Raw.populate failed"


def test_sample_count_populate(common_ephys):
    common_ephys.SampleCount.populate()
    assert len(common_ephys.SampleCount()) == 1, "SampleCount.populate failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_set_lfp_electrodes(common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass


def test_raw_rate_fallback_prefers_rate(common_ephys):
    """Use explicit rate when present on the NWB object."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock(rate=2000.0, timestamps=np.array([0.0, 0.5, 1.0]))

    assert raw_table._rate_fallback(nwb_obj) == 2000.0


def test_raw_rate_fallback_uses_timestamps(common_ephys, monkeypatch):
    """Estimate rate from timestamps when rate is missing."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = np.array([0.0, 0.5, 1.0, 1.5])

    called = {}

    def _fake_estimate(ts, tol, verbose):
        called["ts"] = ts
        called["tol"] = tol
        called["verbose"] = verbose
        return 2.0

    monkeypatch.setattr(common_ephys, "estimate_sampling_rate", _fake_estimate)

    rate = raw_table._rate_fallback(nwb_obj)
    assert rate == 2.0
    assert np.array_equal(called["ts"], nwb_obj.timestamps)
    assert called["tol"] == 1.5


def test_raw_rate_fallback_requires_rate_or_timestamps(common_ephys):
    """Raise when neither rate nor timestamps are available."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = None

    with pytest.raises(ValueError, match="Neither rate nor timestamps"):
        raw_table._rate_fallback(nwb_obj)


def test_raw_valid_times_from_raw_rate_path(common_ephys):
    """Valid times are derived directly when rate is present."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = 1000.0
    nwb_obj.data = np.zeros(3000)

    valid = raw_table._valid_times_from_raw(nwb_obj)
    assert np.array_equal(valid, np.array([[0.0, 3.0]]))


def test_raw_valid_times_from_raw_timestamp_path(common_ephys, monkeypatch):
    """Timestamp fallback delegates to get_valid_intervals."""
    raw_table = common_ephys.Raw()
    nwb_obj = Mock()
    nwb_obj.rate = None
    nwb_obj.timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    monkeypatch.setattr(raw_table, "_rate_fallback", lambda _: 1.0)
    monkeypatch.setattr(
        common_ephys,
        "get_valid_intervals",
        lambda **kwargs: np.array([[0.0, 3.0]]),
    )

    valid = raw_table._valid_times_from_raw(nwb_obj)
    assert np.array_equal(valid, np.array([[0.0, 3.0]]))


def test_lfp_make_compute_returns_none_without_filter_coeff(common_ephys):
    """Return sentinel values when filter coefficients are unavailable."""
    lfp_table = common_ephys.LFP()

    out = lfp_table.make_compute(
        key={"nwb_file_name": "test.nwb"},
        lfp_file_name="analysis.nwb",
        lfp_file_abspath="/tmp/analysis.nwb",
        electrode_keys=[{"electrode_id": 1}],
        rawdata=Mock(),
        sampling_rate=30000,
        interval_list_name="raw data valid times",
        valid_times=Mock(),
        filter={"filter_coeff": np.array([]), "filter_name": "LFP 0-400 Hz"},
    )

    assert out == [None, None]


def test_lfp_make_insert_returns_early_on_none(common_ephys, monkeypatch):
    """Skip all writes when make_compute returned sentinel values."""
    lfp_table = common_ephys.LFP()

    add_calls = []
    insert_interval_calls = []
    insert_calls = []

    monkeypatch.setattr(
        common_ephys.AnalysisNwbfile,
        "add",
        lambda *args, **kwargs: add_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        common_ephys.IntervalList,
        "insert1",
        lambda *args, **kwargs: insert_interval_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        lfp_table,
        "insert1",
        lambda *args, **kwargs: insert_calls.append((args, kwargs)),
    )

    lfp_table.make_insert(
        key={"nwb_file_name": "test.nwb"},
        lfp_valid_times=None,
        added_key=None,
        lfp_file_name="analysis.nwb",
    )

    assert add_calls == []
    assert insert_interval_calls == []
    assert insert_calls == []


def test_set_lfp_electrodes_returns_when_delete_incomplete(
    common_ephys, monkeypatch
):
    """Do not reinsert if post-delete query is still non-empty."""
    inserted = {"session": 0, "part": 0}

    class FakeSessionQuery:
        def delete(self, safemode=True):
            return None

        def fetch(self):
            return [{"nwb_file_name": "still-there.nwb"}]

    class FakeLFPSelection:
        class LFPElectrode:
            @staticmethod
            def insert(*_args, **_kwargs):
                inserted["part"] += 1

        def __and__(self, _key):
            return FakeSessionQuery()

        def insert1(self, _key):
            inserted["session"] += 1

    original_lfp_selection = common_ephys.LFPSelection
    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)

    original_lfp_selection().set_lfp_electrodes("test.nwb", [1, 2])
    assert inserted["session"] == 0
    assert inserted["part"] == 0


def test_lfp_band_selection_invalid_electrode_ids(common_ephys, monkeypatch):
    """Raise when requested electrodes are not in LFPSelection."""

    class FakeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeRel()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)

    with pytest.raises(ValueError, match="electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[99],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[-1],
            lfp_band_sampling_rate=1000,
        )


def test_lfp_band_selection_invalid_divisor(common_ephys, monkeypatch):
    """Raise when band sampling rate is not an integer divisor."""

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)

    with pytest.raises(ValueError, match="integer divisor"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[-1],
            lfp_band_sampling_rate=777,
        )


def test_lfp_band_selection_invalid_reference_length(common_ephys, monkeypatch):
    """Raise when reference list length is neither 1 nor N electrodes."""

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class TrueQuery:
        def __bool__(self):
            return True

    class FakeFirFilterParameters:
        def __and__(self, _key):
            return TrueQuery()

    class FakeIntervalList:
        def __and__(self, _key):
            return TrueQuery()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)
    monkeypatch.setattr(
        common_ephys,
        "FirFilterParameters",
        FakeFirFilterParameters,
    )
    monkeypatch.setattr(common_ephys, "IntervalList", FakeIntervalList)

    with pytest.raises(ValueError, match="reference_electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1, 2],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[1, 2, 3],
            lfp_band_sampling_rate=1000,
        )


def test_lfp_band_selection_invalid_reference_ids(common_ephys, monkeypatch):
    """Raise when reference electrodes are outside allowed IDs + -1."""

    class FakeLFPElectrodeRel:
        def __and__(self, _key):
            return self

        def fetch(self, _field):
            return np.array([1, 2])

    class FakeLFPSelection:
        def LFPElectrode(self):
            return FakeLFPElectrodeRel()

    class FakeLFPRel:
        def __and__(self, _key):
            return self

        def fetch1(self, _field):
            return 30000

    class TrueQuery:
        def __bool__(self):
            return True

    class FakeFirFilterParameters:
        def __and__(self, _key):
            return TrueQuery()

    class FakeIntervalList:
        def __and__(self, _key):
            return TrueQuery()

    monkeypatch.setattr(common_ephys, "LFPSelection", FakeLFPSelection)
    monkeypatch.setattr(common_ephys, "LFP", FakeLFPRel)
    monkeypatch.setattr(
        common_ephys,
        "FirFilterParameters",
        FakeFirFilterParameters,
    )
    monkeypatch.setattr(common_ephys, "IntervalList", FakeIntervalList)

    with pytest.raises(ValueError, match="reference_electrode_list"):
        common_ephys.LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name="test.nwb",
            electrode_list=[1, 2],
            filter_name="LFP 0-400 Hz",
            interval_list_name="lfp valid times",
            reference_electrode_list=[999],
            lfp_band_sampling_rate=1000,
        )
