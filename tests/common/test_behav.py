import pytest
from pandas import DataFrame


def test_invalid_interval(pos_src):
    """Test invalid interval"""
    with pytest.raises(ValueError):
        pos_src.get_pos_interval_name("invalid_interval")


def test_invalid_epoch_num(common):
    """Test invalid epoch num"""
    with pytest.raises(ValueError):
        common.PositionSource.get_epoch_num("invalid_epoch_num")


def test_raw_position_fetchnwb(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch nwb"""
    fetched = DataFrame(
        (common.RawPosition & mini_pos_interval_dict)
        .fetch_nwb()[0]["raw_position"]
        .data
    )
    raw = DataFrame(mini_pos["led_0_series_0"].data)
    # compare with mini_pos
    assert fetched.equals(raw), "RawPosition fetch_nwb failed"


@pytest.mark.skip(reason="No video files in mini")
def test_videofile_no_transaction(common, mini_restr):
    """Test no transaction"""
    common.VideoFile()._no_transaction_make(mini_restr)


@pytest.mark.skip(reason="No video files in mini")
def test_videofile_update_entries(common):
    """Test update entries"""
    common.VideoFile().update_entries()


@pytest.mark.skip(reason="No video files in mini")
def test_videofile_getabspath(common, mini_restr):
    """Test get absolute path"""
    common.VideoFile().getabspath(mini_restr)


def test_posinterval_no_transaction(verbose_context, common, mini_restr):
    """Test no transaction"""
    before = common.PositionIntervalMap().fetch()
    with verbose_context:
        common.PositionIntervalMap()._no_transaction_make(mini_restr)
    after = common.PositionIntervalMap().fetch()
    assert (
        len(after) == len(before) + 2
    ), "PositionIntervalMap no_transaction had unexpected effect"


def test_get_pos_interval_name(pos_src, pos_interval_01):
    """Test get pos interval name"""
    names = [f"pos {x} valid times" for x in range(1)]
    assert pos_interval_01 == names, "get_pos_interval_name failed"


def test_convert_epoch(common, mini_dict, pos_interval_01):
    this_key = (
        common.IntervalList & mini_dict & {"interval_list_name": "01_s1"}
    ).fetch1()
    ret = common.common_behav.convert_epoch_interval_name_to_position_interval_name(
        this_key
    )
    assert (
        ret == pos_interval_01[0]
    ), "convert_epoch_interval_name_to_position_interval_name failed"
