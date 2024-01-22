import pytest
from datajoint.hash import key_hash


@pytest.fixture
def common_position(common):
    yield common.common_position


@pytest.fixture
def interval_position_info(common_position):
    yield common_position.IntervalPositionInfo


@pytest.fixture
def default_param_key():
    yield {"position_info_param_name": "default"}


@pytest.fixture
def interval_key(common):
    yield (common.IntervalList & "interval_list_name LIKE 'pos 0%'").fetch1(
        "KEY"
    )


@pytest.fixture
def param_table(common_position, default_param_key, teardown):
    param_table = common_position.PositionInfoParameters()
    param_table.insert1(default_param_key, skip_duplicates=True)
    yield param_table
    if teardown:
        param_table.delete(safemode=False)


@pytest.fixture
def upsample_position(
    common,
    common_position,
    param_table,
    default_param_key,
    teardown,
    interval_key,
):
    params = (param_table & default_param_key).fetch1()
    upsample_param_key = {"position_info_param_name": "upsampled"}
    param_table.insert1(
        {
            **params,
            **upsample_param_key,
            "is_upsampled": 1,
            "max_separation": 80,
            "upsampling_sampling_rate": 500,
        },
        skip_duplicates=True,
    )
    interval_pos_key = {**interval_key, **upsample_param_key}
    common_position.IntervalPositionInfoSelection.insert1(
        interval_pos_key, skip_duplicates=True
    )
    common_position.IntervalPositionInfo.populate(interval_pos_key)
    yield interval_pos_key
    if teardown:
        (param_table & upsample_param_key).delete(safemode=False)


@pytest.fixture
def interval_pos_key(upsample_position):
    yield upsample_position


def test_interval_position_info_insert(common_position, interval_pos_key):
    assert common_position.IntervalPositionInfo & interval_pos_key


@pytest.fixture
def upsample_position_error(
    upsample_position,
    default_param_key,
    param_table,
    common,
    common_position,
    teardown,
    interval_key,
):
    params = (param_table & default_param_key).fetch1()
    upsample_param_key = {"position_info_param_name": "upsampled error"}
    param_table.insert1(
        {
            **params,
            **upsample_param_key,
            "is_upsampled": 1,
            "max_separation": 1,
            "upsampling_sampling_rate": 500,
        },
        skip_duplicates=True,
    )
    interval_pos_key = {**interval_key, **upsample_param_key}
    common_position.IntervalPositionInfoSelection.insert1(interval_pos_key)
    yield interval_pos_key
    if teardown:
        (param_table & upsample_param_key).delete(safemode=False)


def test_interval_position_info_insert_error(
    interval_position_info, upsample_position_error
):
    with pytest.raises(ValueError):
        interval_position_info.populate(upsample_position_error)


def test_fetch1_dataframe(interval_position_info, interval_pos_key):
    df = (interval_position_info & interval_pos_key).fetch1_dataframe()
    err_msg = "Unexpected output of IntervalPositionInfo.fetch1_dataframe"
    assert df.shape == (5193, 6), err_msg

    df_sums = {c: df[c].iloc[:5].sum() for c in df.columns}
    df_sums_exp = {
        "head_orientation": 4.4300073600180125,
        "head_position_x": 111.25,
        "head_position_y": 141.75,
        "head_speed": 0.6084872579024899,
        "head_velocity_x": -0.4329520555149495,
        "head_velocity_y": 0.42756198762527325,
    }
    for k in df_sums:
        assert k in df_sums_exp, err_msg
        assert df_sums[k] == pytest.approx(df_sums_exp[k], rel=0.02), err_msg


def test_interval_position_info_kwarg_error(interval_position_info):
    with pytest.raises(ValueError):
        interval_position_info._fix_kwargs()


def test_interval_position_info_kwarg_alias(interval_position_info):
    in_tuple = (0, 1, 2, 3)
    out_tuple = interval_position_info._fix_kwargs(
        head_orient_smoothing_std_dev=in_tuple[0],
        head_speed_smoothing_std_dev=in_tuple[1],
        max_separation=in_tuple[2],
        max_speed=in_tuple[3],
    )
    assert (
        out_tuple == in_tuple
    ), "IntervalPositionInfo._fix_kwargs() should alias old arg names."


@pytest.mark.skip(reason="Not testing with video data yet.")
def test_position_video(common_position):
    pass
