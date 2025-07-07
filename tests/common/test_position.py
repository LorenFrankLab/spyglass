import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def interval_pos_key(upsample_position):
    yield upsample_position


def test_interval_position_info_insert(common_position, interval_pos_key):
    assert common_position.IntervalPositionInfo & interval_pos_key


@pytest.fixture(scope="session")
def upsample_position_error(
    upsample_position,
    default_interval_pos_param_key,
    pos_info_param,
    common,
    common_position,
    teardown,
    interval_keys,
):
    interval_key = interval_keys[0]
    params = (pos_info_param & default_interval_pos_param_key).fetch1()
    upsample_param_key = {"position_info_param_name": "upsampled error"}
    pos_info_param.insert1(
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
    common_position.IntervalPositionInfoSelection.insert1(
        interval_pos_key, skip_duplicates=not teardown
    )
    yield interval_pos_key

    (common_position.IntervalPositionInfoSelection & interval_pos_key).delete(
        safemode=False
    )


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
        "head_orientation": 0,
        "head_position_x": 222.5,
        "head_position_y": 283.5,
        "head_speed": 1.2245733375331014,
        "head_velocity_x": -0.865904111029899,
        "head_velocity_y": 0.8551239752505465,
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


@pytest.fixture(scope="session")
def position_video(common_position):
    yield common_position.PositionVideo()


def test_position_video(position_video, upsample_position):
    _ = position_video.populate()
    assert len(position_video) == 2, "Failed to populate PositionVideo table."


def test_convert_to_pixels():
    from spyglass.utils.position import convert_to_pixels

    data = np.array([[2, 4], [6, 8]])
    expect = np.array([[1, 2], [3, 4]])
    output = convert_to_pixels(data, "junk", 2)

    assert np.array_equal(output, expect), "Failed to convert to pixels."


@pytest.fixture(scope="session")
def rename_default_cols(common_position):
    yield common_position._fix_col_names, ["xloc", "yloc", "xloc2", "yloc2"]


@pytest.mark.parametrize(
    "col_type, cols",
    [
        ("DEFAULT_COLS", ["xloc", "yloc", "xloc2", "yloc2"]),
        ("ONE_IDX_COLS", ["xloc1", "yloc1", "xloc2", "yloc2"]),
        ("ZERO_IDX_COLS", ["xloc0", "yloc0", "xloc1", "yloc1"]),
    ],
)
def test_rename_columns(rename_default_cols, col_type, cols):
    _fix_col_names, defaults = rename_default_cols
    df = pd.DataFrame([range(len(cols) + 1)], columns=["junk"] + cols)
    result = _fix_col_names(df).columns.tolist()

    assert result == defaults, f"_fix_col_names failed to rename {col_type}."


def test_rename_three_d(rename_default_cols):
    _fix_col_names, _ = rename_default_cols
    three_d = ["junk", "x", "y", "z"]
    df = pd.DataFrame([range(4)], columns=three_d)
    result = _fix_col_names(df).columns.tolist()

    assert (
        result == three_d[1:]
    ), "_fix_col_names failed to rename THREE_D_COLS."


def test_rename_non_default_columns(monkeypatch, rename_default_cols):
    _fix_col_names, defaults = rename_default_cols
    df = pd.DataFrame([range(4)], columns=["a", "b", "c", "d"])

    # Monkeypatch the input function
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    result = _fix_col_names(df).columns.tolist()

    assert (
        result == defaults
    ), "_fix_col_names failed to rename non-default cols."


def test_rename_non_default_columns_err(monkeypatch, rename_default_cols):
    _fix_col_names, defaults = rename_default_cols
    df = pd.DataFrame([range(4)], columns=["a", "b", "c", "d"])

    monkeypatch.setattr("builtins.input", lambda _: "no")

    with pytest.raises(ValueError):
        _fix_col_names(df)
