import pytest


@pytest.fixture(scope="session")
def decode_merge_key(decode_merge, clusterless_pop):
    _ = clusterless_pop  # ensure population is created
    return decode_merge.fetch("KEY")[0]


@pytest.fixture(scope="session")
def decode_merge_class():
    from spyglass.decoding.decoding_merge import DecodingOutput

    return DecodingOutput


@pytest.fixture(scope="session")
def decode_merge_restr(decode_merge, decode_merge_key):
    return decode_merge & decode_merge_key


def test_decode_merge_fetch_results(
    decode_merge_restr, decode_merge_key, result_coordinates
):

    results = decode_merge_restr.fetch_results(decode_merge_key)
    assert result_coordinates.issubset(
        results.coords._names
    ), "Incorrect coordinates in results"


def test_decode_merge_fetch_model(decode_merge_restr, decode_merge_key):
    # The decoding suite mocks the full decode pipeline (see
    # tests/decoding/conftest.py): ``_save_decoder_results`` /
    # ``load_model`` are patched so the populated row stores and returns the
    # in-memory ``FakeClassifier`` stand-in rather than a real fitted
    # ``non_local_detector`` ``ClusterlessDetector`` (which would require the
    # heavy real decode never run in CI). So we verify ``fetch_model``
    # round-trips the saved classifier and exposes the model interface the
    # downstream consumers use, not its concrete upstream type.
    from tests.decoding.conftest import FakeClassifier

    model = decode_merge_restr.fetch_model(decode_merge_key)
    assert isinstance(
        model, FakeClassifier
    ), "fetch_model did not round-trip the saved (mocked) classifier"
    assert hasattr(model, "initial_conditions_")
    assert hasattr(model, "discrete_state_transitions_")


def test_decode_merge_fetch_env(decode_merge_restr, decode_merge_key):
    from non_local_detector.environment import Environment

    env = decode_merge_restr.fetch_environments(decode_merge_key)[0]

    assert isinstance(env, Environment), "Fetched obj not Environment type"


def test_decode_merge_fetch_pos(decode_merge_restr, decode_merge_key):
    ret = decode_merge_restr.fetch_position_info(decode_merge_key)[0]
    cols = set(ret.columns)
    assert cols == {
        "position_x",
        "velocity_x",
        "orientation",
        "position_y",
        "speed",
        "velocity_y",
    }, "Incorrect columns in position info"


def test_decode_linear_position(decode_merge_restr, decode_merge_key):

    ret = decode_merge_restr.fetch_linear_position_info(decode_merge_key)
    cols = set(ret.columns)
    assert cols == {
        "projected_y_position",
        "speed",
        "velocity_x",
        "orientation",
        "linear_position",
        "position_x",
        "velocity_y",
        "position_y",
        "track_segment_id",
        "projected_x_position",
    }


# @pytest.mark.skip("Errors on unpacking mult from fetch")
def test_decode_view(decode_merge_restr, decode_merge_key):
    ret = decode_merge_restr.create_decoding_view(decode_merge_key)
    assert ret is not None, "Failed to create decoding view"
