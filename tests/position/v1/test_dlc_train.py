import pytest

from tests.conftest import skip_if_no_dlc


def test_existing_params(
    verbose_context, dlc_training_params, training_params_key
):
    params_tbl, params_name = dlc_training_params

    _ = training_params_key  # Ensure populated
    params_query = params_tbl & {"dlc_training_params_name": params_name}
    assert params_query, "Existing params not found"

    with verbose_context:
        params_tbl.insert_new_params(
            paramset_name=params_name,
            params={
                "shuffle": 1,
                "trainingsetindex": 0,
                "net_type": "any",
                "gputouse": None,
            },
            skip_duplicates=False,
        )

    assert len(params_query) == 1, "Existing params duplicated"


def test_insert_params_error(dlc_training_params):
    params_tbl, _ = dlc_training_params
    with pytest.raises(ValueError):
        params_tbl.insert_new_params(
            paramset_name="test",
            params={"shuffle": 1},
        )


@skip_if_no_dlc
def test_get_params(no_dlc, verbose_context, dlc_training_params):
    params_tbl, _ = dlc_training_params
    with verbose_context:
        accepted_params = params_tbl.get_accepted_params()

    assert accepted_params is not None, "Failed to get accepted params"
