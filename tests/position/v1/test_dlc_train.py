import pytest

from tests.conftest import skip_if_no_pose


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


@skip_if_no_pose
def test_get_params(no_pose, verbose_context, dlc_training_params):
    params_tbl, _ = dlc_training_params
    with verbose_context:
        accepted_params = params_tbl.get_accepted_params()

    assert accepted_params is not None, "Failed to get accepted params"


class TestLatestSnapshotIndex:
    """0 reads as a real snapshot index, so it is test-mode only.

    DLC 3's PyTorch engine writes snapshot-*.pt, which the *index* glob does
    not match, so a production run that found nothing used to store 0.
    """

    def test_picks_most_recently_written(self, sgp, tmp_path):
        import os

        from spyglass.position.v1.position_dlc_training import (
            latest_snapshot_index,
        )

        for i, name in enumerate(["snapshot-50.index", "snapshot-10.index"]):
            p = tmp_path / name
            p.touch()
            os.utime(p, (1_600_000_000 + i, 1_600_000_000 + i))

        # snapshot-10 is newer, and "latest" here means most recent, not
        # largest index.
        assert latest_snapshot_index(tmp_path) == 10

    def test_missing_snapshot_raises_outside_test_mode(
        self, sgp, tmp_path, monkeypatch
    ):
        from spyglass.position.v1 import position_dlc_training as mod

        monkeypatch.setattr(mod, "test_mode", False)

        with pytest.raises(FileNotFoundError, match="snapshot-"):
            mod.latest_snapshot_index(tmp_path)

    def test_missing_snapshot_returns_zero_in_test_mode(self, sgp, tmp_path):
        from spyglass.position.v1 import position_dlc_training as mod

        monkeypatch_free = mod.test_mode  # suite already runs in test mode
        assert monkeypatch_free is True
        assert mod.latest_snapshot_index(tmp_path) == 0
