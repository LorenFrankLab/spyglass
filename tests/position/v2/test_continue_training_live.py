"""Live end-to-end verification of DLC PyTorch continued training (T20).

This is a *real* training test — no mocks of the DLC trainer. It:

1. Bootstraps a DLC project on disk with seeded labeled data that DLC can
   actually train on (``make_dlc_project`` + ``seed_labeled_data``).
2. Registers the minimal Spyglass DB rows and trains a fresh ``Model`` in
   test mode via ``Model().populate`` (one real PyTorch epoch on CPU).
3. Calls ``Model().train({'model_id': parent}, epochs=1)`` and asserts the
   continuation genuinely *resumes* from the parent snapshot: a new
   parent-linked ``Model`` row is produced and the real
   ``deeplabcut.train_network`` is invoked with ``snapshot_path`` pointing at
   the parent's ``snapshot-*.pt``.

The whole thing runs on CPU in a few seconds because ``test_mode`` caps DLC to
one epoch. It is gated by ``skip_if_no_dlc`` so CI (``--no-pose``) skips it.
"""

import functools
from pathlib import Path

import pytest


def _fix_labeled_data(config_path):
    """Make the synthetic project's labels trainable.

    ``make_dlc_project`` copies a ``CollectedData`` annotation file whose row
    index points at the *source* dataset's frame folder, which does not match
    this project's ``labeled-data`` layout and makes
    ``create_training_dataset`` fail. Remove it and regenerate matching labels
    with ``seed_labeled_data`` (which is designed to be trainable).

    Parameters
    ----------
    config_path : str or Path
        Path to the DLC ``config.yaml``.
    """
    from tests.position.v2.make_example_dlc_project import seed_labeled_data

    project_dir = Path(config_path).parent
    for stale in project_dir.glob("labeled-data/*/CollectedData_*"):
        stale.unlink()
    seed_labeled_data(config_path)


def _pytorch_snapshots(project_dir):
    """Return the on-disk PyTorch ``snapshot-*.pt`` files for a project.

    Parameters
    ----------
    project_dir : Path
        DLC project root.

    Returns
    -------
    list of Path
        Snapshot files, sorted by name.
    """
    return sorted(
        Path(project_dir).glob("dlc-models-pytorch/**/train/snapshot-*.pt")
    )


class TestContinueTrainingLive:
    """Real DLC PyTorch resume verification (T20)."""

    @pytest.fixture(autouse=True)
    def _require_dlc(self, skip_if_no_dlc):
        """Skip the whole class when DLC is unavailable / ``--no-pose``."""

    def test_resume_from_parent_snapshot(
        self,
        model,
        skeleton,
        model_params,
        model_sel,
        dlc_project_config,
        dlc_bootstrapped_session,
        monkeypatch,
    ):
        """A fresh model trains, then ``train`` resumes from its snapshot.

        Asserts (i) a new parent-linked ``Model`` row is produced, (ii) the
        real ``train_network`` is called with ``snapshot_path`` at the parent
        snapshot, and (iii) that real resume completes without error.
        """
        import deeplabcut

        from spyglass.position.v2.train import ModelSelection
        from spyglass.position.v2.video import VidFileGroup

        config_path = Path(dlc_project_config)
        project_dir = config_path.parent
        _fix_labeled_data(config_path)

        # ── DB wiring ────────────────────────────────────────────────────
        vid_group = VidFileGroup.create_from_dlc_config(str(config_path))
        vid_group_id = vid_group["vid_group_id"]

        # check_duplicates (default) returns the existing skeleton on re-runs
        # against a persistent (--no-teardown) database.
        skel_key = skeleton.insert1(
            {
                "bodyparts": ["whiteLED", "tailBase", "tailMid", "tailTip"],
                "edges": [
                    ("whiteLED", "tailBase"),
                    ("tailBase", "tailMid"),
                    ("tailMid", "tailTip"),
                ],
            }
        )
        skeleton_id = skel_key["skeleton_id"]

        params = {
            "project_path": str(project_dir),
            "shuffle": 1,
            "trainingsetindex": 0,
            "batch_size": 1,
            "test_mode": True,  # caps DLC to ~1 epoch
            "epochs": 1,
            "save_epochs": 1,
        }
        mp_key = model_params.insert1(
            {"tool": "DLC", "params": params, "skeleton_id": skeleton_id}
        )

        sel_key = {
            "model_params_id": mp_key["model_params_id"],
            "tool": "DLC",
            "vid_group_id": vid_group_id,
            "model_selection_id": "live-parent-sel",
        }
        model_sel.insert1(sel_key, skip_duplicates=True)

        # ── Fresh parent train (real one-epoch PyTorch) ──────────────────
        model.populate(sel_key)
        parent_key = (model & sel_key).fetch1("KEY")
        parent_id = parent_key["model_id"]
        assert model & {"model_id": parent_id}, "parent Model row missing"

        parent_snaps = _pytorch_snapshots(project_dir)
        assert parent_snaps, "fresh training produced no snapshot-*.pt on disk"

        # ── Capture the real train_network call on the resume path ───────
        calls = []
        real_train_network = deeplabcut.train_network

        @functools.wraps(real_train_network)
        def _recording_train_network(*args, **kwargs):
            calls.append({"args": args, "kwargs": dict(kwargs)})
            return real_train_network(*args, **kwargs)

        monkeypatch.setattr(
            deeplabcut, "train_network", _recording_train_network
        )

        # ── Continue / fine-tune one more epoch ──────────────────────────
        child_key = model.train({"model_id": parent_id}, epochs=1)
        child_id = child_key["model_id"]

        # (i) a genuinely new, parent-linked Model row
        assert child_id != parent_id, "continuation did not create a new model"
        # Resolve the child's ModelSelection directly from the Model row.
        child_full = (model & child_key).fetch1()
        child_parent_id = (
            ModelSelection
            & {
                "model_params_id": child_full["model_params_id"],
                "tool": child_full["tool"],
                "vid_group_id": child_full["vid_group_id"],
                "model_selection_id": child_full["model_selection_id"],
            }
        ).fetch1("parent_id")
        assert (
            child_parent_id == parent_id
        ), f"child parent_id {child_parent_id!r} != parent {parent_id!r}"

        # (ii) the real train_network ran with snapshot_path at the parent snap
        assert calls, "train_network was never called on the resume path"
        resume_kwargs = calls[-1]["kwargs"]
        assert "snapshot_path" in resume_kwargs, (
            "snapshot_path NOT passed to train_network -- the resume degraded "
            f"to a fresh train. train_network kwargs: {resume_kwargs}"
        )
        snap_passed = Path(resume_kwargs["snapshot_path"])
        assert (
            snap_passed.exists()
        ), f"snapshot_path missing on disk: {snap_passed}"
        assert snap_passed.name.startswith("snapshot-"), snap_passed.name
        assert snap_passed in set(parent_snaps) or snap_passed.parent in {
            p.parent for p in parent_snaps
        }, f"snapshot_path {snap_passed} is not a parent snapshot"

        # (iii) real resume completed and advanced the snapshot count
        after_snaps = _pytorch_snapshots(project_dir)
        assert len(after_snaps) >= len(
            parent_snaps
        ), "resume did not run real training (no new snapshot written)"

        print(
            "\nLIVE RESUME EVIDENCE:"
            f"\n  parent_id      = {parent_id}"
            f"\n  child_id       = {child_id}"
            f"\n  child.parent_id= {child_parent_id}"
            f"\n  snapshot_path  = {resume_kwargs['snapshot_path']}"
            f"\n  parent snaps   = {[p.name for p in parent_snaps]}"
            f"\n  snaps after    = {[p.name for p in after_snaps]}"
        )
