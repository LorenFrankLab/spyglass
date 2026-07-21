"""Live end-to-end test for SLEAP continued training (weight resume).

Exercises the real ``sleap-train --base_checkpoint`` resume path that
``SLEAPStrategy.continue_training`` drives: a parent checkpoint's weights are
loaded into a fresh trainer and training runs for a config-controlled number
of epochs.

Unlike the unit tests (which mock ``subprocess``), this test spawns a genuine
``sleap-train`` process and asserts on its output and on-disk artifacts.  It is
guarded by the ``skip_if_no_sleap`` fixture, so it *skips* wherever SLEAP is
absent (e.g. the ``pv2`` env / CI ``--no-pose``) and *runs* where SLEAP lives
(the ``sl2`` env).

Fixture data (downloaded to the git-ignored ``tests/_data/sleap/`` by
``data_downloader.py``; regenerate with the scripts in ``maintenance_scripts/``):
  - ``small_robot_labeled.slp`` -- 2 user-labeled frames with embedded images,
    single-instance skeleton A/B.  See
    ``maintenance_scripts/make_labeled_slp.py`` for provenance.
  - ``model/`` -- a minimal single-instance UNet parent model
    (``best_model.h5`` + ``training_config.json``) used as the base checkpoint.
"""

import json
import subprocess
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[2] / "_data" / "sleap"
LABELS = DATA_DIR / "small_robot_labeled.slp"
MODEL_DIR = DATA_DIR / "model"
BASE_CONFIG = MODEL_DIR / "training_config.json"
BASE_CKPT = MODEL_DIR / "best_model.h5"


class _StubModel:
    """Minimal stand-in for a Model table instance (logging only)."""

    def _info_msg(self, msg):  # noqa: D401 - trivial logger
        print(f"INFO: {msg}")

    def _warn_msg(self, msg):  # noqa: D401 - trivial logger
        print(f"WARN: {msg}")


def _make_base_config(tmp_path: Path) -> Path:
    """Copy the parent config, pointing runs at *tmp_path* and shrinking work.

    ``_write_epochs_config`` preserves every field except the epoch count, so
    the ``runs_folder`` and the (tiny) per-epoch batch budget must be set here
    on the base config it copies from.  Shrinking keeps the CPU train to a
    couple of batches without changing what is being verified.
    """
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg.setdefault("outputs", {})["runs_folder"] = str(tmp_path / "models")
    cfg["outputs"]["save_outputs"] = True
    opt = cfg.setdefault("optimization", {})
    opt["batch_size"] = 1
    opt["batches_per_epoch"] = 2
    opt["min_batches_per_epoch"] = 1
    opt["val_batches_per_epoch"] = 1
    opt["min_val_batches_per_epoch"] = 1
    out = tmp_path / "base_config.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


class TestLiveSLEAPResume:
    """Real ``sleap-train --base_checkpoint`` resume.  Skipped without SLEAP."""

    @pytest.fixture(autouse=True)
    def skip_if_no_sleap(self, skip_if_no_sleap):  # noqa: F811
        pass

    @pytest.fixture
    def sleap_cli(self):
        import shutil

        cli = shutil.which("sleap-train")
        if cli is None:
            pytest.skip("sleap-train CLI not found in PATH")
        return cli

    @pytest.fixture
    def fixtures_present(self):
        for path in (LABELS, BASE_CONFIG, BASE_CKPT):
            if not path.exists():
                pytest.skip(f"Required SLEAP fixture missing: {path}")

    def test_resume_from_base_checkpoint(
        self, sleap_cli, fixtures_present, tmp_path
    ):
        """A parent checkpoint's weights load and 1 config epoch runs.

        Drives the actual strategy helpers -- ``_resolve_parent_checkpoint``,
        ``_write_epochs_config`` and ``_build_train_cmd`` -- then runs the
        command they produce and asserts on the live SLEAP output:

        1. the command completes (returncode 0);
        2. the base checkpoint's backbone + head weights are loaded (logged);
        3. the epochs override is honoured (``max_epochs=1`` reached);
        4. a fresh ``best.ckpt`` is written to the run directory.
        """
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        strategy = SLEAPStrategy()
        stub = _StubModel()

        # 1. Resolve the parent checkpoint -- must be the .h5 *file*, not dir.
        ckpt = strategy._resolve_parent_checkpoint(MODEL_DIR)
        assert ckpt is not None, "no resumable checkpoint found in model dir"
        assert Path(ckpt).name == "best_model.h5"
        assert Path(ckpt).is_file()

        # 2. Write an epochs=1 override config (strategy's real writer).
        base_cfg = _make_base_config(tmp_path)
        override = strategy._write_epochs_config(base_cfg, 1, stub)
        assert override is not None, "epochs override config not written"
        override = Path(override)
        try:
            assert (
                json.loads(override.read_text())["optimization"]["epochs"] == 1
            )

            # 3. Assemble the command the strategy would run.
            run_name = f"resume_live_{tmp_path.name}"
            cmd = SLEAPStrategy._build_train_cmd(
                override,
                LABELS,
                run_name,
                output_dir="",
                base_checkpoint=ckpt,
            )
            assert "--base_checkpoint" in cmd
            assert str(ckpt) in cmd

            # Explicit val labels avoid the tiny-data train/val split
            # (2 frames -> 0 val -> ZeroDivisionError); --cpu forces CPU.
            cmd = cmd + ["--val_labels", str(LABELS), "--cpu"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        finally:
            override.unlink(missing_ok=True)

        out = (result.stdout or "") + (result.stderr or "")

        # 1. Completed cleanly.
        assert result.returncode == 0, f"sleap-train failed:\n{out}"

        # 2. Base weights were loaded from our checkpoint.
        assert "Loading backbone weights" in out, out[-3000:]
        assert "Loading head weights" in out, out[-3000:]
        assert str(ckpt) in out

        # 3. Epochs override honoured -- trainer stopped at max_epochs=1.
        assert "max_epochs=1" in out, out[-3000:]

        # 4. A fresh checkpoint was written for the resumed run.
        new_ckpt = tmp_path / "models" / run_name / "best.ckpt"
        assert new_ckpt.exists(), f"no best.ckpt at {new_ckpt}\n{out[-2000:]}"
        assert new_ckpt.stat().st_size > 0
