"""DB-free unit tests for continued-training strategy logic.

These tests exercise the tool-agnostic ``epochs`` normalization, the DLC
weight-resume kwarg building, the parent-snapshot degrade path, and the
``NotImplementedError`` raised for tools without continuation support (SLEAP
and the base strategy).

Importing ``spyglass.position`` triggers a DB connection at import, which is
unavailable in this environment. To keep these tests DB-free we load
``tool_strategies`` directly by stubbing its parent packages in ``sys.modules``
(bypassing the package ``__init__`` chain that connects to MySQL).
"""

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import yaml

_SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture(scope="module")
def ts():
    """Import ``tool_strategies`` without triggering a DB connection.

    If the real ``spyglass`` package is already imported (e.g. another test
    module ran first and connected to the DB), just import normally. Otherwise
    stub the parent packages so the ``__init__`` chain that connects to MySQL
    never runs — then tear the stubs down so a later *real* import of
    ``spyglass`` in the same session starts from a clean slate.
    """
    stub_mode = "spyglass" not in sys.modules
    if not stub_mode:  # real package present → import directly
        yield importlib.import_module("spyglass.position.utils.tool_strategies")
        return

    before = set(sys.modules)
    for name, rel in (
        ("spyglass", "spyglass"),
        ("spyglass.position", "spyglass/position"),
        ("spyglass.position.utils", "spyglass/position/utils"),
    ):
        mod = types.ModuleType(name)
        mod.__path__ = [str(_SRC / rel)]
        sys.modules[name] = mod
    try:
        yield importlib.import_module("spyglass.position.utils.tool_strategies")
    finally:
        # Remove every ``spyglass*`` entry we introduced so a later genuine
        # import re-runs the real package (and its DB setup) from scratch.
        for name in set(sys.modules) - before:
            if name == "spyglass" or name.startswith("spyglass."):
                sys.modules.pop(name, None)


class FakeFS:
    """Minimal filesystem double exposing only what the tests need."""

    def __init__(self, config=None, mtimes=None):
        self._config = config or {}
        self._mtimes = mtimes or {}

    def read_yaml(self, path):
        return dict(self._config)

    def getmtime(self, path):
        return self._mtimes.get(str(path), 0.0)

    def exists(self, path):
        return True


# --------------------------------------------------------------------------
# epochs -> native knob
# --------------------------------------------------------------------------


class TestApplyEpochs:
    """``apply_epochs`` maps the generic budget to each tool's native knob."""

    def test_dlc_pytorch_default(self, ts):
        """No engine key defaults to PyTorch, which consumes ``epochs``."""
        out = ts.DLCStrategy().apply_epochs({"shuffle": 1}, 25, config={})
        assert out["epochs"] == 25
        assert "maxiters" not in out

    def test_dlc_pytorch_explicit(self, ts):
        out = ts.DLCStrategy().apply_epochs(
            {}, 10, config={"engine": "pytorch"}
        )
        assert out == {"epochs": 10}

    def test_dlc_tensorflow(self, ts):
        """TF engine consumes ``maxiters``, not ``epochs``."""
        out = ts.DLCStrategy().apply_epochs(
            {}, 5000, config={"engine": "tensorflow"}
        )
        assert out["maxiters"] == 5000
        assert "epochs" not in out

    def test_dlc_tf_legacy_alias(self, ts):
        out = ts.DLCStrategy().apply_epochs({}, 42, config={"engine": "tf"})
        assert out == {"maxiters": 42}

    def test_dlc_none_is_noop(self, ts):
        src = {"shuffle": 1}
        out = ts.DLCStrategy().apply_epochs(src, None, config={})
        assert out is src  # unchanged, same object

    def test_dlc_does_not_mutate_input(self, ts):
        src = {"shuffle": 1}
        ts.DLCStrategy().apply_epochs(src, 7, config={})
        assert "epochs" not in src

    def test_sleap_sets_max_epochs_and_aliases(self, ts):
        out = ts.SLEAPStrategy().apply_epochs({"batch_size": 4}, 200)
        assert out["max_epochs"] == 200
        # aliases expanded for downstream config assembly
        assert out["epochs"] == 200
        assert out["training_epochs"] == 200

    def test_sleap_none_is_noop(self, ts):
        src = {"batch_size": 4}
        out = ts.SLEAPStrategy().apply_epochs(src, None)
        assert out is src

    def test_base_apply_epochs_is_noop(self, ts):
        """Base implementation returns params unchanged (via SLEAP super)."""
        # SLEAP overrides; exercise the base no-op through a bound super call.
        base = ts.PoseToolStrategy.apply_epochs
        out = base(ts.SLEAPStrategy(), {"a": 1}, 5)
        assert out == {"a": 1}

    def test_dlc_accepts_epochs_params(self, ts):
        accepted = ts.DLCStrategy().get_accepted_params()
        assert "epochs" in accepted
        assert "save_epochs" in accepted


# --------------------------------------------------------------------------
# continuation not supported (base strategy only)
# --------------------------------------------------------------------------


class TestContinuationUnsupported:
    """The base strategy rejects continued training with a clear message.

    SLEAP now supports continuation (see ``TestSLEAPContinue`` below), so the
    unsupported behavior is exercised through the *base* ``PoseToolStrategy``
    implementation rather than a concrete tool. Invoking it unbound on a SLEAP
    instance reaches the base ``raise`` without SLEAP's override (mirroring the
    ``test_base_apply_epochs_is_noop`` pattern).
    """

    def test_base_raises_not_implemented(self, ts):
        with pytest.raises(NotImplementedError, match="not supported"):
            ts.PoseToolStrategy.continue_training(
                ts.SLEAPStrategy(), {}, {}, "skel", {}, {}, model_instance=None
            )

    def test_base_message_names_tool(self, ts):
        # The base message interpolates the concrete tool_name.
        with pytest.raises(NotImplementedError, match="SLEAP"):
            ts.PoseToolStrategy.continue_training(
                ts.SLEAPStrategy(), {}, {}, "skel", {}, {}, model_instance=None
            )


# --------------------------------------------------------------------------
# SLEAP continuation (weight-resume via --base_checkpoint)
# --------------------------------------------------------------------------


def _make_sleap(ts, parent_dir, recording=False):
    """Build a SLEAPStrategy with parent-dir resolution stubbed.

    Parameters
    ----------
    ts : module
        The imported ``tool_strategies`` module.
    parent_dir : Path or None
        Directory returned by ``_resolve_parent_model_dir``.
    recording : bool
        When True, ``train_model`` records calls (degrade path) rather than
        shelling out to ``sleap-train``.
    """
    base = ts.SLEAPStrategy
    cls = (
        type("RecordingSLEAP", (_RecordingStrategy, base), {})
        if recording
        else base
    )
    strategy = cls()
    strategy._resolve_parent_model_dir = lambda sel, mi: parent_dir
    return strategy


def _sleap_inputs(tmp_path, config_text=None, config_suffix=".json"):
    """Create labels + optional config files; return (sel_entry, params)."""
    labels = tmp_path / "labels.slp"
    labels.write_text("")
    params = {"run_name": "child_run", "output_dir": str(tmp_path / "models")}
    if config_text is not None:
        config = tmp_path / f"config{config_suffix}"
        config.write_text(config_text)
        params["initial_config"] = str(config)
    sel_entry = {
        "parent_id": "parent_model",
        "training_labels_path": str(labels),
    }
    return sel_entry, params


class TestSLEAPResolveCheckpoint:
    """``_resolve_parent_checkpoint`` finds the resumable checkpoint file."""

    def test_prefers_best_ckpt(self, ts, tmp_path):
        (tmp_path / "best.ckpt").write_text("")
        (tmp_path / "best_model.h5").write_text("")
        ckpt = ts.SLEAPStrategy()._resolve_parent_checkpoint(tmp_path)
        assert ckpt == tmp_path / "best.ckpt"

    def test_legacy_h5_fallback(self, ts, tmp_path):
        (tmp_path / "best_model.h5").write_text("")
        ckpt = ts.SLEAPStrategy()._resolve_parent_checkpoint(tmp_path)
        assert ckpt == tmp_path / "best_model.h5"

    def test_searches_one_level_down(self, ts, tmp_path):
        run = tmp_path / "child_run"
        run.mkdir()
        (run / "best.ckpt").write_text("")
        ckpt = ts.SLEAPStrategy()._resolve_parent_checkpoint(tmp_path)
        assert ckpt == run / "best.ckpt"

    def test_none_when_absent(self, ts, tmp_path):
        assert ts.SLEAPStrategy()._resolve_parent_checkpoint(tmp_path) is None

    def test_none_when_dir_missing(self, ts, tmp_path):
        missing = tmp_path / "nope"
        assert ts.SLEAPStrategy()._resolve_parent_checkpoint(missing) is None

    def test_one_level_down_returns_newest(self, ts, tmp_path):
        """Multiple one-level-down matches → newest-by-mtime wins."""
        import os

        older = tmp_path / "run_old"
        newer = tmp_path / "run_new"
        older.mkdir()
        newer.mkdir()
        (older / "best.ckpt").write_text("")
        (newer / "best.ckpt").write_text("")
        os.utime(older / "best.ckpt", (1000, 1000))
        os.utime(newer / "best.ckpt", (2000, 2000))  # more recent
        ckpt = ts.SLEAPStrategy()._resolve_parent_checkpoint(tmp_path)
        assert ckpt == newer / "best.ckpt"


class TestSLEAPContinue:
    """continue_training wires --base_checkpoint or degrades to fresh."""

    @patch("subprocess.run")
    def test_wires_base_checkpoint(self, mock_run, ts, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        ckpt = parent / "best.ckpt"
        ckpt.write_text("")
        strategy = _make_sleap(ts, parent)
        sel_entry, params = _sleap_inputs(tmp_path, config_text="{}")
        model_dir = tmp_path / "models" / "child_run"

        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            result = strategy.continue_training(
                {}, params, "skel", {}, sel_entry, MagicMock()
            )

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs == {"check": True}
        cmd = mock_run.call_args[0][0]
        assert "--base_checkpoint" in cmd
        assert str(ckpt) in cmd
        # checkpoint value immediately follows the flag
        assert cmd[cmd.index("--base_checkpoint") + 1] == str(ckpt)
        assert result["model_path"] == str(model_dir)

    def test_missing_checkpoint_degrades_to_fresh(self, ts, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()  # no checkpoint inside
        strategy = _make_sleap(ts, parent, recording=True)
        sel_entry, params = _sleap_inputs(tmp_path, config_text="{}")
        msg = MagicMock()

        result = strategy.continue_training(
            {}, params, "skel", {}, sel_entry, msg
        )

        # Fresh (recorded) train_model was called; no base_checkpoint wiring.
        assert len(strategy.train_calls) == 1
        assert result["model_id"] == "child"
        # degrade path warns, naming the parent dir and the fresh fallback
        msg._warn_msg.assert_called_once()
        (warn_arg,) = msg._warn_msg.call_args[0]
        assert str(parent) in warn_arg
        assert "training fresh" in warn_arg

    @patch("subprocess.run")
    def test_epochs_rewrites_yaml_config(self, mock_run, ts, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "best.ckpt").write_text("")
        strategy = _make_sleap(ts, parent)
        cfg_text = yaml.safe_dump({"trainer_config": {"max_epochs": 200}})
        sel_entry, params = _sleap_inputs(
            tmp_path, config_text=cfg_text, config_suffix=".yaml"
        )
        model_dir = tmp_path / "models" / "child_run"

        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            strategy.continue_training(
                {}, params, "skel", {}, sel_entry, MagicMock(), epochs=50
            )

        cmd = mock_run.call_args[0][0]
        used_config = Path(cmd[1])  # config is first positional after program
        assert used_config != Path(params["initial_config"])  # a rewritten copy
        written = yaml.safe_load(used_config.read_text())
        assert written["trainer_config"]["max_epochs"] == 50

    @patch("subprocess.run")
    def test_epochs_rewrites_json_config(self, mock_run, ts, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "best.ckpt").write_text("")
        strategy = _make_sleap(ts, parent)
        sel_entry, params = _sleap_inputs(
            tmp_path, config_text=json.dumps({"optimization": {"epochs": 100}})
        )
        model_dir = tmp_path / "models" / "child_run"

        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            strategy.continue_training(
                {}, params, "skel", {}, sel_entry, MagicMock(), epochs=25
            )

        cmd = mock_run.call_args[0][0]
        written = json.loads(Path(cmd[1]).read_text())
        assert written["optimization"]["epochs"] == 25

    @patch("subprocess.run")
    def test_epochs_from_params_max_epochs(self, mock_run, ts, tmp_path):
        # Model.train(epochs=N) lands as params['max_epochs'] via apply_epochs;
        # continue_training must honor it even without the explicit kwarg.
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "best.ckpt").write_text("")
        strategy = _make_sleap(ts, parent)
        sel_entry, params = _sleap_inputs(
            tmp_path, config_text=json.dumps({"optimization": {"epochs": 1}})
        )
        params["max_epochs"] = 77
        model_dir = tmp_path / "models" / "child_run"

        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            strategy.continue_training(
                {}, params, "skel", {}, sel_entry, MagicMock()
            )

        cmd = mock_run.call_args[0][0]
        written = json.loads(Path(cmd[1]).read_text())
        assert written["optimization"]["epochs"] == 77

    @patch("subprocess.run")
    def test_epochs_without_config_warns(self, mock_run, ts, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "best.ckpt").write_text("")
        strategy = _make_sleap(ts, parent)
        sel_entry, params = _sleap_inputs(tmp_path)  # no initial_config
        model_dir = tmp_path / "models" / "child_run"
        msg = MagicMock()

        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            strategy.continue_training(
                {}, params, "skel", {}, sel_entry, msg, epochs=40
            )

        # epochs not silently dropped: no initial_config to carry the budget,
        # so _write_epochs_config warns and returns None.
        msg._warn_msg.assert_called_once()
        (warn_arg,) = msg._warn_msg.call_args[0]
        assert "epochs budget" in warn_arg
        assert "Ignoring epochs" in warn_arg
        cmd = mock_run.call_args[0][0]
        assert "--base_checkpoint" in cmd  # resume still happens


# --------------------------------------------------------------------------
# DLC weight-resume kwarg building
# --------------------------------------------------------------------------


class TestBuildResumeKwargs:
    """``_build_resume_kwargs`` wires the parent snapshot (PyTorch-only)."""

    def test_pytorch_snapshot_path(self, ts):
        kw = ts.DLCStrategy._build_resume_kwargs(
            snapshot_path="/p/snapshot-050.pt", epochs=30
        )
        assert kw == {
            "snapshot_path": "/p/snapshot-050.pt",
            "epochs": 30,
        }

    def test_no_epochs_only_weight_path(self, ts):
        kw = ts.DLCStrategy._build_resume_kwargs(snapshot_path="/p/s.pt")
        assert kw == {"snapshot_path": "/p/s.pt"}


# --------------------------------------------------------------------------
# DLC continue_training routing (resumable vs degrade-to-fresh)
# --------------------------------------------------------------------------


class _RecordingStrategy:
    """Mixin recording train_model calls without running real training."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.train_calls = []

    def train_model(
        self, key, params, skeleton_id, vid_group, sel_entry, model_instance
    ):
        self.train_calls.append(dict(params))
        return {"model_id": "child", "params": dict(params)}


class _Msg:
    """No-op logger stand-in for model_instance."""

    def _info_msg(self, *a, **k):
        pass

    def _warn_msg(self, *a, **k):
        pass


def _make_dlc(ts, config, snapshot):
    """Build a DLCStrategy whose snapshot resolution is stubbed."""

    cls = type("RecordingDLC", (_RecordingStrategy, ts.DLCStrategy), {})
    strategy = cls(filesystem=FakeFS(config=config))
    strategy._resolve_parent_snapshot = lambda cfg, shuffle=None: snapshot
    return strategy


class TestDLCContinueRouting:
    """continue_training wires the snapshot or degrades to a fresh train."""

    def test_resumable_pytorch_wires_snapshot(self, ts):
        strategy = _make_dlc(
            ts, {"engine": "pytorch"}, Path("/proj/train/snapshot-050.pt")
        )
        params = {"project_path": "/proj", "shuffle": 1, "epochs": 30}
        result = strategy.continue_training(
            {}, params, "skel", {}, {"parent_id": "p"}, _Msg()
        )
        assert len(strategy.train_calls) == 1
        passed = strategy.train_calls[0]
        assert passed["snapshot_path"] == str(
            Path("/proj/train/snapshot-050.pt")
        )
        assert passed["epochs"] == 30  # length knob preserved from params
        assert result["model_id"] == "child"

    def test_tensorflow_raises_not_implemented(self, ts):
        # TF weight-resume can't be wired (init_weights is not a train_network
        # kwarg), so continuation on a TF project raises rather than no-op.
        strategy = _make_dlc(
            ts, {"engine": "tensorflow"}, Path("/proj/train/snapshot-1000")
        )
        params = {"project_path": "/proj", "shuffle": 1, "maxiters": 5000}
        with pytest.raises(NotImplementedError, match="TensorFlow"):
            strategy.continue_training(
                {}, params, "skel", {}, {"parent_id": "p"}, _Msg()
            )
        assert strategy.train_calls == []

    def test_no_snapshot_degrades_to_fresh(self, ts):
        strategy = _make_dlc(ts, {"engine": "pytorch"}, None)
        params = {"project_path": "/proj", "shuffle": 2, "epochs": 30}
        strategy.continue_training(
            {}, params, "skel", {}, {"parent_id": "p"}, _Msg()
        )
        passed = strategy.train_calls[0]
        # Fresh train: parent lineage kept elsewhere, but no snapshot wiring.
        assert "snapshot_path" not in passed
        assert "init_weights" not in passed
        assert passed["epochs"] == 30

    def test_explicit_epochs_kwarg_honored(self, ts):
        # A direct continue_training(..., epochs=N) call must reach the trainer
        # even when params carries no length knob (the make() path relies on
        # params['epochs'] instead — see test_resumable_pytorch_wires_snapshot).
        strategy = _make_dlc(
            ts, {"engine": "pytorch"}, Path("/proj/train/snapshot-050.pt")
        )
        params = {"project_path": "/proj", "shuffle": 1}  # no epochs
        strategy.continue_training(
            {}, params, "skel", {}, {"parent_id": "p"}, _Msg(), epochs=15
        )
        passed = strategy.train_calls[0]
        assert passed["snapshot_path"] == str(
            Path("/proj/train/snapshot-050.pt")
        )
        assert passed["epochs"] == 15

    def test_explicit_epochs_kwarg_overrides_params(self, ts):
        # Explicit kwarg wins over the params-baked length knob.
        strategy = _make_dlc(
            ts, {"engine": "pytorch"}, Path("/proj/train/snapshot-050.pt")
        )
        params = {"project_path": "/proj", "shuffle": 1, "epochs": 30}
        strategy.continue_training(
            {}, params, "skel", {}, {"parent_id": "p"}, _Msg(), epochs=99
        )
        assert strategy.train_calls[0]["epochs"] == 99
