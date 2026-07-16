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
import sys
import types
from pathlib import Path

import pytest

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
# continuation not supported (base + SLEAP)
# --------------------------------------------------------------------------


class TestContinuationUnsupported:
    """Base and SLEAP strategies reject continued training."""

    def test_sleap_continue_raises(self, ts):
        with pytest.raises(NotImplementedError, match="SLEAP"):
            ts.SLEAPStrategy().continue_training(
                {}, {}, "skel", {}, {}, model_instance=None
            )

    def test_base_message_format(self, ts):
        with pytest.raises(
            NotImplementedError, match="Continuation not supported"
        ):
            ts.SLEAPStrategy().continue_training(
                {}, {}, "skel", {}, {}, model_instance=None
            )


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
