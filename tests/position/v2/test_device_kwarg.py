"""Unit tests for the runtime ``device`` kwarg on ``make()``.

``device`` is a *runtime* selection threaded through
``populate(..., make_kwargs={"device": ...})`` -> ``make(key, device=...)``,
never a stored/hashed parameter. These tests drive ``Model.make`` and
``PoseEstim.make`` with heavily mocked table access (no live DB) and assert
that a provided ``device`` is injected into the params handed downstream --
to ``strategy.train_model`` for training and to ``run_inference`` for
inference -- while ``device=None`` leaves those params untouched.
"""

from unittest.mock import MagicMock, patch

import pytest


class _Abort(Exception):
    """Sentinel to stop ``make`` after the downstream call is captured."""


def _run_model_make(device):
    """Drive ``Model.make`` with mocked tables; capture downstream params.

    Parameters
    ----------
    device : str or None
        Value passed as the runtime ``device`` kwarg.

    Returns
    -------
    dict
        The ``params`` dict handed to ``strategy.train_model``.
    """
    from spyglass.position.v2 import train as train_mod

    captured = {}

    def fake_train_model(key, params, skeleton_id, vid_group, sel_entry, inst):
        captured["params"] = params
        return {"model_id": "m1"}

    strategy = MagicMock()
    strategy.supports_training = True
    strategy.train_model.side_effect = fake_train_model

    sel_entry = {
        "model_params_id": "p1",
        "tool": "DLC",
        "vid_group_id": "v1",
        "parent_id": None,
    }
    params_entry = {
        "tool": "DLC",
        "params": {"shuffle": 1},
        "skeleton_id": "s1",
    }

    fake_self = MagicMock()

    with patch.multiple(
        train_mod,
        ModelSelection=MagicMock(),
        ModelParams=MagicMock(),
        VidFileGroup=MagicMock(),
        ToolStrategyFactory=MagicMock(),
    ):
        train_mod.ModelSelection.return_value.__and__.return_value.fetch1.return_value = (  # noqa: E501
            sel_entry
        )
        train_mod.ModelParams.return_value.__and__.return_value.fetch1.return_value = (  # noqa: E501
            params_entry
        )
        train_mod.VidFileGroup.return_value.__and__.return_value.fetch1.return_value = {  # noqa: E501
            "vid_group_id": "v1"
        }
        train_mod.ToolStrategyFactory.create_strategy.return_value = strategy

        # Tri-part make: `device` arrives via make_fetch (DataJoint routes
        # make_kwargs there) and is threaded into make_compute, which is
        # where the strategy params are assembled.
        key = {"model_id": "m1"}
        fetched = train_mod.Model.make_fetch(fake_self, key, device=device)
        train_mod.Model.make_compute(fake_self, key, *fetched)

    return captured["params"]


def _run_pose_estim_make(device):
    """Drive ``PoseEstim.make`` with mocked tables; capture inference params.

    Aborts (via ``_Abort``) inside the mocked ``run_inference`` once the
    forwarded kwargs are captured, so no output-file / NWB machinery runs.

    Parameters
    ----------
    device : str or None
        Value passed as the runtime ``device`` kwarg.

    Returns
    -------
    dict
        The keyword arguments forwarded to ``run_inference``.
    """
    from spyglass.position.v2 import estim as estim_mod

    captured = {}

    def fake_run_inference(
        model_key, videos, destfolder=None, model_info=None, tool=None, **kwargs
    ):
        # model_info/tool are the pre-fetched values make_fetch hands down so
        # run_inference performs no query of its own; only the inference
        # params (which carry `device`) are under test here.
        captured["kwargs"] = kwargs
        raise _Abort

    fake_self = MagicMock()
    fake_self.run_inference.side_effect = fake_run_inference
    fake_self._is_3d_mode.return_value = False

    def fetch_side(arg):
        return [{"vf": 1}] if arg == "KEY" else [0]

    with patch.multiple(
        estim_mod,
        PoseEstimSelection=MagicMock(),
        Model=MagicMock(),
        ModelParams=MagicMock(),
        PoseEstimParams=MagicMock(),
        VidFileGroup=MagicMock(),
        VideoFile=MagicMock(),
        BodyPart=MagicMock(),  # canon_map() now resolved in make_fetch
    ):
        estim_mod.PoseEstimSelection.__and__.return_value.fetch1.return_value = (  # noqa: E501
            "trigger",
            "",
        )
        estim_mod.Model.return_value.__and__.return_value.fetch1.return_value = {  # noqa: E501
            "model_params_id": "p1",
            "tool": "DLC",
        }
        estim_mod.ModelParams.return_value.__and__.return_value.fetch1.return_value = {  # noqa: E501
            "tool": "DLC"
        }
        estim_mod.PoseEstimParams.__and__.return_value.fetch1.return_value = {
            "batch_size": 8
        }
        estim_mod.VidFileGroup.File.__and__.return_value.fetch.side_effect = (
            fetch_side
        )
        estim_mod.VideoFile.__and__.return_value.fetch1.return_value = {
            "path": "/tmp/x.mp4"
        }
        estim_mod.VidFileGroup.return_value.get_nwb_file.return_value = {
            "nwb_file_name": "n.nwb"
        }

        key = {
            "model_id": "m1",
            "vid_group_id": "v1",
            "pose_estim_params_id": "default",
        }
        # Tri-part make: `device` arrives via make_fetch and is folded into
        # inference_params, which make_compute forwards to run_inference.
        fetched = estim_mod.PoseEstim.make_fetch(fake_self, key, device=device)
        with pytest.raises(_Abort):
            estim_mod.PoseEstim.make_compute(fake_self, key, *fetched)

    return captured["kwargs"]


def test_model_make_injects_device():
    """``device`` given -> injected into params passed to ``train_model``."""
    params = _run_model_make(device="cuda:1")
    assert params.get("device") == "cuda:1"
    # Non-device params are preserved.
    assert params.get("shuffle") == 1


def test_model_make_default_no_device():
    """``device=None`` -> params handed downstream are unchanged."""
    params = _run_model_make(device=None)
    assert "device" not in params
    assert params == {"shuffle": 1}


def test_pose_estim_make_injects_device():
    """``device`` given -> forwarded to ``run_inference`` kwargs."""
    kwargs = _run_pose_estim_make(device="cuda:1")
    assert kwargs.get("device") == "cuda:1"
    # Stored inference params still forwarded alongside the runtime device.
    assert kwargs.get("batch_size") == 8


def test_pose_estim_make_default_no_device():
    """``device=None`` -> ``run_inference`` kwargs carry no device."""
    kwargs = _run_pose_estim_make(device=None)
    assert "device" not in kwargs
    assert kwargs == {"batch_size": 8}
