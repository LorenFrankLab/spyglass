"""Example tests demonstrating dependency injection without mock.patch.

These tests show how the SOLID refactor enables clean testing by injecting
test doubles rather than patching global imports.
"""

import pandas as pd

# Test stubs - import these instead of real dependencies in tests
from tests.utils.test_stubs import (
    StubFileSystem,
    StubInferenceRunner,
    StubNWBBuilder,
)


def test_dlc_strategy_with_filesystem_injection():
    """Real DLCStrategy scan routes through an injected stub filesystem.

    ``get_latest_model_info`` is real DLCStrategy code that probes the project
    tree via ``self._fs``. With a stub filesystem where the project resolves but
    holds no ``dlc-models*/iteration-*`` dirs, the real scan yields ``{}`` and
    never touches the real disk.
    """
    from spyglass.position.utils.tool_strategies import DLCStrategy

    # Project dir resolves (its config.yaml is registered) but has no trained
    # model directories.
    stub_fs = StubFileSystem(
        mock_yaml_data={
            "/fake/project/config.yaml": {"project_path": "/fake/project"},
        },
    )

    strategy = DLCStrategy(filesystem=stub_fs)

    # Real code output given the stub input: empty (no trained models).
    assert (
        strategy.get_latest_model_info({"project_path": "/fake/project"}) == {}
    )

    # The real scan probed the injected fs (never the real disk).
    probed = {c["path"] for c in stub_fs.calls if c["method"] == "exists"}
    assert "/fake/project" in probed
    assert any(p.endswith("dlc-models-pytorch") for p in probed)


def test_pose_estim_with_injection():
    """Real PoseEstim DI seam resolves injected stub runner/builder classes.

    The stubs are injected onto the real ``PoseEstim`` (class attrs restored
    afterwards) and the real ``_get_*_cls`` accessors — the seam ``make()``
    relies on — must return them. The resolved doubles are then driven to
    confirm they honour the runner/builder contract the pipeline expects.
    """
    from spyglass.position.v2.estim import PoseEstim

    saved_runner = PoseEstim._inference_runner_cls
    saved_builder = PoseEstim._nwb_builder_cls
    try:
        PoseEstim._inference_runner_cls = StubInferenceRunner
        PoseEstim._nwb_builder_cls = StubNWBBuilder

        # Real accessor code returns the injected classes (not the defaults).
        assert PoseEstim._get_inference_runner_cls() is StubInferenceRunner
        assert PoseEstim._get_nwb_builder_cls() is StubNWBBuilder

        runner_cls = PoseEstim._get_inference_runner_cls()
        builder_cls = PoseEstim._get_nwb_builder_cls()
    finally:
        PoseEstim._inference_runner_cls = saved_runner
        PoseEstim._nwb_builder_cls = saved_builder

    mock_pose_df = pd.DataFrame(
        {
            ("DLC_resnet50", "nose", "x"): [100.0, 101.0],
            ("DLC_resnet50", "nose", "y"): [200.0, 201.0],
            ("DLC_resnet50", "nose", "likelihood"): [0.95, 0.95],
        }
    )

    runner = runner_cls(mock_result=mock_pose_df)
    builder = builder_cls()

    result = runner.run_inference(
        model_info={"model_path": "/fake/model.yaml"},
        video_path="/fake/video.mp4",
    )

    assert isinstance(result, pd.DataFrame)  # guard alongside value
    assert result.equals(mock_pose_df)

    pose_est, skeleton = builder.build_pose_estimation(
        pose_df=result,
        bodyparts=["nose"],
        scorer="DLC_resnet50",
        model_id="test_model",
        skeleton_edges=[],
    )

    # Builder contract: a (pose_estimation, skeleton) pair carrying the
    # model id and body-part count.
    assert pose_est == "MockPoseEstimation(test_model)"
    assert skeleton == "MockSkeleton(1_bodyparts)"

    assert len(runner.calls) == 1
    assert len(builder.calls) == 1
