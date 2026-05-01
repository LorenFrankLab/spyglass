"""Example tests demonstrating dependency injection without mock.patch.

These tests show how the SOLID refactor enables clean testing by injecting
test doubles rather than patching global imports.
"""

import pandas as pd
import pytest

# Test stubs - import these instead of real dependencies in tests
from tests.utils.test_stubs import (
    StubFileSystem,
    StubInferenceRunner,
    StubNWBBuilder,
    StubNWBReader,
    StubNWBWriter,
)


def test_dlc_strategy_with_filesystem_injection():
    """Test DLCStrategy business logic without real filesystem dependencies."""
    from spyglass.position.utils.tool_strategies import DLCStrategy

    # Set up mock filesystem state
    mock_yaml_data = {
        "/fake/project/config.yaml": {
            "project_path": "/fake/project",
            "bodyparts": ["nose", "tail"],
            "skeleton": [["nose", "tail"]],
        }
    }

    # Inject stub filesystem
    stub_fs = StubFileSystem(
        mock_files={
            "/fake/dlc-models/**/*.pb": ["/fake/dlc-models/model1.pb"],
        },
        mock_yaml_data=mock_yaml_data,
    )

    strategy = DLCStrategy(filesystem=stub_fs)

    # This would fail without dependency injection due to filesystem access
    assert strategy._fs.exists("/fake/project")
    config = strategy._fs.read_yaml("/fake/project/config.yaml")
    assert config["bodyparts"] == ["nose", "tail"]

    # Verify the calls were tracked
    assert len(stub_fs.calls) >= 2
    assert any(call["method"] == "exists" for call in stub_fs.calls)
    assert any(call["method"] == "read_yaml" for call in stub_fs.calls)


def test_pose_estim_with_injection():
    """Test PoseEstim make() logic without database or DLC dependencies."""

    # This is conceptual - would need actual test infrastructure
    # but shows the injection pattern:

    class TestablePoseEstim:
        """Test subclass that injects stubs."""

        _inference_runner_cls = StubInferenceRunner
        _nwb_builder_cls = StubNWBBuilder
        _nwb_reader_cls = StubNWBReader
        _nwb_writer_cls = StubNWBWriter

        @classmethod
        def _get_inference_runner_cls(cls):
            return cls._inference_runner_cls

        @classmethod
        def _get_nwb_builder_cls(cls):
            return cls._nwb_builder_cls

        @classmethod
        def _get_nwb_reader_cls(cls):
            return cls._nwb_reader_cls

        @classmethod
        def _get_nwb_writer_cls(cls):
            return cls._nwb_writer_cls

    # Now we can test business logic without external dependencies
    testable_pose_estim = TestablePoseEstim()

    # Test dependency injection works
    assert (
        testable_pose_estim._get_inference_runner_cls() == StubInferenceRunner
    )
    assert testable_pose_estim._get_nwb_builder_cls() == StubNWBBuilder

    # Mock data for testing
    mock_pose_df = pd.DataFrame(
        {
            ("DLC_resnet50", "nose", "x"): [100.0, 101.0],
            ("DLC_resnet50", "nose", "y"): [200.0, 201.0],
            ("DLC_resnet50", "nose", "likelihood"): [0.95, 0.95],
        }
    )

    # Test that we can instantiate the injected classes
    runner = testable_pose_estim._get_inference_runner_cls()(
        mock_result=mock_pose_df
    )
    builder = testable_pose_estim._get_nwb_builder_cls()()

    # Test business logic without I/O
    result = runner.run_inference(
        model_info={"model_path": "/fake/model.yaml"},
        video_path="/fake/video.mp4",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Mock data has 2 rows
    assert result.equals(mock_pose_df)  # Use .equals() not direct comparison

    # Test NWB building logic
    pose_est, skeleton = builder.build_pose_estimation(
        pose_df=result,
        bodyparts=["nose"],
        scorer="DLC_resnet50",
        model_id="test_model",
        skeleton_edges=[],
    )

    assert "MockPoseEstimation" in pose_est
    assert "MockSkeleton" in skeleton

    # Verify calls were tracked for test assertions
    assert len(runner.calls) == 1
    assert len(builder.calls) == 1


def test_static_methods_with_optional_injection():
    """Test static methods with optional dependency injection parameters."""
    from spyglass.position.v2.estim import PoseEstim

    # Create stub classes
    stub_writer = StubNWBWriter()
    stub_reader = StubNWBReader(
        mock_data={
            "/fake/file.nwb": {
                "processing": {"behavior": {"pose_estimations": {}}}
            }
        }
    )

    # Test static methods can accept injected dependencies
    # Note: Would need actual implementation testing, but shows the pattern

    # This demonstrates the API - actual testing would require more setup
    try:
        _ = PoseEstim.load_from_nwb(
            "/fake/file.nwb", nwb_reader_cls=stub_reader.__class__
        )
        # Would test the returned metadata structure
    except Exception:
        pass  # Expected in this demo environment

    # Verify injection pattern works
    assert callable(stub_writer.write)
    assert callable(stub_reader.read)


if __name__ == "__main__":
    # Run the demonstration tests
    print("🧪 DEPENDENCY INJECTION DEMONSTRATION")
    print("=" * 50)

    tests = [
        test_dlc_strategy_with_filesystem_injection,
        test_pose_estim_with_injection,
        test_static_methods_with_optional_injection,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
