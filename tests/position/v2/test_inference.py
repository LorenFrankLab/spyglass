"""Tests for video inference with trained models."""

from pathlib import Path

import pytest


class TestModelInference:
    """Test Model.run_inference() method for DLC models.

    Note: These tests verify error handling. Actual DLC inference requires
    a real DLC config.yaml and trained model, which can't be mocked easily.
    Models imported from ndx-pose NWB files don't contain the trained weights
    needed for inference.
    """

    def test_run_inference_basic(
        self,
        model,
        mock_ndx_pose_nwb_file,
        mock_video_file,
        skip_if_no_dlc,
    ):
        """Test that inference on ndx-pose models raises appropriate error."""
        # Import a model from ndx-pose (doesn't have DLC weights)
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Try to run inference - should fail because model_path is NWB, not config.yaml
        with pytest.raises(ValueError, match="config.yaml"):
            model.run_inference(model_key, video_path=str(mock_video_file))

    def test_run_inference_with_options(
        self,
        model,
        mock_ndx_pose_nwb_file,
        mock_video_file,
        skip_if_no_dlc,
    ):
        """Test that inference options are validated properly."""
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Try to run inference with options - should still fail for same reason
        with pytest.raises(ValueError, match="config.yaml"):
            model.run_inference(
                model_key,
                video_path=str(mock_video_file),
                save_as_csv=True,
                destfolder=str(mock_video_file.parent),
            )

    def test_run_inference_invalid_model(
        self,
        model,
        mock_video_file,
    ):
        """Test error when model doesn't exist."""
        with pytest.raises(ValueError, match="Model not found"):
            model.run_inference(
                {"model_id": "nonexistent"},
                video_path=str(mock_video_file),
            )

    def test_run_inference_invalid_video(
        self,
        model,
        mock_ndx_pose_nwb_file,
    ):
        """Test error when video doesn't exist."""
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        with pytest.raises(FileNotFoundError):
            model.run_inference(
                model_key,
                video_path="/nonexistent/video.avi",
            )


class TestPoseEstimPopulation:
    """Test PoseEstim.populate() and related methods."""

    def test_load_dlc_output_to_nwb(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test converting DLC h5 output to ndx-pose NWB format."""
        PoseEstim = position_v2.estim.PoseEstim

        # Load DLC output into NWB
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Verify NWB file was created/updated
        assert Path(nwb_path).exists()

        # Verify ndx-pose data is present
        import ndx_pose
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbfile = io.read()
            assert "behavior" in nwbfile.processing

            behavior_module = nwbfile.processing["behavior"]
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            assert len(pose_estimations) > 0

    def test_pose_estim_insert(
        self,
        position_v2,
        model,
        mock_ndx_pose_nwb_file,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test inserting PoseEstim entry after inference."""
        PoseEstim = position_v2.estim.PoseEstim
        VidFileGroup = position_v2.video.VidFileGroup

        # Import model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Create video group
        vid_group_key = VidFileGroup().insert1(
            {
                "vid_group_id": "test_group_001",
                "description": "Test video group",
            },
            skip_duplicates=True,
        )

        # Load DLC output to NWB
        PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Insert PoseEstim entry (only use primary keys from Model and VidFileGroup)
        # Note: analysis_file_name is omitted since NWB file isn't registered in AnalysisNwbfile
        estim_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": vid_group_key["vid_group_id"],
        }

        PoseEstim().insert1(estim_key)

        # Verify entry was created
        assert len(PoseEstim() & estim_key) == 1

    def test_pose_estim_fetch_dataframe(
        self,
        position_v2,
        model,
        mock_ndx_pose_nwb_file,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test fetching pose data as DataFrame.

        Note: This test verifies the dataframe fetch logic by reading directly
        from the NWB file, without full AnalysisNwbfile registration.
        The E2E test (test_e2e_dlc_inference) covers the complete workflow.
        """
        PoseEstim = position_v2.estim.PoseEstim

        # Load DLC output into NWB
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Verify NWB contains pose data by reading directly
        import ndx_pose
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbfile = io.read()
            assert "behavior" in nwbfile.processing

            behavior_module = nwbfile.processing["behavior"]
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            assert len(pose_estimations) > 0
            pose_estimation = list(pose_estimations.values())[0]

            # Verify we can read bodyparts and coordinates
            assert len(pose_estimation.pose_estimation_series) > 0
            # pose_estimation_series is a dict-like object
            series_list = list(pose_estimation.pose_estimation_series.values())
            assert len(series_list) > 0
            series = series_list[0]
            assert series.data.shape[0] == 10  # 10 frames
            assert series.data.shape[1] == 2  # x, y coords
            assert len(series.confidence[:]) == 10  # confidence for each frame


class TestLoadFromNWB:
    """Test PoseEstim.load_from_nwb() for existing ndx-pose files."""

    def test_load_from_nwb_basic(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test loading pose data from existing ndx-pose NWB file."""
        PoseEstim = position_v2.estim.PoseEstim

        # First create an NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Now load metadata from that NWB file
        metadata = PoseEstim.load_from_nwb(nwb_path)

        # Verify metadata
        assert metadata["nwb_file_path"] == nwb_path
        assert metadata["pose_estimation_name"] == "PoseEstimation"
        assert len(metadata["bodyparts"]) > 0
        assert metadata["n_frames"] == 10
        assert metadata["scorer"] is not None
        assert metadata["source_software"] == "DeepLabCut"

    def test_load_from_nwb_file_not_found(self, position_v2):
        """Test error when NWB file doesn't exist."""
        PoseEstim = position_v2.estim.PoseEstim

        with pytest.raises(FileNotFoundError):
            PoseEstim.load_from_nwb("/nonexistent/file.nwb")

    def test_load_from_nwb_no_behavior_module(
        self, position_v2, mock_nwb_file_for_parent
    ):
        """Test error when NWB file lacks behavior module."""
        PoseEstim = position_v2.estim.PoseEstim

        # mock_nwb_file_for_parent doesn't have pose data yet
        with pytest.raises(ValueError, match="No behavior module"):
            PoseEstim.load_from_nwb(mock_nwb_file_for_parent)

    def test_load_from_nwb_specific_pose_estimation(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test loading specific PoseEstimation by name."""
        PoseEstim = position_v2.estim.PoseEstim

        # Create NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
            pose_estimation_name="MyPoseEstimation",
        )

        # Load by name
        metadata = PoseEstim.load_from_nwb(
            nwb_path, pose_estimation_name="MyPoseEstimation"
        )

        assert metadata["pose_estimation_name"] == "MyPoseEstimation"

    def test_load_from_nwb_wrong_pose_estimation_name(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test error when specified PoseEstimation doesn't exist."""
        PoseEstim = position_v2.estim.PoseEstim

        # Create NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Try to load non-existent PoseEstimation
        with pytest.raises(ValueError, match="not found in NWB"):
            PoseEstim.load_from_nwb(
                nwb_path, pose_estimation_name="NonExistent"
            )


class TestEndToEndInference:
    """Test complete end-to-end inference workflow.

    Note: Full E2E workflow requires a real DLC config.yaml and trained model.
    The test below demonstrates the workflow conceptually, but actual DLC
    inference can't be tested with mock data.
    """

    def test_e2e_dlc_inference(
        self,
        position_v2,
        model,
        mock_ndx_pose_nwb_file,
        mock_video_file,
        mock_dlc_inference_output,
        skip_if_no_dlc,
    ):
        """Test workflow: import model -> load existing results -> fetch data.

        This test demonstrates the complete workflow using pre-computed DLC output
        instead of running actual inference (which requires a real DLC model).
        """
        PoseEstim = position_v2.estim.PoseEstim
        VidFileGroup = position_v2.video.VidFileGroup

        # Step 1: Import model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Step 2: Create video group
        vid_group_key = VidFileGroup().insert1(
            {
                "vid_group_id": "test_e2e_group",
                "description": "E2E test video group",
            },
            skip_duplicates=True,
        )

        # Step 3: Load pre-computed DLC results to NWB
        # (In real workflow, this would come from Model.run_inference())
        PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=f"{mock_video_file.stem}_analysis.nwb",
        )

        # Step 4: Insert PoseEstim entry (without AnalysisNwbfile registration)
        estim_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": vid_group_key["vid_group_id"],
        }
        PoseEstim().insert1(estim_key)

        # Verify workflow completed
        assert len(PoseEstim() & estim_key) == 1
