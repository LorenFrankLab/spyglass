"""Tests for Model.import_model() with ndx-pose NWB files."""

import pytest
from pynwb import NWBHDF5IO


class TestNdxPoseImport:
    """Test importing models from ndx-pose NWB files."""

    def test_import_ndx_pose_basic(
        self,
        mock_ndx_pose_nwb_file,
        mock_nwb_file_for_parent,
        model,
        skeleton,
        model_params,
        bodypart,
    ):
        """Test complete ndx-pose NWB import with all components verified."""
        # Import model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Verify complete key returned
        assert all(
            k in model_key
            for k in ["model_id", "skeleton_id", "model_params_id"]
        )

        # Verify entries created
        assert len(model & model_key) == 1
        assert len(skeleton & {"skeleton_id": model_key["skeleton_id"]}) == 1
        assert (
            len(
                model_params & {"model_params_id": model_key["model_params_id"]}
            )
            == 1
        )


class TestNdxPoseEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_file_path(self, model, skeleton, model_params, bodypart):
        """Test error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            model.import_model(
                model_path="/nonexistent/file.nwb", nwb_file_name="test.nwb"
            )
