"""Tests verifying that ndx-pose NWB files are rejected by Model.load()
and should instead be ingested via ImportedPose.insert_from_nwbfile().
"""

import pytest


class TestNdxPoseModelLoadRejected:
    """Model.load() must not accept ndx-pose NWB files."""

    def test_nwb_file_raises_value_error(self, mock_ndx_pose_nwb_file, model):
        """model.load(<nwb path>) raises ValueError with ImportedPose guidance."""
        with pytest.raises(ValueError, match="ImportedPose"):
            model.load(model_path=str(mock_ndx_pose_nwb_file))

    def test_nwb_file_raises_with_import_to_v2_hint(
        self, mock_ndx_pose_nwb_file, model
    ):
        """Error message mentions import_to_v2 flag."""
        with pytest.raises(ValueError, match="import_to_v2"):
            model.load(model_path=str(mock_ndx_pose_nwb_file))

    def test_nonexistent_nwb_raises_file_not_found(self, model):
        """Non-existent .nwb path raises FileNotFoundError (path check first)."""
        with pytest.raises(FileNotFoundError):
            model.load(model_path="/nonexistent/file.nwb")
