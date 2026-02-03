"""Tests for Model.verify() method."""

import pytest


class TestModelVerification:
    """Test Model.verify() method."""

    def test_verify_valid_model(
        self,
        model,
        mock_ndx_pose_nwb_file,
    ):
        """Test verification of a valid model."""
        # Import a model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Verify the model
        results = model.verify(model_key)

        # Should pass basic checks
        assert results["valid"] is True
        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is True
        assert results["checks"]["skeleton_valid"] is True
        assert results["checks"]["params_valid"] is True
        assert len(results["errors"]) == 0

    def test_verify_nonexistent_model(self, model):
        """Test verification of non-existent model."""
        results = model.verify({"model_id": "nonexistent_model"})

        assert results["valid"] is False
        assert results["checks"]["model_exists"] is False
        assert len(results["errors"]) > 0
        assert "not found" in results["errors"][0].lower()

    def test_verify_with_missing_file(
        self,
        model,
        mock_ndx_pose_nwb_file,
        tmp_path,
    ):
        """Test verification when model file doesn't exist."""
        # Import a model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Delete the model file
        mock_ndx_pose_nwb_file.unlink()

        # Verify should fail
        results = model.verify(model_key)

        assert results["valid"] is False
        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is False
        assert len(results["errors"]) > 0
        assert "not found" in results["errors"][0].lower()

    def test_verify_with_inference_check(
        self,
        model,
        mock_ndx_pose_nwb_file,
    ):
        """Test verification with inference readiness check."""
        # Import a model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        # Verify with inference check
        results = model.verify(model_key, check_inference=True)

        # ndx-pose models should have warning about inference
        assert results["valid"] is True
        # The specific warning depends on whether DLC is installed
        # Just check that we got warnings
        assert isinstance(results["warnings"], list)

    def test_verify_result_structure(
        self,
        model,
        mock_ndx_pose_nwb_file,
    ):
        """Test that verification results have correct structure."""
        # Import a model
        model_key = model.import_model(
            model_path=str(mock_ndx_pose_nwb_file),
        )

        results = model.verify(model_key)

        # Verify result structure
        assert "valid" in results
        assert "checks" in results
        assert "errors" in results
        assert "warnings" in results
        assert "model_info" in results

        # Verify checks structure
        assert "model_exists" in results["checks"]
        assert "model_path_exists" in results["checks"]
        assert "skeleton_valid" in results["checks"]
        assert "params_valid" in results["checks"]
        assert "inference_ready" in results["checks"]

        # Verify types
        assert isinstance(results["valid"], bool)
        assert isinstance(results["checks"], dict)
        assert isinstance(results["errors"], list)
        assert isinstance(results["warnings"], list)
        assert isinstance(results["model_info"], dict)
