"""Tests for Model.verify() method."""

import pytest


class TestModelVerification:
    """Test Model.verify() method."""

    def test_verify_valid_model(
        self,
        model,
        skip_if_no_dlc,
        dlc_project_config,
    ):
        """Test verification of a valid model (DLC)."""
        model_key = model.load(model_path=str(dlc_project_config))

        results = model.verify(model_key)

        assert results["valid"] is True
        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is True
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
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test verification when model file doesn't exist."""
        # Create a fake yaml file then delete it
        fake_yaml = tmp_path / "fake_model.yaml"
        fake_yaml.write_text("project: fake")
        model_key = model.load(model_path=str(fake_yaml))
        fake_yaml.unlink()

        results = model.verify(model_key)

        assert results["valid"] is False
        assert results["checks"]["model_path_exists"] is False
        assert len(results["errors"]) > 0
        assert "not found" in results["errors"][0].lower()

    def test_verify_result_structure(
        self, model, skip_if_no_dlc, dlc_project_config
    ):
        """Test that verification results have correct structure (DLC model)."""
        model_key = model.load(model_path=str(dlc_project_config))
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
