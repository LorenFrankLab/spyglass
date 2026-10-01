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

        # Model exists in DB even if weights not trained
        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is True
        assert "valid" in results
        assert "errors" in results

    def test_verify_nonexistent_model(self, model):
        """Test verification of non-existent model."""
        results = model.verify({"model_id": "nonexistent_model"})

        assert results["valid"] is False
        assert results["checks"]["model_exists"] is False
        # Missing-model path appends exactly one error and returns early.
        assert len(results["errors"]) == 1
        assert "not found" in results["errors"][0].lower()

    def test_verify_with_missing_file(
        self,
        model,
        skip_if_no_dlc,
        dlc_project_config,
    ):
        """Test verification when model file is removed after insert."""
        import shutil
        import tempfile
        from pathlib import Path

        # Copy the DLC config to a temp location, load it, then delete the copy
        with tempfile.TemporaryDirectory() as tmp:
            tmp_config = Path(tmp) / "config.yaml"
            shutil.copy(dlc_project_config, tmp_config)
            model_key = model.load(model_path=str(tmp_config))
            tmp_config.unlink()

        results = model.verify(model_key)

        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is False
        assert results["valid"] is False

    def test_verify_result_structure(
        self, model, skip_if_no_dlc, dlc_project_config
    ):
        """Test that verification results have correct structure (DLC model)."""
        model_key = model.load(model_path=str(dlc_project_config))
        results = model.verify(model_key)

        # Exact top-level key set (source: Model.verify return dict)
        assert set(results) == {
            "valid",
            "checks",
            "errors",
            "warnings",
            "model_info",
        }
        # Exact per-check key set
        assert set(results["checks"]) == {
            "model_exists",
            "model_path_exists",
            "skeleton_valid",
            "params_valid",
            "inference_ready",
        }

        # Known-exact check values: config is on disk and check_inference
        # defaults off, so inference_ready is never toggled True.
        assert results["checks"]["model_exists"] is True
        assert results["checks"]["model_path_exists"] is True
        assert results["checks"]["inference_ready"] is False

        # model_info echoes the fetched Model row (dict(model_entry)), so it
        # carries the queried model_id.
        assert results["model_info"]["model_id"] == model_key["model_id"]

        # errors / warnings are collected into lists (contents depend on the
        # loaded project, so only the container contract is pinned here).
        assert isinstance(results["errors"], list)
        assert isinstance(results["warnings"], list)
