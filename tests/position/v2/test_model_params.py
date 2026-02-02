"""Tests for ModelParams with SLEAP support."""

import pytest


class TestModelParamsSLEAP:
    """Test SLEAP support in ModelParams."""

    def test_sleap_tool_in_tool_info(self, pv2_train):
        """Test that SLEAP is registered in tool_info."""
        ModelParams = pv2_train.ModelParams

        assert "SLEAP" in ModelParams.tool_info
        assert "required" in ModelParams.tool_info["SLEAP"]
        assert "accepted" in ModelParams.tool_info["SLEAP"]
        assert "skipped" in ModelParams.tool_info["SLEAP"]
        assert "aliases" in ModelParams.tool_info["SLEAP"]

    def test_sleap_required_params(self, pv2_train):
        """Test SLEAP required parameters."""
        ModelParams = pv2_train.ModelParams

        required = ModelParams.tool_info["SLEAP"]["required"]
        assert "model_type" in required

    def test_sleap_accepted_params(self, pv2_train):
        """Test SLEAP accepted parameters."""
        ModelParams = pv2_train.ModelParams

        accepted = ModelParams.tool_info["SLEAP"]["accepted"]

        # Architecture params
        assert "model_type" in accepted
        assert "backbone" in accepted

        # Training params
        assert "max_epochs" in accepted
        assert "batch_size" in accepted
        assert "learning_rate" in accepted

    def test_sleap_default_params_in_contents(self, pv2_train):
        """Test that SLEAP default params are in contents."""
        ModelParams = pv2_train.ModelParams

        # Check contents has SLEAP entry
        sleap_contents = [c for c in ModelParams.contents if c[1] == "SLEAP"]
        assert len(sleap_contents) == 1

        # Check structure
        model_params_id, tool, params, skeleton_id, params_hash = (
            sleap_contents[0]
        )
        assert model_params_id == "sleap_default"
        assert tool == "SLEAP"
        assert "model_type" in params
        assert params["model_type"] == "single_instance"

    def test_insert_sleap_params(self, pv2_train, skeleton):
        """Test inserting SLEAP model parameters."""
        ModelParams = pv2_train.ModelParams

        # Get a skeleton (limit to 1)
        skeleton_key = (skeleton).fetch("KEY", limit=1)[0]

        # Create SLEAP params
        params_key = {
            "tool": "SLEAP",
            "params": {
                "model_type": "centroid",
                "backbone": "resnet50",
                "max_epochs": 100,
                "batch_size": 8,
            },
            "skeleton_id": skeleton_key["skeleton_id"],
        }

        # Should insert successfully
        result = ModelParams().insert1(params_key, accept_default=True)

        assert result is not None
        assert "model_params_id" in result

    def test_sleap_params_validation(self, pv2_train, skeleton):
        """Test that missing required params raises error."""
        ModelParams = pv2_train.ModelParams

        skeleton_key = (skeleton).fetch("KEY", limit=1)[0]

        # Missing required 'model_type'
        params_key = {
            "tool": "SLEAP",
            "params": {
                "backbone": "unet",
                "max_epochs": 100,
            },
            "skeleton_id": skeleton_key["skeleton_id"],
        }

        with pytest.raises(ValueError, match="Missing required params"):
            ModelParams().insert1(params_key, accept_default=True)

    def test_sleap_param_aliases(self, pv2_train, skeleton):
        """Test that SLEAP parameter aliases work."""
        ModelParams = pv2_train.ModelParams

        skeleton_key = (skeleton).fetch("KEY", limit=1)[0]

        # Use alias 'approach' instead of 'model_type'
        params_key = {
            "tool": "SLEAP",
            "params": {
                "approach": "bottom_up",  # alias for model_type
                "backbone": "unet",
            },
            "skeleton_id": skeleton_key["skeleton_id"],
        }

        # Should work due to alias handling
        result = ModelParams().insert1(params_key, accept_default=True)

        assert result is not None
