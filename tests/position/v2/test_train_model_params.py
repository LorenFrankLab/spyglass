"""Tests for model_params table (train.py)."""

import datajoint as dj
import pytest

INSERT_KWARGS = {"skip_duplicates": True, "accept_default": True}


class TestModelParamsInsert:
    """Test model_params.insert1() functionality."""

    def test_insert_with_dlc_tool(self, model_params):
        """Test inserting model_params for DLC tool."""
        params = {
            "net_type": "resnet_50",
            "Task": "reaching",
            "date": "2025-01-01",
        }

        key = model_params.insert1(
            {"tool": "DLC", "params": params},
            **INSERT_KWARGS,
        )

        assert "model_params_id" in key
        entry = (model_params & key).fetch1()
        assert entry["tool"] == "DLC"

    def test_insert_with_custom_model_params_id(self, model_params):
        """Test inserting with custom model_params_id."""
        expected = {"model_params_id": "test-params-001", "tool": "DLC"}
        key = model_params.insert1(
            {
                **expected,
                "params": {"net_type": "test_custom_model_params_id"},
            },
            **INSERT_KWARGS,
        )

        assert key == expected

    def test_insert_with_skeleton_link(self, skeleton, model_params):
        """Test inserting model_params with skeleton link."""
        # Create skeleton first
        skeleton_config = {
            "bodyparts": ["nose", "head"],
            "skeleton": [("nose", "head")],
        }
        skeleton_key = skeleton.insert1(skeleton_config, **INSERT_KWARGS)

        dlc_kwargs = dict(tool="DLC", params={"net_type": "resnet_50"})
        model_params_key = model_params.insert1(
            {**dlc_kwargs, "skeleton_id": skeleton_key["skeleton_id"]},
            **INSERT_KWARGS,
        )
        entry = (model_params & model_params_key).fetch1()
        assert entry["skeleton_id"] == skeleton_key["skeleton_id"]

    def test_insert_with_missing_skeleton(self, model_params):
        """Test inserting model_params with non-existent skeleton_id."""
        dlc_kwargs = dict(
            tool="DLC",
            params={"net_type": "test_missing_skeleton"},
            skeleton_id="fake-skeleton-999",
        )

        with pytest.raises(dj.DataJointError):
            model_params.insert1(dlc_kwargs, **INSERT_KWARGS)

    def test_insert_duplicate_params_skips(self, logger, model_params):
        """Test inserting duplicate params with skip_duplicates."""
        prev_log_level = logger.level
        logger.setLevel("ERROR")  # Suppress expected duplicate warnings

        params = {
            "net_type": "resnet_50",
            "unique_param": "test_duplicate_123",
        }
        insert = dict(key={"tool": "DLC", "params": params}, **INSERT_KWARGS)

        # First insert
        before_key = model_params.insert1(**insert)
        params_count_before = len(model_params)

        # Insert again
        after_key = model_params.insert1(**insert)
        params_count_after = len(model_params)

        # Should not create duplicate
        assert params_count_after == params_count_before
        assert before_key == after_key

        logger.setLevel(prev_log_level)  # Restore previous log level

    def test_insert_validates_tool(self, model_params):
        """Test insert validates tool is supported."""
        params = {"net_type": "resnet_50"}

        # DLC should be supported
        model_key = model_params.insert1(
            {"tool": "DLC", "params": params},
            **INSERT_KWARGS,
        )
        assert "model_params_id" in model_key


class TestModelParamsValidation:
    """Test model_params parameter validation."""

    def test_skipped_params_removed(self, model_params):
        """Test that skipped params are filtered out."""
        # According to train.py:338-385, certain params are skipped
        params = {
            "net_type": "resnet_50",
            "project_path": "/path/to/project",  # Should be skipped
            "video_sets": {"video.mp4": {}},  # Should be skipped
        }

        key = model_params.insert1(
            {"tool": "DLC", "params": params}, **INSERT_KWARGS
        )

        entry = (model_params & key).fetch1()
        stored_params = entry["params"]

        # Skipped params should not be in stored params
        assert "project_path" not in stored_params
        assert "video_sets" not in stored_params
        # Net type should still be there
        assert "net_type" in stored_params

    def test_required_params_validated(self, model_params):
        """Test that required params are validated."""
        insert = {"tool": "DLC", "params": {"Task": "reaching"}}
        with pytest.raises(ValueError):
            model_params.insert1(insert, **INSERT_KWARGS)


class TestModelParamsToolSupport:
    """Test support for different pose estimation tools."""

    def test_tool_info_contains_all_tools(self, model_params):
        """Test tool_info contains all supported tools."""
        assert "DLC" in model_params.tool_info
        assert "ndx-pose" in model_params.tool_info

    def test_dlc_tool_supported(self, model_params):
        """Test DLC tool is supported."""
        model_params.insert1(
            {"tool": "DLC", "params": {"net_type": "resnet_50"}},
            **INSERT_KWARGS,
        )

    def test_ndx_pose_tool_supported(self, model_params):
        """Test ndx-pose tool is supported."""
        params = {"source_software": "DeepLabCut", "model_name": "test"}
        model_params.insert1(
            {"tool": "ndx-pose", "params": params}, **INSERT_KWARGS
        )

    def test_empty_params_dict(self, model_params):
        """Test model_params with empty params dict."""
        # Empty params may fail validation
        with pytest.raises(ValueError):
            model_params.insert1(
                {"tool": "DLC", "params": dict()}, **INSERT_KWARGS
            )

    def test_large_params_dict(self, model_params):
        """Test model_params with large params dict."""
        params = {
            "net_type": "resnet_50",
            **{f"param_{i}": f"value_{i}" for i in range(50)},
        }

        key = model_params.insert1(
            {"tool": "DLC", "params": params},
            skip_duplicates=True,
            accept_default=True,
        )

        entry = (model_params & key).fetch1()
        assert len(entry["params"]) > 10  # Should store large dict
