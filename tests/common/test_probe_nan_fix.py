"""Test for the Probe.Electrode NaN value fix.

This test ensures that NaN values in probe geometry fields are properly
handled by replacing them with -1.0 to avoid DataJoint query formatting issues.
"""

import math

import pytest

from spyglass.common.common_device import _replace_nan_with_default


class TestProbeNaNFix:
    """Test cases for the NaN replacement fix."""

    def test_replace_nan_with_default_basic(self):
        """Test that NaN values are properly replaced with -1.0."""
        test_data = {
            "probe_id": "test_probe",
            "probe_shank": 0,
            "contact_size": float("nan"),
            "probe_electrode": 123,
            "rel_x": float("nan"),
            "rel_y": float("nan"),
            "rel_z": float("nan"),
            "other_field": "normal_value",
        }

        result = _replace_nan_with_default(test_data)

        # Check that NaN values were replaced
        assert result["contact_size"] == -1.0
        assert result["rel_x"] == -1.0
        assert result["rel_y"] == -1.0
        assert result["rel_z"] == -1.0

        # Check that non-NaN values were preserved
        assert result["probe_id"] == "test_probe"
        assert result["probe_shank"] == 0
        assert result["probe_electrode"] == 123
        assert result["other_field"] == "normal_value"

    def test_replace_nan_with_default_custom_value(self):
        """Test that custom default values work correctly."""
        test_data = {
            "rel_x": float("nan"),
            "rel_y": 5.0,  # Normal value should be preserved
        }

        result = _replace_nan_with_default(test_data, default_value=-999.0)

        assert result["rel_x"] == -999.0
        assert result["rel_y"] == 5.0

    def test_replace_nan_with_default_no_nans(self):
        """Test that data without NaN values is unchanged."""
        test_data = {
            "probe_id": "test_probe",
            "contact_size": 25.0,
            "rel_x": 100.0,
            "rel_y": 200.0,
            "rel_z": 300.0,
        }

        result = _replace_nan_with_default(test_data)

        # Should be identical to input
        assert result == test_data

    def test_replace_nan_with_default_non_dict(self):
        """Test that non-dictionary inputs are returned unchanged."""
        assert _replace_nan_with_default("string") == "string"
        assert _replace_nan_with_default(123) == 123
        assert _replace_nan_with_default(None) is None

    def test_replace_nan_with_default_mixed_types(self):
        """Test handling of mixed data types including NaN."""
        test_data = {
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "nan_field": float("nan"),
            "none_field": None,
        }

        result = _replace_nan_with_default(test_data)

        assert result["string_field"] == "test"
        assert result["int_field"] == 42
        assert result["float_field"] == 3.14
        assert result["nan_field"] == -1.0
        assert result["none_field"] is None

    def test_nan_detection(self):
        """Test that NaN detection works correctly."""
        nan_val = float("nan")
        assert math.isnan(nan_val)
        assert not math.isnan(1.0)
        assert not math.isnan(-1.0)

    def test_electrode_key_simulation(self):
        """Test with a key similar to the one described in the issue."""
        # This simulates the exact case from the GitHub issue
        null_val = float("nan")
        key = {
            "probe_id": "nTrode32_probe description",
            "probe_shank": 0,
            "contact_size": null_val,
            "probe_electrode": 194,
            "rel_x": null_val,
            "rel_y": null_val,
            "rel_z": null_val,
        }

        result = _replace_nan_with_default(key)

        # Verify that all NaN values were replaced
        assert result["contact_size"] == -1.0
        assert result["rel_x"] == -1.0
        assert result["rel_y"] == -1.0
        assert result["rel_z"] == -1.0

        # Verify other fields unchanged
        assert result["probe_id"] == "nTrode32_probe description"
        assert result["probe_shank"] == 0
        assert result["probe_electrode"] == 194

        # Verify no NaN values remain
        for key, value in result.items():
            if isinstance(value, float):
                assert not math.isnan(value), f"NaN found in {key}"