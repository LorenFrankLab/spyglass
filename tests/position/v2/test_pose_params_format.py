"""Unit tests for the PoseParams pretty-print formatter.

Exercises the pure ``format_pose_params`` helper on literal dicts only —
no table access, fetch, or inserts. (Importing ``estim`` still needs a live
database connection because of its module-level ``dj.schema`` declaration,
so imports are kept inside the test functions per project convention.)
"""


def _default_params():
    """Return a representative PoseParams-style row dict."""
    return {
        "pose_params_id": "default",
        "orient": {
            "method": "two_pt",
            "bodypart1": "redLED_C",
            "bodypart2": "greenLED",
        },
        "centroid": {
            "method": "2pt",
            "points": {"point1": "redLED_C", "point2": "greenLED"},
            "max_LED_separation": 12.0,
        },
        "smoothing": {
            "interpolate": True,
            "likelihood_thresh": 0.95,
        },
    }


class TestFormatPoseParams:
    """Tests for ``format_pose_params``."""

    def test_returns_string_with_all_sections(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params(_default_params())
        assert isinstance(text, str)
        for section in ("orient", "centroid", "smoothing"):
            assert f"--- {section} ---" in text

    def test_renders_sub_dict_values(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params(_default_params())
        assert "two_pt" in text
        assert "greenLED" in text
        assert "0.95" in text

    def test_name_header_included_when_given(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params(_default_params(), name="default")
        assert "=== PoseParams: default ===" in text

    def test_no_header_when_name_omitted(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params(_default_params())
        assert "PoseParams:" not in text

    def test_missing_sub_dict_reported_gracefully(self):
        from spyglass.position.v2.utils.params import format_pose_params

        # Only orient present; centroid/smoothing absent.
        partial = {"orient": {"method": "none"}}
        text = format_pose_params(partial)
        assert "--- centroid ---" in text
        assert "--- smoothing ---" in text
        assert text.count("(not set)") == 2

    def test_empty_sub_dict_reported_as_not_set(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params(
            {"orient": {}, "centroid": {}, "smoothing": {}}
        )
        assert text.count("(not set)") == 3

    def test_empty_mapping_does_not_raise(self):
        from spyglass.position.v2.utils.params import format_pose_params

        text = format_pose_params({})
        assert text.count("(not set)") == 3
