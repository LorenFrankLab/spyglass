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


class TestCanonicalizeBodyparts:
    """Model.canonicalize_bodyparts resolves surface forms to canonical."""

    def test_list_of_known_surface_forms(self, model):
        """Known variants map to the canonical BodyPart spelling."""
        out = model.canonicalize_bodyparts(["EarR", "green_led", "tailBase"])
        assert out["mapping"] == {
            "EarR": "earR",
            "green_led": "greenLED",
            "tailBase": "tailBase",
        }
        assert out["unresolved"] == []

    def test_unknown_names_land_in_unresolved(self, model):
        """Names with no canonical match are returned, not raised."""
        out = model.canonicalize_bodyparts(["nose", "definitelyNotAPartXyz"])
        assert out["mapping"] == {"nose": "nose"}
        assert out["unresolved"] == ["definitelyNotAPartXyz"]

    def test_accepts_dlc_bodyparts_key(self, model):
        """A DLC config dict resolves via its 'bodyparts' entry."""
        out = model.canonicalize_bodyparts({"bodyparts": ["EarL", "EarR"]})
        assert out["mapping"] == {"EarL": "earL", "EarR": "earR"}

    def test_accepts_ndxpose_sleap_nodes_key(self, model):
        """An ndx-pose / SLEAP config dict resolves via its 'nodes' entry."""
        out = model.canonicalize_bodyparts({"nodes": ["nose", "tail_base"]})
        assert out["mapping"] == {"nose": "nose", "tail_base": "tailBase"}

    def test_bare_path_or_string_raises_typeerror(self, model, tmp_path):
        """A bare path/string is ambiguous and tool-specific -- rejected.

        Reading a tool-specific config file is the import path's job; callers
        pass the parsed names or dict (e.g. load_yaml(path) for DLC).
        """
        cfg = tmp_path / "config.yaml"
        cfg.write_text("bodyparts:\n  - nose\n")
        with pytest.raises(TypeError, match="load_yaml"):
            model.canonicalize_bodyparts(cfg)
        with pytest.raises(TypeError):
            model.canonicalize_bodyparts("earR")
