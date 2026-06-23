"""Tests verifying that ndx-pose NWB files are rejected by Model.load()
and should instead be ingested via ImportedPose.insert_from_nwbfile().
"""

import datajoint as dj
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


class TestCanonicalizeDlcProject:
    """Model._canonicalize_dlc_project rewrites the on-disk DLC config."""

    @staticmethod
    def _write_config(path, bodyparts, skeleton):
        from spyglass.position.utils.yaml_io import dump_yaml

        dump_yaml(path, {"bodyparts": bodyparts, "skeleton": skeleton})

    def test_rewrites_variant_config_and_backs_up(self, model, tmp_path):
        """Variant names are rewritten to canonical; original backed up."""
        from spyglass.position.utils.yaml_io import load_yaml

        cfg = tmp_path / "config.yaml"
        self._write_config(
            cfg, ["green_led", "red_led_c"], [["green_led", "red_led_c"]]
        )

        changes = model._canonicalize_dlc_project(cfg)

        assert changes == {"green_led": "greenLED", "red_led_c": "redLED_C"}
        rewritten = load_yaml(cfg)
        assert rewritten["bodyparts"] == ["greenLED", "redLED_C"]
        assert rewritten["skeleton"] == [["greenLED", "redLED_C"]]
        # original preserved in a timestamped backup beside the config
        backups = list(tmp_path.glob("config.yaml.*.bak"))
        assert len(backups) == 1
        assert load_yaml(backups[0])["bodyparts"] == ["green_led", "red_led_c"]

    def test_repeated_rewrites_keep_every_backup(self, model, tmp_path):
        """Each rewrite-with-changes adds a backup; none are overwritten."""
        cfg = tmp_path / "config.yaml"
        self._write_config(cfg, ["green_led"], [])
        model._canonicalize_dlc_project(cfg)  # green_led -> greenLED, 1 backup

        # introduce a fresh variant and rewrite again
        self._write_config(cfg, ["red_led_c"], [])
        model._canonicalize_dlc_project(cfg)  # red_led_c -> redLED_C, 2 backups

        assert len(list(tmp_path.glob("config.yaml.*.bak"))) == 2

    def test_already_canonical_is_noop(self, model, tmp_path):
        """A fully-canonical config is untouched and leaves no .bak."""
        from spyglass.position.utils.yaml_io import load_yaml

        cfg = tmp_path / "config.yaml"
        self._write_config(
            cfg, ["greenLED", "redLED_C"], [["greenLED", "redLED_C"]]
        )

        assert model._canonicalize_dlc_project(cfg) == {}
        assert list(tmp_path.glob("config.yaml.*.bak")) == []
        assert load_yaml(cfg)["bodyparts"] == ["greenLED", "redLED_C"]

    def test_novel_part_raises_and_leaves_config(self, model, tmp_path):
        """A part with no canonical match raises; config left untouched."""
        cfg = tmp_path / "config.yaml"
        self._write_config(cfg, ["totallyNovelPartXyz"], [])
        original = cfg.read_text()

        with pytest.raises(dj.DataJointError, match="totallyNovelPartXyz"):
            model._canonicalize_dlc_project(cfg)

        assert cfg.read_text() == original
        assert list(tmp_path.glob("config.yaml.*.bak")) == []


class TestImportNameErrorFallback:
    """Name-error guidance points users at normalize_names where it helps."""

    def test_unresolved_error_points_to_admin(self, model):
        import datajoint as dj

        err = dj.DataJointError("Unknown bodypart")
        out = model._augment_name_error(
            err, {"bodyparts": ["totallyNovelPartXyz"]}
        )
        assert "admin" in str(out)
        assert "totallyNovelPartXyz" in str(out)

    def test_resolvable_error_suggests_normalize_names(self, model):
        import datajoint as dj

        err = dj.DataJointError("boom")
        out = model._augment_name_error(err, {"bodyparts": ["green_led"]})
        assert "normalize_names=True" in str(out)
