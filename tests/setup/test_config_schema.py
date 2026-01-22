"""Tests for config schema DRY architecture.

Verifies that:
1. JSON schema is valid
2. Installer and settings.py use the same schema
3. Installer produces config that settings.py can use
4. Directory structures match exactly
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Import from installer
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from install import (
    build_directory_structure,
    determine_tls,
    load_directory_schema,
    load_full_schema,
    validate_schema,
)

# Spyglass imports - lazy loaded in tests to avoid hanging during pytest collection
# DO NOT import SpyglassConfig at module level - it imports datajoint which may
# try to connect to database before fixtures are set up


class TestConfigSchema:
    """Tests for directory_schema.json file."""

    def test_json_schema_is_valid(self):
        """Test that directory_schema.json is valid JSON and has required structure."""
        schema_path = (
            Path(__file__).parent.parent.parent
            / "src/spyglass/directory_schema.json"
        )
        assert (
            schema_path.exists()
        ), "directory_schema.json not found in src/spyglass/"

        with open(schema_path) as f:
            schema = json.load(f)

        # Check top-level structure
        assert isinstance(schema, dict)
        assert "directory_schema" in schema
        assert "tls" in schema

        # Check directory_schema has all prefixes
        dir_schema = schema["directory_schema"]
        assert set(dir_schema.keys()) == {"spyglass", "kachery", "dlc", "moseq"}

    def test_validate_schema_passes_for_valid_schema(self):
        """Test that validate_schema() accepts valid schema."""
        schema = load_full_schema()
        # Should not raise
        validate_schema(schema)

    def test_validate_schema_rejects_invalid_schema(self):
        """Test that validate_schema() rejects invalid schemas."""
        # Missing directory_schema
        with pytest.raises(ValueError, match="missing 'directory_schema'"):
            validate_schema({"other_key": {}})

        # Missing required prefix
        with pytest.raises(ValueError, match="Missing prefixes"):
            validate_schema(
                {
                    "directory_schema": {
                        "spyglass": {"raw": "raw"},
                        "kachery": {"cloud": ".kachery-cloud"},
                        # Missing dlc and moseq
                    }
                }
            )


class TestSchemaConsistency:
    """Tests for schema consistency between installer and settings.py."""

    def test_installer_and_settings_use_same_schema(self):
        """Test that installer and settings.py load identical schemas."""
        from spyglass.settings import SpyglassConfig

        # Load from installer
        installer_schema = load_directory_schema()

        # Load from settings.py
        config = SpyglassConfig()
        settings_schema = config.relative_dirs

        # Should be identical
        assert installer_schema == settings_schema, (
            "Installer and settings.py have different directory schemas. "
            "This violates the DRY principle."
        )

    def test_schema_has_all_required_prefixes(self):
        """Test that schema has all 4 required directory prefixes."""
        schema = load_directory_schema()
        assert set(schema.keys()) == {"spyglass", "kachery", "dlc", "moseq"}

    @pytest.mark.parametrize(
        "prefix,expected_count",
        [
            ("spyglass", 8),
            ("kachery", 3),
            ("dlc", 3),
            ("moseq", 2),
        ],
    )
    def test_schema_has_correct_directory_counts(self, prefix, expected_count):
        """Test that each prefix has expected number of directories."""
        schema = load_directory_schema()
        assert len(schema[prefix]) == expected_count

    @pytest.mark.parametrize(
        "prefix,expected_keys",
        [
            (
                "spyglass",
                {
                    "raw",
                    "analysis",
                    "recording",
                    "sorting",
                    "waveforms",
                    "temp",
                    "video",
                    "export",
                },
            ),
            ("kachery", {"cloud", "temp", "storage"}),
            ("dlc", {"project", "video", "output"}),
            ("moseq", {"project", "video"}),
        ],
    )
    def test_directories_have_correct_keys(self, prefix, expected_keys):
        """Test that each prefix has the correct directory keys."""
        schema = load_directory_schema()
        assert set(schema[prefix].keys()) == expected_keys


class TestInstallerConfig:
    """Tests for installer config generation."""

    def test_build_directory_structure_creates_all_dirs(self):
        """Test that build_directory_structure creates all 16 directories."""
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            dirs = build_directory_structure(
                base_dir, create=True, verbose=False
            )

            # Should return dict with all directories
            assert len(dirs) == 16, f"Expected 16 directories, got {len(dirs)}"

            # All directories should exist
            for name, path in dirs.items():
                assert path.exists(), f"Directory {name} not created at {path}"

    def test_build_directory_structure_dry_run(self):
        """Test that create=False doesn't create directories."""
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            dirs = build_directory_structure(
                base_dir, create=False, verbose=False
            )

            # Should return dict but not create dirs
            assert len(dirs) == 16
            assert not (base_dir / "raw").exists()

    def test_installer_config_has_all_directory_groups(self):
        """Test that installer creates config with all 4 directory groups."""
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            # Simulate what create_database_config does
            dirs = build_directory_structure(
                base_dir, create=True, verbose=False
            )

            config = {
                "custom": {
                    "spyglass_dirs": {
                        "base": str(base_dir),
                        "raw": str(dirs["spyglass_raw"]),
                        "analysis": str(dirs["spyglass_analysis"]),
                        "recording": str(dirs["spyglass_recording"]),
                        "sorting": str(dirs["spyglass_sorting"]),
                        "waveforms": str(dirs["spyglass_waveforms"]),
                        "temp": str(dirs["spyglass_temp"]),
                        "video": str(dirs["spyglass_video"]),
                        "export": str(dirs["spyglass_export"]),
                    },
                    "kachery_dirs": {
                        "cloud": str(dirs["kachery_cloud"]),
                        "storage": str(dirs["kachery_storage"]),
                        "temp": str(dirs["kachery_temp"]),
                    },
                    "dlc_dirs": {
                        "project": str(dirs["dlc_project"]),
                        "video": str(dirs["dlc_video"]),
                        "output": str(dirs["dlc_output"]),
                    },
                    "moseq_dirs": {
                        "project": str(dirs["moseq_project"]),
                        "video": str(dirs["moseq_video"]),
                    },
                }
            }

            # Verify all groups present
            custom = config["custom"]
            assert "spyglass_dirs" in custom
            assert "kachery_dirs" in custom
            assert "dlc_dirs" in custom
            assert "moseq_dirs" in custom

    def test_installer_directory_paths_match_schema(self):
        """Test that installer constructs paths according to schema."""
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            # Get schema
            schema = load_directory_schema()

            # Build directories
            dirs = build_directory_structure(
                base_dir, create=True, verbose=False
            )

            # Verify each path matches schema
            for prefix in schema:
                for key, rel_path in schema[prefix].items():
                    expected_path = base_dir / rel_path
                    actual_path = dirs[f"{prefix}_{key}"]
                    assert expected_path == actual_path, (
                        f"Path mismatch for {prefix}.{key}: "
                        f"expected {expected_path}, got {actual_path}"
                    )

    def test_installer_config_keys_match_settings_expectations(self):
        """Test that installer config keys match what settings.py expects."""
        from spyglass.settings import SpyglassConfig

        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            # Get what settings.py expects
            config_obj = SpyglassConfig()
            expected_structure = config_obj.relative_dirs

            # Build what installer creates
            dirs = build_directory_structure(
                base_dir, create=True, verbose=False
            )

            # Verify each prefix group has correct keys
            for prefix in expected_structure:
                expected_keys = set(expected_structure[prefix].keys())

                # Get actual keys from dirs dict
                actual_keys = set()
                for dir_name in dirs.keys():
                    if dir_name.startswith(f"{prefix}_"):
                        key = dir_name.split("_", 1)[1]
                        actual_keys.add(key)

                assert expected_keys == actual_keys, (
                    f"Key mismatch for {prefix}: "
                    f"expected {expected_keys}, got {actual_keys}"
                )


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_schema_matches_original_hardcoded_structure(self):
        """Test that schema produces same structure as original hard-coded version."""
        # Original structure from settings.py before refactor
        original = {
            "spyglass": {
                "raw": "raw",
                "analysis": "analysis",
                "recording": "recording",
                "sorting": "spikesorting",
                "waveforms": "waveforms",
                "temp": "tmp",
                "video": "video",
                "export": "export",
            },
            "kachery": {
                "cloud": ".kachery-cloud",
                "storage": "kachery_storage",
                "temp": "tmp",
            },
            "dlc": {
                "project": "projects",
                "video": "video",
                "output": "output",
            },
            "moseq": {
                "project": "projects",
                "video": "video",
            },
        }

        # Load current schema
        current = load_directory_schema()

        # Should be identical
        assert current == original, (
            "Schema has changed from original hard-coded structure. "
            "This breaks backwards compatibility."
        )

    def test_settings_produces_original_structure(self):
        """Test that settings.py produces original structure at runtime."""
        from spyglass.settings import SpyglassConfig

        # Original structure
        original = {
            "spyglass": {
                "raw": "raw",
                "analysis": "analysis",
                "recording": "recording",
                "sorting": "spikesorting",
                "waveforms": "waveforms",
                "temp": "tmp",
                "video": "video",
                "export": "export",
            },
            "kachery": {
                "cloud": ".kachery-cloud",
                "storage": "kachery_storage",
                "temp": "tmp",
            },
            "dlc": {
                "project": "projects",
                "video": "video",
                "output": "output",
            },
            "moseq": {
                "project": "projects",
                "video": "video",
            },
        }

        # Get from runtime
        config = SpyglassConfig()
        runtime_structure = config.relative_dirs

        # Should be identical
        assert runtime_structure == original, (
            "Runtime structure differs from original. "
            "This breaks backwards compatibility."
        )


class TestTLSDetermination:
    """Tests for automatic TLS determination."""

    def test_localhost_disables_tls(self):
        """Test that localhost connections disable TLS."""
        assert determine_tls("localhost") is False

    def test_ipv4_localhost_disables_tls(self):
        """Test that 127.0.0.1 disables TLS."""
        assert determine_tls("127.0.0.1") is False

    def test_ipv6_localhost_disables_tls(self):
        """Test that ::1 disables TLS."""
        assert determine_tls("::1") is False

    def test_remote_hostname_enables_tls(self):
        """Test that remote hostnames enable TLS."""
        assert determine_tls("lmf-db.cin.ucsf.edu") is True

    def test_remote_ip_enables_tls(self):
        """Test that remote IP addresses enable TLS."""
        assert determine_tls("192.168.1.100") is True


class TestSchemaVersioning:
    """Tests for schema versioning."""

    def test_schema_has_version(self):
        """Test that schema file includes version."""
        schema = load_full_schema()
        assert "_schema_version" in schema
        assert schema["_schema_version"] == "1.0.0"

    def test_version_history_present(self):
        """Test that version history is documented."""
        schema = load_full_schema()
        assert "_version_history" in schema
        assert "1.0.0" in schema["_version_history"]


class TestConfigCompatibility:
    """Tests for config compatibility between installer and settings.py."""

    def _get_all_keys(self, d: dict, prefix: str = "") -> set:
        """Recursively get all keys in nested dictionary."""
        keys = set()
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict):
                keys.update(self._get_all_keys(v, full_key))
        return keys

    def test_installer_config_has_all_settings_keys(self):
        """Test that installer config includes all keys from settings.py."""
        from spyglass.settings import SpyglassConfig

        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            # Get config from installer
            dir_schema = load_directory_schema()
            dirs = build_directory_structure(
                base_dir, schema=dir_schema, create=True, verbose=False
            )

            installer_config = {
                "database.host": "localhost",
                "database.port": 3306,
                "database.user": "testuser",
                "database.password": "testpass",
                "database.use_tls": False,
                "filepath_checksum_size_limit": 1 * 1024**3,
                "enable_python_native_blobs": True,
                "stores": {
                    "raw": {
                        "protocol": "file",
                        "location": str(dirs["spyglass_raw"]),
                        "stage": str(dirs["spyglass_raw"]),
                    },
                    "analysis": {
                        "protocol": "file",
                        "location": str(dirs["spyglass_analysis"]),
                        "stage": str(dirs["spyglass_analysis"]),
                    },
                },
                "custom": {
                    "debug_mode": False,
                    "test_mode": False,
                    "kachery_zone": "franklab.default",
                    "spyglass_dirs": {
                        "base": str(base_dir),
                        "raw": str(dirs["spyglass_raw"]),
                        "analysis": str(dirs["spyglass_analysis"]),
                        "recording": str(dirs["spyglass_recording"]),
                        "sorting": str(dirs["spyglass_sorting"]),
                        "waveforms": str(dirs["spyglass_waveforms"]),
                        "temp": str(dirs["spyglass_temp"]),
                        "video": str(dirs["spyglass_video"]),
                        "export": str(dirs["spyglass_export"]),
                    },
                    "kachery_dirs": {
                        "cloud": str(dirs["kachery_cloud"]),
                        "storage": str(dirs["kachery_storage"]),
                        "temp": str(dirs["kachery_temp"]),
                    },
                    "dlc_dirs": {
                        "base": str(base_dir / "deeplabcut"),
                        "project": str(dirs["dlc_project"]),
                        "video": str(dirs["dlc_video"]),
                        "output": str(dirs["dlc_output"]),
                    },
                    "moseq_dirs": {
                        "base": str(base_dir / "moseq"),
                        "project": str(dirs["moseq_project"]),
                        "video": str(dirs["moseq_video"]),
                    },
                },
            }

            # Get config from settings.py
            sg_config = SpyglassConfig()
            settings_config = sg_config._generate_dj_config(
                base_dir=str(base_dir),
                database_user="testuser",
                database_password="testpass",
                database_host="localhost",
                database_port=3306,
                database_use_tls=False,
            )

            # Get all keys from both
            installer_keys = self._get_all_keys(installer_config)
            settings_keys = self._get_all_keys(settings_config)

            # Installer must have all settings.py keys
            missing_keys = settings_keys - installer_keys
            assert not missing_keys, (
                f"Installer config is missing keys from settings.py: "
                f"{sorted(missing_keys)}. Update "
                "install.py::create_database_config()"
            )


class TestExampleConfigSync:
    """Tests that dj_local_conf_example.json stays in sync with installer.

    This test ensures the example config file matches the structure that
    install.py::create_database_config() generates. When install.py changes,
    this test will fail, reminding developers to update the example.
    """

    def _get_all_keys(self, d: dict, prefix: str = "") -> set:
        """Recursively get all keys in nested dictionary."""
        keys = set()
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, dict):
                keys.update(self._get_all_keys(v, full_key))
        return keys

    def _normalize_value(self, value):
        """Normalize values for comparison (paths become placeholders)."""
        if isinstance(value, str):
            # Normalize path-like values to placeholder
            if "/" in value or "\\" in value:
                return "/placeholder/path"
            # Normalize localhost variations
            if value in ["localhost", "lmf-db.cin.ucsf.edu"]:
                return "localhost"
        return value

    def test_example_config_structure_matches_installer(self):
        """Test that dj_local_conf_example.json matches installer structure.

        This validates that the example config has the same keys and
        structure as what install.py generates. When new config keys are
        added to install.py, this test will fail, prompting an update to
        the example file.
        """
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "spyglass_data"

            # Get config structure from installer (what create_database_config generates)
            dir_schema = load_directory_schema()
            dirs = build_directory_structure(
                base_dir, schema=dir_schema, create=True, verbose=False
            )

            installer_config = {
                "database.host": "localhost",
                "database.port": 3306,
                "database.user": "root",
                "database.password": "tutorial",
                "database.use_tls": False,
                "database.reconnect": True,
                "connection.init_function": None,
                "connection.charset": "",
                "loglevel": "INFO",
                "safemode": True,
                "fetch_format": "array",
                "display.limit": 12,
                "display.width": 14,
                "display.show_tuple_count": True,
                "add_hidden_timestamp": False,
                "filepath_checksum_size_limit": 1 * 1024**3,
                "enable_python_native_blobs": True,
                "stores": {
                    "raw": {
                        "protocol": "file",
                        "location": str(dirs["spyglass_raw"]),
                        "stage": str(dirs["spyglass_raw"]),
                    },
                    "analysis": {
                        "protocol": "file",
                        "location": str(dirs["spyglass_analysis"]),
                        "stage": str(dirs["spyglass_analysis"]),
                    },
                },
                "custom": {
                    "debug_mode": "false",
                    "test_mode": "false",
                    "kachery_zone": "franklab.default",
                    "spyglass_dirs": {
                        "base": str(base_dir),
                        "raw": str(dirs["spyglass_raw"]),
                        "analysis": str(dirs["spyglass_analysis"]),
                        "recording": str(dirs["spyglass_recording"]),
                        "sorting": str(dirs["spyglass_sorting"]),
                        "waveforms": str(dirs["spyglass_waveforms"]),
                        "temp": str(dirs["spyglass_temp"]),
                        "video": str(dirs["spyglass_video"]),
                        "export": str(dirs["spyglass_export"]),
                    },
                    "kachery_dirs": {
                        "cloud": str(dirs["kachery_cloud"]),
                        "storage": str(dirs["kachery_storage"]),
                        "temp": str(dirs["kachery_temp"]),
                    },
                    "dlc_dirs": {
                        "base": str(base_dir / "deeplabcut"),
                        "project": str(dirs["dlc_project"]),
                        "video": str(dirs["dlc_video"]),
                        "output": str(dirs["dlc_output"]),
                    },
                    "moseq_dirs": {
                        "base": str(base_dir / "moseq"),
                        "project": str(dirs["moseq_project"]),
                        "video": str(dirs["moseq_video"]),
                    },
                },
            }

            # Load example config
            example_path = (
                Path(__file__).parent.parent.parent
                / "dj_local_conf_example.json"
            )
            assert (
                example_path.exists()
            ), "dj_local_conf_example.json not found in repo root"

            with open(example_path) as f:
                example_config = json.load(f)

            # Get all keys from both
            installer_keys = self._get_all_keys(installer_config)
            example_keys = self._get_all_keys(example_config)

            # Check for missing keys in example
            missing_in_example = installer_keys - example_keys
            assert not missing_in_example, (
                f"\ndj_local_conf_example.json is MISSING keys that "
                f"install.py generates:\n"
                f"{sorted(missing_in_example)}\n\n"
                f"Action required:\n"
                f"  1. Update dj_local_conf_example.json to include these keys\n"
                f"  2. See install.py::create_database_config() "
                f"(lines ~1264-1331) for reference\n"
            )

            # Check for extra keys in example (warning, not error)
            extra_in_example = example_keys - installer_keys
            if extra_in_example:
                import warnings

                warnings.warn(
                    f"\ndj_local_conf_example.json has EXTRA keys not in "
                    f"installer:\n"
                    f"{sorted(extra_in_example)}\n\n"
                    f"This might be OK (example may show optional configs), "
                    f"but verify these are intentional.\n",
                    UserWarning,
                )

    def test_example_config_directory_groups_complete(self):
        """Test that example has all 4 required directory groups."""
        example_path = (
            Path(__file__).parent.parent.parent / "dj_local_conf_example.json"
        )

        with open(example_path) as f:
            example_config = json.load(f)

        # Check custom section exists
        assert (
            "custom" in example_config
        ), "dj_local_conf_example.json missing 'custom' section"

        custom = example_config["custom"]

        # Check all 4 directory groups present
        required_groups = [
            "spyglass_dirs",
            "kachery_dirs",
            "dlc_dirs",
            "moseq_dirs",
        ]
        for group in required_groups:
            assert (
                group in custom
            ), f"dj_local_conf_example.json missing '{group}' in custom section"

    def test_example_config_is_valid_json(self):
        """Test that dj_local_conf_example.json is valid JSON."""
        example_path = (
            Path(__file__).parent.parent.parent / "dj_local_conf_example.json"
        )

        assert (
            example_path.exists()
        ), "dj_local_conf_example.json not found in repo root"

        # Should not raise JSONDecodeError
        with open(example_path) as f:
            config = json.load(f)

        assert isinstance(
            config, dict
        ), "Example config should be a JSON object (dict)"
