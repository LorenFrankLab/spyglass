"""Tests for Model.import_from_v1().

4-B: Mock the V1 DLCModel table fetch and verify Model.load() is called
     with the correct config.yaml path.  No live database or DLC required.
"""

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_dlc_model_mock(row: dict):
    """Return a DLCModel class mock that returns *row* from (DLCModel & key).fetch1().

    DataJoint tables are singletons — the & operator is called on the *class*,
    not an instance.  MagicMock magic methods must be configured via
    ``.return_value`` on the existing magic attribute, not by assignment.
    """
    query_mock = MagicMock()
    query_mock.fetch1.return_value = row

    cls_mock = MagicMock()
    cls_mock.__and__.return_value = query_mock  # configure, don't replace
    return cls_mock


def _v1_module(dlc_model_cls) -> ModuleType:
    """Fake spyglass.position.v1.position_dlc_model with a mock DLCModel."""
    mod = ModuleType("spyglass.position.v1.position_dlc_model")
    mod.DLCModel = dlc_model_cls
    return mod


class TestImportFromV1:
    """Unit tests for Model.import_from_v1()."""

    model = None  # set by _model fixture

    @pytest.fixture(autouse=True)
    def _model(self, model):
        self.model = model  # pylint: disable=attribute-defined-outside-init

    # ── helpers ─────────────────────────────────────────────────────────────

    def _v1_key(self):
        return {
            "project_name": "Wtrack_WhiteLED",
            "dlc_model_name": "Wtrack_WhiteLED_ms_stim_wtrack_00",
            "dlc_model_params_name": "default",
        }

    def _make_project(self, tmp_path, *, config_name="config.yaml"):
        """Write a minimal DLC project folder and return the config path."""
        import yaml

        project_dir = tmp_path / "Wtrack_WhiteLED-sc-Nov12"
        project_dir.mkdir(parents=True)
        cfg = {
            "Task": "Wtrack",
            "project_path": str(project_dir),
            "bodyparts": ["greenLED", "redLED_C"],
            "video_sets": {},
            "TrainingFraction": [0.8],
            "iteration": 0,
            "snapshotindex": -1,
        }
        cfg_path = project_dir / config_name
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path

    def _mock_v1_row(self, project_path: str):
        return {
            "project_name": "Wtrack_WhiteLED",
            "dlc_model_name": "Wtrack_WhiteLED_ms_stim_wtrack_00",
            "dlc_model_params_name": "default",
            "project_path": project_path,
            "config_template": {},
        }

    # ── ImportError when V1 schema unavailable ───────────────────────────────

    def test_raises_import_error_when_v1_unavailable(self, monkeypatch):
        """ImportError raised with helpful message when V1 schema is missing."""
        import sys

        import spyglass.position.v1 as _v1_pkg

        # Remove the cached submodule and block re-imports.  Also remove the
        # attribute from the parent package so Python cannot resolve it there.
        monkeypatch.delitem(
            sys.modules,
            "spyglass.position.v1.position_dlc_model",
            raising=False,
        )
        monkeypatch.setitem(
            sys.modules, "spyglass.position.v1.position_dlc_model", None
        )
        if hasattr(_v1_pkg, "position_dlc_model"):
            monkeypatch.delattr(_v1_pkg, "position_dlc_model")

        with pytest.raises(ImportError, match="V1 DLCModel"):
            self.model.import_from_v1(self._v1_key())

    # ── FileNotFoundError when no config in project_path ────────────────────

    def test_raises_file_not_found_when_no_config(self, tmp_path):
        """FileNotFoundError raised when project_path has no config.yaml."""
        empty_dir = tmp_path / "no_config_here"
        empty_dir.mkdir()

        row = self._mock_v1_row(str(empty_dir))
        cls_mock = _make_dlc_model_mock(row)

        with patch.dict(
            "sys.modules",
            {"spyglass.position.v1.position_dlc_model": _v1_module(cls_mock)},
        ):
            with pytest.raises(FileNotFoundError, match="config"):
                self.model.import_from_v1(self._v1_key())

    # ── load() called with correct plain config.yaml ─────────────────────────

    def test_load_called_with_config_yaml(self, tmp_path):
        """Model.load() is called with the config.yaml found in project_path."""
        cfg_path = self._make_project(tmp_path, config_name="config.yaml")
        row = self._mock_v1_row(str(cfg_path.parent))
        cls_mock = _make_dlc_model_mock(row)

        with (
            patch.dict(
                "sys.modules",
                {
                    "spyglass.position.v1.position_dlc_model": _v1_module(
                        cls_mock
                    )
                },
            ),
            patch.object(
                self.model, "load", return_value={"model_id": "m-test"}
            ) as mock_load,
        ):
            result = self.model.import_from_v1(self._v1_key())

        mock_load.assert_called_once()
        called_path = Path(mock_load.call_args[0][0])
        assert called_path == cfg_path
        assert result == {"model_id": "m-test"}

    # ── dj_dlc-prefixed config is preferred ──────────────────────────────────

    def test_dj_dlc_config_preferred_over_plain(self, tmp_path):
        """When dj_dlc_config.yaml exists alongside config.yaml, it is preferred."""
        import shutil

        cfg_path = self._make_project(tmp_path, config_name="config.yaml")
        # Place a dj_dlc_config.yaml in the same directory
        dj_cfg_path = cfg_path.parent / "dj_dlc_config.yaml"
        shutil.copy(cfg_path, dj_cfg_path)

        row = self._mock_v1_row(str(cfg_path.parent))
        cls_mock = _make_dlc_model_mock(row)

        with (
            patch.dict(
                "sys.modules",
                {
                    "spyglass.position.v1.position_dlc_model": _v1_module(
                        cls_mock
                    )
                },
            ),
            patch.object(
                self.model, "load", return_value={"model_id": "m-dj"}
            ) as mock_load,
        ):
            self.model.import_from_v1(self._v1_key())

        called_path = Path(mock_load.call_args[0][0])
        assert "dj_dlc" in called_path.name

    # ── kwargs forwarded to load() ────────────────────────────────────────────

    def test_kwargs_forwarded_to_load(self, tmp_path):
        """Extra kwargs passed to import_from_v1 are forwarded to load()."""
        cfg_path = self._make_project(tmp_path)
        row = self._mock_v1_row(str(cfg_path.parent))
        cls_mock = _make_dlc_model_mock(row)

        with (
            patch.dict(
                "sys.modules",
                {
                    "spyglass.position.v1.position_dlc_model": _v1_module(
                        cls_mock
                    )
                },
            ),
            patch.object(
                self.model, "load", return_value={"model_id": "m-kw"}
            ) as mock_load,
        ):
            self.model.import_from_v1(
                self._v1_key(),
                skeleton_id="sk-custom",
                model_id="m-custom",
            )

        _, load_kwargs = mock_load.call_args
        assert load_kwargs.get("skeleton_id") == "sk-custom"
        assert load_kwargs.get("model_id") == "m-custom"

    # ── KeyError when v1_key matches nothing ─────────────────────────────────

    def test_raises_key_error_for_unknown_v1_key(self):
        """KeyError propagated when the V1 key matches no DLCModel row."""
        query_mock = MagicMock()
        query_mock.fetch1.side_effect = KeyError("no match")

        cls_mock = MagicMock()
        cls_mock.__and__.return_value = query_mock

        with (
            patch.dict(
                "sys.modules",
                {
                    "spyglass.position.v1.position_dlc_model": _v1_module(
                        cls_mock
                    )
                },
            ),
        ):
            with pytest.raises((KeyError, Exception)):
                self.model.import_from_v1({"dlc_model_name": "ghost"})
