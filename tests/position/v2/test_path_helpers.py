"""Tests for spyglass.position.utils.path_helpers."""

from pathlib import Path

import pytest


class TestPathHelpers:
    @pytest.fixture(autouse=True)
    def fns(self):
        from spyglass.position.utils.path_helpers import (
            resolve_model_path,
            to_stored_path,
        )

        self.resolve = resolve_model_path
        self.store = to_stored_path

    def test_absolute_path_returned_unchanged(self):
        p = Path("/absolute/path/to/model.yaml")
        assert self.resolve(str(p)) == p

    def test_returns_path_object(self):
        assert isinstance(self.resolve("/some/path"), Path)

    def test_to_stored_path_returns_string(self, tmp_path):
        assert isinstance(self.store(tmp_path / "model.yaml"), str)
