"""Tests for spyglass.position.utils.yaml_io."""

import pytest


class TestYamlIo:
    @pytest.fixture(autouse=True)
    def fns(self):
        from spyglass.position.utils.yaml_io import dump_yaml, load_yaml

        self.load = load_yaml
        self.dump = dump_yaml

    def test_round_trip(self, tmp_path):
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = tmp_path / "cfg.yaml"
        self.dump(path, data)
        assert self.load(path) == data

    def test_load_returns_dict(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("a: 1\nb: hello\n")
        result = self.load(path)
        assert isinstance(result, dict) and result["a"] == 1

    def test_dump_creates_file(self, tmp_path):
        path = tmp_path / "out.yaml"
        self.dump(path, {"x": 99})
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            self.load(tmp_path / "nonexistent.yaml")

    def test_string_path_accepted(self, tmp_path):
        path = str(tmp_path / "cfg.yaml")
        self.dump(path, {"hello": "world"})
        assert self.load(path)["hello"] == "world"
