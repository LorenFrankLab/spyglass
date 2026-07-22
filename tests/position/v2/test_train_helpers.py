"""Tests for module-level helper functions in spyglass.position.v2.train.

Covers: discover_training_csvs, parse_training_csv, aggregate_training_stats,
validate_skeleton_graph, is_duplicate_skeleton.

All tests are pure (no database fixtures required).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── discover_training_csvs ────────────────────────────────────────────────────


class TestDiscoverTrainingCsvs:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.train import discover_training_csvs

        self.fn = discover_training_csvs

    def test_finds_learning_stats_in_model_dir(self, tmp_path):
        f = tmp_path / "learning_stats.csv"
        f.write_text("1,0.5,0.001\n2,0.4,0.001\n")
        assert f in self.fn(tmp_path)

    def test_finds_csv_in_parent_dir(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        f = tmp_path / "learning_stats.csv"
        f.write_text("1,0.5,0.001\n")
        assert f in self.fn(subdir)

    def test_returns_empty_when_nothing_found(self, tmp_path):
        isolated = tmp_path / "project" / "training" / "snapshot"
        isolated.mkdir(parents=True)
        assert self.fn(isolated) == []

    def test_deduplicates_results(self, tmp_path):
        (tmp_path / "learning_stats.csv").write_text("1,0.5,0.001\n")
        results = self.fn(tmp_path)
        assert len(results) == len(set(results))

    def test_returns_list_of_paths(self, tmp_path):
        (tmp_path / "learning_stats.csv").write_text("1,0.5,0.001\n")
        assert all(isinstance(p, Path) for p in self.fn(tmp_path))

    def test_finds_nested_csv_via_glob(self, tmp_path):
        nested = tmp_path / "train" / "pose_cfg"
        nested.mkdir(parents=True)
        f = nested / "learning_stats.csv"
        f.write_text("1,0.5,0.001\n")
        assert f in self.fn(tmp_path)


# ── parse_training_csv ────────────────────────────────────────────────────────


class TestParseTrainingCsv:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.train import parse_training_csv

        self.fn = parse_training_csv

    def test_headerless_3col(self, tmp_path):
        f = tmp_path / "learning_stats.csv"
        f.write_text("100,0.5,0.001\n200,0.4,0.0005\n")
        result = self.fn(f)
        assert list(result.columns[:3]) == [
            "iteration",
            "loss",
            "learning_rate",
        ]
        assert len(result) == 2

    def test_header_row_standard_names(self, tmp_path):
        f = tmp_path / "log.csv"
        f.write_text("iteration,loss,learning_rate\n100,0.5,0.001\n")
        result = self.fn(f)
        assert result["iteration"].iloc[0] == 100

    def test_source_file_column_added(self, tmp_path):
        f = tmp_path / "learning_stats.csv"
        f.write_text("1,0.5,0.001\n")
        assert "source_file" in self.fn(f).columns

    def test_returns_dataframe(self, tmp_path):
        f = tmp_path / "learning_stats.csv"
        f.write_text("1,0.5,0.001\n")
        assert isinstance(self.fn(f), pd.DataFrame)

    def test_two_column_csv(self, tmp_path):
        f = tmp_path / "loss.csv"
        f.write_text("0,1.0\n100,0.5\n")
        result = self.fn(f)
        assert "iteration" in result.columns and "loss" in result.columns

    def test_lr_column_detected_by_name(self, tmp_path):
        f = tmp_path / "log.csv"
        f.write_text("step,train_loss,learning_rate\n0,1.0,0.001\n")
        assert "learning_rate" in self.fn(f).columns

    def test_none_for_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert self.fn(f) is None

    def test_none_for_single_column(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("100\n200\n")
        assert self.fn(f) is None


# ── aggregate_training_stats ──────────────────────────────────────────────────


class TestAggregateTrainingStats:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.train import aggregate_training_stats

        self.fn = aggregate_training_stats

    def _df(self, iterations, losses, source="test.csv"):
        return pd.DataFrame(
            {"iteration": iterations, "loss": losses, "source_file": source}
        )

    def test_combines_multiple_dataframes(self):
        result = self.fn(
            [self._df([0, 100], [1.0, 0.8]), self._df([200, 300], [0.6, 0.4])]
        )
        assert len(result) == 4

    def test_sorted_by_iteration(self):
        result = self.fn(
            [self._df([200, 300], [0.6, 0.4]), self._df([0, 100], [1.0, 0.8])]
        )
        assert list(result["iteration"]) == [0, 100, 200, 300]

    def test_single_dataframe(self):
        assert len(self.fn([self._df([0, 100, 200], [1.0, 0.8, 0.6])])) == 3

    def test_returns_dataframe(self):
        assert isinstance(self.fn([self._df([0], [1.0])]), pd.DataFrame)

    def test_empty_list_returns_empty_df(self):
        result = self.fn([])
        assert isinstance(result, pd.DataFrame) and len(result) == 0

    def test_preserves_source_file_column(self):
        result = self.fn([self._df([0], [1.0], source="mine.csv")])
        assert result["source_file"].iloc[0] == "mine.csv"


# ── validate_skeleton_graph ───────────────────────────────────────────────────


class TestValidateSkeletonGraph:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.train import validate_skeleton_graph

        self.fn = validate_skeleton_graph

    def test_valid_graph_does_not_raise(self):
        self.fn(["nose", "tail"], [("nose", "tail")])

    def test_empty_bodyparts_raises(self):
        with pytest.raises((ValueError, Exception)):
            self.fn([], [])

    def test_edge_with_unknown_bodypart_raises(self):
        with pytest.raises((ValueError, Exception)):
            self.fn(["nose"], [("nose", "ghost")])

    def test_single_node_no_edges_valid(self):
        self.fn(["nose"], [])

    def test_chain_graph_valid(self):
        self.fn(["nose", "body", "tail"], [("nose", "body"), ("body", "tail")])


# ── is_duplicate_skeleton ─────────────────────────────────────────────────────


class TestIsDuplicateSkeleton:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.train import is_duplicate_skeleton

        self.fn = is_duplicate_skeleton

    def test_identical_is_duplicate(self):
        bps = ["nose", "tail"]
        edges = [("nose", "tail")]
        assert self.fn(bps, edges, bps, edges, threshold=0.85)

    def test_different_node_count_not_duplicate(self):
        assert not self.fn(
            ["nose"], [], ["nose", "tail"], [("nose", "tail")], 0.85
        )

    def test_case_insensitive_match(self):
        assert self.fn(
            ["Nose", "Tail"],
            [("Nose", "Tail")],
            ["nose", "tail"],
            [("nose", "tail")],
            threshold=0.85,
        )

    def test_unrelated_names_not_duplicate(self):
        assert not self.fn(
            ["head", "body"],
            [("head", "body")],
            ["alpha", "beta"],
            [("alpha", "beta")],
            threshold=0.85,
        )

    def test_returns_bool(self):
        result = self.fn(
            ["a", "b"], [("a", "b")], ["a", "b"], [("a", "b")], 0.85
        )
        assert isinstance(result, bool)
