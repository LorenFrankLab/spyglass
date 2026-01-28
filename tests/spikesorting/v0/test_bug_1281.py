"""Tests for bug 1281: AutomaticCuration.get_labels early return.

The bug caused get_labels() to return after processing only the first
metric in label_params, silently dropping labels from subsequent metrics.

These tests verify the fix and define the expected behavior for the
fix_1281 repair function.
"""

import pytest


# -- Helpers ----------------------------------------------------------


@pytest.fixture(scope="session")
def get_labels(spike_v0):
    """Wrapper: sorting arg is unused in label logic."""

    def _get_labels(parent_labels, quality_metrics, label_params):
        return spike_v0.AutomaticCuration.get_labels(
            sorting=None,
            parent_labels=parent_labels,
            quality_metrics=quality_metrics,
            label_params=label_params,
        )

    yield _get_labels


# -- Task 5.1: get_labels unit tests ---------------------------------


class TestGetLabelsMultiMetric:
    """Verify get_labels processes ALL metrics in label_params."""

    two_metric_params = {
        "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
        "isi_violation": [">", 0.5, ["mua"]],
    }
    quality_metrics = {
        "nn_noise_overlap": {"0": 0.2, "1": 0.05, "2": 0.3},
        "isi_violation": {"0": 0.8, "1": 0.9, "2": 0.1},
    }

    def test_both_metrics_applied(self, get_labels):
        """Unit matching both metrics gets labels from both."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        # unit 0: nn_noise_overlap 0.2 > 0.1 -> noise,reject
        #         isi_violation    0.8 > 0.5 -> mua
        assert "noise" in result["0"]
        assert "reject" in result["0"]
        assert "mua" in result["0"]

    def test_second_metric_only(self, get_labels):
        """Unit matching only second metric still gets its labels."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        # unit 1: nn_noise_overlap 0.05 <= 0.1 -> no label
        #         isi_violation    0.9  >  0.5 -> mua
        assert "1" in result
        assert "mua" in result["1"]
        assert "noise" not in result.get("1", [])

    def test_no_match_no_label(self, get_labels):
        """Unit matching neither metric gets no labels."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        # unit 2: nn_noise_overlap 0.3 > 0.1 -> noise,reject
        #         isi_violation    0.1 <= 0.5 -> no label
        assert "noise" in result["2"]
        assert "reject" in result["2"]
        assert "mua" not in result["2"]

    def test_all_units_covered(self, get_labels):
        """All units present in quality_metrics are evaluated."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        assert set(result.keys()) >= {"0", "1"}


class TestGetLabelsSingleMetric:
    """Single-metric label_params works identically pre/post fix."""

    single_metric_params = {
        "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
    }
    quality_metrics = {
        "nn_noise_overlap": {"0": 0.2, "1": 0.05},
    }

    def test_single_metric_labels(self, get_labels):
        result = get_labels({}, self.quality_metrics, self.single_metric_params)
        assert "noise" in result["0"]
        assert "reject" in result["0"]

    def test_single_metric_no_match(self, get_labels):
        result = get_labels({}, self.quality_metrics, self.single_metric_params)
        assert "1" not in result


class TestGetLabelsEmpty:
    """Empty label_params returns parent_labels unchanged."""

    def test_empty_params_returns_parent(self, get_labels):
        parent = {"0": ["accept"]}
        result = get_labels(parent, {"metric": {"0": 1.0}}, {})
        assert result == {"0": ["accept"]}

    def test_empty_params_empty_parent(self, get_labels):
        result = get_labels({}, {"metric": {"0": 1.0}}, {})
        assert result == {}


class TestGetLabelsParentPreservation:
    """Existing parent labels are preserved and extended."""

    def test_parent_labels_preserved(self, get_labels):
        parent = {"0": ["accept"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels(parent, qm, params)
        assert "accept" in result["0"]
        assert "noise" in result["0"]

    def test_no_duplicate_labels(self, get_labels):
        parent = {"0": ["noise"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels(parent, qm, params)
        assert result["0"].count("noise") == 1

    def test_parent_labels_extended_by_second_metric(self, get_labels):
        """Parent labels from first metric are extended by second."""
        parent = {}
        params = {
            "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
            "isi_violation": [">", 0.5, ["mua"]],
        }
        qm = {
            "nn_noise_overlap": {"0": 0.2},
            "isi_violation": {"0": 0.8},
        }
        result = get_labels(parent, qm, params)
        assert set(result["0"]) == {"noise", "reject", "mua"}


class TestGetLabelsMetricSkipping:
    """Metrics not in quality_metrics are skipped, not errored."""

    def test_missing_metric_skipped(self, get_labels):
        params = {
            "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
            "nonexistent_metric": [">", 0.5, ["mua"]],
        }
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels({}, qm, params)
        assert "noise" in result["0"]
        assert "mua" not in result.get("0", [])

    def test_only_overlapping_metrics_apply(self, get_labels):
        """With 3 metrics, only 2 present in qm are applied."""
        params = {
            "nn_noise_overlap": [">", 0.1, ["noise"]],
            "missing": [">", 0.5, ["artifact"]],
            "isi_violation": [">", 0.5, ["mua"]],
        }
        qm = {
            "nn_noise_overlap": {"0": 0.2},
            "isi_violation": {"0": 0.8},
        }
        result = get_labels({}, qm, params)
        assert "noise" in result["0"]
        assert "mua" in result["0"]
        assert "artifact" not in result.get("0", [])


class TestGetLabelsComparisonOperators:
    """All comparison operators work correctly."""

    @pytest.mark.parametrize(
        "op,threshold,value,should_label",
        [
            (">", 0.5, 0.6, True),
            (">", 0.5, 0.5, False),
            (">=", 0.5, 0.5, True),
            ("<", 0.5, 0.4, True),
            ("<", 0.5, 0.5, False),
            ("<=", 0.5, 0.5, True),
            ("==", 0.5, 0.5, True),
            ("==", 0.5, 0.6, False),
        ],
    )
    def test_operator(self, get_labels, op, threshold, value, should_label):
        params = {"nn_noise_overlap": [op, threshold, ["reject"]]}
        qm = {"nn_noise_overlap": {"0": value}}
        result = get_labels({}, qm, params)
        if should_label:
            assert "0" in result and "reject" in result["0"]
        else:
            assert "0" not in result
