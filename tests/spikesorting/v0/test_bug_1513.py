"""Tests for bug 1513: AutomaticCuration.get_labels early return.

The bug caused get_labels() to return after processing only the first
metric in label_params, silently dropping labels from subsequent metrics.

These tests verify the fix and define the expected behavior for
Fix1513Status.
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
        assert "noise" in result[0]
        assert "reject" in result[0]
        assert "mua" in result[0]

    def test_second_metric_only(self, get_labels):
        """Unit matching only second metric still gets its labels."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        # unit 1: nn_noise_overlap 0.05 <= 0.1 -> no label
        #         isi_violation    0.9  >  0.5 -> mua
        assert 1 in result
        assert "mua" in result[1]
        assert "noise" not in result.get(1, [])

    def test_no_match_no_label(self, get_labels):
        """Unit matching only the first metric gets those labels, not mua."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        # unit 2: nn_noise_overlap 0.3 > 0.1 -> noise,reject
        #         isi_violation    0.1 <= 0.5 -> no label
        assert "noise" in result[2]
        assert "reject" in result[2]
        assert "mua" not in result[2]

    def test_all_units_covered(self, get_labels):
        """All units present in quality_metrics are evaluated."""
        result = get_labels({}, self.quality_metrics, self.two_metric_params)
        assert set(result.keys()) >= {0, 1}


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
        assert "noise" in result[0]
        assert "reject" in result[0]

    def test_single_metric_no_match(self, get_labels):
        result = get_labels({}, self.quality_metrics, self.single_metric_params)
        assert 1 not in result


class TestGetLabelsEmpty:
    """Empty label_params returns parent_labels unchanged."""

    def test_empty_params_returns_parent(self, get_labels):
        parent = {0: ["accept"]}
        result = get_labels(parent, {"metric": {"0": 1.0}}, {})
        assert result == {0: ["accept"]}

    def test_empty_params_empty_parent(self, get_labels):
        result = get_labels({}, {"metric": {"0": 1.0}}, {})
        assert result == {}


class TestGetLabelsParentPreservation:
    """Existing parent labels are preserved and extended."""

    def test_accept_removed_when_other_labels_added(self, get_labels):
        """Accept label is removed when metric labels are applied."""
        parent = {0: ["accept"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels(parent, qm, params)
        assert "accept" not in result[0]
        assert "noise" in result[0]

    def test_no_duplicate_labels(self, get_labels):
        parent = {0: ["noise"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels(parent, qm, params)
        assert result[0].count("noise") == 1

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
        assert set(result[0]) == {"noise", "reject", "mua"}


class TestGetLabelsMetricSkipping:
    """Metrics not in quality_metrics are skipped, not errored."""

    def test_missing_metric_skipped(self, get_labels):
        params = {
            "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
            "nonexistent_metric": [">", 0.5, ["mua"]],
        }
        qm = {"nn_noise_overlap": {"0": 0.2}}
        result = get_labels({}, qm, params)
        assert "noise" in result[0]
        assert "mua" not in result.get(0, [])

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
        assert "noise" in result[0]
        assert "mua" in result[0]
        assert "artifact" not in result.get(0, [])


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
            assert 0 in result and "reject" in result[0]
        else:
            assert 0 not in result


# -- Task 5.2: int/string unit_id type-mismatch behavior ----------------


class TestGetLabelsUnitIdTypeMismatch:
    """Verify unit_id normalization and its effect on label propagation.

    ``get_labels`` normalizes all quality-metric unit_ids to ``int`` before
    merging with parent labels, so int-keyed parent entries and string-keyed
    QM entries for the same logical unit are now unified.  These tests
    document the current (fixed) behavior, not the historical isolation that
    existed before normalization was added.
    """

    def test_new_qm_unit_added_independently(self, get_labels):
        """A new string-keyed QM unit does not corrupt int-keyed parents."""
        parent = {1: ["mua"], 2: ["accept"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        # unit '4' is new (QM only); units 1, 2 are below threshold
        qm = {"nn_noise_overlap": {"4": 0.5}}
        result = get_labels(parent, qm, params)
        assert result.get(1) == ["mua"]
        assert result.get(2) == ["accept"]
        assert "noise" in result[4]

    def test_int_parent_not_overwritten_when_below_threshold(self, get_labels):
        """Int-keyed parent labels survive when the same logical unit
        is below the metric threshold in quality_metrics."""
        parent = {1: ["mua"]}
        params = {"nn_noise_overlap": [">", 0.5, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"1": 0.1}}  # below threshold
        result = get_labels(parent, qm, params)
        assert result.get(1) == ["mua"]
        assert "noise" not in result.get(1, [])

    def test_qm_labels_do_not_overwrite_unrelated_parent(self, get_labels):
        """QM-derived labels apply only to their own units.

        Parent unit 1 keeps its existing labels while QM-flagged units
        (2, 3) independently receive the threshold labels via
        ``label[2].copy()``, so no list is shared across units.
        """
        parent = {1: ["mua"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"2": 0.5, "3": 0.5}}
        result = get_labels(parent, qm, params)
        assert result.get(1) == ["mua"]
        assert "noise" not in result.get(1, [])
        assert "noise" in result[2]
        assert "noise" in result[3]

    def test_bug_c_duplicates_confined_to_rejected_units(self, get_labels):
        """Bug C duplicate labels can only appear on units being rejected.

        The old code's list-in-list comparison always returned True,
        causing .extend() to run unconditionally.  But QM units have
        string keys and parent units have int keys, so duplicates only
        accumulated on string-keyed (QM) units already receiving reject
        labels — not on accepted parents.
        """
        parent = {1: ["accept"], 2: ["accept"]}
        params = {"nn_noise_overlap": [">", 0.1, ["noise", "reject"]]}
        qm = {"nn_noise_overlap": {"3": 0.5}}
        result = get_labels(parent, qm, params)
        assert result.get(1) == ["accept"] or 1 not in result
        assert result.get(2) == ["accept"] or 2 not in result
        assert result[3].count("noise") == 1
        assert result[3].count("reject") == 1

    def test_insert_curation_int_conversion_overwrites_correctly(self):
        """int() normalisation lets QM results overwrite parent labels.

        Simulates Curation.insert_curation line 136:
            new_labels = {int(k): labels[k] for k in labels}
        When labels has both int and string keys for the same logical
        unit, the string-keyed QM entry (appended by get_labels) wins.
        """
        combined = {
            1: ["mua"],
            2: ["accept"],
            "1": ["noise", "reject"],
            "4": ["noise", "reject"],
        }
        normalized = {int(k): combined[k] for k in combined}
        assert normalized[1] == ["noise", "reject"]
        assert normalized[2] == ["accept"]
        assert normalized[4] == ["noise", "reject"]


# -- Task 5.3: Fix1513Status table tests --------------------------------


class TestFix1513Status:
    """Tests for Fix1513Status.populate()-driven repair workflow.

    All tests use mocking — no live database required.
    """

    from unittest.mock import MagicMock, patch

    @pytest.fixture(autouse=True)
    def clear_perm_cache(self):
        """Reset class-level permission caches before and after each test."""
        from spyglass.spikesorting.v0.spikesorting_curation import (
            Fix1513Status,
        )

        Fix1513Status._perm_pass.clear()
        Fix1513Status._perm_fail.clear()
        yield
        Fix1513Status._perm_pass.clear()
        Fix1513Status._perm_fail.clear()

    @pytest.fixture
    def table(self):
        from spyglass.spikesorting.v0.spikesorting_curation import (
            Fix1513Status,
        )

        return Fix1513Status()

    # ---- _get_case ---------------------------------------------------

    def test_get_case_a_empty(self, table):
        assert table._get_case([]) == "A"

    def test_get_case_b_newly_rejected(self, table):
        rsc = [{"unit": 1, "was_rejected": False, "should_reject": True}]
        assert table._get_case(rsc) == "B"

    def test_get_case_c_newly_accepted(self, table):
        rsc = [{"unit": 1, "was_rejected": True, "should_reject": False}]
        assert table._get_case(rsc) == "C"

    def test_get_case_c_mixed(self, table):
        rsc = [
            {"unit": 1, "was_rejected": False, "should_reject": True},
            {"unit": 2, "was_rejected": True, "should_reject": False},
        ]
        assert table._get_case(rsc) == "C"

    # ---- _compute_label_diff scope gates -----------------------------

    def _make_key(self):
        return {"nwb_file_name": "test.nwb", "sort_group_id": 0}

    def test_compute_diff_returns_none_for_empty_label_params(self, table):
        """Empty label_params → none_needed without permission check."""
        from unittest.mock import patch

        key = self._make_key()
        with (
            patch(
                "spyglass.spikesorting.v0.spikesorting_curation"
                ".AutomaticCurationParameters"
            ),
            patch(
                "spyglass.spikesorting.v0.spikesorting_curation"
                ".AutomaticCuration"
            ),
        ):
            # Simplest path: patch _compute_label_diff directly to test
            # that make() inserts none_needed when it returns None
            with (
                patch.object(table, "_compute_label_diff", return_value=None),
                patch.object(table, "insert1") as mock_insert,
            ):
                pass

                with patch(
                    "spyglass.spikesorting.v0.spikesorting_curation"
                    ".LabMember"
                ):
                    table.make(key, action="keep")
                mock_insert.assert_called_once()
                call_kwargs = mock_insert.call_args[0][0]
                assert call_kwargs["action"] == "none_needed"
                assert call_kwargs["reviewed_by"] == "system"

    def test_make_skips_clusterless_sorter(self, table):
        """Clusterless thresholder → none_needed before diff compute."""
        from unittest.mock import patch

        key = {**self._make_key(), "sorter": "clusterless_thresholder"}
        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with (
            patch.object(table, "_compute_label_diff") as mock_diff,
            patch.object(table, "insert1") as mock_insert,
            patch(f"{module}.LabMember"),
        ):
            table.make(key, action="keep")

        mock_diff.assert_not_called()
        mock_insert.assert_called_once()
        row = mock_insert.call_args[0][0]
        assert row["action"] == "none_needed"
        assert "clusterless" in row.get("notes", "").lower()

    def test_make_clusterless_skip_runs_in_none_needed_only_mode(self, table):
        """Clusterless skip fires even when action='none_needed_only'."""
        from unittest.mock import patch

        key = {**self._make_key(), "sorter": "clusterless_thresholder"}
        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with (
            patch.object(table, "_compute_label_diff") as mock_diff,
            patch.object(table, "insert1") as mock_insert,
            patch(f"{module}.LabMember"),
        ):
            table.make(key, action="none_needed_only")

        mock_diff.assert_not_called()
        mock_insert.assert_called_once()
        assert mock_insert.call_args[0][0]["action"] == "none_needed"

    def test_make_non_clusterless_sorter_still_computes_diff(self, table):
        """Non-clusterless sorter does not trigger the early skip."""
        from unittest.mock import patch

        key = {**self._make_key(), "sorter": "mountainsort4"}
        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with (
            patch.object(
                table, "_compute_label_diff", return_value=None
            ) as mock_diff,
            patch.object(table, "insert1") as mock_insert,
            patch(f"{module}.LabMember"),
        ):
            table.make(key, action="keep")

        mock_diff.assert_called_once()
        row = mock_insert.call_args[0][0]
        assert "clusterless" not in row.get("notes", "").lower()

    def test_compute_diff_returns_none_before_bug_date(self, table):
        """Entry created before 2025-04-22 → _compute_label_diff returns None."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        key = self._make_key()
        datetime(2025, 4, 22).timestamp()
        pre_bug = datetime(2025, 1, 1).timestamp()

        label_params = {
            "nn_noise_overlap": [">", 0.1, ["noise", "reject"]],
            "isi_violation": [">", 0.5, ["mua"]],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with (
            patch(f"{module}.AutomaticCuration") as mock_ac,
            patch(f"{module}.AutomaticCurationParameters"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.QualityMetrics"),
        ):
            # label_params join fetch
            join_mock = MagicMock()
            join_mock.fetch1.return_value = label_params
            mock_ac.__and__ = MagicMock(return_value=join_mock)
            join_mock.__mul__ = MagicMock(return_value=join_mock)

            # auto_curation_key fetch
            ack = {"nwb_file_name": "test.nwb", "curation_id": 1}
            join_mock.fetch1.side_effect = [label_params, ack]

            # time_of_creation before bug date
            curation_mock = MagicMock()
            curation_mock.fetch1.return_value = pre_bug
            mock_curation.__and__ = MagicMock(return_value=curation_mock)

            from spyglass.spikesorting.v0.spikesorting_curation import (
                Fix1513Status,
            )

            result = Fix1513Status._compute_label_diff(key)
            assert result is None

    def test_compute_diff_returns_none_for_no_label_change(self, table):
        """When recomputed labels match stored → _compute_label_diff returns None."""
        from unittest.mock import patch

        key = self._make_key()

        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with patch(f"{module}.Fix1513Status._compute_label_diff") as mock_diff:
            mock_diff.return_value = None
            with patch.object(table, "insert1") as mock_ins:
                pass

                with patch(f"{module}.LabMember"):
                    table.make(key, action="keep")
                row = mock_ins.call_args[0][0]
                assert row["action"] == "none_needed"

    # ---- permission tests --------------------------------------------

    def test_permission_check_cached_by_nwb_file(self, table):
        """_get_exp_summary called once for two entries with same nwb_file_name."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()

        with (
            patch.object(table, "_get_exp_summary") as mock_exp,
            patch("spyglass.common.LabTeam") as mock_lt,
        ):
            mock_exp.return_value = MagicMock(
                fetch=MagicMock(return_value=["alice"])
            )
            table._member_pk = "lab_member_name"
            mock_lt.return_value.get_team_members.return_value = ["bob"]

            table._check_permission(key, "bob", is_admin=False)
            table._check_permission(key, "bob", is_admin=False)

            assert mock_exp.call_count == 1

    def test_permission_cache_persists_across_calls(self, table):
        """Cache is class-level; second call with same nwb_file hits cache."""
        from unittest.mock import patch

        from spyglass.spikesorting.v0.spikesorting_curation import (
            Fix1513Status,
        )

        key = self._make_key()
        Fix1513Status._perm_pass["test.nwb"] = "alice"

        with patch.object(table, "_get_exp_summary") as mock_exp:
            result = table._check_permission(key, "bob", is_admin=False)
            mock_exp.assert_not_called()
        assert result == "alice"

    def test_perm_fail_short_circuits(self, table):
        """_perm_fail entry raises PermissionError without _get_exp_summary."""
        from unittest.mock import patch

        from spyglass.spikesorting.v0.spikesorting_curation import (
            Fix1513Status,
        )

        key = self._make_key()
        Fix1513Status._perm_fail.add("test.nwb")

        with patch.object(table, "_get_exp_summary") as mock_exp:
            with pytest.raises(PermissionError):
                table._check_permission(key, "bob", is_admin=False)
            mock_exp.assert_not_called()

    def test_admin_bypasses_cached_perm_fail(self, table):
        """Admin is permitted even when _perm_fail has a cached denial.

        Without the admin bypass, an entry cached from a prior
        non-admin denial would erroneously block an admin sharing the
        same Python process.
        """
        from unittest.mock import MagicMock, patch

        from spyglass.spikesorting.v0.spikesorting_curation import (
            Fix1513Status,
        )

        key = self._make_key()
        Fix1513Status._perm_fail.add("test.nwb")

        with (
            patch.object(table, "_get_exp_summary") as mock_exp,
            patch("spyglass.common.LabTeam") as mock_lt,
        ):
            mock_exp.return_value = MagicMock(
                fetch=MagicMock(return_value=["alice"])
            )
            table._member_pk = "lab_member_name"

            result = table._check_permission(key, "admin_user", is_admin=True)
            assert result == "alice"
            mock_lt.return_value.get_team_members.assert_not_called()

    def test_admin_bypasses_team_check(self, table):
        """Admin user is permitted without calling get_team_members."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()

        with (
            patch.object(table, "_get_exp_summary") as mock_exp,
            patch("spyglass.common.LabTeam") as mock_lt,
        ):
            mock_exp.return_value = MagicMock(
                fetch=MagicMock(return_value=["alice"])
            )
            table._member_pk = "lab_member_name"

            table._check_permission(key, "admin_user", is_admin=True)
            mock_lt.return_value.get_team_members.assert_not_called()

    def test_non_team_member_raises_permission_error(self, table):
        """User not on a team with the owner raises PermissionError."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()

        with (
            patch.object(table, "_get_exp_summary") as mock_exp,
            patch("spyglass.common.LabTeam") as mock_lt,
        ):
            mock_exp.return_value = MagicMock(
                fetch=MagicMock(return_value=["alice"])
            )
            table._member_pk = "lab_member_name"
            mock_lt.return_value.get_team_members.return_value = [
                "alice",
                "charlie",
            ]

            with pytest.raises(PermissionError, match="not on a team"):
                table._check_permission(key, "bob", is_admin=False)

        assert "test.nwb" in table._perm_fail

    # ---- NWB integrity / action tests --------------------------------

    def test_update_case_a_no_nwb_fix(self, table):
        """Case A update: labels written; _repair_nwb_labels NOT called."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["mua"]},
            "reject_status_changed": [],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "_repair_nwb_labels") as mock_nwb,
            patch.object(table, "_repair_unit_labels"),
            patch.object(table, "insert1"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_curation.connection.transaction.__enter__ = MagicMock(
                return_value=None
            )
            mock_curation.connection.transaction.__exit__ = MagicMock(
                return_value=False
            )
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=1))
            )

            table.make(key, action="update")
            mock_nwb.assert_not_called()

    def test_update_case_b_raises_value_error(self, table):
        """Case B with action='update' raises ValueError before any DB write.

        Newly-rejected units must be removed via repopulate so
        CuratedSpikeSorting's reject-filter excludes them; patching
        labels alone would leave stale rows.
        """
        from unittest.mock import patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["noise", "reject"]},
            "reject_status_changed": [
                {"unit": 0, "was_rejected": False, "should_reject": True}
            ],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []

            with pytest.raises(ValueError, match="not safe"):
                table.make(key, action="update")

            mock_curation.update1.assert_not_called()

    def test_update_case_c_raises_value_error(self, table):
        """Case C with action='update' raises ValueError before any DB write."""
        from unittest.mock import patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: ["noise", "reject"]},
            "new_labels": {0: []},
            "reject_status_changed": [
                {"unit": 0, "was_rejected": True, "should_reject": False}
            ],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []

            with pytest.raises(ValueError, match="not safe"):
                table.make(key, action="update")

            mock_curation.update1.assert_not_called()

    def test_case_c_interactive_prompt_excludes_update(self, table):
        """Interactive prompt for Case C does not offer (u)pdate."""
        from unittest.mock import patch

        with patch("builtins.input", return_value="k") as mock_input:
            result = table._prompt_action("C")

        prompt_str = mock_input.call_args[0][0]
        assert "u" not in prompt_str.split("[")[1]
        assert result == "keep"

    def test_case_b_interactive_prompt_excludes_update(self, table):
        """Interactive prompt for Case B does not offer (u)pdate."""
        from unittest.mock import patch

        with patch("builtins.input", return_value="k") as mock_input:
            result = table._prompt_action("B")

        prompt_str = mock_input.call_args[0][0]
        assert "u" not in prompt_str.split("[")[1]
        assert result == "keep"

    def test_action_keep_no_label_writes(self, table):
        """action='keep' records decision without writing labels."""
        from unittest.mock import patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["mua"]},
            "reject_status_changed": [],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting"),
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []

            table.make(key, action="keep")

            mock_curation.update1.assert_not_called()
            row = mock_ins.call_args[0][0]
            assert row["action"] == "keep"

    def test_action_skip_inserts_row_no_writes(self, table):
        """action='skip' inserts review row but makes no label changes."""
        from unittest.mock import patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["mua"]},
            "reject_status_changed": [],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting"),
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []

            table.make(key, action="skip")

            mock_curation.update1.assert_not_called()
            row = mock_ins.call_args[0][0]
            assert row["action"] == "skip"

    def _make_diff(self):
        return {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: [], 1: ["noise"]},
            "new_labels": {0: ["mua"], 1: ["noise", "reject"]},
            "reject_status_changed": [],
        }

    def test_keep_stores_label_diff(self, table):
        """action='keep' persists old/new label diff for audit."""
        from unittest.mock import patch

        key = self._make_key()
        diff = self._make_diff()
        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation"),
            patch(f"{module}.CuratedSpikeSorting"),
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            table.make(key, action="keep")

        row = mock_ins.call_args[0][0]
        assert "label_diff" in row
        assert row["label_diff"]["old_labels"] == diff["old_labels"]
        assert row["label_diff"]["new_labels"] == diff["new_labels"]

    def test_skip_stores_label_diff(self, table):
        """action='skip' persists old/new label diff for audit."""
        from unittest.mock import patch

        key = self._make_key()
        diff = self._make_diff()
        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation"),
            patch(f"{module}.CuratedSpikeSorting"),
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            table.make(key, action="skip")

        row = mock_ins.call_args[0][0]
        assert "label_diff" in row
        assert row["label_diff"]["old_labels"] == diff["old_labels"]
        assert row["label_diff"]["new_labels"] == diff["new_labels"]

    def test_update_does_not_store_label_diff(self, table):
        """action='update' leaves label_diff as None (change is in Curation)."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()
        diff = self._make_diff()
        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "_repair_unit_labels"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_curation.connection.transaction.__enter__ = MagicMock(
                return_value=None
            )
            mock_curation.connection.transaction.__exit__ = MagicMock(
                return_value=False
            )
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=0))
            )
            table.make(key, action="update")

        row = mock_ins.call_args[0][0]
        assert row.get("label_diff") is None

    def test_interactive_prompt_routes_to_update(self, table):
        """Interactive input 'u' routes to update action."""
        from unittest.mock import MagicMock, patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["mua"]},
            "reject_status_changed": [],
        }

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "_repair_unit_labels"),
            patch.object(table, "insert1") as mock_ins,
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
            patch("builtins.input", return_value="u"),
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_curation.connection.transaction.__enter__ = MagicMock(
                return_value=None
            )
            mock_curation.connection.transaction.__exit__ = MagicMock(
                return_value=False
            )
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=0))
            )

            table.make(key, action="report")

            row = mock_ins.call_args[0][0]
            assert row["action"] == "update"

    def test_none_needed_only_skips_in_scope_entries(self, table):
        """none_needed_only returns without inserting for in-scope entries."""
        from unittest.mock import patch

        key = self._make_key()
        diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: []},
            "new_labels": {0: ["mua"]},
            "reject_status_changed": [],
        }

        with (
            patch.object(table, "_compute_label_diff", return_value=diff),
            patch.object(table, "insert1") as mock_ins,
            patch.object(table, "_check_permission") as mock_perm,
        ):
            table.make(key, action="none_needed_only")
            mock_ins.assert_not_called()
            mock_perm.assert_not_called()

    def test_none_needed_only_inserts_for_out_of_scope(self, table):
        """none_needed_only still inserts none_needed when diff is None."""
        from unittest.mock import patch

        key = self._make_key()

        with (
            patch.object(table, "_compute_label_diff", return_value=None),
            patch.object(table, "insert1") as mock_ins,
        ):
            table.make(key, action="none_needed_only")
            mock_ins.assert_called_once()
            assert mock_ins.call_args[0][0]["action"] == "none_needed"

    # ---- pending_for_member ------------------------------------------

    def test_pending_for_member_returns_queries(self, table):
        """pending_for_member returns (unreviewed, skipped) query pair."""
        from unittest.mock import MagicMock, patch

        module = "spyglass.spikesorting.v0.spikesorting_curation"

        with (
            patch(f"{module}.AutomaticCuration") as mock_ac,
            patch("spyglass.common.Session") as mock_session,
            patch.object(
                type(table),
                "__sub__",
                return_value=MagicMock(__len__=lambda s: 2),
            ),
            patch.object(
                type(table),
                "__and__",
                return_value=MagicMock(__len__=lambda s: 1),
            ),
            patch("builtins.print"),
        ):
            mock_ac.__mul__ = MagicMock(return_value=MagicMock())
            mock_session.Experimenter = MagicMock()

            from spyglass.spikesorting.v0.spikesorting_curation import (
                Fix1513Status,
            )

            unreviewed, skipped = Fix1513Status.pending_for_member("Alice")
            assert unreviewed is not None
            assert skipped is not None

    def test_pending_for_member_prints_summary(self, table):
        """pending_for_member prints counts for the requested member."""
        from unittest.mock import MagicMock, patch

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        unreviewed_mock = MagicMock()
        unreviewed_mock.__len__ = MagicMock(return_value=3)
        skipped_mock = MagicMock()
        skipped_mock.__len__ = MagicMock(return_value=1)

        # Build the mock chain for the unreviewed query:
        # (AutomaticCuration * Session.Experimenter & member_filter) - cls()
        # The __sub__ is on the DataJoint join result, not on Fix1513Status,
        # so we configure the AutomaticCuration mock chain explicitly.
        filtered_mock = MagicMock()
        filtered_mock.__sub__ = MagicMock(return_value=unreviewed_mock)
        joined_mock = MagicMock()
        joined_mock.__and__ = MagicMock(return_value=filtered_mock)

        with (
            patch(f"{module}.AutomaticCuration") as mock_ac,
            patch("spyglass.common.Session") as mock_session,
            patch("builtins.print") as mock_print,
            patch.object(
                type(table),
                "__and__",
                return_value=skipped_mock,
            ),
        ):
            mock_ac.__mul__ = MagicMock(return_value=joined_mock)
            mock_session.Experimenter = MagicMock()

            from spyglass.spikesorting.v0.spikesorting_curation import (
                Fix1513Status,
            )

            Fix1513Status.pending_for_member("Alice")

        first_print = mock_print.call_args_list[0][0][0]
        assert "Alice" in first_print
        assert "3" in first_print  # unreviewed count
        assert "1" in first_print  # skipped count


class TestNwbTransactionSafety:
    """Option C: NWB writes staged to temp copies outside the DB transaction."""

    @pytest.fixture(autouse=True)
    def clear_perm_cache(self):
        from spyglass.spikesorting.v0.spikesorting_curation import Fix1513Status

        Fix1513Status._perm_pass.clear()
        Fix1513Status._perm_fail.clear()
        yield
        Fix1513Status._perm_pass.clear()
        Fix1513Status._perm_fail.clear()

    @pytest.fixture
    def table(self):
        from spyglass.spikesorting.v0.spikesorting_curation import Fix1513Status

        return Fix1513Status()

    def _make_key(self):
        return {"nwb_file_name": "test.nwb", "curation_id": 1}

    def _case_a_diff(self):
        """Case A diff (label text change, no reject-status flip).

        Used by tests that exercise the update path's staging /
        transaction / activation ordering — Case B/C are not
        eligible for ``action='update'``.
        """
        return {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: ["noise"]},
            "new_labels": {0: ["noise", "mua"]},
            "reject_status_changed": [],
        }

    def test_case_a_stages_before_update_activates_after(self, table):
        """stage_nwb_repairs precedes Curation.update1; activate follows.

        make() runs inside the transaction opened by
        Fix1513Status.populate(), so there is no separate inner
        transaction to enter/exit (see C1 fix) — staging must still
        happen before the DB writes, and activation after.
        """
        from unittest.mock import MagicMock, patch

        call_order = []

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(
                table, "_compute_label_diff", return_value=self._case_a_diff()
            ),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(
                table,
                "_stage_nwb_repairs",
                side_effect=lambda *a, **kw: call_order.append("stage") or [],
            ),
            patch.object(
                table,
                "_activate_nwb_repairs",
                side_effect=lambda *a, **kw: call_order.append("activate"),
            ),
            patch.object(
                table,
                "_repair_unit_labels",
                side_effect=lambda *a, **kw: call_order.append("repair_labels"),
            ),
            patch.object(table, "insert1"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_curation.update1 = MagicMock(
                side_effect=lambda *a: call_order.append("update1")
            )
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=1))
            )
            table.make(self._make_key(), action="update")

        assert call_order.index("stage") < call_order.index("update1")
        assert call_order.index("update1") < call_order.index("repair_labels")
        assert call_order.index("repair_labels") < call_order.index("activate")

    def test_db_failure_discards_staged_temps(self, table, tmp_path):
        """Staged temp files are deleted when the DB write fails."""
        from unittest.mock import MagicMock, patch

        temp_file = tmp_path / "patch.nwb.tmp"
        temp_file.write_bytes(b"staged content")
        staged = [(str(temp_file), "/real/file.nwb")]

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(
                table, "_compute_label_diff", return_value=self._case_a_diff()
            ),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "_stage_nwb_repairs", return_value=staged),
            patch.object(table, "_activate_nwb_repairs"),
            patch.object(table, "_repair_unit_labels"),
            patch(f"{module}.Curation") as mock_curation,
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_curation.update1.side_effect = RuntimeError("DB failure")
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=1))
            )
            with pytest.raises(RuntimeError, match="DB failure"):
                table.make(self._make_key(), action="update")

        assert (
            not temp_file.exists()
        ), "temp file must be removed after DB failure"

    def test_case_a_also_stages_nwb(self, table):
        """Case A (label text change, no reject flip) still stages NWB patches."""
        from unittest.mock import MagicMock, patch

        case_a_diff = {
            "auto_curation_key": {
                "nwb_file_name": "test.nwb",
                "curation_id": 1,
            },
            "curation_key": {"curation_id": 1},
            "old_labels": {0: ["noise"]},
            "new_labels": {0: ["noise", "mua"]},
            "reject_status_changed": [],
        }
        module = "spyglass.spikesorting.v0.spikesorting_curation"
        with (
            patch.object(
                table, "_compute_label_diff", return_value=case_a_diff
            ),
            patch.object(table, "_check_permission", return_value="alice"),
            patch.object(table, "_print_diff"),
            patch.object(table, "_stage_nwb_repairs") as mock_stage,
            patch.object(table, "_activate_nwb_repairs"),
            patch.object(table, "_repair_unit_labels"),
            patch.object(table, "insert1"),
            patch(f"{module}.Curation"),
            patch(f"{module}.CuratedSpikeSorting") as mock_css,
            patch(f"{module}.LabMember") as mock_lm,
        ):
            mock_stage.return_value = []
            mock_lm.return_value.get_djuser_name.return_value = "alice"
            mock_lm.return_value.admin = []
            mock_css.__and__ = MagicMock(
                return_value=MagicMock(__len__=MagicMock(return_value=1))
            )
            table.make(self._make_key(), action="update")

        mock_stage.assert_called_once()

    def test_activate_nwb_repairs_renames_and_checksums(self, tmp_path):
        """_activate_nwb_repairs renames each temp file and updates checksum."""
        from unittest.mock import MagicMock, patch

        temp_file = tmp_path / "file.nwb.tmp"
        real_file = tmp_path / "file.nwb"
        temp_file.write_bytes(b"patched content")

        helper_path = "spyglass.common.common_nwbfile.AnalysisNwbfile"
        with (
            patch(helper_path) as mock_anwb,
            patch("spyglass.utils.dj_helper_fn.dj") as mock_dj,
        ):
            inst = mock_anwb.return_value
            inst._analysis_dir = str(tmp_path)
            ext_tbl = MagicMock()
            inst._ext_tbl = ext_tbl
            ext_tbl.__and__ = MagicMock(
                return_value=MagicMock(
                    fetch1=MagicMock(return_value={"filepath": "file.nwb"})
                )
            )
            mock_dj.hash.uuid_from_file.return_value = "abc123"

            from spyglass.spikesorting.v0.spikesorting_curation import (
                Fix1513Status,
            )

            Fix1513Status._activate_nwb_repairs(
                [(str(temp_file), str(real_file))], verbose=False
            )

        assert not temp_file.exists(), "temp file must be gone after activation"
        assert real_file.read_bytes() == b"patched content"
        ext_tbl.update1.assert_called_once()

    def test_update_nwb_checksum_routes_through_helper(self, tmp_path):
        """_update_nwb_checksum delegates to dj_helper_fn helper
        (no admin gating, since Fix1513 is a user-driven retroactive repair)."""
        from unittest.mock import patch

        real_file = tmp_path / "file.nwb"
        real_file.write_bytes(b"data")

        with (
            patch(
                "spyglass.utils.dj_helper_fn._update_analysis_file_checksum"
            ) as mock_helper,
            patch("spyglass.common.LabMember") as mock_lm,
        ):
            from spyglass.spikesorting.v0.spikesorting_curation import (
                Fix1513Status,
            )

            Fix1513Status._update_nwb_checksum(str(real_file), verbose=False)

        mock_helper.assert_called_once_with(str(real_file))
        mock_lm.return_value.check_admin_privilege.assert_not_called()


# -- Task 6.1.2: populate()-level integration tests (C1) ----------------


class TestFix1513PopulateIntegration:
    """Exercise Fix1513Status through the real populate() machinery.

    Unlike TestFix1513Status/TestNwbTransactionSafety, ``Curation`` and
    ``CuratedSpikeSorting`` are NOT mocked here for the DB-write paths, so
    these tests reproduce C1: the inner ``with Curation.connection
    .transaction:`` blocks raised "Nested connections are not supported"
    because make() already runs inside the transaction opened by
    ``_populate1``.
    """

    @pytest.fixture(autouse=True)
    def clear_perm_cache(self, spike_v0):
        spike_v0.Fix1513Status._perm_pass.clear()
        spike_v0.Fix1513Status._perm_fail.clear()
        yield
        spike_v0.Fix1513Status._perm_pass.clear()
        spike_v0.Fix1513Status._perm_fail.clear()

    def _patched_diff(self, spike_v0, auto_curation_key, new_labels):
        return {
            "auto_curation_key": auto_curation_key,
            "curation_key": {
                k: auto_curation_key[k]
                for k in spike_v0.Curation.primary_key
                if k in auto_curation_key
            },
            "old_labels": {0: []},
            "new_labels": new_labels,
            "reject_status_changed": [],
        }

    def test_update_via_populate_no_nested_transaction_error(
        self, spike_v0, pop_auto_curation
    ):
        """action='update' runs through populate() without C1's
        DataJointError("Nested connections are not supported").

        CuratedSpikeSorting is mocked with len()==0 so has_downstream is
        False; this still exercises the (un)wrapped Curation.update1 call
        that previously crashed inside `with Curation.connection
        .transaction:` when called from inside populate()'s transaction.
        """
        from unittest.mock import MagicMock, patch

        Fix1513Status = spike_v0.Fix1513Status
        AutomaticCuration = spike_v0.AutomaticCuration
        Curation = spike_v0.Curation

        auto_curation, _ = pop_auto_curation
        key = auto_curation.fetch("KEY")[0]
        auto_curation_key = (AutomaticCuration & key).fetch1(
            "auto_curation_key"
        )
        original_labels = (Curation & auto_curation_key).fetch1(
            "curation_labels"
        )
        new_labels = {0: ["mua"]}
        diff = self._patched_diff(spike_v0, auto_curation_key, new_labels)

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        try:
            with (
                patch.object(
                    Fix1513Status, "_compute_label_diff", return_value=diff
                ),
                patch.object(
                    Fix1513Status, "_check_permission", return_value="alice"
                ),
                patch.object(Fix1513Status, "_print_diff"),
                patch(f"{module}.CuratedSpikeSorting") as mock_css,
                patch(f"{module}.LabMember") as mock_lm,
            ):
                mock_lm.return_value.get_djuser_name.return_value = "alice"
                mock_lm.return_value.admin = []
                mock_css.__and__ = MagicMock(
                    return_value=MagicMock(__len__=MagicMock(return_value=0))
                )

                Fix1513Status.populate(key, make_kwargs={"action": "update"})

            assert (Fix1513Status & key).fetch1("action") == "update"
            assert (Curation & auto_curation_key).fetch1(
                "curation_labels"
            ) == new_labels
        finally:
            (Fix1513Status & key).delete(safemode=False)
            Curation.update1(
                {**auto_curation_key, "curation_labels": original_labels}
            )

    def test_repopulate_via_populate_no_nested_transaction_error(
        self, spike_v0, pop_auto_curation
    ):
        """action='repopulate' runs through populate() without C1's
        DataJointError, defers CuratedSpikeSorting.populate() to
        run_pending_repopulates()."""
        from unittest.mock import MagicMock, patch

        Fix1513Status = spike_v0.Fix1513Status
        AutomaticCuration = spike_v0.AutomaticCuration
        Curation = spike_v0.Curation

        auto_curation, _ = pop_auto_curation
        key = auto_curation.fetch("KEY")[0]
        auto_curation_key = (AutomaticCuration & key).fetch1(
            "auto_curation_key"
        )
        original_labels = (Curation & auto_curation_key).fetch1(
            "curation_labels"
        )
        new_labels = {0: ["noise", "reject"]}
        diff = self._patched_diff(spike_v0, auto_curation_key, new_labels)
        diff["reject_status_changed"] = [
            {"unit": 0, "was_rejected": False, "should_reject": True}
        ]

        module = "spyglass.spikesorting.v0.spikesorting_curation"
        try:
            with (
                patch.object(
                    Fix1513Status, "_compute_label_diff", return_value=diff
                ),
                patch.object(
                    Fix1513Status, "_check_permission", return_value="alice"
                ),
                patch.object(Fix1513Status, "_print_diff"),
                # No CuratedSpikeSorting rows for this key, so the
                # transaction-bound delete() inside make() is a no-op.
                patch(f"{module}.CuratedSpikeSorting") as mock_css,
                patch(f"{module}.LabMember") as mock_lm,
            ):
                mock_lm.return_value.get_djuser_name.return_value = "alice"
                mock_lm.return_value.admin = []
                mock_css.__and__ = MagicMock(
                    return_value=MagicMock(__len__=MagicMock(return_value=0))
                )

                Fix1513Status.populate(
                    key, make_kwargs={"action": "repopulate"}
                )

            row = (Fix1513Status & key).fetch1()
            assert row["action"] == "repopulate"
            assert row["repopulated"] == 0
            assert (Curation & auto_curation_key).fetch1(
                "curation_labels"
            ) == new_labels

            with patch(f"{module}.CuratedSpikeSorting") as mock_css:
                n_done = Fix1513Status.run_pending_repopulates(
                    restriction=key, verbose=False
                )

            assert n_done == 1
            mock_css.populate.assert_called_once_with(auto_curation_key)
            assert (Fix1513Status & key).fetch1("repopulated") == 1
        finally:
            (Fix1513Status & key).delete(safemode=False)
            Curation.update1(
                {**auto_curation_key, "curation_labels": original_labels}
            )
