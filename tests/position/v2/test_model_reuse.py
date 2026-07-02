"""Guardrails against DLC model proliferation.

Pure-logic and mocked-branch tests (no training). The DB join methods
(`with_skeleton`/`for_subject`/`reusable_for`) are exercised against real rows
in the integration test at the bottom (skipped without DLC).
"""

import pytest


class TestReuseCandidates:
    """Pure grouping/annotation helper (no DB)."""

    @pytest.fixture(autouse=True)
    def _cls(self, pv2_train):
        self.Model = pv2_train.Model

    def test_groups_rows_by_model_with_subjects(self):
        rows = [
            {"model_id": "m1", "subject_id": "A"},
            {"model_id": "m1", "subject_id": "B"},
            {"model_id": "m2", "subject_id": "A"},
        ]
        out = self.Model._reuse_candidates(rows)
        by_id = {c["model_id"]: c for c in out}
        assert by_id["m1"]["subjects"] == ["A", "B"]
        assert by_id["m2"]["subjects"] == ["A"]

    def test_overlap_reflects_query_subjects(self):
        rows = [
            {"model_id": "m1", "subject_id": "A"},
            {"model_id": "m1", "subject_id": "B"},
        ]
        (cand,) = self.Model._reuse_candidates(rows, subjects={"B", "Z"})
        assert cand["overlap"] == ["B"]

    def test_model_with_no_subject_kept_with_empty_set(self):
        rows = [{"model_id": "m1", "subject_id": None}]
        (cand,) = self.Model._reuse_candidates(rows)
        assert cand["subjects"] == [] and cand["overlap"] == []

    def test_sorted_and_deduped(self):
        rows = [
            {"model_id": "mb", "subject_id": "B"},
            {"model_id": "ma", "subject_id": "A"},
            {"model_id": "ma", "subject_id": "A"},  # dup
        ]
        out = self.Model._reuse_candidates(rows)
        assert [c["model_id"] for c in out] == ["ma", "mb"]
        assert out[0]["subjects"] == ["A"]  # deduped


class TestRedundantModelMessage:
    """The actionable redundant-model message (pure)."""

    @pytest.fixture(autouse=True)
    def _cls(self, pv2_train):
        self.MS = pv2_train.ModelSelection

    def test_message_lists_models_and_escape_hatch(self):
        cands = [
            {"model_id": "m1", "subjects": ["A"], "overlap": ["A"]},
            {"model_id": "m2", "subjects": ["B"], "overlap": []},
        ]
        msg = self.MS._redundant_model_msg("skelX", cands)
        assert "skelX" in msg
        assert "m1" in msg and "m2" in msg
        assert "overlaps ['A']" in msg  # overlap annotated
        assert "allow_redundant_model=True" in msg  # escape hatch documented


class TestGuardBranches:
    """Guard decision logic with DB touchpoints mocked."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, pv2_train):
        self.MS = pv2_train.ModelSelection
        self.Model = pv2_train.Model
        self.ms = self.MS()
        # Default: a skeleton exists, no own models, empty subjects.
        monkeypatch.setattr(
            self.MS, "_skeleton_for_params", staticmethod(lambda p: "skel1")
        )
        monkeypatch.setattr(
            self.MS, "_subjects", classmethod(lambda cls, row: set())
        )
        monkeypatch.setattr(
            self.MS, "_own_model_ids", classmethod(lambda cls, row: [])
        )
        self.monkeypatch = monkeypatch

    def _set_candidates(self, candidates):
        self.monkeypatch.setattr(
            self.Model,
            "reusable_for",
            classmethod(
                lambda cls, skel, subjects=None, exclude=None: candidates
            ),
        )

    def _row(self, **over):
        row = {"model_params_id": "mp1"}
        row.update(over)
        return row

    def test_raises_when_candidates_exist(self):
        self._set_candidates(
            [{"model_id": "m1", "subjects": ["A"], "overlap": ["A"]}]
        )
        with pytest.raises(ValueError, match="m1"):
            self.ms._guard_redundant_model(self._row(), False)

    def test_allow_flag_bypasses(self):
        self._set_candidates(
            [{"model_id": "m1", "subjects": [], "overlap": []}]
        )
        self.ms._guard_redundant_model(self._row(), True)  # no raise

    def test_fine_tune_exempt(self):
        self._set_candidates(
            [{"model_id": "m1", "subjects": [], "overlap": []}]
        )
        # parent_id set → intentional lineage, never gated
        self.ms._guard_redundant_model(self._row(parent_id="m0"), False)

    def test_no_skeleton_exempt(self):
        self.monkeypatch.setattr(
            self.MS, "_skeleton_for_params", staticmethod(lambda p: None)
        )
        self._set_candidates(
            [{"model_id": "m1", "subjects": [], "overlap": []}]
        )
        self.ms._guard_redundant_model(self._row(), False)  # no raise

    def test_no_candidates_passes(self):
        self._set_candidates([])
        self.ms._guard_redundant_model(self._row(), False)  # no raise

    def test_non_dict_row_skipped(self):
        self._set_candidates(
            [{"model_id": "m1", "subjects": [], "overlap": []}]
        )
        self.ms._guard_redundant_model(("mp1", "dlc"), False)  # no raise


class TestCreateProjectNudge:
    """Early warn (never raises) in create_project."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, model):
        self.model = model
        self.monkeypatch = monkeypatch

    def _patch_session(self, monkeypatch, subjects):
        import spyglass.common as common

        class _FakeSession:
            def __and__(self, restr):
                return self

            def fetch(self, *a, **k):
                return list(subjects)

        # method does a local `from spyglass.common import Session`
        monkeypatch.setattr(common, "Session", _FakeSession(), raising=False)

    def test_warns_when_existing_model_for_subject(self, monkeypatch):
        warned = {}
        monkeypatch.setattr(
            type(self.model),
            "_warn_msg",
            lambda self, msg: warned.setdefault("msg", msg),
        )
        self._patch_session(monkeypatch, ["subjX"])

        class _Q:
            def fetch(self, *a, **k):
                return ["mZ"]

        monkeypatch.setattr(
            type(self.model), "for_subject", classmethod(lambda cls, s: _Q())
        )

        self.model._warn_existing_models_for_videos(
            [{"nwb_file_name": "x_.nwb", "epoch": 1}]
        )
        assert "mZ" in warned.get("msg", "")
        assert "subjX" in warned["msg"]

    def test_silent_when_none(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            type(self.model),
            "_warn_msg",
            lambda self, msg: called.setdefault("msg", msg),
        )
        self._patch_session(monkeypatch, [])
        self.model._warn_existing_models_for_videos(
            [{"nwb_file_name": "x_.nwb", "epoch": 1}]
        )
        assert "msg" not in called
