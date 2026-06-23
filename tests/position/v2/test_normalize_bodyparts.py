"""Round-trip tests for body-part name canonicalization.

Validates the full normalize path end to end: tool-native (messy) names in,
canonical names out at every stage, with pose values unchanged. Function- and
DB-level round-trips run unconditionally; the live DLC import round-trip is
gated by ``skip_if_no_dlc``.
"""

import datajoint as dj
import numpy as np
import pandas as pd
import pytest

SCORER = "scorer"
N = 6


def _messy_df(name_to_xy):
    """Build a (scorer, bodypart, coord) df from {name: (x_arr, y_arr)}.

    Likelihood is a fixed ramp so it can be asserted to round-trip too.
    """
    tuples, columns = [], []
    like = np.linspace(0.6, 1.0, N)
    for bp, (x, y) in name_to_xy.items():
        for coord, col in (("x", x), ("y", y), ("likelihood", like)):
            tuples.append((SCORER, bp, coord))
            columns.append(col)
    cols = pd.MultiIndex.from_tuples(
        tuples, names=["scorer", "bodyparts", "coords"]
    )
    return pd.DataFrame(np.column_stack(columns), columns=cols)


def _series_from(df, bp):
    """Build an ndx-pose series named '<bp>_pose' from a canonical df."""
    import ndx_pose

    return ndx_pose.PoseEstimationSeries(
        name=f"{bp}_pose",
        description="round-trip",
        data=np.column_stack(
            [df[(SCORER, bp, "x")].to_numpy(), df[(SCORER, bp, "y")].to_numpy()]
        ),
        unit="pixels",
        reference_frame="(0,0)",
        timestamps=np.arange(N, dtype=float),
        confidence=df[(SCORER, bp, "likelihood")].to_numpy(),
    )


class _Container:
    def __init__(self, series):
        self.pose_estimation_series = {s.name: s for s in series}


class TestInMemoryRoundTrip:
    """Messy names in -> canonical names out, pose values bit-identical."""

    def test_write_then_read_canonicalizes_without_changing_values(self):
        from spyglass.position.v2.estim import (
            canonicalize_pose_columns,
            pose_estimation_to_dataframe,
        )
        from spyglass.position.v2.utils.skeleton import build_canonical_map

        rng = np.random.default_rng(0)
        green_xy = (rng.normal(size=N), rng.normal(size=N))
        red_xy = (rng.normal(size=N), rng.normal(size=N))
        messy = _messy_df({"green_led": green_xy, "red_led_c": red_xy})

        canon_map = build_canonical_map(["greenLED", "redLED_C"])

        # write boundary: tool-native columns -> canonical columns
        canon_df, bodyparts = canonicalize_pose_columns(
            messy, ["green_led", "red_led_c"], canon_map
        )
        assert bodyparts == ["greenLED", "redLED_C"]

        # NWB shape + read boundary
        pe = _Container([_series_from(canon_df, bp) for bp in bodyparts])
        read_df = pose_estimation_to_dataframe(pe, SCORER, False, canon_map)

        # every stage is canonical
        assert list(read_df.columns.get_level_values("bodyparts").unique()) == [
            "greenLED",
            "redLED_C",
        ]

        # values unchanged end to end (messy 'green_led' == read 'greenLED')
        np.testing.assert_array_equal(
            read_df[(SCORER, "greenLED", "x")].to_numpy(), green_xy[0]
        )
        np.testing.assert_array_equal(
            read_df[(SCORER, "redLED_C", "y")].to_numpy(), red_xy[1]
        )

    def test_matches_canonical_control_run(self):
        """A messy run and an already-canonical run give identical output."""
        from spyglass.position.v2.estim import (
            canonicalize_pose_columns,
            pose_estimation_to_dataframe,
        )
        from spyglass.position.v2.utils.skeleton import build_canonical_map

        rng = np.random.default_rng(1)
        xy = (rng.normal(size=N), rng.normal(size=N))
        canon_map = build_canonical_map(["greenLED"])

        messy = _messy_df({"green_led": xy})
        canon = _messy_df({"greenLED": xy})

        m_df, m_bp = canonicalize_pose_columns(messy, ["green_led"], canon_map)
        c_df, c_bp = canonicalize_pose_columns(canon, ["greenLED"], canon_map)

        m_read = pose_estimation_to_dataframe(
            _Container([_series_from(m_df, b) for b in m_bp]),
            SCORER,
            False,
            canon_map,
        )
        c_read = pose_estimation_to_dataframe(
            _Container([_series_from(c_df, b) for b in c_bp]),
            SCORER,
            False,
            canon_map,
        )
        pd.testing.assert_frame_equal(m_read, c_read)


class TestSkeletonNamespaceRoundTrip:
    """A messy config produces a canonical, internally-consistent skeleton."""

    def test_bodyparts_canonical_and_edges_subset(self, skeleton):
        sid = "nb10-namespace"
        config = {
            "skeleton_id": sid,
            "bodyparts": ["green_led", "red_led_c", "nose"],
            "skeleton": [["green_led", "red_led_c"], ["nose", "green_led"]],
        }
        try:
            skeleton.insert1(config, skip_duplicates=True)
            bodyparts = set(skeleton.get_bodyparts(sid))
            assert bodyparts == {"greenLED", "redLED_C", "nose"}
            edges = (skeleton & {"skeleton_id": sid}).fetch1("edges")
            edge_labels = {lbl for edge in edges for lbl in edge}
            assert edge_labels <= bodyparts
        finally:
            (skeleton & {"skeleton_id": sid}).delete(safemode=False)


class TestOnDiskRoundTrip:
    """normalize_names rewrites the project; result is canonical + idempotent."""

    @staticmethod
    def _write(path, bodyparts, skeleton):
        from spyglass.position.utils.yaml_io import dump_yaml

        dump_yaml(path, {"bodyparts": bodyparts, "skeleton": skeleton})

    def test_rewrite_is_canonical_and_idempotent(self, model, tmp_path):
        from spyglass.position.utils.yaml_io import load_yaml

        cfg = tmp_path / "config.yaml"
        self._write(
            cfg, ["green_led", "red_led_c"], [["green_led", "red_led_c"]]
        )

        model._canonicalize_dlc_project(cfg)
        rewritten = load_yaml(cfg)
        assert rewritten["bodyparts"] == ["greenLED", "redLED_C"]
        assert rewritten["skeleton"] == [["greenLED", "redLED_C"]]
        assert len(list(tmp_path.glob("config.yaml.*.bak"))) == 1

        # rewritten config now resolves to itself, and re-running is a no-op
        resolved = model.canonicalize_bodyparts(rewritten)
        assert resolved["unresolved"] == []
        assert all(k == v for k, v in resolved["mapping"].items())
        assert model._canonicalize_dlc_project(cfg) == {}
        assert len(list(tmp_path.glob("config.yaml.*.bak"))) == 1


class TestPermissionFallbackRoundTrip:
    """Non-admin imports of resolvable names succeed; novel names guide to admin."""

    def test_nonadmin_resolvable_config_inserts(self, skeleton, monkeypatch):
        from spyglass.common import LabMember

        monkeypatch.setattr(
            LabMember, "user_is_admin", property(lambda self: False)
        )
        sid = "nb10-nonadmin"
        config = {
            "skeleton_id": sid,
            "bodyparts": ["green_led", "red_led_c"],
            "skeleton": [["green_led", "red_led_c"]],
        }
        try:
            # variant spellings resolve to existing parts -> no BodyPart insert
            # is needed, so a non-admin import succeeds.
            skeleton.insert1(config, skip_duplicates=True)
            assert set(skeleton.get_bodyparts(sid)) == {"greenLED", "redLED_C"}
        finally:
            (skeleton & {"skeleton_id": sid}).delete(safemode=False)

    def test_novel_name_error_points_to_admin(self, model):
        err = dj.DataJointError("Unknown bodypart")
        out = model._augment_name_error(
            err, {"bodyparts": ["totallyNovelPartXyz"]}
        )
        assert "admin" in str(out) and "totallyNovelPartXyz" in str(out)


class TestDlcImportRoundTrip:
    """Live DLC import threads normalize_names without breaking the import."""

    @pytest.fixture(autouse=True)
    def _require_dlc(self, skip_if_no_dlc):
        """Skip the whole class when DLC is unavailable."""

    def test_load_with_normalize_names_succeeds(
        self, model, dlc_project_config, dlc_bootstrapped_session
    ):
        model_key = model.load(str(dlc_project_config), normalize_names=True)
        assert "model_id" in model_key
        assert model & model_key
