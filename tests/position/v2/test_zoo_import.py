"""Importing pretrained models whose training videos are not in Spyglass.

Plan: `.claude/TASKS_ZOO.md`. Two cases drive this — pilot/scoping videos that
should not be registered as sessions, and DLC Model Zoo models whose training
videos do not exist at all.

Every test here is RED until the matching task lands. Task IDs in the class
docstrings map to TASKS_ZOO.md §4.

Fixture rules (from the #1680 post-mortem, `CLAUDE.md`): no session-scoped
fixture may insert `Session`/`Nwbfile`, and teardown uses
``super_delete(warn=False, safemode=False)`` — plain ``delete()`` triggers
``cautious_delete``'s *global* external-file sweep and deletes other suites'
analysis files.
"""

import pathlib

import pytest

from tests.conftest import skip_if_no_pose

# DLC ships four SuperAnimal project configs. Names are 16-24 chars, which is
# why they live in `model_name` and never in the 32-char `model_id` (a prefix
# that long truncates the identifying hash -- see TASKS_ZOO.md §3.6).
ZOO_MODELS = [
    "superanimal_topviewmouse",
    "superanimal_quadruped",
    "superanimal_bird",
    "superanimal_humanbody",
]


@pytest.fixture
def requires_network():
    """Skip unless the SuperAnimal weights are already cached on disk.

    Deliberately checks the cache rather than connectivity. One model is
    ~646 MB (152 MB pose + 494 MB detector for topviewmouse/hrnet_w32), so a
    test run must never fetch it implicitly — CI also passes ``--no-pose``, so
    these never run there regardless.

    Pre-fetch once with::

        python -m spyglass.position.v2.utils.fetch_dlc_zoo superanimal_topviewmouse

    Returns
    -------
    callable
        ``_require(dataset, model_name=None)`` -- skips if absent. ``None``
        means "whichever backbone the dataset offers first", since backbones
        differ per dataset.
    """

    def _require(dataset, model_name=None):
        from spyglass.position.v2.utils import fetch_dlc_zoo

        if not fetch_dlc_zoo.zoo_catalog():
            pytest.skip("deeplabcut 3.x / dlclibrary not installed")
        if not fetch_dlc_zoo.is_cached(dataset, model_name):
            pytest.skip(
                f"zoo weights not cached for {dataset}. Pre-fetch with "
                "`python -m spyglass.position.v2.utils.fetch_dlc_zoo "
                f"{dataset}`"
            )
        return fetch_dlc_zoo.snapshot_path(dataset, model_name)

    return _require


class TestZooCatalog:
    """`fetch_dlc_zoo` — catalog lookup, the basis of `Model.load` dispatch.

    No weights needed: the catalog ships with dlclibrary, and snapshot paths
    are deterministic whether or not the file exists.
    """

    @pytest.fixture
    def zoo(self):
        from spyglass.position.v2.utils import fetch_dlc_zoo

        if not fetch_dlc_zoo.zoo_catalog():
            pytest.skip("deeplabcut 3.x / dlclibrary not installed")
        return fetch_dlc_zoo

    def test_catalog_lists_superanimals(self, zoo):
        catalog = zoo.zoo_catalog()
        assert set(ZOO_MODELS) <= set(catalog)
        assert all(backbones for backbones in catalog.values())

    def test_is_zoo_model(self, zoo):
        assert zoo.is_zoo_model("superanimal_topviewmouse")
        assert not zoo.is_zoo_model("/some/path/config.yaml")
        assert not zoo.is_zoo_model("superanimal_notathing")

    def test_default_backbone_is_dataset_specific(self, zoo):
        """No single default works: bird has only resnet_50, humanbody only
        rtmpose_x. A hardcoded `hrnet_w32` is wrong for half the catalog."""
        catalog = zoo.zoo_catalog()
        for dataset, backbones in catalog.items():
            assert zoo.default_backbone(dataset) in backbones

    def test_resolve_backbone_rejects_unavailable(self, zoo):
        catalog = zoo.zoo_catalog()
        dataset = "superanimal_bird"
        if "hrnet_w32" in catalog.get(dataset, []):
            pytest.skip("catalog changed; bird now offers hrnet_w32")
        with pytest.raises(ValueError, match="offers"):
            zoo.resolve_backbone(dataset, "hrnet_w32")

    def test_unknown_dataset_lists_the_catalog(self, zoo):
        with pytest.raises(ValueError, match="superanimal"):
            zoo.default_backbone("superanimal_notathing")

    def test_detector_defaults_per_dataset(self, zoo):
        """PyTorch SuperAnimal inference is top-down and needs a detector.

        `video_inference_superanimal` raises "You have to specify a
        detector_name when using the Pytorch framework" without one -- found by
        running it for real, not from the docs.
        """
        for dataset in ("superanimal_topviewmouse", "superanimal_quadruped"):
            chosen = zoo.resolve_detector(dataset)
            assert chosen in zoo.available_detectors(dataset)

    def test_humanbody_has_no_detector(self, zoo):
        """Bottom-up model: None is the right answer, not an error."""
        if zoo.available_detectors("superanimal_humanbody"):
            pytest.skip("catalog changed; humanbody now offers detectors")
        assert zoo.resolve_detector("superanimal_humanbody") is None

    def test_rejects_detector_not_offered(self, zoo):
        with pytest.raises(ValueError, match="offers"):
            zoo.resolve_detector("superanimal_topviewmouse", "ssdlite")

    def test_snapshot_path_does_not_download(self, zoo):
        """Deterministic path, so callers can check before committing ~646MB."""
        path = zoo.snapshot_path("superanimal_topviewmouse")
        assert path.name.startswith("superanimal_topviewmouse")
        assert path.suffix == ".pt"
        assert zoo.is_cached("superanimal_topviewmouse") == path.exists()

    def test_is_cached_is_false_for_unknown(self, zoo):
        """Unknown names must answer False, not raise -- `load` dispatch
        calls this on arbitrary strings."""
        assert zoo.is_cached("superanimal_notathing") is False


class TestExternalVideoTable:
    """Z01 — `VidFileGroup.ExternalVideo` records unregistered training media.

    The point is honesty: the videos are documented and queryable without
    claiming they are recording sessions. A fabricated `Session` would be
    indistinguishable from real data downstream.
    """

    def test_part_table_exists(self, VidFileGroup):
        assert hasattr(VidFileGroup, "ExternalVideo")

    def test_path_is_nullable(self, VidFileGroup):
        """Zoo configs ship `video_sets:` empty — source without paths."""
        heading = VidFileGroup.ExternalVideo().heading
        assert "path" in heading.names
        assert heading.attributes["path"].nullable

    def test_records_source_without_path(self, VidFileGroup, make_vid_group):
        """A zoo import has provenance but no file paths."""
        gid = make_vid_group("zoo_extvid_probe", [])
        VidFileGroup.ExternalVideo().insert1(
            {
                "vid_group_id": gid,
                "external_video_id": 0,
                "path": None,
                "source": "dlc-modelzoo:superanimal_topviewmouse",
                "note": "training videos not publicly available",
            }
        )
        row = (VidFileGroup.ExternalVideo & {"vid_group_id": gid}).fetch1()
        assert row["path"] is None
        assert row["source"].endswith("superanimal_topviewmouse")

    def test_coexists_with_fileless_group(self, VidFileGroup, make_vid_group):
        """External rows are not `File` rows — the group stays session-less.

        This is what keeps the inference guard meaningful: `get_nwb_file()`
        must still refuse this group, so it can never back a `PoseV2` entry.
        """
        gid = make_vid_group("zoo_fileless_probe", [])
        VidFileGroup.ExternalVideo().insert1(
            {
                "vid_group_id": gid,
                "external_video_id": 0,
                "path": "/elsewhere/pilot_video.mp4",
                "source": "external",
                "note": "",
            }
        )
        assert len(VidFileGroup.File & {"vid_group_id": gid}) == 0
        with pytest.raises(ValueError, match="no video files"):
            VidFileGroup().get_nwb_file(gid)


class TestModelName:
    """Z02 — `Model` gains a human-readable name.

    There is currently no such field: `model_id` is a date+hash, so `Model()`
    previews are unreadable. For a zoo model the name *is* the identity.
    """

    def test_attribute_exists(self, pv2_train):
        assert "model_name" in pv2_train.Model().heading.names

    def test_nullable_for_existing_rows(self, pv2_train):
        """Back-compat: pre-existing rows must survive the migration."""
        assert pv2_train.Model().heading.attributes["model_name"].nullable

    def test_populated_from_task_on_import(
        self,
        pv2_train,
        dlc_project_config,
        dlc_bootstrapped_session,
        skip_if_no_dlc,
    ):
        """Project imports default the name to the DLC `Task`.

        Exercises a real `Model.load`; `stub_model` is a direct insert and
        would only prove the column default.
        """
        import yaml

        task = yaml.safe_load(pathlib.Path(dlc_project_config).read_text())[
            "Task"
        ]
        key = pv2_train.Model().load(dlc_project_config)
        assert (pv2_train.Model & key).fetch1("model_name") == task


class TestTrainingMode:
    """Z04 — declared continuation capability.

    Three states, not a boolean: DLC's zoo fine-tune entry point is
    `build_weight_init(super_animal=...)`, which *seeds a new project* rather
    than resuming the imported model.
    """

    def test_attribute_exists(self, pv2_train):
        assert "training_mode" in pv2_train.Model().heading.names

    def test_enum_values(self, pv2_train):
        attr = pv2_train.Model().heading.attributes["training_mode"]
        for value in ("resumable", "weights_only", "inference_only"):
            assert value in attr.type

    def test_defaults_to_resumable(self, stub_model, pv2_train):
        """A row inserted without the field is assumed locally trained.

        That is the overwhelmingly common case -- `Model.make()` trains in
        place. Import paths set it explicitly rather than relying on this.
        """
        mode = (pv2_train.Model & {"model_id": stub_model["model_id"]}).fetch1(
            "training_mode"
        )
        assert mode == "resumable"

    def test_project_without_snapshot_is_inference_only(
        self, pv2_train, tmp_path, skip_if_no_dlc
    ):
        """Derivation, not the default: no `snapshot-*.pt` means no resume.

        Uses its own project directory. The shared `dlc_project_config` is
        session-scoped and ships a snapshot, and earlier tests train into it,
        so asserting the absence of one there depended on suite order.
        """
        from spyglass.position.utils.tool_strategies import DLCStrategy

        project = tmp_path / "no_snapshot_project"
        (project / "dlc-models" / "iteration-0" / "m" / "train").mkdir(
            parents=True
        )
        cfg_path = project / "config.yaml"
        cfg_path.touch()

        mode = DLCStrategy.project_training_mode(
            {"project_path": str(project)}, cfg_path
        )
        assert mode == "inference_only"

    def test_tensorflow_project_is_inference_only(self, pv2_train):
        """DLC-TF continuation raises NotImplementedError -- say so up front."""
        from spyglass.position.utils.tool_strategies import DLCStrategy

        mode = DLCStrategy.project_training_mode(
            {"engine": "tensorflow", "project_path": "/nonexistent"},
            "/nonexistent/config.yaml",
        )
        assert mode == "inference_only"


class TestLoadExternalVideos:
    """Z05 — the pilot-video case: import a real project, skip registration."""

    @pytest.fixture
    def unregistered_project(self, tmp_path_factory):
        """A DLC project whose video cannot match any `VideoFile` row.

        The shared example project's video basename *is* registered by
        `dlc_bootstrapped_session`, and `create_from_dlc_config` falls back to
        basename matching -- so reusing it would resolve and silently test the
        wrong branch. Give the video a unique name instead.
        """
        import yaml

        from tests.position.v2.make_example_dlc_project import make_dlc_project

        out = tmp_path_factory.mktemp("unreg_project", numbered=True)
        config_path = make_dlc_project(out)

        cfg = yaml.safe_load(config_path.read_text())
        old_video = next(iter(cfg["video_sets"]))
        unique = (
            pathlib.Path(old_video).parent / "pilot_scoping_unregistered.avi"
        )
        pathlib.Path(old_video).rename(unique)
        cfg["video_sets"] = {str(unique): {"crop": "0, 640, 0, 480"}}
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return config_path

    def test_load_rejects_unregistered_by_default(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """Default stays strict — opting out must be explicit."""
        with pytest.raises(ValueError, match="could not be unambiguously"):
            pv2_train.Model().load(unregistered_project, external_videos=False)

    def test_external_videos_records_config_paths(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """`video_sets` entries land in ExternalVideo, not VideoFile."""
        from spyglass.position.v2.video import VidFileGroup

        key = pv2_train.Model().load(unregistered_project, external_videos=True)
        gid = (pv2_train.Model & key).fetch1("vid_group_id")

        assert len(VidFileGroup.File & {"vid_group_id": gid}) == 0
        ext = VidFileGroup.ExternalVideo & {"vid_group_id": gid}
        assert len(ext) == 1
        assert "pilot_scoping_unregistered" in ext.fetch1("path")

    def test_external_model_is_still_usable_for_inference(
        self,
        pv2_train,
        pv2_estim,
        unregistered_project,
        single_session_group,
        skip_if_no_dlc,
    ):
        """The point of the feature: import without sessions, then *use* it.

        The safety argument rests on inference never reading the model's
        *training* group -- it reads only `model_path` / `tool` / params, and
        resolves sessions from the **inference** group in
        `PoseEstimSelection`. That was verified by reading the code; this
        asserts it, so a future change cannot quietly couple them and strand
        every externally-imported model.
        """
        key = pv2_train.Model().load(unregistered_project, external_videos=True)

        sel = {
            "model_id": key["model_id"],
            "vid_group_id": single_session_group,  # registered, one session
            "pose_estim_params_id": "default",
        }
        tbl = pv2_estim.PoseEstimSelection()
        tbl.insert1({**sel, "task_mode": "load", "output_dir": ""})
        try:
            assert len(tbl & sel) == 1
        finally:
            (tbl & sel).super_delete(warn=False, safemode=False)

    def test_external_group_cannot_back_inference(
        self, pv2_train, unregistered_project, skip_if_no_dlc
    ):
        """The session guard is untouched: no File rows, no inference.

        This is what makes relaxing the import safe -- an externally-sourced
        model still cannot produce a `PoseV2` entry from unregistered data.
        """
        from spyglass.position.v2.video import VidFileGroup

        key = pv2_train.Model().load(unregistered_project, external_videos=True)
        gid = (pv2_train.Model & key).fetch1("vid_group_id")
        with pytest.raises(ValueError, match="no video files"):
            VidFileGroup().get_nwb_file(gid)


@skip_if_no_pose
class TestLoadDispatch:
    """Z06 — one entry point for *any* pretrained model.

    The original request was pretrained models generally, not the zoo
    specifically. ``Model.load`` therefore takes a string and decides: an
    existing path is a project import; otherwise it is looked up in the DLC
    zoo catalog. `load_from_dlc_zoo` stays as the explicit form.

    Dispatch is checked without weights; only the tests that build a real row
    need the download.
    """

    def test_zoo_name_dispatches_without_importing(
        self, pv2_train, monkeypatch
    ):
        """`load` routes a catalog name to the zoo path.

        Stubs the import: a real one downloads a few hundred MB of weights,
        which a dispatch test has no business doing.
        """
        seen = {}

        def _fake(self, dataset, *args, **kwargs):
            seen["dataset"] = dataset
            return {"model_id": "zoo-stub"}

        monkeypatch.setattr(pv2_train.Model, "load_from_dlc_zoo", _fake)
        out = pv2_train.Model().load("superanimal_topviewmouse")

        assert seen["dataset"] == "superanimal_topviewmouse"
        assert out["model_id"] == "zoo-stub"

    def test_unknown_string_names_both_possibilities(self, pv2_train):
        """Neither a path nor a zoo entry -- say so, and list the catalog."""
        with pytest.raises(FileNotFoundError) as exc:
            pv2_train.Model().load("not_a_path_nor_a_zoo_model")
        msg = str(exc.value)
        assert "not_a_path_nor_a_zoo_model" in msg
        assert "superanimal" in msg  # catalog surfaced to the user

    def test_existing_path_still_imports_as_project(
        self,
        pv2_train,
        dlc_project_config,
        dlc_bootstrapped_session,
        skip_if_no_dlc,
    ):
        """Regression: the path branch is unchanged.

        Needs ``dlc_bootstrapped_session`` -- the project's videos must be
        registered, or this fails on the very constraint this work relaxes
        rather than on dispatch.
        """
        key = pv2_train.Model().load(dlc_project_config)
        assert pv2_train.Model & key

    def test_explicit_zoo_method_exists(self, pv2_train):
        assert hasattr(pv2_train.Model, "load_from_dlc_zoo")

    def test_rejects_unknown_zoo_name(self, pv2_train):
        """Fail on a typo before any download."""
        with pytest.raises(ValueError, match="superanimal"):
            pv2_train.Model().load_from_dlc_zoo("superanimal_notathing")

    def test_rejects_backbone_not_offered(self, pv2_train):
        """Validated before any download."""
        with pytest.raises(ValueError, match="offers"):
            pv2_train.Model().load_from_dlc_zoo(
                "superanimal_bird", model_name="hrnet_w32"
            )

    @pytest.mark.parametrize("zoo_name", ZOO_MODELS)
    def test_model_id_prefix_preserves_hash(
        self, pv2_train, zoo_name, requires_network
    ):
        """`zoo-` prefix keeps the id under 32 chars with the hash intact.

        The zoo name itself must never be the prefix:
        `superanimal_topviewmouse-2026091` truncates the hash and two models
        would collide on `model_id`.
        """
        requires_network(zoo_name)
        key = pv2_train.Model().load(zoo_name)
        model_id = key["model_id"]
        assert model_id.startswith("zoo-")
        assert len(model_id) <= 32
        assert len(model_id.split("-")[-1]) == 8

    def test_records_name_mode_and_provenance(
        self, pv2_train, requires_network
    ):
        from spyglass.position.v2.video import VidFileGroup

        requires_network("superanimal_topviewmouse")
        key = pv2_train.Model().load("superanimal_topviewmouse")
        row = (pv2_train.Model & key).fetch1()
        assert row["model_name"] == "superanimal_topviewmouse"
        assert row["training_mode"] == "weights_only"

        ext = VidFileGroup.ExternalVideo & {"vid_group_id": row["vid_group_id"]}
        assert len(ext) == 1
        assert "superanimal_topviewmouse" in ext.fetch1("source")

    def test_weights_stored_portably(self, pv2_train, requires_network):
        """`model_path` must not point into the per-env DLC package cache.

        `get_super_animal_snapshot_path` resolves inside the installed
        deeplabcut package, so storing it verbatim yields a path valid only in
        the importing user's conda env -- a shared-database footgun.
        """
        from pathlib import Path

        from spyglass.settings import pose_project_dir

        requires_network("superanimal_topviewmouse")
        key = pv2_train.Model().load("superanimal_topviewmouse")
        stored = (pv2_train.Model & key).fetch1("model_path")
        assert "site-packages" not in stored
        assert not Path(stored).is_absolute() or str(pose_project_dir) in str(
            stored
        )

    def test_skeleton_parts_marked_imported(
        self, pv2_train, bodypart, requires_network
    ):
        """Zoo vocabulary must not silently widen the curated list (Z08)."""
        requires_network("superanimal_topviewmouse")
        pv2_train.Model().load("superanimal_topviewmouse")
        sources = set((bodypart & {"bodypart": "mouse_center"}).fetch("source"))
        assert sources == {"imported"}

    def test_tf_backed_zoo_model_is_inference_only(self, pv2_train):
        """`dlcrnet` is a TensorFlow backend; DLC-TF continuation is won't-do.

        Such a model must import as `inference_only` rather than let the user
        discover the dead end at train time.
        """
        from spyglass.position.utils.tool_strategies import DLCStrategy

        mode = DLCStrategy.zoo_training_mode("dlcrnet")
        assert mode == "inference_only"

    def test_second_import_not_blocked_by_reuse_guard(
        self, pv2_train, requires_network
    ):
        """`ModelSelection` blocks redundant models per skeleton (§3.8)."""
        requires_network("superanimal_topviewmouse")
        pv2_train.Model().load("superanimal_topviewmouse")
        pv2_train.Model().load(
            "superanimal_topviewmouse", allow_redundant_model=True
        )


class TestZooInferenceDispatch:
    """Z10 — a zoo model routes to `video_inference_superanimal`.

    `analyze_videos` needs a project (config + shuffle + trainingsetindex); a
    zoo model is a bare checkpoint and has none. Stubbed throughout: real
    inference needs weights *and* a GPU-minutes budget, and what matters here
    is which function gets called with what.
    """

    @pytest.fixture
    def runner(self, pv2_estim):
        from spyglass.position.v2.utils.nwb_io import PoseInferenceRunner

        return PoseInferenceRunner()

    @pytest.fixture
    def video(self, mock_video_file):
        """A real file: `run_dlc_inference` validates videos before branching,
        which is correct -- project and zoo paths both need a readable one."""
        return str(mock_video_file)

    @pytest.fixture
    def capture_zoo(self, monkeypatch):
        """Intercept DLC's zoo entry point and record its kwargs."""
        import deeplabcut.modelzoo.video_inference as vi

        seen = {}

        def _fake(**kwargs):
            seen.update(kwargs)
            return "/tmp/out.h5"

        monkeypatch.setattr(vi, "video_inference_superanimal", _fake)
        return seen

    def test_zoo_model_routes_to_superanimal(self, runner, capture_zoo, video):
        out = runner.run_dlc_inference(
            {
                "model_path": "irrelevant.pt",
                "superanimal_name": "superanimal_topviewmouse",
                "zoo_backbone": "hrnet_w32",
            },
            video,
            destfolder="/out",
        )
        assert out == "/tmp/out.h5"
        assert capture_zoo["superanimal_name"] == "superanimal_topviewmouse"
        assert capture_zoo["model_name"] == "hrnet_w32"

    def test_passes_a_detector(self, runner, capture_zoo, video):
        """Without one DLC refuses to run the PyTorch path at all."""
        runner.run_dlc_inference(
            {
                "model_path": "x.pt",
                "superanimal_name": "superanimal_topviewmouse",
                "zoo_backbone": "hrnet_w32",
            },
            video,
        )
        from spyglass.position.v2.utils import fetch_dlc_zoo

        assert capture_zoo[
            "detector_name"
        ] in fetch_dlc_zoo.available_detectors("superanimal_topviewmouse")

    def test_defaults_to_one_individual(self, runner, capture_zoo, video):
        """The scoping decision: zoo output is pinned single-animal.

        SuperAnimal models are multi-animal by construction, so without this
        every zoo result would carry an `individuals` level the pipeline
        cannot use.
        """
        runner.run_dlc_inference(
            {
                "model_path": "x.pt",
                "superanimal_name": "superanimal_topviewmouse",
                "zoo_backbone": "hrnet_w32",
            },
            video,
        )
        assert capture_zoo["max_individuals"] == 1

    def test_backbone_resolved_when_absent(self, runner, capture_zoo, video):
        """A row predating `zoo_backbone` still resolves a valid backbone."""
        runner.run_dlc_inference(
            {
                "model_path": "x.pt",
                "superanimal_name": "superanimal_bird",
                "zoo_backbone": None,
            },
            video,
        )
        from dlclibrary import get_available_models

        assert capture_zoo["model_name"] in list(
            get_available_models("superanimal_bird")
        )

    def test_more_individuals_warns(self, runner, capture_zoo, caplog, video):
        """Opting out is allowed, loudly -- it will fail downstream.

        Captured at DEBUG: `BaseMixin._warn_msg` demotes to debug under
        `test_mode` to keep test output quiet, so a WARNING-level caplog would
        silently miss it and the assertion would pass vacuously.
        """
        with caplog.at_level("DEBUG", logger="spyglass"):
            runner.run_dlc_inference(
                {
                    "model_path": "x.pt",
                    "superanimal_name": "superanimal_topviewmouse",
                    "zoo_backbone": "hrnet_w32",
                },
                video,
                max_individuals=5,
            )
        assert capture_zoo["max_individuals"] == 5
        assert "one animal per entry" in caplog.text

    def test_project_model_still_uses_analyze_videos(
        self, runner, capture_zoo, tmp_path
    ):
        """Regression: a non-zoo model must not take the zoo branch."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("Task: t\n")
        with pytest.raises(Exception) as exc:
            runner.run_dlc_inference(
                {"model_path": str(cfg), "superanimal_name": None},
                str(tmp_path / "v.mp4"),
            )
        # Fails somewhere in the project path, never having called the zoo API
        assert not capture_zoo
        assert not isinstance(exc.value, AssertionError)
