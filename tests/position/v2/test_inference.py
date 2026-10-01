"""Tests for video inference with trained models."""

from pathlib import Path

import numpy as np
import pytest


class TestModelInference:
    """Test Model.run_inference() method for DLC models.

    Note: These tests verify error handling. Actual DLC inference requires
    a real DLC config.yaml and trained model, which can't be mocked easily.
    """

    def test_run_inference_basic(
        self,
        model,
        dlc_project_config,
        dlc_bootstrapped_session,
        mock_video_file,
        skip_if_no_dlc,
    ):
        """run_inference dispatches DLC inference and returns the output path.

        ``analyze_videos`` is patched: its real behavior needs a trained
        shuffle this fake project lacks (and which only exists once a live
        training test has run), so a real call is order-dependent. Mocking it
        keeps the test deterministic and focused on the spyglass wiring — tool
        dispatch, path handling, and output resolution.
        """
        from unittest.mock import patch

        model_key = model.load(model_path=str(dlc_project_config))

        def _fake_analyze(**kwargs):
            video = Path(kwargs["videos"][0])
            dest = Path(kwargs.get("destfolder") or video.parent)
            (dest / f"{video.stem}DLC_test.h5").write_text("mock output")

        with patch("deeplabcut.analyze_videos", side_effect=_fake_analyze):
            output = model.run_inference(
                model_key, video_path=str(mock_video_file)
            )

        assert output, "run_inference returned no output path"
        assert Path(output).exists(), f"Reported output missing: {output}"

    def test_run_inference_with_options(
        self,
        model,
        dlc_project_config,
        dlc_bootstrapped_session,
        mock_video_file,
        skip_if_no_dlc,
    ):
        """run_inference forwards save_as_csv/destfolder to DLC analyze_videos."""
        from unittest.mock import patch

        model_key = model.load(model_path=str(dlc_project_config))
        destfolder = mock_video_file.parent
        captured = {}

        def _fake_analyze(**kwargs):
            captured.update(kwargs)
            video = Path(kwargs["videos"][0])
            dest = Path(kwargs.get("destfolder") or video.parent)
            (dest / f"{video.stem}DLC_test.h5").write_text("mock output")

        with patch("deeplabcut.analyze_videos", side_effect=_fake_analyze):
            output = model.run_inference(
                model_key,
                video_path=str(mock_video_file),
                save_as_csv=True,
                destfolder=str(destfolder),
            )

        assert output, "run_inference returned no output path"
        assert Path(output).exists(), f"Reported output missing: {output}"
        # Options are forwarded to DLC's analyze_videos.
        assert captured.get("save_as_csv") is True
        assert Path(captured.get("destfolder")) == destfolder

    def test_run_inference_cuda_unpickling_error_hint(
        self,
        model,
        dlc_project_config,
        dlc_bootstrapped_session,
        mock_video_file,
        skip_if_no_dlc,
        caplog,
    ):
        """A CUDA-related UnpicklingError propagates with an actionable hint.

        torch's ``weights_only`` unpickler wraps any snapshot-load failure
        (including a low-level "device busy/unavailable" CUDA driver error)
        in a generic ``pickle.UnpicklingError``. run_dlc_inference must not
        swallow or reclassify it, but should log an actionable hint pointing
        at GPU contention rather than a corrupt/untrusted file.
        """
        import logging
        import pickle
        from unittest.mock import patch

        model_key = model.load(model_path=str(dlc_project_config))

        def _fake_analyze(**kwargs):
            raise pickle.UnpicklingError(
                "Weights only load failed... WeightsUnpickler error: CUDA "
                "error: CUDA-capable device(s) is/are busy or unavailable"
            )

        with (
            # spyglass's own logger has its level pinned to INFO (see
            # spyglass.utils.logging), which filters out the .debug() calls
            # _err_msg uses in test mode before caplog's root-level setting
            # ever sees them. Override this specific logger by name.
            caplog.at_level(logging.DEBUG, logger="spyglass"),
            patch("deeplabcut.analyze_videos", side_effect=_fake_analyze),
            pytest.raises(pickle.UnpicklingError),
        ):
            model.run_inference(model_key, video_path=str(mock_video_file))

        assert any(
            "GPU availability problem" in rec.message for rec in caplog.records
        ), "Expected an actionable GPU-contention hint in the logs"

    def test_run_inference_invalid_model(
        self,
        model,
        mock_video_file,
    ):
        """Test error when model doesn't exist."""
        with pytest.raises(ValueError, match="Model not found"):
            model.run_inference(
                {"model_id": "nonexistent"},
                video_path=str(mock_video_file),
            )

    def test_run_inference_invalid_video(
        self,
        model,
        skip_if_no_dlc,
        dlc_project_config,
        dlc_bootstrapped_session,
    ):
        """Test error when video doesn't exist."""
        model_key = model.load(model_path=str(dlc_project_config))

        with pytest.raises((FileNotFoundError, ValueError)):
            model.run_inference(
                model_key,
                video_path="/nonexistent/video.avi",
            )


class TestCheckGpuAvailable:
    """check_gpu_available: the non-cuda guard short-circuits before
    touching torch at all (see issue #1676).

    Lives alongside PoseInferenceRunner in nwb_io.py (DLC 3.x/PyTorch is
    the only V2 backend), not in the tool-agnostic position/utils/ package
    -- a bare torch.cuda.is_available() check would be silently wrong for
    a hypothetical V1/TensorFlow caller.

    The torch.cuda-touching branch itself is marked ``# pragma: no cover``
    in source and deliberately untested here: CI never has a GPU, and
    mocking ``torch.cuda.is_available()`` to assert an ``if not X: raise``
    fires is mock theatre -- it tests Python's control flow, not our
    logic. The one thing worth asserting is the real branching decision:
    a non-CUDA device never reaches that code at all.
    """

    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.v2.utils.nwb_io import check_gpu_available

        self.fn = check_gpu_available

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_non_cuda_device_is_noop(self, device):
        """Non-CUDA (or absent) device requests never touch torch.cuda."""
        from unittest.mock import patch

        with patch("torch.cuda.is_available") as mock_avail:
            self.fn(device)
        mock_avail.assert_not_called()


class _FakeCap:
    """Minimal stand-in for ``cv2.VideoCapture``."""

    def __init__(self, opened=True, count=10):
        self._opened = opened
        self._count = count

    def isOpened(self):
        return self._opened

    def get(self, prop):  # cv2.CAP_PROP_FRAME_COUNT
        return self._count

    def release(self):
        pass


class TestFrameCountValidation:
    """Inference safety net for videos DLC cannot count frames for.

    OpenCV returns a non-positive ``CAP_PROP_FRAME_COUNT`` for videos whose
    metadata it cannot read; DLC then crashes deep inside ``tqdm(video)``
    with ``ValueError: __len__() should return >= 0``. Raw elementary
    streams (.h264/.h265/.hevc) are converted upstream by the shared
    ``general.ensure_mp4``; this residual check turns the opaque crash into
    an actionable error for *container* videos with unreadable metadata.
    These fake ``cv2.VideoCapture`` so they need neither DLC nor a real
    unreadable video, and run under ``--no-pose``.
    """

    def _patch_cap(self, monkeypatch, opened=True, count=10):
        import cv2

        monkeypatch.setattr(
            cv2,
            "VideoCapture",
            lambda *a, **k: _FakeCap(opened=opened, count=count),
        )

    def test_reported_frame_count(self, monkeypatch):
        """dlc_reported_frame_count returns the OpenCV metadata count."""
        from spyglass.position.utils.general import dlc_reported_frame_count

        self._patch_cap(monkeypatch, count=42)
        assert dlc_reported_frame_count("any.mp4") == 42

    def test_reported_frame_count_unopenable(self, monkeypatch):
        """None is returned when OpenCV cannot open the video."""
        from spyglass.position.utils.general import dlc_reported_frame_count

        self._patch_cap(monkeypatch, opened=False)
        assert dlc_reported_frame_count("any.mp4") is None

    def test_valid_count_passes(self, monkeypatch):
        """A positive frame count passes validation silently."""
        from spyglass.position.v2.utils import nwb_io

        self._patch_cap(monkeypatch, count=10)
        # Should not raise.
        nwb_io.PoseInferenceRunner()._assert_countable_videos(["good.mp4"])

    def test_negative_count_raises_actionable(self, monkeypatch):
        """A non-positive frame count raises a clear, actionable error."""
        from spyglass.position.v2.utils import nwb_io

        self._patch_cap(monkeypatch, count=-1)
        with pytest.raises(ValueError, match=r"__len__.*>= 0"):
            nwb_io.PoseInferenceRunner()._assert_countable_videos(["bad.mp4"])

    def test_unopenable_raises_actionable(self, monkeypatch):
        """An unopenable video raises the actionable error."""
        from spyglass.position.v2.utils import nwb_io

        self._patch_cap(monkeypatch, opened=False)
        with pytest.raises(ValueError, match="Re-encode"):
            nwb_io.PoseInferenceRunner()._assert_countable_videos(["bad.mp4"])


class TestPoseEstimPopulation:
    """Test PoseEstim.populate() and related methods."""

    def test_load_dlc_output_to_nwb(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test converting DLC h5 output to ndx-pose NWB format."""
        PoseEstim = position_v2.estim.PoseEstim

        # Load DLC output into NWB
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Verify NWB file was created/updated
        assert Path(nwb_path).exists()

        # Verify ndx-pose data is present
        import ndx_pose
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbfile = io.read()
            assert "behavior" in nwbfile.processing

            behavior_module = nwbfile.processing["behavior"]
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            # load_dlc_output creates exactly one PoseEstimation object.
            assert len(pose_estimations) == 1

    def test_pose_estim_insert(
        self,
        position_v2,
        model,
        skip_if_no_dlc,
        dlc_project_config,
        dlc_bootstrapped_session,
        mini_restr,
    ):
        """Test insert1 with a properly registered analysis_file_name.

        Uses the minirec session NWB (registered by mini_insert, valid pynwb)
        to create an AnalysisNwbfile entry, then calls insert1 with that key.
        The DLC-bootstrapped VidFileGroup satisfies the FK chain.
        """
        from spyglass.common import AnalysisNwbfile, Nwbfile

        PoseEstim = position_v2.estim.PoseEstim
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        VidFileGroup = position_v2.video.VidFileGroup

        model_key = model.load(model_path=str(dlc_project_config))

        # Link VidFileGroup to the bootstrapped VideoFile entries.
        vid_group_key = VidFileGroup.create_from_dlc_config(
            str(dlc_project_config)
        )

        # Use the minirec session NWB (already a valid pynwb file) as the
        # parent for AnalysisNwbfile.create() — avoids touching the 0-byte
        # bootstrap NWB whose checksum is already tracked by DataJoint.
        mini_nwb = (Nwbfile & mini_restr).fetch1("nwb_file_name")
        analysis_file_name = AnalysisNwbfile().create(mini_nwb)
        AnalysisNwbfile().add(mini_nwb, analysis_file_name)

        # Set up PoseEstimSelection so insert1 can reference it.
        selection_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": vid_group_key["vid_group_id"],
            "pose_estim_params_id": "default",
        }
        PoseEstimSelection().insert1(
            {**selection_key, "task_mode": "load", "output_dir": ""},
            skip_duplicates=True,
        )

        # Insert with a proper analysis_file_name — this is the primary goal.
        estim_key = {**selection_key, "analysis_file_name": analysis_file_name}
        PoseEstim().insert1(estim_key)

        assert len(PoseEstim() & selection_key) == 1

    def test_pose_estim_fetch_dataframe(
        self,
        position_v2,
        model,
        mock_ndx_pose_nwb_file,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test fetching pose data as DataFrame.

        Note: This test verifies the dataframe fetch logic by reading directly
        from the NWB file, without full AnalysisNwbfile registration.
        The E2E test (test_e2e_dlc_inference) covers the complete workflow.
        """
        PoseEstim = position_v2.estim.PoseEstim

        # Load DLC output into NWB
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Verify NWB contains pose data by reading directly
        import ndx_pose
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(nwb_path), mode="r") as io:
            nwbfile = io.read()
            assert "behavior" in nwbfile.processing

            behavior_module = nwbfile.processing["behavior"]
            pose_estimations = {
                name: obj
                for name, obj in behavior_module.data_interfaces.items()
                if isinstance(obj, ndx_pose.PoseEstimation)
            }

            # load_dlc_output creates exactly one PoseEstimation object.
            assert len(pose_estimations) == 1
            pose_estimation = list(pose_estimations.values())[0]

            # Verify we can read bodyparts and coordinates: the mock output
            # has 4 bodyparts, so one series per bodypart.
            assert len(pose_estimation.pose_estimation_series) == 4
            # pose_estimation_series is a dict-like object
            series_list = list(pose_estimation.pose_estimation_series.values())
            assert len(series_list) == 4
            series = series_list[0]
            assert series.data.shape[0] == 10  # 10 frames
            assert series.data.shape[1] == 2  # x, y coords
            assert len(series.confidence[:]) == 10  # confidence for each frame


class TestLoadFromNWB:
    """Test PoseEstim.load_from_nwb() for existing ndx-pose files."""

    def test_load_from_nwb_basic(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test loading pose data from existing ndx-pose NWB file."""
        PoseEstim = position_v2.estim.PoseEstim

        # First create an NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Now load metadata from that NWB file
        metadata = PoseEstim.load_from_nwb(nwb_path)

        # Verify metadata
        assert metadata["nwb_file_path"] == nwb_path
        assert metadata["pose_estimation_name"] == "PoseEstimation"
        # The mock DLC output declares these 4 bodyparts and this scorer.
        assert sorted(metadata["bodyparts"]) == [
            "bodypart1",
            "bodypart2",
            "bodypart3",
            "objectA",
        ]
        assert metadata["n_frames"] == 10
        assert metadata["scorer"] == "DLC_resnet50_TESTSep8shuffle1_6"
        assert metadata["source_software"] == "DeepLabCut"

    def test_load_from_nwb_file_not_found(self, position_v2):
        """Test error when NWB file doesn't exist."""
        PoseEstim = position_v2.estim.PoseEstim

        with pytest.raises(FileNotFoundError):
            PoseEstim.load_from_nwb("/nonexistent/file.nwb")

    def test_load_from_nwb_no_behavior_module(
        self, position_v2, mock_nwb_file_for_parent
    ):
        """Test error when NWB file lacks behavior module."""
        PoseEstim = position_v2.estim.PoseEstim

        # mock_nwb_file_for_parent doesn't have pose data yet
        with pytest.raises(ValueError, match="No behavior module"):
            PoseEstim.load_from_nwb(mock_nwb_file_for_parent)

    def test_load_from_nwb_specific_pose_estimation(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test loading specific PoseEstimation by name."""
        PoseEstim = position_v2.estim.PoseEstim

        # Create NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
            pose_estimation_name="MyPoseEstimation",
        )

        # Load by name
        metadata = PoseEstim.load_from_nwb(
            nwb_path, pose_estimation_name="MyPoseEstimation"
        )

        assert metadata["pose_estimation_name"] == "MyPoseEstimation"

    def test_load_from_nwb_wrong_pose_estimation_name(
        self,
        position_v2,
        mock_dlc_inference_output,
        mock_nwb_file_for_parent,
    ):
        """Test error when specified PoseEstimation doesn't exist."""
        PoseEstim = position_v2.estim.PoseEstim

        # Create NWB file with pose data
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(mock_dlc_inference_output["h5"]),
            nwb_file_name=mock_nwb_file_for_parent.name,
        )

        # Try to load non-existent PoseEstimation
        with pytest.raises(ValueError, match="not found in NWB"):
            PoseEstim.load_from_nwb(
                nwb_path, pose_estimation_name="NonExistent"
            )


class TestEndToEndInference:
    """Test complete end-to-end inference workflow via populate() in load mode."""

    def test_e2e_dlc_inference(
        self,
        position_v2,
        model,
        skip_if_no_dlc,
        dlc_project_config,
        dlc_bootstrapped_session,
        mock_dlc_inference_output,
        mini_restr,
    ):
        """Test workflow: import model -> populate(load mode) -> verify entry.

        Uses a pre-computed DLC h5 as the output_dir source.
        - VidFileGroup.get_nwb_file is patched to return minirec (valid pynwb)
          so AnalysisNwbfile.create() can open it.
        - _fetch_meters_per_pixel is patched since the test VideoFile NWB lacks
          real CameraDevice metadata.
        """
        from unittest.mock import patch

        from spyglass.common import Nwbfile

        PoseEstim = position_v2.estim.PoseEstim
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        VidFileGroup = position_v2.video.VidFileGroup

        # Step 1: Import model.
        model_key = model.load(model_path=str(dlc_project_config))

        # Step 2: Link VidFileGroup to bootstrapped VideoFile entries.
        vid_group_key = VidFileGroup.create_from_dlc_config(
            str(dlc_project_config)
        )

        # Step 3: Register PoseEstimSelection in load mode.
        h5_dir = str(mock_dlc_inference_output["h5"].parent)
        selection_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": vid_group_key["vid_group_id"],
            "pose_estim_params_id": "default",
        }
        PoseEstimSelection().insert1(
            {**selection_key, "task_mode": "load", "output_dir": h5_dir},
            skip_duplicates=True,
        )

        # Step 4: populate() → make() → NWB creation + insert.
        # Redirect the NWB parent to minirec (valid pynwb, already in Nwbfile)
        # so AnalysisNwbfile.create() can open it.
        mini_nwb = (Nwbfile & mini_restr).fetch1("nwb_file_name")
        n_pose = len(mock_dlc_inference_output["dataframe"])
        with (
            patch.object(
                VidFileGroup,
                "get_nwb_file",
                return_value={"nwb_file_name": mini_nwb},
            ),
            patch.object(
                PoseEstim, "_fetch_meters_per_pixel", return_value=1.0
            ),
            # The mock DLC output has n_pose frames, but the patched minirec
            # video carries a different timestamp count; align them so the
            # load-mode timestamp/pose-length guard in make() is satisfied.
            patch.object(
                PoseEstim,
                "_fetch_video_timestamps",
                return_value=np.arange(n_pose) / 30.0,
            ),
        ):
            PoseEstim().populate(selection_key)

        assert len(PoseEstim() & selection_key) == 1


class TestFetchVideoTimestamps:
    """PoseEstim._fetch_video_timestamps: NWB lookup + video-file fallback
    (see issue #1676). Two bugs fixed here:

    1. The NWB lookup's exception was silently swallowed (``ts = None``
       with no logging), so a real failure (e.g. an NFS/HDF5 file-lock
       hiccup) was indistinguishable from "no NWB data, fall back as
       expected" -- both just fell through to the same generic error.
    2. The video-file fallback read frame count from the *raw* registered
       ``VideoFile.path`` (e.g. a ``.h264`` elementary stream) instead of
       the already-converted mp4 -- OpenCV cannot read a frame count from
       a raw H.264 stream (it returns a nonsensical negative value rather
       than raising), so the fallback silently failed too.
    """

    KEY = {"vid_group_id": "vg1"}
    VF_KEY = {
        "vid_group_id": "vg1",
        "nwb_file_name": "n.nwb",
        "epoch": 1,
        "video_file_num": 1,
    }

    def _patch_vf_keys(self, estim_mod):
        estim_mod.VidFileGroup.File.__and__.return_value.fetch.return_value = [
            self.VF_KEY
        ]

    def test_nwb_lookup_success_returns_timestamps(self):
        """The common path: NWB lookup succeeds, fallback never runs."""
        from unittest.mock import MagicMock, patch

        from spyglass.position.v2 import estim as estim_mod

        with patch.multiple(
            estim_mod, VidFileGroup=MagicMock(), VideoFile=MagicMock()
        ):
            self._patch_vf_keys(estim_mod)
            fake_video_file = MagicMock()
            fake_video_file.timestamps = [1.0, 2.0, 3.0]
            estim_mod.VideoFile.__and__.return_value.fetch_nwb.return_value = [
                {"video_file": fake_video_file}
            ]
            result = estim_mod.PoseEstim._fetch_video_timestamps(self.KEY)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_nwb_lookup_failure_is_logged(self, caplog):
        """A real NWB-lookup exception is logged, not silently swallowed."""
        import logging
        from unittest.mock import MagicMock, patch

        from spyglass.position.v2 import estim as estim_mod

        with patch.multiple(
            estim_mod, VidFileGroup=MagicMock(), VideoFile=MagicMock()
        ):
            self._patch_vf_keys(estim_mod)
            estim_mod.VideoFile.__and__.return_value.fetch_nwb.side_effect = (
                OSError("unable to lock file")
            )
            estim_mod.VideoFile.__and__.return_value.fetch1.return_value = (
                "/tmp/raw.h264"
            )
            with (
                patch(
                    "spyglass.position.utils.general.ensure_mp4",
                    return_value=["/tmp/converted.mp4"],
                ),
                patch("cv2.VideoCapture") as mock_cap,
                # spyglass's logger is pinned to INFO; _err_msg/_warn_msg use
                # .debug() in test mode, which INFO would filter before
                # caplog's root-level setting ever sees it.
                caplog.at_level(logging.DEBUG, logger="spyglass"),
            ):
                mock_cap.return_value.get.side_effect = [30.0, 10]
                estim_mod.PoseEstim._fetch_video_timestamps(self.KEY)

        assert any(
            "NWB timestamp lookup failed" in r.message
            and "unable to lock file" in r.message
            for r in caplog.records
        ), "Expected the real NWB-lookup exception to be logged"

    def test_fallback_uses_converted_mp4_not_raw_source(self):
        """The video-file fallback counts frames on the converted mp4."""
        from unittest.mock import MagicMock, patch

        from spyglass.position.v2 import estim as estim_mod

        with patch.multiple(
            estim_mod, VidFileGroup=MagicMock(), VideoFile=MagicMock()
        ):
            self._patch_vf_keys(estim_mod)
            estim_mod.VideoFile.__and__.return_value.fetch_nwb.side_effect = (
                OSError("no nwb data")
            )
            estim_mod.VideoFile.__and__.return_value.fetch1.return_value = (
                "/tmp/raw.h264"
            )
            with (
                patch(
                    "spyglass.position.utils.general.ensure_mp4",
                    return_value=["/tmp/converted.mp4"],
                ) as mock_ensure_mp4,
                patch("cv2.VideoCapture") as mock_cap,
            ):
                mock_cap.return_value.get.side_effect = [30.0, 10]
                result = estim_mod.PoseEstim._fetch_video_timestamps(self.KEY)

        mock_ensure_mp4.assert_called_once()
        assert mock_ensure_mp4.call_args[0][0] == ["/tmp/raw.h264"]
        mock_cap.assert_called_once_with("/tmp/converted.mp4")
        np.testing.assert_array_equal(result, np.arange(10) / 30.0)


class TestInsertEstimationTaskValidation:
    """Tests for PoseEstimSelection.insert_estimation_task() validation."""

    def test_missing_vid_group_id_raises(self, position_v2):
        """insert_estimation_task() raises when vid_group_id is absent."""
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        with pytest.raises(ValueError, match="vid_group_id"):
            PoseEstimSelection().insert_estimation_task(
                {"model_id": "some_model_id"}
            )

    def test_invalid_vid_group_id_raises(self, position_v2):
        """insert_estimation_task() raises when vid_group_id not in VidFileGroup."""
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        with pytest.raises(ValueError, match="not found"):
            PoseEstimSelection().insert_estimation_task(
                {
                    "model_id": "some_model_id",
                    "vid_group_id": "nonexistent_group_xyz99",
                }
            )


class TestInsertFromVideoFile:
    """PoseEstimSelection.insert_from_videofile() — VideoFile-keyed entry."""

    def test_neither_videos_nor_restriction_raises(self, position_v2):
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        with pytest.raises(ValueError, match="exactly one"):
            PoseEstimSelection().insert_from_videofile(model_id="m")

    def test_both_videos_and_restriction_raises(self, position_v2):
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        with pytest.raises(ValueError, match="exactly one"):
            PoseEstimSelection().insert_from_videofile(
                model_id="m",
                videos={"nwb_file_name": "x"},
                restriction={"nwb_file_name": "x"},
            )

    def test_empty_restriction_raises(self, position_v2):
        """A restriction matching no VideoFile rows gives an actionable error."""
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        with pytest.raises(ValueError, match="No VideoFile rows match"):
            PoseEstimSelection().insert_from_videofile(
                model_id="m",
                restriction={"nwb_file_name": "definitely_absent_xyz_.nwb"},
            )

    def test_restriction_and_keys_create_tasks(
        self,
        position_v2,
        model,
        dlc_project_config,
        dlc_bootstrapped_session,
        skip_if_no_dlc,
    ):
        """A VideoFile restriction (and a key list) queue one task per video."""
        from spyglass.common import VideoFile

        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        VidFileGroup = position_v2.video.VidFileGroup

        model_key = model.load(model_path=str(dlc_project_config))
        nwb_file_name = dlc_bootstrapped_session
        restr = {"nwb_file_name": nwb_file_name}
        vf_keys = (VideoFile & restr).fetch("KEY", as_dict=True)
        assert vf_keys, "bootstrap did not populate VideoFile"

        # restriction form
        keys = PoseEstimSelection().insert_from_videofile(
            model_id=model_key["model_id"], restriction=restr
        )
        assert len(keys) == len(vf_keys)  # one task per matched video
        pk_fields = ("model_id", "vid_group_id", "pose_estim_params_id")
        for k in keys:
            assert VidFileGroup() & {"vid_group_id": k["vid_group_id"]}
            # restrict on the PK only — task_mode is a secondary attr a
            # pre-existing row (shared container) may already hold.
            assert PoseEstimSelection() & {f: k[f] for f in pk_fields}

        # explicit key-list form is idempotent (skip_duplicates) and matches
        keys2 = PoseEstimSelection().insert_from_videofile(
            model_id=model_key["model_id"], videos=vf_keys
        )
        assert {k["vid_group_id"] for k in keys2} == {
            k["vid_group_id"] for k in keys
        }


class TestPoseEstimMakeValidation:
    """Tests for PoseEstim.make() fail-fast validation."""

    def test_make_raises_when_no_nwb_parent(
        self,
        position_v2,
        model,
        dlc_project_config,
        dlc_bootstrapped_session,
        tmp_path,
        skip_if_no_dlc,
    ):
        """PoseEstim.make() raises ValueError when VidFileGroup has no NWB parent.

        Set up all required DB entries (model, vid_group, params, selection)
        and a real h5 output file, but deliberately leave the VidFileGroup
        unlinked from any Session/Nwbfile.  The make() call must fail at the
        AnalysisNwbfile storage step with a descriptive ValueError, not silently
        store partial data.
        """
        import pandas as pd

        PoseEstim = position_v2.estim.PoseEstim
        PoseEstimParams = position_v2.estim.PoseEstimParams
        PoseEstimSelection = position_v2.estim.PoseEstimSelection
        VidFileGroup = position_v2.video.VidFileGroup

        # 1. Import model via DLC (skip_if_no_dlc gate is already applied)
        model_key = model.load(str(dlc_project_config))

        # 2. Create a VidFileGroup with no File entries (no session link)
        vg_id = "pem_validate_no_nwb_8910"
        VidFileGroup().insert1(
            {"vid_group_id": vg_id, "description": "make() validation test"},
            skip_duplicates=True,
        )

        # 3. Use the default PoseEstimParams entry (always present from contents)
        params_id = (
            PoseEstimParams & {"pose_estim_params_id": "default"}
        ).fetch1("pose_estim_params_id")

        # 4. Create a real DLC-style h5 output in tmp_path
        scorer = "DLC_resnet50_TestSep8shuffle1_1"
        bodyparts = ["nose"]
        coords = ["x", "y", "likelihood"]
        columns = pd.MultiIndex.from_product(
            [[scorer], bodyparts, coords],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame([[1.0, 2.0, 0.99]], columns=columns)
        h5_path = tmp_path / f"{scorer}DLC_output.h5"
        df.to_hdf(str(h5_path), key="df_with_missing", mode="w")

        # 5. Insert PoseEstimSelection with output_dir pointing at tmp_path
        selection_key = {
            "model_id": model_key["model_id"],
            "vid_group_id": vg_id,
            "pose_estim_params_id": params_id,
        }
        PoseEstimSelection().insert1(
            {
                **selection_key,
                "task_mode": "load",
                "output_dir": str(tmp_path),
            },
            skip_duplicates=True,
        )

        # 6. make_fetch() fails fast: the missing Nwbfile parent is detected
        #    while reading upstream inputs, before any inference runs.
        with pytest.raises(ValueError, match="Cannot store pose estimation"):
            PoseEstim().make_fetch(selection_key)


class TestLoadDLCOutputTimestamps:
    """load_dlc_output must refuse frame-index fallback."""

    def test_raises_when_no_time_index_and_no_timestamps(
        self, position_v2, tmp_path
    ):
        """load_dlc_output raises ValueError when DLC h5 has no named time
        index and no timestamps are provided."""
        import pandas as pd

        PoseEstim = position_v2.estim.PoseEstim

        scorer = "DLC_resnet50_test"
        bodyparts = ["nose"]
        cols = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame([[1.0, 2.0, 0.99]], columns=cols)
        h5_path = tmp_path / "no_timestamps.h5"
        df.to_hdf(str(h5_path), key="df_with_missing", mode="w")

        with pytest.raises(ValueError, match="[Tt]imestamp"):
            PoseEstim.load_dlc_output(
                dlc_output_path=str(h5_path),
                nwb_file_name=str(tmp_path / "out.nwb"),
            )

    def test_uses_provided_timestamps(self, position_v2, tmp_path):
        """load_dlc_output uses explicitly-provided timestamps."""
        import pandas as pd

        PoseEstim = position_v2.estim.PoseEstim

        n = 5
        scorer = "DLC_resnet50_test"
        bodyparts = ["nose"]
        cols = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            [[float(i), float(i), 0.99] for i in range(n)], columns=cols
        )
        h5_path = tmp_path / "with_external_ts.h5"
        df.to_hdf(str(h5_path), key="df_with_missing", mode="w")

        ts = np.linspace(0.0, 1.0, n)
        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(h5_path),
            nwb_file_name=str(tmp_path / "with_ts.nwb"),
            timestamps=ts,
        )
        assert Path(nwb_path).exists()

    def test_uses_named_time_index(self, position_v2, tmp_path):
        """load_dlc_output uses df.index when it is named 'time'."""
        import pandas as pd

        PoseEstim = position_v2.estim.PoseEstim

        n = 5
        scorer = "DLC_resnet50_test"
        bodyparts = ["nose"]
        cols = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            [[float(i), float(i), 0.99] for i in range(n)], columns=cols
        )
        df.index = pd.Index(np.linspace(0.0, 1.0, n), name="time")
        h5_path = tmp_path / "with_time_index.h5"
        df.to_hdf(str(h5_path), key="df_with_missing", mode="w")

        nwb_path = PoseEstim.load_dlc_output(
            dlc_output_path=str(h5_path),
            nwb_file_name=str(tmp_path / "with_time_idx.nwb"),
        )
        assert Path(nwb_path).exists()
