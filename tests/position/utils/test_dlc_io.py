"""Tests for DLC I/O and parsing utilities."""

import numpy as np
import pandas as pd
import pytest

# Check for pytables availability
pytables_available = True
try:
    import tables  # noqa: F401
except ImportError:
    pytables_available = False

skip_if_no_pytables = pytest.mark.skipif(
    not pytables_available, reason="pytables not available"
)

# Check for DLC availability
dlc_available = True
try:
    import deeplabcut  # noqa: F401
except ImportError:
    dlc_available = False


@pytest.fixture
def skip_if_no_dlc():
    if not dlc_available:
        pytest.skip("deeplabcut not installed")


class TestDLCOutputParsing:
    """Test DLC output file parsing utilities."""

    @pytest.fixture
    def mock_dlc_dataframe(self):
        """Create a mock DLC DataFrame with MultiIndex columns."""
        # Create MultiIndex columns: [scorer, bodypart, coords]
        scorer = "DLC_resnet50_test_projectJun1shuffle1_100000"
        bodyparts = ["nose", "leftear", "rightear", "tailbase"]
        coords = ["x", "y", "likelihood"]

        columns = pd.MultiIndex.from_product(
            [[scorer], bodyparts, coords],
            names=["scorer", "bodypart", "coords"],
        )

        # Create sample data
        np.random.seed(42)  # For reproducible tests
        n_frames = 100
        data = np.random.rand(n_frames, len(columns))

        # Set likelihood values to be between 0 and 1
        likelihood_cols = [
            i for i, col in enumerate(columns) if col[2] == "likelihood"
        ]
        for i in likelihood_cols:
            data[:, i] = np.random.rand(n_frames) * 0.8 + 0.1  # 0.1 to 0.9

        # Set x/y coordinates to reasonable ranges
        xy_cols = [i for i, col in enumerate(columns) if col[2] in ["x", "y"]]
        for i in xy_cols:
            data[:, i] = np.random.rand(n_frames) * 500 + 100  # 100 to 600

        df = pd.DataFrame(data, columns=columns)
        return df, scorer, bodyparts

    @skip_if_no_pytables
    @pytest.fixture
    def mock_dlc_h5_file(self, tmp_path, mock_dlc_dataframe):
        """Create a mock DLC H5 file."""
        df, scorer, bodyparts = mock_dlc_dataframe
        h5_path = tmp_path / "test_DLCproject_output.h5"
        df.to_hdf(h5_path, key="df_with_missing", mode="w")
        return h5_path, df, scorer, bodyparts

    @pytest.fixture
    def mock_dlc_csv_file(self, tmp_path, mock_dlc_dataframe):
        """Create a mock DLC CSV file."""
        df, scorer, bodyparts = mock_dlc_dataframe
        csv_path = tmp_path / "test_DLCproject_output.csv"
        df.to_csv(csv_path)
        return csv_path, df, scorer, bodyparts

    @skip_if_no_pytables
    def test_parse_dlc_h5_output_with_metadata(
        self, mock_dlc_h5_file, parse_dlc_h5_output
    ):
        """Test parsing DLC H5 file with metadata extraction."""
        h5_path, original_df, expected_scorer, expected_bodyparts = (
            mock_dlc_h5_file
        )

        result = parse_dlc_h5_output(h5_path, return_metadata=True)

        assert len(result) == 3
        df, scorer, bodyparts = result

        assert scorer == expected_scorer
        assert set(bodyparts) == set(expected_bodyparts)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == original_df.shape[0]

        # Check that DataFrame has proper structure
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.names == ["scorer", "bodypart", "coords"]

    @skip_if_no_pytables
    def test_parse_dlc_h5_output_no_metadata(
        self, mock_dlc_h5_file, parse_dlc_h5_output
    ):
        """Test parsing DLC H5 file without metadata."""
        h5_path, original_df, _, _ = mock_dlc_h5_file

        df = parse_dlc_h5_output(h5_path, return_metadata=False)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == original_df.shape[0]
        assert isinstance(df.columns, pd.MultiIndex)

    def test_parse_dlc_csv_output(self, mock_dlc_csv_file, parse_dlc_h5_output):
        """Test parsing DLC CSV file."""
        csv_path, original_df, expected_scorer, expected_bodyparts = (
            mock_dlc_csv_file
        )

        df, scorer, bodyparts = parse_dlc_h5_output(
            csv_path, return_metadata=True
        )

        assert scorer == expected_scorer
        assert set(bodyparts) == set(expected_bodyparts)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == original_df.shape[0]

    @skip_if_no_pytables
    def test_parse_dlc_output_bodypart_filter(
        self, mock_dlc_h5_file, parse_dlc_h5_output
    ):
        """Test filtering specific bodyparts during parsing."""
        h5_path, _, expected_scorer, _ = mock_dlc_h5_file

        filter_bodyparts = ["nose", "leftear"]
        df, scorer, bodyparts = parse_dlc_h5_output(
            h5_path, bodyparts=filter_bodyparts, return_metadata=True
        )

        assert scorer == expected_scorer
        assert set(bodyparts) == set(filter_bodyparts)

        # Check that only requested bodyparts are in DataFrame
        df_bodyparts = df.columns.get_level_values("bodypart").unique()
        assert set(df_bodyparts) == set(filter_bodyparts)

    def test_parse_dlc_output_file_not_found(self, parse_dlc_h5_output):
        """Test error when DLC output file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            parse_dlc_h5_output("/nonexistent/file.h5")

    @skip_if_no_pytables
    def test_get_dlc_bodyparts(self, mock_dlc_h5_file, get_dlc_bodyparts):
        """Test extracting bodypart names from DLC DataFrame."""
        h5_path, df, _, expected_bodyparts = mock_dlc_h5_file

        bodyparts = get_dlc_bodyparts(h5_path)

        assert set(bodyparts) == set(expected_bodyparts)
        assert len(bodyparts) == len(expected_bodyparts)

    @skip_if_no_pytables
    def test_get_dlc_scorer(self, mock_dlc_h5_file, get_dlc_scorer):
        """Test extracting scorer name from DLC DataFrame."""
        h5_path, df, expected_scorer, _ = mock_dlc_h5_file

        scorer = get_dlc_scorer(h5_path)

        assert scorer == expected_scorer

    def test_validate_dlc_output_structure_valid(
        self, mock_dlc_dataframe, validate_dlc_output_structure
    ):
        """Test validation of valid DLC DataFrame structure."""
        df, _, _ = mock_dlc_dataframe

        # Should not raise exception
        validate_dlc_output_structure(df)

    def test_validate_dlc_output_structure_no_multiindex(
        self, validate_dlc_output_structure
    ):
        """Test validation fails for DataFrame without MultiIndex."""
        # Create DataFrame with regular columns
        df = pd.DataFrame(
            {"x1": [1, 2, 3], "y1": [4, 5, 6], "likelihood1": [0.9, 0.8, 0.7]}
        )

        with pytest.raises(ValueError, match="MultiIndex columns"):
            validate_dlc_output_structure(df)

    def test_validate_dlc_output_structure_wrong_levels(
        self, validate_dlc_output_structure
    ):
        """Test validation fails for MultiIndex with wrong level names."""
        # Create MultiIndex with wrong names
        columns = pd.MultiIndex.from_product(
            [["scorer"], ["nose"], ["x", "y", "prob"]],
            names=["scorer", "bodypart", "wrong_name"],  # Should be "coords"
        )
        df = pd.DataFrame(np.random.rand(5, 3), columns=columns)

        with pytest.raises(ValueError, match="column levels"):
            validate_dlc_output_structure(df)

    def test_convert_dlc_to_position_df(
        self, mock_dlc_dataframe, convert_dlc_to_position_df
    ):
        """Test converting DLC format to position DataFrame."""
        df, _, bodyparts = mock_dlc_dataframe

        pos_df = convert_dlc_to_position_df(df)

        # Check structure
        assert isinstance(pos_df, pd.DataFrame)
        assert len(pos_df) == len(df)

        # Check that columns include x/y for each bodypart
        expected_cols = []
        for bp in bodyparts:
            expected_cols.extend([f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"])

        assert set(pos_df.columns) == set(expected_cols)

    def test_convert_dlc_to_position_df_likelihood_threshold(
        self, mock_dlc_dataframe, convert_dlc_to_position_df
    ):
        """Test converting DLC format with likelihood threshold."""
        df, _, _ = mock_dlc_dataframe

        # Set some likelihood values below threshold
        likelihood_threshold = 0.5
        pos_df = convert_dlc_to_position_df(
            df, likelihood_thresh=likelihood_threshold
        )

        # Check that low likelihood values are set to NaN
        for col in pos_df.columns:
            if col.endswith("_likelihood"):
                bp_name = col.replace("_likelihood", "")
                x_col = f"{bp_name}_x"
                y_col = f"{bp_name}_y"

                # Where likelihood < threshold, x/y should be NaN
                low_likelihood_mask = pos_df[col] < likelihood_threshold
                if low_likelihood_mask.any():
                    assert pos_df.loc[low_likelihood_mask, x_col].isna().all()
                    assert pos_df.loc[low_likelihood_mask, y_col].isna().all()


class TestDLCFileHandling:
    """Test DLC file format handling and edge cases."""

    def test_parse_dlc_output_corrupted_h5(self, tmp_path, parse_dlc_h5_output):
        """Test handling of corrupted H5 file."""
        # Create a file that looks like H5 but is corrupted
        h5_path = tmp_path / "corrupted.h5"
        h5_path.write_text("This is not a valid HDF5 file")

        with pytest.raises(Exception):  # Could be various exceptions
            parse_dlc_h5_output(h5_path)

    @skip_if_no_pytables
    def test_parse_dlc_output_empty_file(self, tmp_path, parse_dlc_h5_output):
        """Test handling of empty DLC file."""
        h5_path = tmp_path / "empty.h5"

        # Create empty DataFrame with correct structure
        columns = pd.MultiIndex.from_product(
            [["scorer"], ["nose"], ["x", "y", "likelihood"]],
            names=["scorer", "bodypart", "coords"],
        )
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_hdf(h5_path, key="df_with_missing", mode="w")

        df = parse_dlc_h5_output(h5_path, return_metadata=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert isinstance(df.columns, pd.MultiIndex)

    @skip_if_no_pytables
    def test_parse_dlc_output_missing_bodyparts(
        self, mock_dlc_h5_file, parse_dlc_h5_output
    ):
        """Test parsing when requested bodyparts don't exist."""
        h5_path, _, _, _ = mock_dlc_h5_file

        # Request bodyparts that don't exist
        nonexistent_bodyparts = ["tail", "head"]

        with pytest.raises(ValueError, match="not found"):
            parse_dlc_h5_output(
                h5_path, bodyparts=nonexistent_bodyparts, return_metadata=True
            )


class TestGetDLCModelEval:
    """Tests for get_dlc_model_eval error handling."""

    def test_missing_eval_folder_raises(self, tmp_path, skip_if_no_dlc):
        """RuntimeError when evaluation folder does not exist."""
        from unittest.mock import patch

        from spyglass.position.utils.dlc_io import get_dlc_model_eval

        project_path = tmp_path / "project"
        project_path.mkdir()
        yml_path = project_path / "config.yaml"
        yml_path.touch()

        dlc_config = {"TrainingFraction": [0.95]}

        with (
            patch(
                "spyglass.position.utils.dlc_io.evaluate_network",
                return_value=None,
            ),
            patch(
                "spyglass.position.utils.dlc_io.get_evaluation_folder",
                return_value="dlc-models/eval-nonexistent",
            ),
            pytest.raises(RuntimeError, match="Evaluation folder not found"),
        ):
            get_dlc_model_eval(str(yml_path), "", 1, 0, dlc_config)

    def test_parse_failure_raises(self, tmp_path, skip_if_no_dlc):
        """RuntimeError when eval CSV cannot be parsed."""
        from unittest.mock import patch

        from spyglass.position.utils.dlc_io import get_dlc_model_eval

        project_path = tmp_path / "project"
        project_path.mkdir()
        yml_path = project_path / "config.yaml"
        yml_path.touch()
        eval_dir = project_path / "eval"
        eval_dir.mkdir()
        # Put a non-CSV file there so get_most_recent_file fails or CSV parse fails
        (eval_dir / "results.txt").write_text("not a csv")

        dlc_config = {"TrainingFraction": [0.95]}

        with (
            patch(
                "spyglass.position.utils.dlc_io.evaluate_network",
                return_value=None,
            ),
            patch(
                "spyglass.position.utils.dlc_io.get_evaluation_folder",
                return_value="eval",
            ),
            patch(
                "spyglass.position.utils.dlc_io.get_most_recent_file",
                side_effect=FileNotFoundError("no csv"),
            ),
            pytest.raises(RuntimeError, match="Failed to parse evaluation"),
        ):
            get_dlc_model_eval(str(yml_path), "", 1, 0, dlc_config)


class TestDLCProjectReader:
    """Tests for DLCProjectReader construction and property access."""

    @pytest.fixture
    def dlc_output_dir(self, tmp_path):
        """Create a minimal synthetic DLC output directory."""
        import pickle

        import pandas as pd
        import yaml

        out_dir = tmp_path / "dlc_output"
        out_dir.mkdir()

        scorer = "DLC_resnet50_testJun1shuffle1_50000"
        bodyparts = ["nose", "tail"]
        coords = ["x", "y", "likelihood"]
        columns = pd.MultiIndex.from_product(
            [[scorer], bodyparts, coords],
            names=["scorer", "bodypart", "coords"],
        )
        df = pd.DataFrame(
            [[float(i)] * len(columns) for i in range(5)], columns=columns
        )
        h5_path = out_dir / "videoDLC_resnet50_testJun1shuffle1_50000.h5"
        df.to_hdf(h5_path, key="df_with_missing", mode="w")

        meta = {
            "data": {
                "Scorer": scorer,
                "fps": 30,
                "nframes": 5,
                "training set fraction": 0.95,
                "iteration (active-learning)": 0,
            }
        }
        pkl_path = (
            out_dir / "videoDLC_resnet50_testJun1shuffle1_50000meta.pickle"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(meta, f)

        cfg = {
            "Task": "test",
            "date": "Jun1",
            "TrainingFraction": [0.95],
            "snapshotindex": -1,
            "engine": "tensorflow",
        }
        yml_path = out_dir / "dj_dlc_config.yaml"
        with open(yml_path, "w") as f:
            yaml.dump(cfg, f)

        return out_dir

    def test_init_finds_files(self, dlc_output_dir, skip_if_no_dlc):
        """DLCProjectReader finds pkl, h5, and yaml files."""
        from spyglass.position.utils.dlc_io import DLCProjectReader

        reader = DLCProjectReader(dlc_output_dir)
        assert reader.pkl_path.exists()
        assert reader.h5_path.exists()
        assert reader.yml_path.exists()

    def test_missing_dir_raises(self, tmp_path, skip_if_no_dlc):
        """FileNotFoundError when dlc_dir does not exist."""
        from spyglass.position.utils.dlc_io import DLCProjectReader

        with pytest.raises(FileNotFoundError):
            DLCProjectReader(tmp_path / "nonexistent")

    def test_model_dict_populated(self, dlc_output_dir, skip_if_no_dlc):
        """model dict contains expected keys after init."""
        from spyglass.position.utils.dlc_io import DLCProjectReader

        reader = DLCProjectReader(dlc_output_dir)
        assert "Scorer" in reader.model
        assert "Task" in reader.model
        assert "shuffle" in reader.model
        assert reader.fps == 30
        assert reader.nframes == 5

    def test_body_parts_property(self, dlc_output_dir, skip_if_no_dlc):
        """body_parts property returns expected bodypart names."""
        from spyglass.position.utils.dlc_io import DLCProjectReader

        reader = DLCProjectReader(dlc_output_dir)
        bps = reader.body_parts
        assert "nose" in bps
        assert "tail" in bps


class TestDoPoseEstimation:
    """Tests for do_pose_estimation parameter routing."""

    def test_pytorch_engine_omits_tf_params(self, tmp_path, skip_if_no_dlc):
        """PyTorch engine must NOT receive TF-only params like TFGPUinference."""
        import sys
        from unittest.mock import MagicMock, patch

        import yaml

        config = {"engine": "pytorch"}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        dlc_model = {
            "config_template": {"project_path": str(tmp_path)},
            "shuffle": 1,
            "trainingsetindex": 0,
            "model_prefix": None,
        }

        mock_torch_analyze = MagicMock()
        mock_tf_analyze = MagicMock()

        fake_torch = MagicMock()
        fake_torch.analyze_videos = mock_torch_analyze
        fake_tf = MagicMock()
        fake_tf.analyze_videos = mock_tf_analyze

        with (
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(config_path),
            ),
            patch.dict(
                sys.modules,
                {
                    "deeplabcut.pose_estimation_pytorch": fake_torch,
                    "deeplabcut.pose_estimation_tensorflow": fake_tf,
                },
            ),
        ):
            from spyglass.position.utils.dlc_io import do_pose_estimation

            do_pose_estimation(
                video_filepaths=[str(tmp_path / "video.mp4")],
                dlc_model=dlc_model,
                project_path=str(tmp_path),
                output_dir=str(tmp_path),
            )

        mock_torch_analyze.assert_called_once()
        call_kwargs = mock_torch_analyze.call_args[1]
        assert "TFGPUinference" not in call_kwargs
        assert "gputouse" not in call_kwargs
