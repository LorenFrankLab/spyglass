"""Tests for DLC I/O and parsing utilities."""

import numpy as np
import pandas as pd
import pytest


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

    def test_get_dlc_bodyparts(self, mock_dlc_h5_file, get_dlc_bodyparts):
        """Test extracting bodypart names from DLC DataFrame."""
        h5_path, df, _, expected_bodyparts = mock_dlc_h5_file

        bodyparts = get_dlc_bodyparts(h5_path)

        assert set(bodyparts) == set(expected_bodyparts)
        assert len(bodyparts) == len(expected_bodyparts)

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
            df, likelihood_threshold=likelihood_threshold
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
