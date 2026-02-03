"""Tests for VidFileGroup helper methods."""

import pytest


class TestVidFileGroupHelpers:
    """Test VidFileGroup helper methods for creating video groups."""

    def test_create_from_files_basic(self, position_v2, tmp_path):
        """Test creating video group from list of files."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Create some mock video files
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"
        video1.touch()
        video2.touch()

        # Create the group (videos won't be added since not in VideoFile table)
        group_key = VidFileGroup.create_from_files(
            video_files=[video1, video2], description="Test video group"
        )

        # Group should be created even if videos can't be added
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], int)

    def test_create_from_files_empty_list(self, position_v2):
        """Test that empty file list raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        with pytest.raises(ValueError, match="cannot be empty"):
            VidFileGroup.create_from_files(
                video_files=[], description="Empty group"
            )

    def test_create_from_directory_not_exists(self, position_v2):
        """Test that non-existent directory raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        with pytest.raises(FileNotFoundError):
            VidFileGroup.create_from_directory(
                directory="/nonexistent/path", description="Test group"
            )

    def test_create_from_directory_basic(self, position_v2, tmp_path):
        """Test creating video group from directory."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Create some mock video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.avi").touch()

        # Create group (videos won't be added since not in VideoFile table)
        group_key = VidFileGroup.create_from_directory(
            directory=tmp_path, description="Test video group", pattern="*.mp4"
        )

        # Group should be created
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], int)

    def test_create_from_directory_no_matches(self, position_v2, tmp_path):
        """Test that no matching files raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Empty directory
        with pytest.raises(ValueError, match="No video files found"):
            VidFileGroup.create_from_directory(
                directory=tmp_path, description="Empty group", pattern="*.mp4"
            )

    def test_create_from_directory_recursive(self, position_v2, tmp_path):
        """Test recursive directory search."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "video1.mp4").touch()
        (subdir / "video2.mp4").touch()

        # Test that recursive finds files in subdirectories
        group_key = VidFileGroup.create_from_directory(
            directory=tmp_path,
            description="Recursive group",
            pattern="*.mp4",
            recursive=True,
        )

        # Group should be created
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], int)

    def test_add_files_group_not_exists(self, position_v2):
        """Test adding files to non-existent group raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        with pytest.raises(ValueError, match="not found"):
            VidFileGroup.add_files(
                vid_group_id=999999999, video_files=["/fake/path.mp4"]
            )

    def test_insert1_with_string_id(self, position_v2):
        """Test that string IDs are hashed to integers."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {
                "vid_group_id": "my_string_id",
                "description": "Test group with string ID",
            },
            skip_duplicates=True,
        )

        # Should return integer ID
        assert isinstance(group_key["vid_group_id"], int)
        assert group_key["vid_group_id"] > 0

    def test_insert1_with_int_id(self, position_v2):
        """Test that integer IDs are preserved."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {"vid_group_id": 42, "description": "Test group with int ID"},
            skip_duplicates=True,
        )

        assert group_key["vid_group_id"] == 42

    def test_insert1_auto_generated_id(self, position_v2):
        """Test that ID is auto-generated from description."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {"description": "Test group auto ID"}, skip_duplicates=True
        )

        # Should generate an ID
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], int)
        assert group_key["vid_group_id"] > 0
