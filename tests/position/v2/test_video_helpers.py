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
        assert isinstance(group_key["vid_group_id"], str)

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
                position_v2.video.VideoGroupParams(
                    directory="/nonexistent/path", description="Test group"
                )
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
            position_v2.video.VideoGroupParams(
                directory=tmp_path,
                description="Test video group",
                pattern="*.mp4",
            )
        )

        # Group should be created
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], str)

    def test_create_from_directory_no_matches(self, position_v2, tmp_path):
        """Test that no matching files raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Empty directory
        with pytest.raises(ValueError, match="No video files found"):
            VidFileGroup.create_from_directory(
                position_v2.video.VideoGroupParams(
                    directory=tmp_path,
                    description="Empty group",
                    pattern="*.mp4",
                )
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
            position_v2.video.VideoGroupParams(
                directory=tmp_path,
                description="Recursive group",
                pattern="*.mp4",
                recursive=True,
            )
        )

        # Group should be created
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], str)

    def test_create_from_directory_legacy_interface(
        self, position_v2, tmp_path
    ):
        """Test backward compatibility with legacy parameter interface."""
        VidFileGroup = position_v2.video.VidFileGroup

        # Create some mock video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()

        # Test legacy interface still works
        group_key = VidFileGroup.create_from_directory_legacy(
            directory=tmp_path,
            description="Legacy test group",
            pattern="*.mp4",
            recursive=False,
        )

        # Group should be created
        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], str)

    def test_add_files_group_not_exists(self, position_v2):
        """Test adding files to non-existent group raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        with pytest.raises(ValueError, match="not found"):
            VidFileGroup.add_files(
                vid_group_id="nonexistent_group", video_files=["/fake/path.mp4"]
            )

    def test_insert1_with_string_id(self, position_v2):
        """Test that string IDs are stored as-is."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {
                "vid_group_id": "my_string_id",
                "description": "Test group with string ID",
            },
            skip_duplicates=True,
        )

        assert group_key["vid_group_id"] == "my_string_id"

    def test_insert1_with_int_id(self, position_v2):
        """Test that integer IDs are cast to str."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {"vid_group_id": 42, "description": "Test group with int ID"},
            skip_duplicates=True,
        )

        assert group_key["vid_group_id"] == "42"

    def test_insert1_auto_generated_id(self, position_v2):
        """Test that ID is auto-generated from description via key_hash."""
        VidFileGroup = position_v2.video.VidFileGroup

        group_key = VidFileGroup().insert1(
            {"description": "Test group auto ID"}, skip_duplicates=True
        )

        assert "vid_group_id" in group_key
        assert isinstance(group_key["vid_group_id"], str)
        assert len(group_key["vid_group_id"]) == 32


class TestVidFileGroupGetNwbFile:
    """Tests for VidFileGroup.get_nwb_file() method."""

    def test_nonexistent_group_raises(self, position_v2):
        """vid_group_id not in VidFileGroup raises ValueError."""
        VidFileGroup = position_v2.video.VidFileGroup
        with pytest.raises(ValueError, match="Video group not found"):
            VidFileGroup().get_nwb_file("definitely_nonexistent_xyzzy_99")

    def test_empty_group_raises(self, position_v2):
        """VidFileGroup entry exists but has no File parts → ValueError."""
        VidFileGroup = position_v2.video.VidFileGroup
        VidFileGroup().insert1(
            {
                "vid_group_id": "gnf_empty_group_test",
                "description": "get_nwb_file empty group",
            },
            skip_duplicates=True,
        )
        with pytest.raises(ValueError, match="Video group not found"):
            VidFileGroup().get_nwb_file("gnf_empty_group_test")

    def test_returns_nwb_name_with_session_link(
        self, position_v2, mini_insert, mini_dict
    ):
        """Happy path: group with VideoFile entry linked to a session returns name."""
        from spyglass.common import VideoFile

        VidFileGroup = position_v2.video.VidFileGroup

        # Get videos from the same session only to avoid multiple NWB files
        target_nwb = mini_dict["nwb_file_name"]
        video_entries = (VideoFile & {"nwb_file_name": target_nwb}).fetch(
            as_dict=True, limit=1
        )
        if not video_entries:
            pytest.skip("No VideoFile entries in target session")

        # Use a unique group ID based on the session to avoid conflicts
        test_group_id = f"gnf_linked_test_{target_nwb.replace('.nwb', '').replace('.', '_')}"

        VidFileGroup().insert1(
            {
                "vid_group_id": test_group_id,
                "description": "get_nwb_file session-linked group",
            },
            skip_duplicates=True,
        )

        # VidFileGroup.File PK: vid_group_id + VideoFile PK fields
        video_pk = {
            k: v
            for k, v in video_entries[0].items()
            if k in ("nwb_file_name", "epoch", "video_file_num")
        }
        VidFileGroup.File().insert1(
            {"vid_group_id": test_group_id, **video_pk},
            skip_duplicates=True,
        )

        result = VidFileGroup().get_nwb_file(test_group_id)
        assert "nwb_file_name" in result
        assert result["nwb_file_name"] == mini_dict["nwb_file_name"]
