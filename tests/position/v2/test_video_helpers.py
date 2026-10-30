"""Tests for VidFileGroup helper methods."""

import pytest

# VidFileGroup.insert1 derives vid_group_id by hashing the resolved VideoFile
# keys together with any external media paths. This digest is the empty case,
# reachable only when neither is given: requesting files that do not resolve
# raises, so a group is never created ahead of its videos.
_EMPTY_FILES_HASH = "f6383428b86fcf5dd16653be0ee5ff67"


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

        # Routed through insert1, which resolves each path and names the one
        # it cannot match. A group is no longer created ahead of its videos.
        with pytest.raises(ValueError, match="No unique entry in VideoFile"):
            VidFileGroup.create_from_files(
                video_files=[video1, video2], description="Test video group"
            )

    def test_create_from_files_empty_list(self, position_v2):
        """Test that empty file list raises error."""
        VidFileGroup = position_v2.video.VidFileGroup

        with pytest.raises(ValueError, match="cannot be empty"):
            VidFileGroup.create_from_files(
                video_files=[], description="Empty group"
            )

    def test_group_id_identifies_the_resolved_files(
        self, position_v2, two_sessions
    ):
        """The auto ID derives from the files, not from an empty list.

        `create_from_files` used to insert the group before adding files, so
        every auto-generated ID was `key_hash({"files": []})` and two groups
        with different videos collided on it.
        """
        VidFileGroup = position_v2.video.VidFileGroup
        key_a, key_b = two_sessions["keys"]

        group_a = VidFileGroup.create_from_files(
            video_files=[dict(key_a)], description="group a"
        )
        group_b = VidFileGroup.create_from_files(
            video_files=[dict(key_b)], description="group b"
        )

        try:
            assert group_a["vid_group_id"] != _EMPTY_FILES_HASH
            assert group_a["vid_group_id"] != group_b["vid_group_id"]
            assert len(VidFileGroup.File() & group_a) == 1
            assert len(VidFileGroup.File() & group_b) == 1
        finally:
            for group in (group_a, group_b):
                (VidFileGroup & group).super_delete(warn=False, safemode=False)

    def test_external_only_groups_do_not_collide(self, position_v2):
        """External media is an external-only group's whole identity.

        Such a group has no `File` rows, so hashing the resolved files alone
        gave every one the same ID and `skip_duplicates` returned the first.
        Two pilot-video imports silently became one group.
        """
        VidFileGroup = position_v2.video.VidFileGroup

        first = VidFileGroup().insert1(
            {
                "description": "pilot footage A",
                "external_videos": ["/nowhere/pilot_a.mp4"],
            },
            skip_duplicates=True,
        )
        second = VidFileGroup().insert1(
            {
                "description": "pilot footage B",
                "external_videos": ["/nowhere/pilot_b.mp4"],
            },
            skip_duplicates=True,
        )

        try:
            assert first["vid_group_id"] != second["vid_group_id"]
            for group in (first, second):
                assert len(VidFileGroup.ExternalVideo() & group) == 1
        finally:
            for group in (first, second):
                (VidFileGroup & group).super_delete(warn=False, safemode=False)

    def test_group_id_ignores_input_order(self, position_v2, two_sessions):
        """The ID identifies the set, so argument order cannot change it."""
        VidFileGroup = position_v2.video.VidFileGroup
        key_a, key_b = (dict(k) for k in two_sessions["keys"])

        forward = VidFileGroup.create_from_files(
            video_files=[key_a, key_b], description="both, forward"
        )
        try:
            reversed_ = VidFileGroup.create_from_files(
                video_files=[key_b, key_a], description="both, reversed"
            )
            assert reversed_["vid_group_id"] == forward["vid_group_id"]
        finally:
            (VidFileGroup & forward).super_delete(warn=False, safemode=False)

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

        # Delegates to create_from_files, so unregistered videos raise there.
        with pytest.raises(ValueError, match="No unique entry in VideoFile"):
            VidFileGroup.create_from_directory(
                position_v2.video.VideoGroupParams(
                    directory=tmp_path,
                    description="Test video group",
                    pattern="*.mp4",
                )
            )

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

        # Recursion reaches the subdirectory, so resolution is attempted for
        # a file under it; neither is registered, so the first one raises.
        with pytest.raises(ValueError) as err:
            VidFileGroup.create_from_directory(
                position_v2.video.VideoGroupParams(
                    directory=tmp_path,
                    description="Recursive group",
                    pattern="*.mp4",
                    recursive=True,
                )
            )

        assert "No unique entry in VideoFile" in str(err.value)
        assert ".mp4" in str(err.value)

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

        # Auto-generated from key_hash({"files": []}) (no files provided).
        assert group_key["vid_group_id"] == _EMPTY_FILES_HASH
        assert len(group_key["vid_group_id"]) == 32


class TestVidFileGroupGetNwbFile:
    """Tests for VidFileGroup.get_nwb_file() method."""

    def test_nonexistent_group_raises(self, position_v2):
        """vid_group_id not in VidFileGroup raises ValueError."""
        VidFileGroup = position_v2.video.VidFileGroup
        with pytest.raises(ValueError, match="Video group not found"):
            VidFileGroup().get_nwb_file("definitely_nonexistent_xyzzy_99")

    def test_empty_group_raises(self, position_v2):
        """VidFileGroup entry exists but has no File parts → ValueError.

        Distinct from the nonexistent-group case: ``VidFileGroup.insert1``
        explicitly permits creating a group with zero files (it warns and
        continues), so this is a reachable user error that deserves its own
        message.
        """
        VidFileGroup = position_v2.video.VidFileGroup
        VidFileGroup().insert1(
            {
                "vid_group_id": "gnf_empty_group_test",
                "description": "get_nwb_file empty group",
            },
            skip_duplicates=True,
        )
        with pytest.raises(ValueError, match="no video files"):
            VidFileGroup().get_nwb_file("gnf_empty_group_test")

    def test_multi_session_group_raises(
        self, position_v2, two_sessions, multi_session_group
    ):
        """A group spanning two sessions has no single NWB parent.

        This is the invariant the inference path depends on: a ``PoseV2``
        entry must derive from exactly one ``Nwbfile``.
        """
        VidFileGroup = position_v2.video.VidFileGroup
        name_a, name_b = two_sessions["names"]

        with pytest.raises(ValueError) as exc:
            VidFileGroup().get_nwb_file(multi_session_group)

        msg = str(exc.value)
        assert "Expected exactly 1 common NWB file" in msg
        assert name_a in msg and name_b in msg  # names both listed

    def test_single_session_group_returns_name(
        self, position_v2, two_sessions, single_session_group
    ):
        """Positive control: one session in, that session's name out."""
        VidFileGroup = position_v2.video.VidFileGroup
        result = VidFileGroup().get_nwb_file(single_session_group)
        assert result == {"nwb_file_name": two_sessions["names"][0]}

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


class TestGroupConstruction:
    """Group identity and camera assignment must follow the caller's input.

    Four defects sat here: camera indices were paired with a relational
    fetch's order rather than the input's, `create_from_files` inserted the
    group before its files so every auto ID hashed an empty list, the
    session/basename passes accepted their first hit across sessions, and
    caller overrides were stored without resolving to one `VideoFile`.
    """

    def test_auto_id_differs_per_file_set(self, position_v2, video_keys):
        """Two different file sets must not collapse onto one auto ID."""
        VidFileGroup = position_v2.VidFileGroup

        first = VidFileGroup().insert1(
            {"files": [video_keys[0]], "description": "one"},
            skip_duplicates=True,
        )
        both = VidFileGroup().insert1(
            {"files": list(video_keys[:2]), "description": "two"},
            skip_duplicates=True,
        )

        assert first["vid_group_id"] != both["vid_group_id"]

    def test_auto_id_ignores_input_order(self, position_v2, video_keys):
        """The ID names a set of files, so order must not change it."""
        VidFileGroup = position_v2.VidFileGroup
        if len(video_keys) < 2:
            pytest.skip("needs two registered videos")

        forward = VidFileGroup().insert1(
            {"files": list(video_keys[:2])}, skip_duplicates=True
        )
        reverse = VidFileGroup().insert1(
            {"files": list(reversed(video_keys[:2]))}, skip_duplicates=True
        )

        assert forward["vid_group_id"] == reverse["vid_group_id"]

    def test_camera_index_length_checked_against_request(
        self, position_v2, video_keys
    ):
        """The count is rejected against requested files, not resolved ones."""
        VidFileGroup = position_v2.VidFileGroup

        with pytest.raises(ValueError, match="requested video files"):
            VidFileGroup().insert1(
                {
                    "files": [video_keys[0]],
                    "camera_indices": [0, 1],
                    "vid_group_id": "cam_len_mismatch",
                }
            )


class TestOverrideResolution:
    """A caller-supplied override must name exactly one `VideoFile`.

    Overrides were stored verbatim, so a key naming nothing counted as
    resolved and produced a group missing that video, while a partial key
    could name several recordings.
    """

    @staticmethod
    def _config(tmp_path, video_path):
        """A minimal DLC config naming one video."""
        import yaml

        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "project_path": str(tmp_path),
                    "video_sets": {video_path: {"crop": "0,1,0,1"}},
                    "bodyparts": ["nose"],
                }
            )
        )
        return path

    def test_override_naming_nothing_stays_unresolved(
        self, position_v2, tmp_path
    ):
        VidFileGroup = position_v2.video.VidFileGroup
        video = "/nowhere/absent_clip.mp4"
        config = self._config(tmp_path, video)

        with pytest.raises(ValueError) as err:
            VidFileGroup.create_from_dlc_config(
                config_path=config,
                vid_file_matches={
                    video: {"nwb_file_name": "no_such_file_.nwb"}
                },
            )

        assert "absent_clip.mp4" in str(err.value)

    def test_valid_override_resolves(self, position_v2, tmp_path, video_keys):
        """A full, real key is accepted and lands in the group."""
        VidFileGroup = position_v2.video.VidFileGroup
        video = "/nowhere/unmatchable_name.mp4"
        config = self._config(tmp_path, video)

        group = VidFileGroup.create_from_dlc_config(
            config_path=config,
            vid_file_matches={video: dict(video_keys[0])},
        )

        assert len(VidFileGroup.File() & group) == 1
