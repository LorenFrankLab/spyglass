"""Unit tests for DlcConfig (pure config editing, no DB / no DLC)."""

from pathlib import Path


def _cfg(DlcConfig, **overrides):
    config = {
        "video_sets": {
            "/proj/videos/clip.1.h264": {"crop": "0,1,0,1"},
            "/proj/videos/clip.mp4": {"crop": "0,2,0,2"},
        },
        "bodyparts": ["a", "b"],
        "numframes2pick": 5,
    }
    config.update(overrides)
    return DlcConfig("/proj", config)


def test_video_names_returns_basenames(DlcConfig):
    assert _cfg(DlcConfig).video_names() == {"clip.1.h264", "clip.mp4"}


def test_keep_videos_drops_stale_entries(DlcConfig):
    cfg = _cfg(DlcConfig).keep_videos({"clip.mp4"})
    assert set(cfg.video_sets) == {"/proj/videos/clip.mp4"}


def test_keep_videos_is_chainable_and_in_place(DlcConfig):
    cfg = _cfg(DlcConfig)
    returned = cfg.keep_videos({"clip.mp4"})
    assert returned is cfg  # chainable
    assert cfg.config["video_sets"] == {
        "/proj/videos/clip.mp4": {"crop": "0,2,0,2"}
    }


def test_set_bodyparts_replaces_list(DlcConfig):
    cfg = _cfg(DlcConfig).set_bodyparts(("greenLED", "redLED_C"))
    assert cfg.config["bodyparts"] == ["greenLED", "redLED_C"]


def test_set_arbitrary_key(DlcConfig):
    cfg = _cfg(DlcConfig).set("numframes2pick", 12)
    assert cfg.config["numframes2pick"] == 12


def test_edits_chain(DlcConfig):
    cfg = (
        _cfg(DlcConfig)
        .keep_videos({"clip.mp4"})
        .set_bodyparts(["x"])
        .set("numframes2pick", 3)
    )
    assert set(cfg.video_sets) == {"/proj/videos/clip.mp4"}
    assert cfg.config["bodyparts"] == ["x"]
    assert cfg.config["numframes2pick"] == 3


def test_missing_video_sets_defaults_empty(DlcConfig):
    cfg = DlcConfig("/proj", {"bodyparts": []})
    assert cfg.video_sets == {}
    assert cfg.video_names() == set()


def test_project_dir_is_path(DlcConfig):
    assert _cfg(DlcConfig).project_dir == Path("/proj")
