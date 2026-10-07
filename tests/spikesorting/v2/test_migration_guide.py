"""The migration guide covers the public v1-to-v2 migration categories."""

from pathlib import Path


def test_migration_guide_covers_v1_to_v2_categories():
    guide = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "src"
        / "Features"
        / "SpikeSortingV2_Migration.md"
    ).read_text()
    lower = guide.lower()
    for marker in (
        "sorter_param_name",
        "recording_id",
        "noise_levels",
        "off-by-one",
        "multi-channel",
        "seed",
        "amplitude_thresh_uv",
        "metriccuration",
        "spikesorting_artifact_detection_v2",
    ):
        assert marker in lower, f"missing migration-guide marker: {marker!r}"
