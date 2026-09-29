"""DB-free unit tests for the ``describe_run`` receipt and ``CurationLabel``.

``describe_run`` is a pure transform over the dict / list that
``run_v2_pipeline`` and ``run_v2_pipeline_session`` return, so it needs no
DataJoint connection. These tests pin the receipt's row schema -- the property
that makes a zero-unit sort or a failed group a first-class row instead of a
value buried in a nested dict.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spyglass.spikesorting.v2 import CurationLabel
from spyglass.spikesorting.v2._pipeline_reporting import (
    _RUN_COLUMNS,
    describe_run,
)


def _run_summary(*, n_units=3, warnings=None, analysis=False):
    """A minimal run_v2_pipeline-shaped summary.

    ``analysis=False`` is a root-only run (``analysis_*`` are ``None``);
    ``analysis=True`` mimics ``auto_curate=True``, where ``analysis_*`` point at
    the auto-labeled child.
    """
    return {
        "pipeline_preset": "preset_x",
        "recording_id": "rec",
        "artifact_detection_id": "art",
        "sorting_id": "sort",
        "root_curation_id": 0,
        "root_merge_id": "root-merge-1",
        "auto_labeled_curation_id": 1 if analysis else None,
        "auto_labeled_merge_id": "analysis-merge-1" if analysis else None,
        "n_units": n_units,
        "recording_status": "computed",
        "artifact_detection_status": "computed",
        "sorting_status": "reused",
        "curation_status": "computed",
        "stage_seconds": {
            "recording": 1.0,
            "artifact_detection": 2.0,
            "sorting": 0.0,
            "curation": 0.5,
        },
        "warnings": list(warnings or []),
    }


def test_describe_run_single_columns_and_rows():
    frame = describe_run(_run_summary())
    assert list(frame.columns) == _RUN_COLUMNS

    by_type = frame["row_type"].tolist()
    assert by_type[0] == "summary"
    # one stage row per stage_seconds entry
    assert by_type.count("stage") == 4
    assert "warning" not in by_type

    summary = frame.iloc[0]
    assert summary["n_units"] == 3
    assert summary["root_merge_id"] == "root-merge-1"
    # A root-only run has no analysis-ready id, and the status says so plainly
    # (so a user can't mistake the root for the downstream-science handle).
    assert summary["auto_labeled_merge_id"] is None
    assert summary["status"] == "root only"
    assert summary["seconds"] == pytest.approx(3.5)  # 1 + 2 + 0 + 0.5

    stage_rows = frame[frame["row_type"] == "stage"].set_index("stage")
    assert stage_rows.loc["sorting", "status"] == "reused"
    assert stage_rows.loc["recording", "seconds"] == pytest.approx(1.0)


def test_describe_run_single_auto_curated_status():
    # auto_curate=True fills analysis_*; the receipt shows the analysis merge id
    # and flips the summary status to "auto-labeled".
    frame = describe_run(_run_summary(analysis=True))
    summary = frame.iloc[0]
    assert summary["root_merge_id"] == "root-merge-1"
    assert summary["auto_labeled_merge_id"] == "analysis-merge-1"
    assert summary["status"] == "auto-labeled"


def test_describe_run_concat_lists_each_member_merge_id():
    """Concat receipts expose one explicit row per session-safe output."""
    summary = {
        **_run_summary(),
        "source_mode": "concat",
        "root_merge_id": None,
        "member_curation_status": "computed",
        "member_merge_ids": {1: "merge-b", 0: "merge-a"},
        "stage_seconds": {
            **_run_summary()["stage_seconds"],
            "member_curation": 0.25,
        },
    }
    frame = describe_run(summary)
    members = frame[frame["row_type"] == "member"]
    assert members["member_index"].tolist() == [0, 1]
    assert members["member_merge_id"].tolist() == ["merge-a", "merge-b"]


def _unit_match_summary():
    """A minimal run_v2_unit_match-shaped summary (no root/analysis merge id)."""
    return {
        "session_group_owner": "owner",
        "session_group_name": "grp",
        "matcher_params_name": "unitmatch_default",
        "unit_match_id": "um-1",
        "unit_match_status": "computed",
        "n_pairs": 3,
        "tracked_unit_status": "computed",
        "n_tracked_units": 2,
        "stage_seconds": {"unit_match": 1.0, "tracked_unit": 0.5},
        "warnings": [],
    }


def test_describe_run_unit_match_summary_not_labeled_root_only():
    # run_v2_unit_match summaries flow through describe_run too, but have no
    # auto_labeled_merge_id. The "root only" / "auto-labeled" status is a
    # run_v2_pipeline concept -- a UnitMatch receipt must NOT be mislabeled
    # "root only"; its summary status stays blank.
    frame = describe_run(_unit_match_summary())
    summary = frame.iloc[0]
    assert summary["row_type"] == "summary"
    assert pd.isna(summary["status"])
    assert summary["root_merge_id"] is None
    assert summary["auto_labeled_merge_id"] is None
    # The match stages still render as their own rows.
    stages = set(frame[frame["row_type"] == "stage"]["stage"])
    assert {"unit_match", "tracked_unit"} <= stages


def test_describe_run_lists_each_unit_match_input():
    """A unit-match receipt renders one input row per matching input, in
    input_index order: its kind and source, sessions, pinned curation, and
    whether its waveforms came from motion-corrected traces (with the
    corrected recording's id)."""
    import uuid

    from spyglass.spikesorting.v2._pipeline_types import UnitMatchInputSummary

    ids = {name: uuid.UUID(int=i + 1) for i, name in enumerate("abcdefg")}
    inputs = (
        UnitMatchInputSummary(
            input_index=0,
            sorting_id=ids["a"],
            curation_id=2,
            curation_uuid=ids["b"],
            source_kind="concatenated_recording",
            source_id=ids["c"],
            nwb_file_names=("day1.nwb", "day1.nwb"),
            interval_list_names=("epoch 1", "epoch 2"),
            n_recordings=2,
            motion_corrected_recording_id=ids["d"],
            waveform_traces="motion_corrected_recording",
        ),
        UnitMatchInputSummary(
            input_index=1,
            sorting_id=ids["e"],
            curation_id=0,
            curation_uuid=ids["f"],
            source_kind="recording",
            source_id=ids["g"],
            nwb_file_names=("day2.nwb",),
            interval_list_names=("epoch 1",),
            n_recordings=1,
            motion_corrected_recording_id=None,
            waveform_traces="recording",
        ),
    )
    frame = describe_run({**_unit_match_summary(), "inputs": inputs})
    rows = frame[frame["row_type"] == "input"]
    assert rows["setting"].tolist() == ["input_0", "input_1"]
    assert rows["nwb_file_name"].tolist() == ["day1.nwb, day1.nwb", "day2.nwb"]
    assert rows["value"].tolist() == [
        f"concatenated_recording {ids['c']}; 2 recording(s) from day1.nwb, "
        f"day1.nwb; curation sorting_id={ids['a']}, curation_id=2 "
        f"(curation_uuid={ids['b']}); motion_corrected_recording traces "
        f"(motion corrected, motion_corrected_recording_id={ids['d']})",
        f"recording {ids['g']}; 1 recording(s) from day2.nwb; curation "
        f"sorting_id={ids['e']}, curation_id=0 (curation_uuid={ids['f']}); "
        "recording traces (not motion corrected)",
    ]


def test_describe_run_single_warning_is_its_own_row():
    frame = describe_run(
        _run_summary(n_units=0, warnings=["zero units found on this shank"])
    )
    warn_rows = frame[frame["row_type"] == "warning"]
    assert len(warn_rows) == 1
    assert "zero units" in warn_rows.iloc[0]["warning"]
    # the zero-unit count is visible on the summary row, not hidden
    assert frame.iloc[0]["n_units"] == 0


def test_describe_run_session_counts_and_group_rows():
    ok = {**_run_summary(n_units=5), "sort_group_id": 0, "outcome": "ok"}
    zero = {
        **_run_summary(n_units=0, warnings=["zero units"]),
        "sort_group_id": 1,
        "outcome": "ok",
    }
    failed = {
        "sort_group_id": 2,
        "pipeline_preset": "preset_x",
        "outcome": "failed",
        "error": "ZeroUnitSortError: ...",
        "partial_run_summary": None,
    }
    frame = describe_run([ok, zero, failed])

    header = frame.iloc[0]
    assert header["row_type"] == "summary"
    assert header["status"] == "2 ok, 1 failed, 1 zero-unit, 1 with warnings"

    groups = frame[frame["row_type"] == "group"]
    assert groups["sort_group_id"].tolist() == [0, 1, 2]
    failed_row = groups[groups["sort_group_id"] == 2].iloc[0]
    assert failed_row["status"] == "failed"
    assert "ZeroUnitSortError" in failed_row["error"]
    # seconds tolerated as missing on a partial-less failed group (NaN, not a
    # spurious 0.0)
    assert pd.isna(failed_row["seconds"])

    # the zero-unit group's warning is surfaced as its own row
    warn_rows = frame[frame["row_type"] == "warning"]
    assert warn_rows["sort_group_id"].tolist() == [1]


def test_describe_run_session_surfaces_partial_summary_seconds():
    # A failed group that completed some stages carries a partial summary; its
    # seconds and stable metadata should still aggregate.
    failed_partial = {
        "sort_group_id": 0,
        "outcome": "failed",
        "error": "boom",
        "partial_run_summary": {
            "stage_seconds": {"recording": 4.0},
            "n_units": 0,
            "root_merge_id": "root-merge-before-failure",
            "warnings": ["zero units before curation failure"],
        },
    }
    frame = describe_run([failed_partial])
    group = frame[frame["row_type"] == "group"].iloc[0]
    assert group["seconds"] == pytest.approx(4.0)
    assert group["n_units"] == 0
    assert group["root_merge_id"] == "root-merge-before-failure"
    header = frame.iloc[0]
    assert header["status"] == "0 ok, 1 failed, 1 zero-unit, 1 with warnings"
    warning = frame[frame["row_type"] == "warning"].iloc[0]
    assert "zero units" in warning["warning"]


def test_describe_run_rejects_unexpected_type():
    with pytest.raises(TypeError):
        describe_run("not a run summary")


def test_describe_run_rejects_list_entry_without_outcome():
    # A raw run summary wrapped in a list (no 'outcome' key) must not be
    # silently counted as an ok group -- that would inflate the ok tally.
    with pytest.raises(ValueError, match="outcome"):
        describe_run([_run_summary()])


def test_curation_label_export_and_order():
    # DB-free: CurationLabel is re-exported from the stdlib-only _enums module.
    assert [label.value for label in CurationLabel] == [
        "accept",
        "mua",
        "noise",
        "artifact",
        "reject",
    ]
    # lowercase, as documented (CurationLabel.mua, not CurationLabel.MUA)
    assert CurationLabel.mua.value == "mua"


def test_describe_run_renders_sorter_config_rows():
    """The receipt shows what the sort executed, from ``sorter_config``.

    One ``config`` row per effective setting -- the SI kwargs, whiten routing,
    seed, job kwargs and backend the run resolved once via
    ``resolve_sort_config`` -- so a user reads the effective configuration
    off the receipt instead of re-deriving it from parameter rows.
    """
    summary = _run_summary()
    summary["sorter_config"] = {
        "sorter": "mountainsort5",
        "scientific_params": {"whiten": True, "scheme": "2"},
        "si_sorter_params": {"whiten": False, "scheme": "2"},
        "external_whiten": True,
        "random_seed": 0,
        "job_kwargs": {"n_jobs": 4, "chunk_duration": "1s"},
        "execution_backend": "local",
        "container_image": None,
    }
    frame = describe_run(summary)
    config = frame[frame["row_type"] == "config"].set_index("setting")
    assert config.loc["sorter", "value"] == "'mountainsort5'"
    assert config.loc["external_whiten", "value"] == "True"
    assert config.loc["job_kwargs", "value"] == repr(
        {"n_jobs": 4, "chunk_duration": "1s"}
    )
    assert config.loc["si_sorter_params", "value"] == repr(
        {"whiten": False, "scheme": "2"}
    )
    # A receipt without the key (e.g. a unit-match manifest) renders no config rows.
    assert "config" not in describe_run(_run_summary())["row_type"].tolist()


def _motion_fields(mode):
    """The motion receipt of a run in ``mode`` (ids as strings)."""
    applied = mode == "apply"
    estimated = mode != "off"
    return {
        "motion_mode": mode,
        "motion_correction_params_name": (
            "dredge_fast_v1" if estimated else None
        ),
        "motion_estimate_id": "estimate-1" if estimated else None,
        "motion_estimate_supplied": False,
        "motion_estimation_preset": "dredge_fast" if estimated else None,
        "motion_corrected_recording_id": "corrected-1" if applied else None,
        "motion_removed_channel_ids": [0, 31] if applied else None,
        "motion_spans_without_evidence": [] if estimated else None,
    }


@pytest.mark.parametrize("mode", ["off", "estimate", "apply"])
def test_describe_run_shows_the_motion_receipt(mode):
    """Every motion receipt field is a ``config`` row (``"None"`` where the
    mode produces none), and the motion stages sit between the source
    stages and the sort."""
    summary = {**_run_summary(), **_motion_fields(mode)}
    stages = {"motion_estimate": 3.0} if mode != "off" else {}
    if mode == "apply":
        stages["motion_corrected_recording"] = 4.0
    for stage in stages:
        summary[f"{stage}_status"] = "computed"
    summary["stage_seconds"] = {**summary["stage_seconds"], **stages}

    frame = describe_run(summary)

    config = frame[frame["row_type"] == "config"].set_index("setting")["value"]
    for field, value in _motion_fields(mode).items():
        assert config[field] == str(value)
    stage_order = frame.loc[frame["row_type"] == "stage", "stage"].tolist()
    assert stage_order == [
        "recording",
        "artifact_detection",
        *stages,
        "sorting",
        "curation",
    ]


def test_describe_run_without_a_motion_receipt_adds_no_motion_rows():
    """A summary with no motion keys (e.g. a unit-match manifest) renders no
    motion rows."""
    frame = describe_run(_run_summary())
    assert not frame["setting"].fillna("").str.startswith("motion_").any()
