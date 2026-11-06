"""Import-boundary and facade contracts for the DB-free service modules.

Service modules import without ``spyglass.common`` or a v2 schema module,
so a cold import opens no DB connection. Typed pipeline contracts are
re-exported beside ``run_v2_pipeline``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Service modules that MUST be importable without the DataJoint DB layer.
_SERVICE_ROOT = (
    Path(__file__).resolve().parents[3] / "src/spyglass/spikesorting/v2"
)
_DB_FREE_SERVICE_MODULES = [
    "bad_channels",
    *sorted(
        ".".join(
            (
                path.parent
                if path.name == "__init__.py"
                else path.with_suffix("")
            )
            .relative_to(_SERVICE_ROOT)
            .parts
        )
        for path in _SERVICE_ROOT.rglob("*.py")
        if path != _SERVICE_ROOT / "__init__.py"
        and path.relative_to(_SERVICE_ROOT).parts[0].startswith("_")
    ),
]


def test_cold_import_probes_cover_all_private_services():
    """New service modules must retain the worker-import contract."""
    service_dir = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spyglass"
        / "spikesorting"
        / "v2"
    )
    assert service_dir.is_dir()
    assert (service_dir / "_orchestration").is_dir()
    assert not (service_dir / "_internal").exists()
    services = {
        ".".join(
            (
                path.parent
                if path.name == "__init__.py"
                else path.with_suffix("")
            )
            .relative_to(service_dir)
            .parts
        )
        for path in service_dir.rglob("*.py")
        if path != service_dir / "__init__.py"
        and path.relative_to(service_dir).parts[0].startswith("_")
    }
    missing = services - set(_DB_FREE_SERVICE_MODULES)
    assert not missing, f"Missing cold-import probes: {sorted(missing)!r}"
    assert len(_DB_FREE_SERVICE_MODULES) == len(set(_DB_FREE_SERVICE_MODULES))
    assert "_motion" in _DB_FREE_SERVICE_MODULES
    assert "_recording" in _DB_FREE_SERVICE_MODULES
    assert "_params" in _DB_FREE_SERVICE_MODULES


@pytest.mark.parametrize("modname", _DB_FREE_SERVICE_MODULES)
def test_service_module_imports_without_db_layer(modname):
    """Each private service and package imports with no DB-layer dependency.

    Schema modules connect at import; spawned workers and DB-isolated nodes
    must be able to import services. Checked by cold import in a subprocess.
    """
    probe = textwrap.dedent(f"""
        import importlib.abc
        import sys

        schema_mods = {{
            "spyglass.spikesorting.v2.artifact_output",
            "spyglass.spikesorting.v2.artifact",
            "spyglass.spikesorting.v2.concat_member_curation",
            "spyglass.spikesorting.v2.figpack_curation",
            "spyglass.spikesorting.v2.metric_curation",
            "spyglass.spikesorting.v2.motion",
            "spyglass.spikesorting.v2.recompute",
            "spyglass.spikesorting.v2.review_profile",
            "spyglass.spikesorting.v2.session_group",
            "spyglass.spikesorting.v2.sorting",
            "spyglass.spikesorting.v2.recording",
            "spyglass.spikesorting.v2.curation",
            "spyglass.spikesorting.v2.unit_annotation",
            "spyglass.spikesorting.v2.unit_matching",
        }}
        attempted = []

        class RejectSchemaImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("spyglass.common") or fullname in schema_mods:
                    attempted.append(fullname)
                    raise ImportError("Forbidden DB-layer import: " + fullname)

        # Fail before schema activation can connect to a developer's database.
        # Keep attempts as well so catching ImportError cannot hide a violation.
        sys.meta_path.insert(0, RejectSchemaImports())
        import spyglass.spikesorting.v2.{modname}  # noqa: F401

        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common") or m in schema_mods
        )
        assert not leaked, "cold-import pulled in DB-layer modules: " + repr(
            leaked
        )
        assert not attempted, "cold-import attempted DB-layer imports: " + repr(
            attempted
        )
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"{modname} must import without a DB dependency\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_pipeline_type_contracts_are_reexported_from_facade():
    """Typed pipeline contracts are discoverable beside ``run_v2_pipeline``."""
    from spyglass.spikesorting.v2._orchestration import types as pipeline_types
    from spyglass.spikesorting.v2 import pipeline

    for name in pipeline_types.__all__:
        assert getattr(pipeline, name) is getattr(pipeline_types, name)

    options = {
        "pipeline_preset",
        "curation_description",
        "require_units",
        "auto_curate",
        "preflight",
        "manual_excluded_times",
        "motion_mode",
        "motion_correction_params_name",
        "motion_estimate_id",
    }
    assert pipeline_types.RunV2PipelineInputs.__required_keys__ == {
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
    }
    assert pipeline_types.RunV2PipelineInputs.__optional_keys__ == options
    assert pipeline_types.RunV2ConcatPipelineInputs.__required_keys__ == {
        "session_group_owner",
        "session_group_name",
    }
    assert pipeline_types.RunV2ConcatPipelineInputs.__optional_keys__ == options
    assert pipeline_types.RunV2PipelineSessionInputs.__required_keys__ == {
        "nwb_file_name",
        "interval_list_name",
        "team_name",
        "pipeline_preset",
    }
    assert pipeline_types.RunV2PipelineSessionInputs.__optional_keys__ == {
        "sort_group_ids",
        "curation_description",
        "require_units",
        "auto_curate",
        "preflight",
        "continue_on_error",
        "manual_excluded_times",
        "motion_mode",
        "motion_correction_params_name",
    }
    # NotRequired keys must appear in __optional_keys__; postponed annotations
    # in _pipeline_types would make every key required.
    assert pipeline_types.PipelineStageSeconds.__required_keys__ == {
        "sorting",
        "curation",
    }
    assert pipeline_types.PipelineStageSeconds.__optional_keys__ == {
        "recording",
        "artifact_detection",
        "member_recording",
        "member_artifact_detection",
        "concat_recording",
        "member_curation",
        "motion_estimate",
        "motion_corrected_recording",
        "auto_curation",
    }
    # RunV2PipelineSummary is a union on ``source_mode``, so check each arm.
    # Mode-specific source keys are required on their arm; auto-curation keys
    # are optional on both.
    _SUMMARY_BASE_REQUIRED = {
        "pipeline_preset",
        "scientific_config",
        "source_mode",
        "sorting_id",
        "root_curation_id",
        "root_merge_id",
        "auto_labeled_curation_id",
        "auto_labeled_merge_id",
        "root_curation_uuid",
        "auto_labeled_curation_uuid",
        "sorter_config",
        "n_units",
        "sorting_status",
        "curation_status",
        "stage_seconds",
        "warnings",
        "motion_mode",
        "motion_correction_params_name",
        "motion_estimate_id",
        "motion_estimate_supplied",
        "motion_corrected_recording_id",
        "motion_estimation_preset",
        "motion_removed_channel_ids",
        "motion_spans_without_evidence",
    }
    _SUMMARY_OPTIONAL = {
        "motion_estimate_status",
        "motion_corrected_recording_status",
        "curation_evaluation_id",
        "auto_curation_status",
    }
    assert pipeline_types.RunV2SingleSessionSummary.__required_keys__ == (
        _SUMMARY_BASE_REQUIRED
        | {
            "recording_id",
            "recording_status",
            "artifact_detection_id",
            "artifact_detection_status",
        }
    )
    assert (
        pipeline_types.RunV2SingleSessionSummary.__optional_keys__
        == _SUMMARY_OPTIONAL
    )
    assert pipeline_types.RunV2ConcatSummary.__required_keys__ == (
        _SUMMARY_BASE_REQUIRED
        | {
            "member_recording_status",
            "member_recording_ids",
            "member_artifact_detection_status",
            "member_artifacts",
            "artifact_masked_duration_s",
            "concat_recording_id",
            "concat_recording_status",
            "member_curation_status",
            "member_merge_ids",
        }
    )
    assert (
        pipeline_types.RunV2ConcatSummary.__optional_keys__ == _SUMMARY_OPTIONAL
    )
    # A session-runner success entry is a single-session summary, so it
    # carries every receipt key (the motion keys included).
    assert pipeline_types.RunV2PipelineSessionOk.__required_keys__ == (
        pipeline_types.RunV2SingleSessionSummary.__required_keys__
        | {"sort_group_id", "outcome"}
    )
    assert pipeline_types.UnitMatchStageSeconds.__required_keys__ == {
        "unit_match",
        "tracked_unit",
    }
    # ``estimate_motion``'s receipt: the estimate keys always, the source keys
    # of whichever input mode ran.
    assert pipeline_types.EstimateMotionReceipt.__required_keys__ == {
        "pipeline_preset",
        "source_mode",
        "motion_correction_params_name",
        "motion_estimate_id",
        "motion_estimation_preset",
        "motion_estimate_status",
        "motion_spans_without_evidence",
        "motion_diagnostics",
        "stage_seconds",
        "warnings",
    }
    assert pipeline_types.EstimateMotionReceipt.__optional_keys__ == {
        "recording_id",
        "recording_status",
        "artifact_detection_id",
        "artifact_detection_status",
        "member_recording_ids",
        "member_recording_status",
        "member_artifact_detection_status",
        "member_artifacts",
        "artifact_masked_duration_s",
        "concat_recording_id",
        "concat_recording_status",
    }


def test_package_root_imports_without_optional_extra_modules():
    """Package-root import must not import optional curation/matching extras."""
    probe = textwrap.dedent("""
        import sys

        import spyglass.spikesorting.v2  # noqa: F401

        optional_roots = {
            "UnitMatchPy",
            "figpack",
            "figpack_spike_sorting",
            "mat73",
        }
        leaked = sorted(
            modname
            for modname in sys.modules
            if modname.split(".", 1)[0] in optional_roots
        )
        assert not leaked, (
            "package-root import pulled optional extra modules: "
            + repr(leaked)
        )
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        "spyglass.spikesorting.v2 must cold-import without optional "
        "curation/matching extras\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
