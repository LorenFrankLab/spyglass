"""Import-boundary and facade contracts for the DB-free service modules.

The table-class internals are factored into dependency-light service
modules. The load-bearing property the whole extraction rests on is that
the ``_*`` service modules import without pulling in the DB layer
(``spyglass.common`` or a v2 *schema* module), so a cold import opens no
DB connection. These tests pin that import boundary and check that the
typed pipeline contracts are re-exported beside ``run_v2_pipeline``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Service modules that MUST be importable without the DataJoint DB layer.
_DB_FREE_SERVICE_MODULES = [
    "_analyzer_cache",
    "_artifact_compute",
    "_artifact_intervals",
    "_artifact_naming",
    "bad_channels",
    "_concat_recording",
    "_curation_plan",
    "_curation_routing",
    "_curation_transforms",
    "_enums",
    "_figpack_curation",
    "_lookup_validation",
    "_nwb_metadata_helpers",
    "_pipeline_presets",
    "_pipeline_types",
    "_recipe_catalog",
    "_recording_geometry",
    "_recording_nwb",
    "_recording_preprocessing",
    "_recording_restriction",
    "_reference_resolution",
    "_selection_identity",
    "_selection_plan",
    "_shared_artifact_group",
    "_signal_math",
    "_sort_group_planning",
    "_sorting_analyzer",
    "_sorting_artifact_mask",
    "_sorting_dispatch",
    "_sorting_units",
    "_unitmatch_backend",
    "_units_nwb",
]


@pytest.mark.parametrize("modname", _DB_FREE_SERVICE_MODULES)
def test_service_module_imports_without_db_layer(modname):
    """Each ``_*`` service module cold-imports with no DB-layer dependency.

    Importing a v2 *schema* module (artifact / sorting / recording /
    curation) activates ``dj.schema(...)`` and the ``from spyglass.common
    import ...`` side effect, both of which open a DB connection AT IMPORT.
    A service module must do neither, so a fresh interpreter (the view a
    spawned ``n_jobs>1`` worker or a DB-isolated compute node gets) can
    import it and run its pure / IO logic without a database. Cold-import
    in a subprocess and assert it pulled in neither ``spyglass.common`` nor
    any v2 schema module. Hermetic -- no DB, no container.
    """
    probe = textwrap.dedent(f"""
        import sys

        import spyglass.spikesorting.v2.{modname}  # noqa: F401

        schema_mods = {{
            "spyglass.spikesorting.v2.artifact",
            "spyglass.spikesorting.v2.sorting",
            "spyglass.spikesorting.v2.recording",
            "spyglass.spikesorting.v2.curation",
        }}
        leaked = sorted(
            m
            for m in sys.modules
            if m.startswith("spyglass.common") or m in schema_mods
        )
        assert not leaked, "cold-import pulled in DB-layer modules: " + repr(
            leaked
        )
        """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"{modname} must import without a DB dependency\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_pipeline_type_contracts_are_reexported_from_facade():
    """Typed pipeline contracts are discoverable beside ``run_v2_pipeline``."""
    from spyglass.spikesorting.v2 import _pipeline_types as pipeline_types
    from spyglass.spikesorting.v2 import pipeline

    for name in pipeline_types.__all__:
        assert getattr(pipeline, name) is getattr(pipeline_types, name)

    # Every input key is optional: the call accepts exactly one of two modes
    # (single-session or concat), validated at runtime, so neither field group
    # is unconditionally required.
    assert pipeline_types.RunV2PipelineInputs.__required_keys__ == frozenset()
    assert pipeline_types.RunV2PipelineInputs.__optional_keys__ == {
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "team_name",
        "concat_session_group_owner",
        "concat_session_group_name",
        "pipeline_preset",
        "curation_description",
        "require_units",
        "auto_curate",
        "preflight",
        "build_figpack_view",
        "figpack_label_options",
    }
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
    }
    # The result/stage contracts mix always-present and mode/opt-in keys: their
    # NotRequired keys MUST surface as optional in the runtime TypedDict metadata
    # (codegen + callers inspect __optional_keys__). This silently breaks under
    # postponed annotations -- ``from __future__ import annotations`` in
    # _pipeline_types would stringize the ``NotRequired`` wrapper and reclassify
    # every key as required -- so pin the split here.
    assert pipeline_types.PipelineStageSeconds.__required_keys__ == {
        "sorting",
        "curation",
    }
    assert pipeline_types.PipelineStageSeconds.__optional_keys__ == {
        "recording",
        "artifact_detection",
        "member_recording",
        "concat_recording",
        "auto_curation",
        "figpack",
    }
    # RunV2PipelineSummary is a single/concat tagged union (discriminated by
    # ``source_mode``), so pin each arm's keys rather than the alias (a union has
    # no __required_keys__). The mode-specific source keys are REQUIRED on their
    # arm -- the whole point of the union is that an illegal mix (e.g. a concat
    # summary carrying recording_id) is unrepresentable. The flag-gated
    # auto-curation / figpack keys stay optional on both arms.
    _SUMMARY_BASE_REQUIRED = {
        "pipeline_preset",
        "source_mode",
        "sorting_id",
        "root_curation_id",
        "root_merge_id",
        "analysis_curation_id",
        "analysis_merge_id",
        "n_units",
        "sorting_status",
        "curation_status",
        "stage_seconds",
        "warnings",
    }
    _SUMMARY_OPTIONAL = {
        "curation_evaluation_id",
        "auto_curation_id",
        "auto_merge_id",
        "auto_curation_status",
        "figpack_uri",
        "figpack_status",
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
            "concat_recording_id",
            "concat_recording_status",
        }
    )
    assert (
        pipeline_types.RunV2ConcatSummary.__optional_keys__ == _SUMMARY_OPTIONAL
    )
    assert pipeline_types.UnitMatchStageSeconds.__required_keys__ == {
        "unit_match",
        "tracked_unit",
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
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "spyglass.spikesorting.v2 must cold-import without optional "
        "curation/matching extras\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
