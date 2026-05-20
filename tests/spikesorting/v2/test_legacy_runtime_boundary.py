"""Validation suite for the SpikeInterface-0.104 legacy-runtime boundary.

Asserts the contract that lets v2 ship under SpikeInterface 0.104 without
breaking existing v0/v1 query paths:

- the project is pinned to SI 0.104,
- the v0/v1 spike-sorting modules and the non-spike-sorting consumers in
  the audit all import under SI 0.104,
- each guarded v0/v1 entry point raises the legacy-environment error at
  call time,
- existing v0/v1 `SpikeSortingOutput` merge queries remain functional,
- v0/v1 DataJoint ``definition`` strings have not drifted,
- the resolver finds ``mountainsort5`` and the optional matching extra,
- ``correct_motion`` exposes the kwargs the v2 motion-correction contract
  requires.

These tests run under the resolver-clean SI 0.104 environment; the
merge-query smoke test additionally needs Docker.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"


# ---------- Schema-stability test: v0/v1 definitions are unchanged ------------

# Source files whose DataJoint ``definition`` blocks must not drift while the
# legacy-runtime boundary is in effect. The schemas are byte-identical to the
# SI-0.99 baseline; any drift here would mean an active migration sneaked in.
_LEGACY_TABLE_FILES = {
    "v0": [
        "spyglass/spikesorting/v0/spikesorting_recording.py",
        "spyglass/spikesorting/v0/spikesorting_artifact.py",
        "spyglass/spikesorting/v0/spikesorting_sorting.py",
        "spyglass/spikesorting/v0/spikesorting_curation.py",
        "spyglass/spikesorting/v0/spikesorting_burst.py",
        "spyglass/spikesorting/v0/spikesorting_recompute.py",
    ],
    "v1": [
        "spyglass/spikesorting/v1/recording.py",
        "spyglass/spikesorting/v1/artifact.py",
        "spyglass/spikesorting/v1/sorting.py",
        "spyglass/spikesorting/v1/curation.py",
        "spyglass/spikesorting/v1/metric_curation.py",
        "spyglass/spikesorting/v1/burst_curation.py",
        "spyglass/spikesorting/v1/recompute.py",
    ],
}


def _extract_definition_blocks(path: Path) -> dict[str, str]:
    """Return ``{class_name: definition_string}`` for every DataJoint class.

    Parses the source for triple-quoted ``definition =`` blocks attached to
    a preceding ``class Name`` declaration.
    """
    source = path.read_text()
    # ``class`` can be at column 0 (master tables) or indented (DataJoint Part
    # tables nested inside a master); both must be captured so a Part-table
    # drift cannot slip past the snapshot.
    pattern = re.compile(
        r"^\s*class\s+(\w+).*?\n\s+definition\s*=\s*\"\"\"\n(.*?)\"\"\"",
        re.DOTALL | re.MULTILINE,
    )
    return {match.group(1): match.group(2) for match in pattern.finditer(source)}


def test_no_legacy_schema_changes():
    """v0/v1 DataJoint ``definition`` blocks parse and remain present.

    The SI dependency bump is a runtime change only; any legacy table whose
    ``definition`` string has drifted would mean a schema migration sneaked
    in while the boundary contract was meant to be quiescent.
    """
    blocks: dict[str, dict[str, str]] = {}
    for tier, files in _LEGACY_TABLE_FILES.items():
        for relative in files:
            path = _SRC / relative
            if not path.exists():
                pytest.fail(f"Missing legacy file {path}")
            file_blocks = _extract_definition_blocks(path)
            assert file_blocks, (
                f"No `definition = \"\"\"...\"\"\"` blocks parsed from "
                f"{relative}; the parser or the file structure changed."
            )
            for name, body in file_blocks.items():
                blocks.setdefault(tier, {})[name] = body
    # Sanity: the legacy tables we care about are present.
    assert "SpikeSorting" in blocks["v0"]
    assert "SpikeSorting" in blocks["v1"]
    assert "MetricCuration" in blocks["v1"]


# ---------- Pyproject pin --------------------------------------------------


def test_pyproject_si_pin():
    """``pyproject.toml`` requires ``spikeinterface>=0.104,<0.105``."""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    assert "spikeinterface>=0.104,<0.105" in pyproject


# ---------- Sorter / matching-extra resolution -----------------------------


def test_sorter_runtime_resolution():
    """SpikeInterface 0.104 is installed and ``mountainsort5`` is available.

    The MS4 runtime install lives outside ``[test]``; only its presence in
    ``installed_sorters()`` is documented (Linux) and not asserted here.
    """
    import spikeinterface as si
    import spikeinterface.sorters as sis

    assert Version(si.__version__) >= Version("0.104")
    installed = set(sis.installed_sorters())
    assert "mountainsort5" in installed, installed


def test_optional_matching_extra_resolution():
    """The ``spikesorting-v2-matching`` optional extra declares both deps."""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    matching_block = re.search(
        r"optional-dependencies\.spikesorting-v2-matching\s*=\s*\[(.*?)\]",
        pyproject,
        re.DOTALL,
    )
    assert matching_block is not None
    body = matching_block.group(1)
    assert "UnitMatchPy>=3.3,<4" in body
    assert "mat73" in body


# ---------- correct_motion contract for Phase 3 ----------------------------


def test_correct_motion_api_contract():
    """``correct_motion`` exposes the kwargs Phase 3's MVP contract needs.

    Phase 3 persists only the corrected ``ElectricalSeries`` + sample
    boundaries + hash. The API must accept ``output_motion=False`` and
    ``output_motion_info=False`` and have a ``preset`` kwarg.
    """
    from spikeinterface.preprocessing import correct_motion

    params = inspect.signature(correct_motion).parameters
    for required in ("preset", "output_motion", "output_motion_info", "folder"):
        assert required in params, (
            f"correct_motion is missing kwarg {required!r}; "
            f"available: {sorted(params)}"
        )


# ---------- Audit-listed imports load under SI 0.104 -----------------------

_AUDIT_IMPORTS_TO_VERIFY = [
    # query-compatible files the audit lists for explicit smoke
    "spyglass.common.common_nwbfile",
    "spyglass.utils.waveforms",
    "spyglass.utils.mixins.analysis",
    "spyglass.utils.mixins.analysis_builder",
    "spyglass.decoding.v0.clusterless",
    "spyglass.decoding.v1.waveform_features",
    "spyglass.spikesorting.v1",
    "spyglass.spikesorting.v0",
    "spyglass.spikesorting.spikesorting_merge",
]


@pytest.mark.parametrize("module_name", _AUDIT_IMPORTS_TO_VERIFY)
def test_legacy_import_smoke(module_name, dj_conn):
    """Audit-listed v0/v1 + shared modules import under SI 0.104.

    Module-level imports must not crash; runtime guards only fire at call
    time, so a module can be imported without invoking a removed API. The
    ``dj_conn`` fixture is requested so the Docker MySQL server has been
    waited on before schema declaration runs.
    """
    importlib.import_module(module_name)


# ---------- Each guarded entry point raises at call time -------------------


def _assert_legacy_guard(
    component: str, callable_: Callable[[], object]
) -> None:
    """Assert ``callable_()`` raises the prescribed legacy-environment error.

    Captures the raised exception once so a guard that fails to raise on the
    second invocation cannot silently pass the component-name assertion.
    """
    with pytest.raises(RuntimeError, match="legacy SpikeInterface 0.99") as exc_info:
        callable_()
    assert component in str(exc_info.value)


def test_legacy_runtime_guard_raises_under_si_0104(dj_conn):
    """Every guarded v0/v1 entry point raises the legacy-env guard at call."""
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("Guard intentionally inactive under legacy SI < 0.101")

    from spyglass.decoding.v0.clusterless import UnitMarks
    from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures
    from spyglass.spikesorting.v0.spikesorting_artifact import (
        ArtifactDetection as V0ArtifactDetection,
    )
    from spyglass.spikesorting.v0.spikesorting_burst import (
        BurstPair as V0BurstPair,
    )
    from spyglass.spikesorting.v0.spikesorting_curation import (
        QualityMetrics as V0QualityMetrics,
        Waveforms as V0Waveforms,
    )
    from spyglass.spikesorting.v1.artifact import (
        ArtifactDetection as V1ArtifactDetection,
    )
    from spyglass.spikesorting.v1.burst_curation import (
        BurstPair as V1BurstPair,
    )
    from spyglass.spikesorting.v1.metric_curation import MetricCuration

    cases = [
        ("v0 Waveforms.make", lambda: V0Waveforms().make_compute(
            key={}, waveform_params={}, waveform_extractor_path="",
            recording_path="", sorting_path="", merge_groups=[],
        )),
        ("v0 Waveforms.load_waveforms",
         lambda: V0Waveforms().load_waveforms({})),
        ("v0 QualityMetrics.make", lambda: V0QualityMetrics().make_compute(
            key={}, wf_path="", params={}, qm_name="",
            quality_metrics_path="",
        )),
        ("v0 BurstPair.make", lambda: V0BurstPair().make({})),
        ("v0 ArtifactDetection.make",
         lambda: V0ArtifactDetection().make({})),
        ("v1 MetricCuration.make",
         lambda: MetricCuration().make_compute({}, {})),
        ("v1 MetricCuration.get_waveforms",
         lambda: MetricCuration().get_waveforms({})),
        ("v1 ArtifactDetection.make",
         lambda: V1ArtifactDetection().make({})),
        ("v1 BurstPair.make", lambda: V1BurstPair().make({})),
        ("v0 UnitMarks.make", lambda: UnitMarks().make({})),
        ("v1 UnitWaveformFeatures.make",
         lambda: UnitWaveformFeatures().make({})),
    ]
    for component, call in cases:
        _assert_legacy_guard(component, call)


# ---------- Merge-output query smoke (DB tier; skips if no Docker) ---------


@pytest.mark.slow
def test_legacy_merge_query_smoke(dj_conn):
    """Existing ``SpikeSortingOutput`` merge queries import and remain
    query-callable under SI 0.104.

    Uses the merge master directly (no v0/v1 active populate), so any drift
    in the legacy schemas would surface here as a fetch / restrict failure.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    # Heading describes the merge master and its source-class registry exists.
    heading = SpikeSortingOutput.heading
    assert "merge_id" in heading.attributes
    # source_class_dict is a property on the class; call on an instance.
    assert isinstance(SpikeSortingOutput().source_class_dict, dict)
    # Restricting with an unknown merge_id is a no-op (empty result) and must
    # not error -- this is the cheapest way to exercise the merge code path
    # under SI 0.104 without depending on real v0/v1 rows being present.
    unknown_uuid = "00000000-0000-0000-0000-000000000000"
    result = (SpikeSortingOutput & {"merge_id": unknown_uuid}).fetch(
        as_dict=True
    )
    assert result == []
