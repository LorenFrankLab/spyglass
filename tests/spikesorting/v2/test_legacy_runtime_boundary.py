"""v0/v1 query paths keep working under SpikeInterface 0.104.

- the project is pinned to SI 0.104,
- the v0/v1 spike-sorting modules and the shared modules that reference
  SpikeInterface all import under SI 0.104,
- each guarded v0/v1 entry point raises the legacy-environment error at
  call time,
- existing v0/v1 `SpikeSortingOutput` merge queries remain functional,
- v0/v1 DataJoint ``definition`` strings have not drifted,
- the resolver finds ``mountainsort5`` and the optional matching extra.

The merge-query smoke test needs Docker.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_LEGACY_SCHEMAS_BASELINE = (
    _REPO_ROOT
    / "tests"
    / "spikesorting"
    / "v2"
    / "baselines"
    / "legacy_schemas.json"
)


# ---------- Schema-stability test: v0/v1 definitions are unchanged ------------

# Source files whose DataJoint ``definition`` blocks must match the SI 0.99
# baseline.
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
    # ``class`` may be indented (Part tables nested in a master).
    pattern = re.compile(
        r"^\s*class\s+(\w+).*?\n\s+definition\s*=\s*\"\"\"\n(.*?)\"\"\"",
        re.DOTALL | re.MULTILINE,
    )
    return {
        match.group(1): match.group(2) for match in pattern.finditer(source)
    }


def test_no_legacy_schema_changes():
    """v0/v1 DataJoint ``definition`` blocks are byte-identical to baseline.

    Baseline: ``tests/spikesorting/v2/baselines/legacy_schemas.json``. A
    deliberate definition change must regenerate it in the same change.
    """
    assert _LEGACY_SCHEMAS_BASELINE.exists(), (
        f"Missing baseline {_LEGACY_SCHEMAS_BASELINE}; regenerate with the "
        "snapshot script in tests/spikesorting/v2/baselines/README.md."
    )
    baseline = json.loads(_LEGACY_SCHEMAS_BASELINE.read_text())

    current: dict[str, dict[str, str]] = {}
    for tier, files in _LEGACY_TABLE_FILES.items():
        current[tier] = {}
        for relative in files:
            path = _SRC / relative
            if not path.exists():
                pytest.fail(f"Missing legacy file {path}")
            file_blocks = _extract_definition_blocks(path)
            assert file_blocks, (
                f"No triple-quoted `definition =` blocks parsed from "
                f"{relative}; the parser or the file structure changed."
            )
            for name, body in file_blocks.items():
                current[tier][name] = body

    for tier in baseline:
        baseline_names = set(baseline[tier])
        current_names = set(current.get(tier, {}))
        assert baseline_names == current_names, (
            f"Class set drift in {tier}: "
            f"added={current_names - baseline_names} "
            f"removed={baseline_names - current_names}"
        )
        for name, expected in baseline[tier].items():
            actual = current[tier][name]
            assert actual == expected, (
                f"Definition of {tier}.{name} drifted from the legacy "
                "baseline. If this is intentional, regenerate "
                f"{_LEGACY_SCHEMAS_BASELINE.name} in the same change."
            )


# ---------- Sorter / matching-extra resolution -----------------------------


def test_sorter_runtime_resolution():
    """SpikeInterface 0.104 is installed and ``mountainsort5`` is available.

    Also records the MountainSort 4 (``mountainsort4``) status. Both native
    sorters are included in the ``spikesorting-v2`` extra; the standalone
    runtime test exercises MS4's algorithm and compiled dependencies.
    """
    import spikeinterface as si
    import spikeinterface.sorters as sis

    assert Version(si.__version__) >= Version("0.104")
    installed = set(sis.installed_sorters())
    assert "mountainsort5" in installed, installed
    # MS4 is exercised by test_mountainsort4_runtime.py.
    print(
        f"mountainsort4 in installed_sorters(): "
        f"{'mountainsort4' in installed}"
    )


def test_default_merge_sources_skip_v2_when_unavailable(dj_conn, monkeypatch):
    """``get_restricted_merge_ids`` must not force v2 in a v0/v1-only env.

    The default resolves to the available sources; an explicit list is
    honored verbatim. (``dj_conn``: importing the merge module declares its
    schema.)
    """
    from spyglass.spikesorting import spikesorting_merge as m

    monkeypatch.setattr(m, "CurationV2", None)
    monkeypatch.setattr(m, "ConcatMemberCuration", None)
    assert m._available_merge_sources() == ["v0", "v1"]

    monkeypatch.setattr(m, "CurationV2", object())
    assert m._available_merge_sources() == ["v0", "v1", "v2"]


def test_v2_artifact_restriction_warns_when_unresolved(dj_conn, monkeypatch):
    """``restrict_by_artifact=True`` with a non-artifact interval name and
    no ``artifact_detection_id`` warns that v2 results are NOT artifact-restricted,
    instead of silently returning unrestricted merge ids.
    """
    from spyglass.spikesorting import spikesorting_merge as m

    captured = []
    monkeypatch.setattr(
        m.logger, "warning", lambda msg, *a, **k: captured.append(msg)
    )
    m.SpikeSortingOutput()._get_restricted_merge_ids_v2(
        {"interval_list_name": "raw data valid times"},
        restrict_by_artifact=True,
    )
    assert any("artifact-restricted" in msg for msg in captured), captured


# ---------- v0/v1 and shared SpikeInterface-using modules load under SI 0.104

_SI_DEPENDENT_IMPORTS_TO_VERIFY = [
    # query-compatible modules that reference SpikeInterface, smoke-imported
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


@pytest.mark.parametrize("module_name", _SI_DEPENDENT_IMPORTS_TO_VERIFY)
def test_legacy_import_smoke(module_name, dj_conn):
    """v0/v1 + shared SpikeInterface-using modules import under SI 0.104.

    Runtime guards fire at call time, not import. ``dj_conn`` ensures MySQL is
    ready before schema declaration.
    """
    importlib.import_module(module_name)


# ---------- Each guarded entry point raises at call time -------------------


def _assert_legacy_guard(
    component: str, callable_: Callable[[], object]
) -> None:
    """Assert ``callable_()`` raises the legacy-environment guard's error.

    Captures the raised exception once so a guard that fails to raise on the
    second invocation cannot silently pass the component-name assertion.
    """
    with pytest.raises(
        RuntimeError, match="legacy SpikeInterface 0.99"
    ) as exc_info:
        callable_()
    assert component in str(exc_info.value)


def test_legacy_runtime_guard_raises_under_si_0104(dj_conn, monkeypatch):
    """Every guarded v0/v1 entry point raises the legacy-env guard at call."""
    import spikeinterface as si

    if Version(si.__version__) < Version("0.101"):
        pytest.skip("Guard intentionally inactive under legacy SI < 0.101")

    from spyglass.decoding.v0.clusterless import UnitMarks
    from spyglass.decoding.v1.waveform_features import (
        UnitWaveformFeatures,
        WaveformFeaturesParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
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

    # UnitWaveformFeatures.make guards only after source dispatch (non-v2
    # branch), and make({}) fails earlier on fetch1("params"). Insert one
    # param row and stub merge_get_parent to report a CurationV1 source.
    WaveformFeaturesParams().insert1(
        [
            "legacy_guard_amplitude",
            {
                "waveform_features_params": {
                    "amplitude": {
                        "peak_sign": "neg",
                        "estimate_peak_time": False,
                    }
                },
                "waveform_extraction_params": {
                    "ms_before": 0.5,
                    "ms_after": 0.5,
                    "max_spikes_per_unit": None,
                    "n_jobs": 5,
                    "chunk_duration": "1000s",
                },
            },
        ],
        skip_duplicates=True,
    )

    class _NonV2Source:
        # ``to_camel_case("__curation_v1") == "CurationV1"`` (not CurationV2),
        # so ``UnitWaveformFeatures.make`` takes the legacy ``else`` branch.
        table_name = "__curation_v1"

        def fetch1(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(
        SpikeSortingOutput,
        "merge_get_parent",
        lambda self, *a, **k: _NonV2Source(),
    )

    waveform_guard_key = {
        "spikesorting_merge_id": "00000000-0000-0000-0000-000000000000",
        "features_param_name": "legacy_guard_amplitude",
    }

    cases = [
        (
            "v0 Waveforms.make",
            lambda: V0Waveforms().make_compute(
                key={},
                waveform_params={},
                waveform_extractor_path="",
                recording_path="",
                sorting_path="",
                merge_groups=[],
            ),
        ),
        (
            "v0 QualityMetrics.make",
            lambda: V0QualityMetrics().make_compute(
                key={},
                wf_path="",
                params={},
                qm_name="",
                quality_metrics_path="",
            ),
        ),
        ("v0 BurstPair.make", lambda: V0BurstPair().make({})),
        ("v0 ArtifactDetection.make", lambda: V0ArtifactDetection().make({})),
        (
            "v1 MetricCuration.make",
            lambda: MetricCuration().make_compute({}, {}),
        ),
        ("v1 ArtifactDetection.make", lambda: V1ArtifactDetection().make({})),
        ("v1 BurstPair.make", lambda: V1BurstPair().make({})),
        ("v0 UnitMarks.make", lambda: UnitMarks().make({})),
        (
            "v1 UnitWaveformFeatures.make",
            lambda: UnitWaveformFeatures().make(waveform_guard_key),
        ),
    ]
    try:
        for component, call in cases:
            _assert_legacy_guard(component, call)
    finally:
        # Drop the param row inserted above so it doesn't linger in a shared
        # DB. No FK consumers (the guard fires before any selection insert).
        (
            WaveformFeaturesParams
            & {"features_param_name": "legacy_guard_amplitude"}
        ).delete(safemode=False)


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
    # An unknown merge_id restricts to an empty result without error.
    unknown_uuid = "00000000-0000-0000-0000-000000000000"
    result = (SpikeSortingOutput & {"merge_id": unknown_uuid}).fetch(
        as_dict=True
    )
    assert result == []
