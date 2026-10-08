"""Keep implementation ownership and dependency direction explicit."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3] / "src/spyglass/spikesorting/v2"
PREFIX = "spyglass.spikesorting.v2"
DOMAIN_PACKAGES = {
    "_core",
    "_recording",
    "_artifacts",
    "_sorting",
    "_storage",
    "_curation",
    "_review",
    "_matching",
    "_motion",
    "_orchestration",
}


def test_private_implementations_have_domain_owners():
    assert not (ROOT / "_internal").exists()
    assert {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    } == DOMAIN_PACKAGES | {"_params"}
    for domain in DOMAIN_PACKAGES:
        assert set((ROOT / domain).rglob("__init__.py")) == {
            ROOT / domain / "__init__.py"
        }, f"Nested private package in {domain}"
    assert {
        path.name for path in ROOT.glob("_*.py") if path.name != "__init__.py"
    } == set()
    assert not (ROOT / "utils.py").exists()
    assert not (ROOT / "_core" / "legacy_imports.py").exists()
    implementations = [
        path
        for domain in DOMAIN_PACKAGES
        for path in (ROOT / domain).rglob("*.py")
        if path.name != "__init__.py"
    ]
    assert implementations
    for path in implementations:
        assert path.parent.parent == ROOT, f"Nested implementation: {path}"
        assert not path.stem.startswith(
            "_"
        ), f"Private filename in private package: {path}"
        owner = path.parent.name.removeprefix("_")
        assert not path.stem.startswith(
            f"{owner}_"
        ), f"Repeated domain prefix: {path}"


def _imported_modules(node, package):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    module = node.module or ""
    if node.level:
        module = importlib.util.resolve_name("." * node.level + module, package)
    if module in {PREFIX, *(f"{PREFIX}.{name}" for name in DOMAIN_PACKAGES)}:
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module]


def test_internal_dependencies_use_owners_and_keep_orchestration_outward():
    violations = []
    for path in (
        path
        for domain in DOMAIN_PACKAGES
        for path in (ROOT / domain).glob("*.py")
    ):
        package = f"{PREFIX}.{path.parent.name}"
        for node in ast.walk(ast.parse(path.read_text())):
            for module in _imported_modules(node, package):
                reason = None
                if module == f"{PREFIX}.utils" or module.startswith(
                    f"{PREFIX}.utils."
                ):
                    reason = "imports removed utilities facade"
                elif module.startswith(f"{PREFIX}._internal"):
                    reason = "imports previous implementation namespace"
                elif (
                    path.parent.name != "_orchestration"
                    and module.startswith(f"{PREFIX}._orchestration.")
                    and module.rsplit(".", 1)[-1] not in {"types", "exports"}
                ):
                    reason = "imports orchestration"
                if reason:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: {reason}"
                    )
    assert not violations, "\n".join(violations)


def test_private_domain_initializers_keep_imports_lightweight():
    """Domain namespaces contain only their explanatory docstring."""
    for domain in DOMAIN_PACKAGES:
        path = ROOT / domain / "__init__.py"
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Constant
            ):
                assert isinstance(node.value.value, str)
                continue
            pytest.fail(f"Unexpected initializer work in {path}:{node.lineno}")


def test_v2_source_imports_resolve_to_current_modules():
    """Table adapters, public APIs and services use current source locations."""
    violations = []
    for path in ROOT.rglob("*.py"):
        package = (
            f"{PREFIX}.{'.'.join(path.parent.relative_to(ROOT).parts)}".rstrip(
                "."
            )
        )
        for node in ast.walk(ast.parse(path.read_text())):
            for module in _imported_modules(node, package):
                if not module.startswith(f"{PREFIX}._"):
                    continue
                relative = Path(*module.removeprefix(f"{PREFIX}.").split("."))
                if not (
                    (ROOT / relative).with_suffix(".py").exists()
                    or (ROOT / relative / "__init__.py").exists()
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: {module}"
                    )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("wrapper", ["acquisition", "estimation"])
def test_recording_serialization_preserves_traces_and_clock(wrapper):
    """Current extractor dictionaries and pickles retain samples and timing."""
    import pickle

    import numpy as np
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2._motion.estimation import (
        EstimationClockRecording,
        build_estimation_clock,
    )
    from spyglass.spikesorting.v2._recording.acquisition_spans import (
        AcquisitionSpanRecording,
    )

    traces = np.arange(30, dtype=np.float32).reshape(10, 3)
    parent = sc.NumpyRecording(traces_list=[traces], sampling_frequency=1000.0)
    if wrapper == "acquisition":
        view = AcquisitionSpanRecording(parent, 2, 7)
        expected_class = (
            f"{PREFIX}._recording.acquisition_spans.AcquisitionSpanRecording"
        )
        expected_traces = traces[2:7]
        expected_times = np.arange(2, 7) / 1000.0
    else:
        clock = build_estimation_clock(
            [(0, 10)], [0.0], [0.009], 1000.0, max_gap_s=1.0
        )
        view = EstimationClockRecording(parent, clock)
        expected_class = f"{PREFIX}._motion.estimation.EstimationClockRecording"
        expected_traces = traces
        expected_times = np.arange(10) / 1000.0
    saved = view.to_dict(recursive=True)
    assert saved["class"] == expected_class
    restored = sc.load(saved)
    np.testing.assert_array_equal(restored.get_traces(), expected_traces)
    np.testing.assert_allclose(
        restored.get_times(), expected_times, rtol=0, atol=1e-12
    )

    payload = pickle.dumps(view, protocol=pickle.HIGHEST_PROTOCOL)
    canonical_module = type(view).__module__.encode()
    assert canonical_module in payload
    restored_pickle = pickle.loads(payload)
    np.testing.assert_array_equal(restored_pickle.get_traces(), expected_traces)
    np.testing.assert_allclose(
        restored_pickle.get_times(), expected_times, rtol=0, atol=1e-12
    )
