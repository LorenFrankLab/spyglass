"""Runtime receipts describe the producing host without leaking ambient state."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from spyglass.spikesorting.v2._core import runtime

pytestmark = pytest.mark.unit


def _pool(*, api="blas", threads=2):
    return {
        "user_api": api,
        "internal_api": "openblas" if api == "blas" else "openmp",
        "prefix": "libopenblas" if api == "blas" else "libgomp",
        "version": "0.3.30" if api == "blas" else None,
        "num_threads": threads,
        "threading_layer": "pthreads" if api == "blas" else None,
        "architecture": "NEHALEM" if api == "blas" else None,
    }


@pytest.fixture
def receipt(monkeypatch):
    """A complete synthetic receipt, independent of installed optional tools."""
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.11.0")
    monkeypatch.setattr(
        runtime.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(runtime.platform, "python_compiler", lambda: "GCC 13")
    monkeypatch.setattr(
        runtime,
        "_platform_identity",
        lambda: {
            "system": "TestOS",
            "release": "1.0",
            "machine": "x86_64",
            "cpu_model": "SyntheticCPU",
            "cpu_count": 8,
            "cpu_affinity": [0, 1],
        },
    )
    monkeypatch.setattr(runtime, "version", lambda name: "1.0")
    monkeypatch.setattr(
        runtime,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version="1.0")
            for name in runtime._PACKAGES
        ]
        + [
            SimpleNamespace(
                metadata={"Name": "indirect-numerical"}, version="3.0"
            )
        ],
    )
    monkeypatch.setattr(
        runtime,
        "_source_identity",
        lambda: {
            "git_commit": "a" * 40,
            "source_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_gpu_inventory",
        lambda: {
            "status": "unavailable",
            "devices": [],
        },
    )
    monkeypatch.setattr(runtime, "_native_pools", lambda: [_pool()])
    for name in (*runtime._THREAD_ENV, *runtime._DEVICE_ENV):
        monkeypatch.delenv(name, raising=False)
    return runtime.capture_runtime_environment(
        job_kwargs={
            "n_jobs": np.int64(2),
            "chunk_duration": "1s",
            "progress_bar": False,
        },
        execution_params={"docker_image": "ghcr.io/example/sorter:1.0"},
    )


def test_capture_is_json_roundtrippable_and_ignores_arbitrary_environment(
    receipt, monkeypatch
):
    monkeypatch.setenv("SPYGLASS_TEST_SECRET", "do-not-record-this-token")
    monkeypatch.setenv("SPYGLASS_TEST_PATH", "/private/unrelated/path")
    captured = runtime.capture_runtime_environment(
        job_kwargs=receipt["job_kwargs"],
        execution_params=receipt["execution_params"],
    )
    encoded = json.dumps(captured, allow_nan=False)
    restored = json.loads(encoded)
    assert restored == captured
    assert restored["job_kwargs"]["n_jobs"] == 2
    assert "SPYGLASS_TEST_SECRET" not in encoded
    assert "do-not-record-this-token" not in encoded
    assert "/private/unrelated/path" not in encoded
    assert set(captured["threading"]["environment"]) == set(runtime._THREAD_ENV)
    assert set(captured["accelerators"]["environment"]) == set(
        runtime._DEVICE_ENV
    )
    assert runtime.runtime_environment_fingerprint(
        restored
    ) == runtime.runtime_environment_fingerprint(receipt)


def test_native_pool_capture_omits_absolute_library_paths(monkeypatch):
    import threadpoolctl

    pool = {
        **_pool(),
        "filepath": "/private/native/libopenblas.dylib",
        "unexpected": "not-a-receipt-field",
    }
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [pool])
    actual = runtime._native_pools()
    assert actual == [_pool()]
    encoded = json.dumps(actual)
    assert "filepath" not in encoded
    assert "/private/native" not in encoded
    assert "unexpected" not in encoded


def test_real_capture_is_stable_without_optional_import_warming():
    """The first capture warms mandatory BLAS dependencies before taking identity."""
    first = runtime.capture_runtime_environment(job_kwargs={"n_jobs": 1})
    second = runtime.capture_runtime_environment(job_kwargs={"n_jobs": 1})
    assert runtime.runtime_environment_fingerprint(
        first
    ) == runtime.runtime_environment_fingerprint(second)
    assert first["threading"]["blas_pools"] == second["threading"]["blas_pools"]
    assert first["source"] == second["source"]
    assert json.loads(json.dumps(first, allow_nan=False)) == first


@pytest.mark.parametrize(
    "path, replacement",
    [
        (("python", "version"), "3.12.0"),
        (("python", "implementation"), "PyPy"),
        (("python", "compiler"), "Clang 18"),
        (("packages", "numpy"), "2.0"),
        (("platform", "cpu_model"), "AnotherCPU"),
        (("platform", "cpu_count"), 16),
        (("platform", "cpu_affinity"), [0]),
        (("platform", "machine"), "arm64"),
        (("platform", "release"), "2.0"),
        (("threading", "environment", "OMP_NUM_THREADS"), "4"),
        (("threading", "blas_pools", 0, "num_threads"), 4),
        (("threading", "blas_pools", 0, "version"), "0.3.31"),
        (("accelerators", "environment", "CUDA_VISIBLE_DEVICES"), "0"),
        (
            ("accelerators", "nvidia"),
            {
                "status": "available",
                "devices": [
                    {
                        "name": "Test GPU",
                        "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                        "driver_version": "550.0",
                    }
                ],
            },
        ),
        (("job_kwargs", "n_jobs"), 4),
        (("execution_params", "docker_image"), "ghcr.io/example/sorter:2.0"),
        (("source", "source_sha256"), "c" * 64),
        (("source", "git_commit"), "d" * 40),
    ],
)
def test_reproducibility_inputs_change_identity(receipt, path, replacement):
    changed = copy.deepcopy(receipt)
    parent = changed
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = replacement
    assert runtime.runtime_environment_fingerprint(
        changed
    ) != runtime.runtime_environment_fingerprint(receipt)


def test_lazy_optional_native_pool_observations_do_not_change_identity(receipt):
    changed = copy.deepcopy(receipt)
    changed["observed_native_pools"].append(_pool(api="openmp", threads=8))
    changed["observed_native_pools"][0]["num_threads"] = 12
    assert runtime.runtime_environment_fingerprint(
        changed
    ) == runtime.runtime_environment_fingerprint(receipt)


def test_unknown_optional_package_versions_and_unavailable_hardware_are_explicit(
    receipt, monkeypatch
):
    def installed_version(name):
        if name in {"numpy", "scipy", "spikeinterface", "spyglass-neuro"}:
            return "1.0"
        raise runtime.PackageNotFoundError(name)

    monkeypatch.setattr(runtime, "version", installed_version)
    versions = runtime._package_versions()
    assert versions["numpy"] == "1.0"
    assert versions["torch"] is None
    assert versions["UnitMatchPy"] is None
    changed = copy.deepcopy(receipt)
    changed["packages"] = versions
    changed["source"]["git_commit"] = None
    changed["platform"]["cpu_model"] = None
    changed["platform"]["cpu_affinity"] = None
    assert len(runtime.runtime_environment_fingerprint(changed)) == 64


def test_source_identity_changes_when_package_code_changes(
    tmp_path, monkeypatch
):
    package = tmp_path / "checkout" / "src" / "spyglass"
    module = package / "spikesorting" / "v2" / "_core" / "runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("# runtime\n")
    scientific = package / "scientific.py"
    scientific.write_text("threshold = 5\n")
    monkeypatch.setattr(runtime, "__file__", str(module))
    first = runtime._source_identity()
    scientific.write_text("threshold = 6\n")
    second = runtime._source_identity()
    assert first["git_commit"] is None
    assert second["git_commit"] is None
    assert first["source_sha256"] != second["source_sha256"]
    scientific.with_suffix(".md").write_text("non-source documentation\n")
    assert runtime._source_identity() == second
    moved = scientific.with_name("renamed.py")
    scientific.rename(moved)
    assert (
        runtime._source_identity()["source_sha256"] != second["source_sha256"]
    )


def test_requested_container_is_configuration_not_producer_environment(receipt):
    assert receipt["scope"] == "host_orchestrator"
    assert (
        receipt["execution_params"]["docker_image"]
        == "ghcr.io/example/sorter:1.0"
    )
    assert receipt["python"]["implementation"] == "CPython"
    assert receipt["platform"]["cpu_model"] == "SyntheticCPU"
    assert receipt["packages"]["spikeinterface"] == "1.0"
    assert "container_environment" not in receipt
    changed = copy.deepcopy(receipt)
    changed["scope"] = "container_producer"
    with pytest.raises(ValueError, match="scope"):
        runtime.runtime_environment_fingerprint(changed)


@pytest.mark.parametrize(
    "path",
    [
        ("python", "version"),
        ("python", "compiler"),
        ("platform", "cpu_count"),
        ("platform", "cpu_affinity"),
        ("packages", "numpy"),
        ("source", "git_commit"),
        ("source", "source_sha256"),
        ("threading", "environment", "OMP_NUM_THREADS"),
        ("threading", "blas_pools"),
        ("threading", "blas_pools", 0, "num_threads"),
        ("accelerators", "environment", "CUDA_VISIBLE_DEVICES"),
        ("accelerators", "nvidia", "status"),
        ("accelerators", "nvidia", "devices"),
    ],
)
def test_fingerprint_rejects_missing_nested_receipt_fields(receipt, path):
    changed = copy.deepcopy(receipt)
    parent = changed
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]
    with pytest.raises(ValueError):
        runtime.runtime_environment_fingerprint(changed)


@pytest.mark.parametrize(
    "path, replacement",
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("python", "version"), None),
        (("python", "implementation"), 1),
        (("platform", "system"), ""),
        (("platform", "cpu_count"), 0),
        (("platform", "cpu_count"), True),
        (("platform", "cpu_affinity"), "all"),
        (("packages", "numpy"), 123),
        (("source", "source_sha256"), "not-a-digest"),
        (("source", "git_commit"), 123),
        (("threading", "environment", "OMP_NUM_THREADS"), 2),
        (("threading", "blas_pools"), [{}]),
        (("threading", "blas_pools", 0, "num_threads"), -1),
        (("threading", "blas_pools", 0, "num_threads"), True),
        (("threading", "blas_pools", 0, "user_api"), "openmp"),
        (("observed_native_pools",), [{}]),
        (("accelerators", "environment", "CUDA_VISIBLE_DEVICES"), [0]),
        (("accelerators", "nvidia"), {"status": "unknown", "devices": []}),
        (("accelerators", "nvidia"), {"status": [], "devices": []}),
        (("accelerators", "nvidia"), {"status": "available", "devices": [{}]}),
        (("job_kwargs",), []),
        (("execution_params",), []),
    ],
)
def test_fingerprint_rejects_malformed_nested_receipt_values(
    receipt, path, replacement
):
    changed = copy.deepcopy(receipt)
    parent = changed
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = replacement
    with pytest.raises(ValueError):
        runtime.runtime_environment_fingerprint(changed)


@pytest.mark.parametrize("container", ["threading", "accelerators"])
def test_fingerprint_rejects_unlisted_environment_capture(receipt, container):
    changed = copy.deepcopy(receipt)
    changed[container]["environment"]["SECRET_TOKEN"] = "not-permitted"
    with pytest.raises(ValueError):
        runtime.runtime_environment_fingerprint(changed)


def test_gpu_capture_handles_unavailable_cli_and_records_device_driver(
    monkeypatch,
):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    assert runtime._gpu_inventory() == {"status": "unavailable", "devices": []}
    monkeypatch.setattr(
        runtime.shutil, "which", lambda name: "/private/bin/nvidia-smi"
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout="Test GPU, GPU-001, 550.0\n")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime._gpu_inventory() == {
        "status": "available",
        "devices": [
            {"name": "Test GPU", "uuid": "GPU-001", "driver_version": "550.0"}
        ],
    }
    assert commands[0][1] == "--query-gpu=name,uuid,driver_version"


def test_runtime_provenance_captures_resolved_execution_and_checks_integrity(
    receipt,
):
    provenance = runtime.runtime_environment_provenance(
        job_kwargs=receipt["job_kwargs"],
        execution_params=receipt["execution_params"],
    )
    assert provenance["runtime_environment"] == receipt
    assert provenance[
        "runtime_environment_sha256"
    ] == runtime.runtime_environment_fingerprint(receipt)
    assert runtime.validate_runtime_provenance(provenance) is None
    changed = copy.deepcopy(provenance)
    changed["runtime_environment"]["job_kwargs"]["n_jobs"] = 8
    with pytest.raises(ValueError, match="fingerprint does not match"):
        runtime.validate_runtime_provenance(changed)


@pytest.mark.parametrize(
    "field", ["runtime_environment", "runtime_environment_sha256"]
)
def test_runtime_provenance_rejects_missing_receipt_or_digest(receipt, field):
    provenance = {
        "runtime_environment": receipt,
        "runtime_environment_sha256": runtime.runtime_environment_fingerprint(
            receipt
        ),
    }
    del provenance[field]
    with pytest.raises(ValueError, match="missing required fields"):
        runtime.validate_runtime_provenance(provenance)


def test_runtime_provenance_rejects_malformed_receipt_with_an_existing_digest(
    receipt,
):
    provenance = {
        "runtime_environment": copy.deepcopy(receipt),
        "runtime_environment_sha256": runtime.runtime_environment_fingerprint(
            receipt
        ),
    }
    del provenance["runtime_environment"]["python"]["version"]
    with pytest.raises(ValueError, match="python"):
        runtime.validate_runtime_provenance(provenance)


def test_indirect_distribution_version_changes_runtime_identity(receipt):
    changed = copy.deepcopy(receipt)
    changed["installed_distributions"]["indirect-numerical"] = ["3.1"]
    assert changed["packages"] == receipt["packages"]
    assert runtime.runtime_environment_fingerprint(
        changed
    ) != runtime.runtime_environment_fingerprint(receipt)


def test_distribution_inventory_normalizes_names_and_keeps_distinct_versions(
    monkeypatch,
):
    distributions = [
        SimpleNamespace(
            metadata={"Name": "My_Numerical.Dep"},
            version="2.0",
            _path="/private/site-a/dist-info",
        ),
        SimpleNamespace(
            metadata={"Name": "my-numerical-dep"},
            version="1.0",
            _path="/private/site-b/dist-info",
        ),
        SimpleNamespace(
            metadata={"Name": "MY__NUMERICAL...DEP"},
            version="2.0",
            _path="/private/site-c/dist-info",
        ),
        SimpleNamespace(
            metadata={"Name": "Another"},
            version="3.0",
            _path="/private/site-d/dist-info",
        ),
    ]
    monkeypatch.setattr(runtime, "distributions", lambda: distributions)
    inventory = runtime._installed_distributions()
    assert inventory == {"another": ["3.0"], "my-numerical-dep": ["1.0", "2.0"]}
    assert list(inventory) == ["another", "my-numerical-dep"]
    assert "/private/" not in json.dumps(inventory)
    assert json.loads(json.dumps(inventory)) == inventory
    monkeypatch.setattr(
        runtime, "distributions", lambda: reversed(distributions)
    )
    assert runtime._installed_distributions() == inventory


def test_fingerprint_requires_installed_distribution_inventory(receipt):
    changed = copy.deepcopy(receipt)
    del changed["installed_distributions"]
    with pytest.raises(ValueError, match="missing required fields"):
        runtime.runtime_environment_fingerprint(changed)


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {},
        [],
        {"numpy": "1.0"},
        {"numpy": []},
        {"numpy": [None]},
        {"numpy": [123]},
        {"numpy": [""]},
        {"numpy": ["1.0", "1.0"]},
        {"numpy": ["2.0", "1.0"]},
        {"NumPy": ["1.0"]},
        {"my_dep": ["1.0"]},
        {"": ["1.0"]},
    ],
    ids=[
        "none",
        "empty",
        "wrong_container",
        "scalar_version",
        "empty_versions",
        "none_version",
        "numeric_version",
        "empty_version",
        "duplicate_versions",
        "unsorted_versions",
        "uppercase_name",
        "underscore_name",
        "empty_name",
    ],
)
def test_fingerprint_rejects_malformed_installed_distribution_inventory(
    receipt, inventory
):
    changed = copy.deepcopy(receipt)
    changed["installed_distributions"] = inventory
    with pytest.raises(ValueError):
        runtime.runtime_environment_fingerprint(changed)


@pytest.mark.parametrize(
    "name, version",
    [(None, "1.0"), ("", "1.0"), ("numpy", None), ("numpy", "")],
)
def test_distribution_capture_rejects_incomplete_metadata(
    monkeypatch, name, version
):
    monkeypatch.setattr(
        runtime,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}, version=version)],
    )
    with pytest.raises(ValueError, match="lacks a name or version"):
        runtime._installed_distributions()
