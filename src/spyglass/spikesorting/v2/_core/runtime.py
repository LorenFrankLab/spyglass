"""Structured runtime receipts and stable environment identities, without DB I/O.

Capture records the host orchestrator; a requested container image is execution
configuration, not proof of the environment inside that image. Producing sorter
versions remain recorded by the sorter adapter. Optional native-pool observations
describe libraries loaded at capture time and do not change environment identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path

from spyglass.spikesorting.v2._core.selection_identity import sha256_json

RUNTIME_SCHEMA_VERSION = 1
_PACKAGES = (
    "spyglass-neuro",
    "numpy",
    "scipy",
    "spikeinterface",
    "pynwb",
    "hdmf",
    "datajoint",
    "numba",
    "scikit-learn",
    "torch",
    "mountainsort4",
    "mountainsort5",
    "dredge",
    "UnitMatchPy",
    "figpack",
    "figpack-spike-sorting",
    "threadpoolctl",
)
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "PYTHONHASHSEED",
)
_DEVICE_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
_POOL_FIELDS = (
    "user_api",
    "internal_api",
    "prefix",
    "version",
    "num_threads",
    "threading_layer",
    "architecture",
)


def _json_value(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Runtime receipt cannot encode {type(value).__name__}.")


def _normalized(values):
    return json.loads(json.dumps(values, default=_json_value, allow_nan=False))


def _package_versions():
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_distributions():
    """Capture indirect dependencies and ambiguous duplicate metadata versions."""
    inventory = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        installed_version = distribution.version
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(installed_version, str)
            or not installed_version.strip()
        ):
            raise ValueError(
                "Installed distribution metadata lacks a name or version."
            )
        name = re.sub(r"[-_.]+", "-", name).lower()
        inventory.setdefault(name, set()).add(installed_version)
    return {
        name: sorted(versions) for name, versions in sorted(inventory.items())
    }


def _source_identity():
    package = Path(__file__).resolve().parents[3]
    files = {
        path.relative_to(package)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    if not files:
        raise ValueError("Cannot fingerprint Spyglass source: no Python files.")
    source_sha = sha256_json(files)
    root = package.parent.parent
    commit = None
    if (root / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {"git_commit": commit, "source_sha256": source_sha}


def _cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        try:
            return (
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                or None
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or None


def _platform_identity():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    }


def _gpu_inventory():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "devices": []}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise ValueError("Unexpected nvidia-smi device inventory.")
        devices.append(dict(zip(("name", "uuid", "driver_version"), fields)))
    return {"status": "available", "devices": devices}


def _native_pools():
    # Only mandatory linear-algebra dependencies are loaded. Importing optional
    # Torch here would change benchmark timing and GPU/process initialization.
    import numpy.linalg  # noqa: F401
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info

    pools = [
        {field: pool.get(field) for field in _POOL_FIELDS}
        for pool in threadpool_info()
    ]
    return sorted(pools, key=lambda pool: json.dumps(pool, sort_keys=True))


def capture_runtime_environment(*, job_kwargs=None, execution_params=None):
    """Capture JSON-safe source, dependency, platform and execution metadata.

    ``job_kwargs`` must be the resolved settings used by computation. Environment
    variable capture is restricted to thread and accelerator configuration.
    No timestamps, hostnames, arbitrary environment variables or absolute native
    library paths enter the receipt. NumPy/SciPy BLAS settings enter identity;
    other loaded pools remain observations to allow lazy optional-library imports.
    """
    for name, values in (
        ("job_kwargs", job_kwargs),
        ("execution_params", execution_params),
    ):
        if values is not None and not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping or None.")
    pools = _native_pools()
    snapshot = _normalized(
        {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "scope": "host_orchestrator",
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
            },
            "platform": _platform_identity(),
            "packages": _package_versions(),
            "installed_distributions": _installed_distributions(),
            "source": _source_identity(),
            "threading": {
                "environment": {
                    name: os.environ.get(name) for name in _THREAD_ENV
                },
                "blas_pools": [
                    pool for pool in pools if pool["user_api"] == "blas"
                ],
            },
            "accelerators": {
                "environment": {
                    name: os.environ.get(name) for name in _DEVICE_ENV
                },
                "nvidia": _gpu_inventory(),
            },
            "job_kwargs": dict(job_kwargs or {}),
            "execution_params": dict(execution_params or {}),
            "observed_native_pools": pools,
        }
    )
    runtime_environment_fingerprint(snapshot)
    return snapshot


def _require_string(value, field, *, optional=False, empty=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"runtime environment.{field} must be a string.")


def _validate_native_pool(pool):
    if not isinstance(pool, Mapping) or set(_POOL_FIELDS) - pool.keys():
        raise ValueError("runtime environment native pool is incomplete.")
    for field in _POOL_FIELDS:
        if field != "num_threads":
            _require_string(
                pool[field],
                f"native_pool.{field}",
                optional=field
                in {"version", "threading_layer", "architecture"},
            )
    if type(pool["num_threads"]) is not int or pool["num_threads"] <= 0:
        raise ValueError(
            "runtime environment native pool num_threads must be positive."
        )


def _validate_runtime_values(snapshot):
    inventory = snapshot["installed_distributions"]
    if not inventory:
        raise ValueError(
            "runtime environment installed distribution inventory is empty."
        )
    for name, versions in inventory.items():
        _require_string(name, "installed_distributions.name")
        if (
            not isinstance(versions, list)
            or not versions
            or any(
                not isinstance(value, str) or not value.strip()
                for value in versions
            )
        ):
            raise ValueError(
                "runtime environment installed distribution versions are invalid."
            )
        if (
            versions != sorted(set(versions))
            or name != re.sub(r"[-_.]+", "-", name).lower()
        ):
            raise ValueError(
                "runtime environment installed distribution inventory must be normalized."
            )
    for field in ("version", "implementation", "compiler"):
        _require_string(snapshot["python"][field], f"python.{field}")
    for field in ("system", "release", "machine", "cpu_model"):
        _require_string(
            snapshot["platform"][field],
            f"platform.{field}",
            optional=field == "cpu_model",
        )
    count = snapshot["platform"]["cpu_count"]
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError(
            "runtime environment.platform.cpu_count must be positive."
        )
    affinity = snapshot["platform"]["cpu_affinity"]
    if affinity is not None and (
        not isinstance(affinity, list)
        or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
        or len(set(affinity)) != len(affinity)
    ):
        raise ValueError(
            "runtime environment.platform.cpu_affinity is invalid."
        )
    _require_string(
        snapshot["source"]["git_commit"], "source.git_commit", optional=True
    )
    for group, keys in (
        ("threading", _THREAD_ENV),
        ("accelerators", _DEVICE_ENV),
    ):
        values = snapshot[group]["environment"]
        if not isinstance(values, Mapping) or set(keys) != values.keys():
            raise ValueError(
                f"runtime environment.{group}.environment is incomplete or contains unknown fields."
            )
        for name, value in values.items():
            _require_string(
                value, f"{group}.environment.{name}", optional=True, empty=True
            )
    for pool in snapshot["threading"]["blas_pools"]:
        _validate_native_pool(pool)
        if pool["user_api"] != "blas":
            raise ValueError(
                "runtime environment.threading.blas_pools must describe BLAS."
            )
    for pool in snapshot["observed_native_pools"]:
        _validate_native_pool(pool)
    nvidia = snapshot["accelerators"]["nvidia"]
    if not isinstance(nvidia, Mapping) or {"status", "devices"} - nvidia.keys():
        raise ValueError(
            "runtime environment accelerator inventory is incomplete."
        )
    if (
        not isinstance(nvidia["status"], str)
        or nvidia["status"] not in {"available", "unavailable"}
        or not isinstance(nvidia["devices"], list)
    ):
        raise ValueError(
            "runtime environment accelerator inventory is invalid."
        )
    if nvidia["status"] == "unavailable" and nvidia["devices"]:
        raise ValueError(
            "Unavailable accelerator inventory cannot contain devices."
        )
    for device in nvidia["devices"]:
        if (
            not isinstance(device, Mapping)
            or {"name", "uuid", "driver_version"} - device.keys()
        ):
            raise ValueError(
                "runtime environment accelerator device is incomplete."
            )
        for field in ("name", "uuid", "driver_version"):
            _require_string(device[field], f"accelerators.nvidia.{field}")


def runtime_environment_fingerprint(snapshot):
    """Validate a current receipt and hash its reproducibility identity."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("runtime environment must be a mapping.")
    required = {
        "schema_version",
        "scope",
        "python",
        "platform",
        "packages",
        "installed_distributions",
        "source",
        "threading",
        "accelerators",
        "job_kwargs",
        "execution_params",
        "observed_native_pools",
    }
    if required - snapshot.keys():
        raise ValueError("runtime environment is missing required fields.")
    if (
        type(snapshot["schema_version"]) is not int
        or snapshot["schema_version"] != RUNTIME_SCHEMA_VERSION
    ):
        raise ValueError("runtime environment requires schema_version=1.")
    if snapshot["scope"] != "host_orchestrator":
        raise ValueError(
            "runtime environment requires host_orchestrator scope."
        )
    fields = {
        "python": {"version", "implementation", "compiler"},
        "platform": {
            "system",
            "release",
            "machine",
            "cpu_model",
            "cpu_count",
            "cpu_affinity",
        },
        "packages": set(_PACKAGES),
        "installed_distributions": set(),
        "source": {"git_commit", "source_sha256"},
        "threading": {"environment", "blas_pools"},
        "accelerators": {"environment", "nvidia"},
        "job_kwargs": set(),
        "execution_params": set(),
    }
    for name, keys in fields.items():
        value = snapshot[name]
        if not isinstance(value, Mapping) or keys - value.keys():
            raise ValueError(f"runtime environment.{name} is incomplete.")
    for group, keys in (
        ("threading", _THREAD_ENV),
        ("accelerators", _DEVICE_ENV),
    ):
        values = snapshot[group]["environment"]
        if not isinstance(values, Mapping) or set(keys) - values.keys():
            raise ValueError(
                f"runtime environment.{group}.environment is incomplete."
            )
    if not isinstance(
        snapshot["threading"]["blas_pools"], list
    ) or not isinstance(snapshot["observed_native_pools"], list):
        raise ValueError("runtime environment native pools must be lists.")
    for name, value in snapshot["packages"].items():
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"runtime environment package {name!r} has an invalid version."
            )
    source_sha = snapshot["source"]["source_sha256"]
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(char not in "0123456789abcdef" for char in source_sha)
    ):
        raise ValueError(
            "runtime environment.source.source_sha256 must be SHA-256."
        )
    identity = {
        key: value
        for key, value in snapshot.items()
        if key != "observed_native_pools"
    }
    _validate_runtime_values(snapshot)
    try:
        return sha256_json(identity, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime environment must contain finite JSON values."
        ) from exc


def runtime_environment_provenance(*, job_kwargs=None, execution_params=None):
    """Return a runtime receipt and its identity for current scientific outputs."""
    snapshot = capture_runtime_environment(
        job_kwargs=job_kwargs, execution_params=execution_params
    )
    return {
        "runtime_environment": snapshot,
        "runtime_environment_sha256": runtime_environment_fingerprint(snapshot),
    }


def validate_runtime_provenance(values):
    """Require an intact runtime receipt on current scientific provenance."""
    if (
        not isinstance(values, Mapping)
        or {"runtime_environment", "runtime_environment_sha256"} - values.keys()
    ):
        raise ValueError("Runtime provenance is missing required fields.")
    actual = runtime_environment_fingerprint(values["runtime_environment"])
    if values["runtime_environment_sha256"] != actual:
        raise ValueError(
            "Runtime provenance fingerprint does not match its receipt."
        )
