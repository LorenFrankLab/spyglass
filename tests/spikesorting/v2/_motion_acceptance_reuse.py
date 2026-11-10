"""Reuse bookkeeping outside the scientific motion benchmark's pinned harness.

The pinned runner and gates remain unchanged. This wrapper stamps new results
with production source bytes and the numerical runtime that executed them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from spyglass.spikesorting.v2._core.runtime import (
    capture_runtime_environment,
    runtime_environment_fingerprint,
)
from tests.spikesorting.v2._motion_acceptance import (
    case_tag,
    load_manifest,
    result_is_reusable,
)

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"


def production_source_fingerprint(source_root=SOURCE_ROOT) -> str:
    """Hash each relative Python path and its bytes in deterministic order."""
    root = Path(source_root)
    files = {
        path.relative_to(root)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    if not files:
        raise ValueError(f"No production Python sources found in {root}.")
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def benchmark_runtime_environment(args) -> dict:
    """Capture the explicit job and sorter settings the pinned runner uses."""
    from spyglass.spikesorting.v2._params.sorter import MountainSort5Schema

    def option(name):
        return args[args.index(name) + 1]

    if args[0] == "case":
        manifest = load_manifest(Path(option("--manifest")))
        job_kwargs = dict(manifest.evaluation.job_kwargs)
        recipe = option("--recipe")
        seed = int(option("--seed"))
        execution_params = {
            "scenario": option("--scenario"),
            "seed": seed,
            "recipe": recipe,
            "recipe_params": manifest.recipes[recipe].model_dump(),
            "noise_levels_seed": (
                seed
                if manifest.noise_levels_seed == "case_seed"
                else int(manifest.noise_levels_seed)
            ),
            "sorter": manifest.sorter.name,
            "sorter_params": MountainSort5Schema(
                **manifest.sorter.params
            ).model_dump(),
            "sorter_job_kwargs": {"random_seed": manifest.sorter.random_seed},
        }
    elif args[0] == "representative":
        from tests.spikesorting.v2._motion_fixtures import JOB_KWARGS

        job_kwargs = dict(JOB_KWARGS)
        execution_params = {
            "sort": "--sort" in args,
            "shank": int(option("--shank")) if "--shank" in args else 2,
        }
        if execution_params["sort"]:
            execution_params.update(
                sorter="mountainsort5",
                sorter_params=MountainSort5Schema().model_dump(),
                sorter_job_kwargs={"random_seed": 0},
            )
    else:
        raise ValueError(f"Unsupported benchmark command {args[0]!r}.")
    return capture_runtime_environment(
        job_kwargs=job_kwargs, execution_params=execution_params
    )


def source_result_is_reusable(
    result,
    *,
    manifest_sha,
    fingerprint,
    source_sha,
    runtime_environment,
):
    """Require intact receipts for the harness, sources and current runtime."""
    if not isinstance(result, dict):
        return False
    try:
        current_sha = runtime_environment_fingerprint(runtime_environment)
        saved_sha = runtime_environment_fingerprint(
            result["runtime_environment"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        result_is_reusable(
            result, manifest_sha=manifest_sha, fingerprint=fingerprint
        )
        and result.get("production_source_sha256") == source_sha
        and result.get("runtime_environment_sha256") == saved_sha == current_sha
    )


def run_with_source_fingerprint(args, runner, *, source_root=SOURCE_ROOT):
    """Stamp only fresh evidence from stable production and runtime inputs."""

    def option(name):
        return args[args.index(name) + 1]

    out = Path(option("--out"))
    if args[0] == "case":
        name = case_tag(
            option("--scenario"), int(option("--seed")), option("--recipe")
        )
    elif args[0] == "representative":
        name = f"representative_shank{option('--shank')}"
    else:
        raise ValueError(f"Unsupported benchmark command {args[0]!r}.")
    result_path = out / f"{name}.json"
    source_sha = production_source_fingerprint(source_root)
    runtime_environment = benchmark_runtime_environment(args)
    runtime_sha = runtime_environment_fingerprint(runtime_environment)
    # The runner must create fresh metrics. A failed or incomplete invocation
    # must never attach a new receipt to an earlier run's cache entry.
    result_path.unlink(missing_ok=True)
    runner(args)
    if production_source_fingerprint(source_root) != source_sha:
        raise RuntimeError(
            "Production source changed during the motion benchmark."
        )
    if (
        runtime_environment_fingerprint(benchmark_runtime_environment(args))
        != runtime_sha
    ):
        raise RuntimeError(
            "Runtime environment changed during the motion benchmark."
        )
    result = json.loads(result_path.read_text())
    result["production_source_sha256"] = source_sha
    result["runtime_environment"] = runtime_environment
    result["runtime_environment_sha256"] = runtime_sha
    result_path.write_text(json.dumps(result, indent=1))


def main():
    from tests.spikesorting.v2._motion_acceptance_run import main as runner

    run_with_source_fingerprint(sys.argv[1:], runner)


if __name__ == "__main__":
    main()
