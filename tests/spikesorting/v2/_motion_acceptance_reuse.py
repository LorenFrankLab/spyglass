"""Reuse bookkeeping outside the scientific motion benchmark's pinned harness.

The pinned runner and gates remain unchanged. This wrapper stamps new results
with the current production Python source bytes, including uncommitted edits.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tests.spikesorting.v2._motion_acceptance import (
    case_tag,
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


def source_result_is_reusable(result, *, manifest_sha, fingerprint, source_sha):
    """Require both the pinned harness identity and current production bytes."""
    return (
        result_is_reusable(
            result, manifest_sha=manifest_sha, fingerprint=fingerprint
        )
        and result.get("production_source_sha256") == source_sha
    )


def run_with_source_fingerprint(args, runner, *, source_root=SOURCE_ROOT):
    """Run the pinned entrypoint and stamp only evidence from stable sources."""

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
    source_sha = production_source_fingerprint(source_root)
    runner(args)
    if production_source_fingerprint(source_root) != source_sha:
        raise RuntimeError(
            "Production source changed during the motion benchmark."
        )
    result_path = out / f"{name}.json"
    result = json.loads(result_path.read_text())
    result["production_source_sha256"] = source_sha
    result_path.write_text(json.dumps(result, indent=1))


def main():
    from tests.spikesorting.v2._motion_acceptance_run import main as runner

    run_with_source_fingerprint(sys.argv[1:], runner)


if __name__ == "__main__":
    main()
