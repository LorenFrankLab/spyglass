"""Regenerate ``legacy_schemas.json`` from the current source.

Reads every v0/v1 DataJoint ``definition`` block (master tables and Part
tables) and writes a JSON snapshot. The snapshot is the baseline
``test_no_legacy_schema_changes`` compares against. Run only when a v0/v1
``definition`` legitimately needs to change; commit the result in the same
change that edits the source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / "src"
_BASELINE = Path(__file__).with_name("legacy_schemas.json")

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

_PATTERN = re.compile(
    r"^\s*class\s+(\w+).*?\n\s+definition\s*=\s*\"\"\"\n(.*?)\"\"\"",
    re.DOTALL | re.MULTILINE,
)


def main() -> int:
    snapshot: dict[str, dict[str, str]] = {}
    for tier, files in _LEGACY_TABLE_FILES.items():
        snapshot[tier] = {}
        for relative in files:
            source = (_SRC / relative).read_text()
            for name, body in _PATTERN.findall(source):
                snapshot[tier][name] = body
    _BASELINE.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(
        f"wrote {_BASELINE.name}: "
        f"v0={len(snapshot['v0'])} classes, v1={len(snapshot['v1'])} classes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
