# Pre-refactor baseline bundle (local-only)

This directory holds reference artifacts captured by running the v2 spike-sorting
pipeline against the 60s MEArec polymer fixture on unmodified pre-refactor code.
The validation slice compares the post-refactor pipeline output to these
artifacts on deterministic paths (the `Recording` trace + timestamps and
`clusterless_thresholder` spike samples).

> **CI gating reality.** Only the lightweight `MANIFEST.json` and
> `stage_metrics.json` are committed; the heavy `.npz` / `.pkl` payloads are
> gitignored. Default CI therefore **skips** every test that depends on the
> `phase1_baseline_artifacts` fixture. This is a **local / manual** regression
> gate: a developer regenerates the bundle on pre-refactor tip, runs the
> dependent validation tests locally to confirm bit-equivalence, then commits
> the refactor. CI re-runs the `phase1_baseline_consumer` smoke checks
> (which load the small metadata files) but cannot exercise the full
> trace/spike-time comparison without the heavy artifacts.

## What's committed vs local

- `MANIFEST.json` — captured SpikeInterface / NumPy / pynwb versions + the
  git SHA at regen time. Small; kept under git so reviewers can sanity-check
  the baseline environment without running the regen.
- `stage_metrics.json` — wall-clock + peak RSS per stage. Informational only.
  Small; kept under git.
- `recording_artifacts.npz`, `sorting_spike_times.pkl`,
  `curation_spike_times.pkl` — the heavy artifacts (~480 MB combined for the
  60s polymer fixture). **Gitignored**; regenerate locally per the workflow
  below.

## Regenerating the bundle

The bundle must be (re)generated on **unmodified Phase 1 code** so it captures
the behavior the Phase 1b refactor is preserving:

    1. git stash any in-progress refactor edits.
    2. git rev-parse HEAD and confirm it matches the baseline-source SHA in
       MANIFEST.json. If you are creating a new baseline, this is the SHA the
       regen records.
    3. rm -rf tests/spikesorting/v2/_fixtures/phase1_baseline/{*.npz,*.pkl}
       (leave MANIFEST.json / stage_metrics.json in place; they will be
       overwritten by the regen).
    4. pytest tests/spikesorting/v2/test_phase1_baseline_regen.py -q
    5. git status — confirm only MANIFEST.json / stage_metrics.json
       (or the heavy files outside of git) updated; commit the small ones.
    6. git stash pop to restore in-progress edits.

If `MANIFEST.json`'s recorded SpikeInterface / NumPy / pynwb versions differ
from the active environment, the `phase1_baseline_artifacts` fixture skips
the dependent validation tests with a clear pointer — bit-equivalence loses
its meaning if the dependency surface changes.
