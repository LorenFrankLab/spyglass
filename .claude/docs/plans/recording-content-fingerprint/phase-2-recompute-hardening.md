# Phase 2 — v1-parity operational hardening for recording recompute

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Additive operational maturity on top of the Phase-1 fix, porting the proven
patterns from v1's `RecordingRecompute` (design §10). None of this is required
for the round-trip correctness Phase 1 delivers; it makes the recompute usable
at archive scale and avoids wasting work on known-incompatible attempts.

**Inputs to read first:**

- [../spikesorting-v2/recording-content-fingerprint-design.md](../spikesorting-v2/recording-content-fingerprint-design.md)
  §10 — the adopt/reject list (and *why* object-id preservation and per-attempt
  rounding are rejected).
- [v1/recompute.py:93-116](../../../../src/spyglass/spikesorting/v1/recompute.py) —
  `this_env` / `_has_matching_env` (the env-compat gate pattern).
- [v1/recompute.py:310](../../../../src/spyglass/spikesorting/v1/recompute.py) —
  `attempt_all(limit, force_attempt)` with `dj.condition.Top(limit,
  order_by="RAND()")`.
- [v1/recompute.py:365](../../../../src/spyglass/spikesorting/v1/recompute.py) —
  `_check_xfail(skip_probe, skip_pynwb_api, skip_nwb_spec)`.
- [recompute.py:261-300](../../../../src/spyglass/spikesorting/v2/recompute.py) —
  v2 `RecordingArtifactRecomputeSelection.attempt_all` (post-Phase-1, with
  `rounding` already removed); `RecordingArtifactVersions` already inventories
  `nwb_deps`.

## Tasks

### 1. Env-compatibility gate on `attempt_all`

Add a `this_env`-style filter so `RecordingArtifactRecomputeSelection.attempt_all`
plans attempts only for artifacts whose inventoried `nwb_deps`
(PyNWB/HDMF/namespace versions in `RecordingArtifactVersions`) are compatible
with the current environment — mirroring v1's
[`_has_matching_env`](../../../../src/spyglass/spikesorting/v1/recompute.py).
Add `force_attempt: bool = False` to override the gate for deliberate audits.
Default `force_attempt=False`.

### 2. `limit` throttling on `attempt_all`

Add `limit: int | None = None`; when set, restrict the source to a random subset
(`source & dj.condition.Top(limit=limit, order_by="RAND()")`) before inserting —
for large retrospective audits. Not correctness-critical.

### 3. Narrow known-xfail handling

Port a `_check_xfail`-style helper that marks structural impossibilities into the
existing `xfail_reason` column **before** a recompute attempt is scheduled:
missing probe info, PyNWB API / NWB-spec incompatibility. Keep it narrow — this
is for known structural impossibilities, **not** a broad skip mechanism (design
§10). Reuse v1's flag shape (`skip_probe`, `skip_pynwb_api`, `skip_nwb_spec`).

### 4. At-creation environment provenance (logging only)

Have `Recording` creation version-log the artifact environment into
`RecordingArtifactVersions` (mirrors
[v1/recording.py:264](../../../../src/spyglass/spikesorting/v1/recording.py)).
**Logging only — never deletion authority.** Deletion still requires a fresh
recompute match against `Recording.content_hash` (Phase 1, design §3.6 Medium 1).
See overview Open Question 1 — if lazy population is judged sufficient, this task
may be dropped without affecting correctness.

### 5. Docs

- `CHANGELOG.md` — note the `attempt_all` env-compat gate + `limit` (operator-
  facing behavior change: `attempt_all` now skips incompatible-env artifacts by
  default; pass `force_attempt=True` for the old behavior).
- Docstring `attempt_all` with the gate / `limit` / `force_attempt` semantics.

## Deliberately not in this phase

- **v1's object-id preservation** — rejected (design §10); v2 uses the semantic
  fingerprint, no object-id rewriting.
- **Per-attempt rounding** — already removed in Phase 1; the fixed constants are
  the contract.
- Any change to the Phase-1 fingerprint / reconciliation / delete path.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_attempt_all_skips_incompatible_env` *(slow)* | With an inventoried `nwb_deps` incompatible with the current env, `attempt_all()` inserts no selection; `attempt_all(force_attempt=True)` does. |
| `test_attempt_all_limit` *(slow)* | `attempt_all(limit=k)` inserts at most `k` selections from a larger eligible set. |
| `test_check_xfail_marks_structural` *(medium)* | A recording with a known structural impossibility (e.g. missing probe info) is marked with `xfail_reason` and not scheduled as a normal attempt. |
| `test_creation_logs_provenance` *(slow, if Task 4 kept)* | Recording creation populates `RecordingArtifactVersions` env provenance; provenance alone does **not** authorize `delete_files`. |

## Fixtures

Reuse Phase-1 fixtures. The env-compat test fabricates an incompatible
`nwb_deps` inventory row (synthesized, no new large fixture); the xfail test uses
a recording fixture stripped of probe info (synthesized in `conftest.py`).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The env-compat gate defaults to filtering (not attempting); `force_attempt`
  override works and is documented.
- Known-xfail stays narrow (structural impossibilities only) — not a general
  skip.
- At-creation provenance is logging only and never gates deletion.
- No regression to the Phase-1 round-trip (re-run the Phase-1 delete→rebuild
  integration test).
- Tests aren't trivial; shared setup in fixtures (`testing-anti-patterns`).
- No docstring / test / module name references this plan or "Phase 2".
- CHANGELOG updated for the operator-facing `attempt_all` change.
