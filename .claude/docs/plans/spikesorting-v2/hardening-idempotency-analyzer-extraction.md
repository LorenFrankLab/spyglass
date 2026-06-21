# Spike-Sorting v2 — Hardening Plan: Selection Idempotency, Analyzer Paths, Compute Extraction

**Author:** Claude (plan-evaluation session); revised by Codex
**Date:** 2026-06-12 (proposed); **as-built update:** 2026-06-12
**Status:** IMPLEMENTED / AS-BUILT. Phases 0, A, B, and C are all landed on
branch `spikesorting-v2` (this branch is pre-production). The full v2 suite is
green. Remaining work is test-hardening + docs, not new architecture — see
**Implementation status (as-built)** immediately below. The original
forward-looking plan text is preserved unchanged after that section for
provenance; where it says "do X" / "steps", read it as "X was done".
**Scope:** Three independent post-audit hardening tracks on top of completed Phases 0–1
(+ partial 1b) and the [AUDIT-REPORT.md](AUDIT-REPORT.md) remediations. This is **not**
part of the numbered phase roadmap in [PLAN.md](PLAN.md); it is pre-release architecture
hardening before v2 becomes public API.
**Relationship:** Complements [ARCHITECTURE-EVALUATION.md](ARCHITECTURE-EVALUATION.md)
(layering/cohesion) and continues the "thin DataJoint shell over pure/IO services" direction
established by the artifact-kernel extraction (`_artifact_compute.py`, commit `32ba091a`).

---

## Implementation status (as-built)

All four phases are **implemented and landed** on `spikesorting-v2`. Summary:

| Phase | Status | As-built notes |
|---|---|---|
| **0 — docstring cleanup** | ✅ Done | Module docstrings corrected (no more "stub raises NotImplementedError" overclaim for implemented methods). |
| **A — deterministic selection identity** | ✅ Done | `_selection_identity.py` (UUID5 `deterministic_id` / `assert_supplied_id_matches`); `RecordingSelection` / `ArtifactDetectionSelection` / `SortingSelection` `insert_selection` rewritten to content-addressed IDs with duplicate-PK race recovery + source-part collision checks. |
| **B — analyzer cache contract** | ✅ Done | `_analyzer_cache.py` (`analyzer_path` / root resolution); `Sorting.analyzer_folder` is transient state (threaded `make_compute → make_insert`), not a persisted absolute path; orphan scan computes the canonical path. |
| **C — service extraction** | ✅ Done | DB-free service modules: `_artifact_compute`, `_artifact_intervals`, `_curation_transforms`, `_units_nwb` (incl. `build_lazy_merged_sorting`), `_sorting_compute`, `_recording_restriction`, `_recording_geometry`, `_recording_preprocessing`, `_recording_nwb`, `_signal_math`, `_enums`, `_analyzer_cache`, `_selection_identity`. Table classes are thin orchestrators; delegators kept only where a test / external caller pins the surface. |

**Deliberately NOT done (rejected, do not re-attempt):** the broad
`_curation_queries.py` extraction (resolve_restriction / get_sort_metadata /
get_merge_groups / get_merged_sorting / get_matchable_unit_ids /
get_unit_brain_regions / get_sort_group_info / _upstream_recording_row /
has_unapplied_proposed_merges). Those are DataJoint relation API, not DB-free
service logic — all are DB-bound, 8 are externally/test-pinned, and a module
would still depend on `CurationV2` while forcing compatibility delegators
(churn without payoff). The "single owner of join topology" is already true
because they are centralized methods. The better move (taken instead) was to
extract only the genuine compute core, `build_lazy_merged_sorting`, and leave
the table methods in place. Decision rule going forward: extract only when it
removes hidden DB reads, creates directly testable pure/IO logic, or removes
real duplication — not for LOC reduction.

**Remaining TODO (test-hardening / docs, not architecture):**

- Direct service-level tests for the newest helpers (`build_lazy_merged_sorting`,
  the `_artifact_intervals` pure functions) + cold-import checks asserting the
  DB-free modules open no DB connection at import.
- Keep this doc in sync as further service tests land.

The forward-looking proposal text below is preserved for provenance.

---

## How to read this

This started as a 3-PR proposal. The user does **not** want PRs; this is a phased,
suite-gated, local-commit plan. Every "current state" assertion below was checked against
the code — **the original proposal's line numbers were ~50 lines stale and have been
corrected here**. The substantive claims all held up.

Verdicts up front:

| Track | Value | Risk | Status |
|---|---|---|---|
| **Phase 0 — docstring cleanup** | Low (clarity) | None | Ready; ~5-min commit |
| **Phase A — deterministic selection identity** | High (schema invariant) | Low–moderate | Ready; allow v2 table reset |
| **Phase B — analyzer cache contract** | High (removes path drift class) | Moderate | Ready; schema cleanup preferred |
| **Phase C — service extraction** | High (stabilizes v2 architecture) | Diffuse | Do incrementally after A/B |

Pre-production stance:

- Prefer one clean invariant over compatibility shims for random-UUID rows or stale analyzer
  paths.
- Keep the useful single-column selection handles, but make them content-addressed and
  deterministic.
- Treat `SortingAnalyzer` folders as regeneratable cache, not persisted database artifacts.
- Use table resets/migrations now to avoid carrying awkward transitional logic into v2's public
  API.

Executor defaults:

- Proceed without asking for more design approval unless an implementation fact below is false.
- Use deterministic UUID primary keys as the Phase A enforcement mechanism. Do **not** add
  `identity_hash` / `unique index` in the first implementation pass; keep that as a later audit
  enhancement only if deterministic PKs prove insufficient.
- Use `dj.config["custom"]["spikesorting_v2_analyzer_dir"]` as the optional analyzer-root
  override, falling back to `Path(temp_dir) / "spikesorting_v2" / "analyzers"`.
- Remove `Sorting.analyzer_folder` from the v2 schema rather than preserving a dual-path
  compatibility mode.
- Keep Phase C extractions behavior-preserving and one service at a time.

---

## Verified current state (the facts this plan rests on)

### Selection idempotency (Phase A)

| Table | PK + `uuid4()` site | Existing find-existing dedup | Transaction |
|---|---|---|---|
| `RecordingSelection` | `recording_id`, [recording.py:753](../../../../src/spyglass/spikesorting/v2/recording.py#L753) | [recording.py:733-743](../../../../src/spyglass/spikesorting/v2/recording.py#L733), matches `nwb_file_name, sort_group_id, interval_list_name, preprocessing_params_name, team_name` | **none** around check+insert |
| `ArtifactDetectionSelection` | `artifact_detection_id`, [artifact.py:564](../../../../src/spyglass/spikesorting/v2/artifact.py#L564) | [artifact.py:519-549](../../../../src/spyglass/spikesorting/v2/artifact.py#L519), matches `artifact_detection_params_name` + (`recording_id` \| `shared_artifact_group_name`) via source-part join | insert wrapped ([artifact.py:570-572](../../../../src/spyglass/spikesorting/v2/artifact.py#L570)); **check is before the txn** |
| `SortingSelection` | `sorting_id`, [sorting.py:675](../../../../src/spyglass/spikesorting/v2/sorting.py#L675) | [sorting.py:613-653](../../../../src/spyglass/spikesorting/v2/sorting.py#L613), matches `sorter, sorter_params_name, recording_id` + optional `artifact_detection_id` (normalized to `uuid.UUID`) | insert wrapped ([sorting.py:681-690](../../../../src/spyglass/spikesorting/v2/sorting.py#L681)); **check is before the txn** |

- The proposal's logical-identity field lists **match the code exactly** (`source_kind` is not a
  stored column; it is implicit in which source part exists, already handled by the joins).
- There is a **real check-then-insert TOCTOU race**: the dedup query runs *before* the insert
  (and outside any transaction for `RecordingSelection`). Two concurrent callers for the same
  logical selection both see "no existing row" and both insert with *different* random UUIDs →
  two logical-duplicate master rows.
- No `try/except IntegrityError` exists today; dedup is assumed sufficient.
- These helpers are called directly (via `run_pipeline()` in
  [pipeline.py](../../../../src/spyglass/spikesorting/v2/pipeline.py)) and could be called
  concurrently (e.g. HPC job arrays each "ensure the selection exists" — the same usage class
  that motivated the DB-free artifact kernels).
- No existing canonicalize/`uuid5`/hash helper in v2. One idempotency test exists:
  `test_insert_selection_dedup_accepts_str_artifact_detection_id`
  ([test_merge_id_artifact_resolution.py:196](../../../../tests/spikesorting/v2/test_merge_id_artifact_resolution.py#L196)) —
  it documents that a str-vs-`UUID` `artifact_detection_id` mismatch already caused a duplicate-sort bug.
  **That is the canonicalization footgun Phase A must respect.**

### Analyzer paths (Phase B)

- `Sorting` stores `analyzer_folder` as an **absolute path** (not-null `varchar(255)`),
  [sorting.py:859-867](../../../../src/spyglass/spikesorting/v2/sorting.py#L859).
- `_analyzer_path(key)` computes `{temp_dir}/spikesorting_v2/analyzers/{sorting_id}.analyzer`,
  reading `temp_dir` **at call time**,
  [utils.py:660-684](../../../../src/spyglass/spikesorting/v2/utils.py#L660).
- **6 of 7 paths recompute from `temp_dir` and ignore the stored column:**
  - `get_analyzer` recomputes ([sorting.py:1386](../../../../src/spyglass/spikesorting/v2/sorting.py#L1386)).
  - `_rebuild_analyzer_folder` recomputes ([sorting.py:1450](../../../../src/spyglass/spikesorting/v2/sorting.py#L1450)).
  - `_build_analyzer` recomputes and has **no** `folder=` arg ([sorting.py:2245](../../../../src/spyglass/spikesorting/v2/sorting.py#L2245)).
  - `delete` recomputes ([sorting.py:1491](../../../../src/spyglass/spikesorting/v2/sorting.py#L1491)).
  - **Only** `find_orphaned_analyzer_folders` reads the stored column
    ([sorting.py:1559-1567](../../../../src/spyglass/spikesorting/v2/sorting.py#L1559)).
- `temp_dir` **can change between runs** (configured value; test mode uses a per-session temp
  base). A change orphans previously-stored paths and makes `get_analyzer` rebuild a duplicate
  in the new root while the stored column points at the old one.

### Stale docstrings (Phase 0)

- [artifact.py:28](../../../../src/spyglass/spikesorting/v2/artifact.py#L28): calls
  `insert_selection`, `make`, `get_artifact_removed_intervals`, `delete`,
  `SharedArtifactGroup.insert_group` "forward-declared stubs that raise NotImplementedError" —
  **all implemented now.**
- [sorting.py:14](../../../../src/spyglass/spikesorting/v2/sorting.py#L14): calls `make` /
  `insert_selection` / accessors "forward-declared stubs" — **implemented now.**
- **Do not blanket-delete:** `SharedArtifactGroup.insert_group`
  ([session_group.py:88](../../../../src/spyglass/spikesorting/v2/session_group.py#L88)) and
  the concat path are *genuine* remaining `NotImplementedError` stubs. The fix corrects the
  overclaim about *implemented* methods only.

### File sizes (Phase C motivation)

`sorting.py` 2558 lines · `recording.py` 2016 · `curation.py` 1776 · `artifact.py` 1384.

---

## Phase 0 — Docstring cleanup (free, do first)

**Goal:** stop sending readers on an archaeological dig after methods that are implemented.

**Change:** rewrite the two module docstrings ([artifact.py:24-29](../../../../src/spyglass/spikesorting/v2/artifact.py#L24),
[sorting.py:12-16](../../../../src/spyglass/spikesorting/v2/sorting.py#L12)) to describe the
methods as implemented, **preserving** the accurate note that the concatenated-recording /
`SharedArtifactGroup.insert_group` paths remain gated `NotImplementedError` stubs.

**Risk:** none (comments only). **Tests:** none needed; black-clean.

---

## Phase A — Deterministic selection identity

**Goal:** make every logical selection resolve to one stable, content-addressed UUID under
serial, repeated, concurrent, and worker-retry insertion. Because v2 is pre-production, do
**not** carry legacy random-UUID compatibility in the steady-state code.

**Chosen shape:** keep the compact single-column handles (`recording_id`, `artifact_detection_id`,
`sorting_id`) because downstream FKs, interval names, NWB object names, and user-facing keys are
much simpler with them. Make those handles deterministic from the canonical logical identity.

Natural composite PKs would be the most literal database model, but the blast radius is high:
every downstream FK grows, and the alternate-source tables (`RecordingSource` vs
`SharedGroupSource`, artifact/no-artifact sorting) become awkward. Content-addressed
UUIDs give the pipeline the important invariant without making every downstream table carry the
whole input tuple.

**Mechanism:** derive each selection PK as
`uuid.uuid5(V2_NAMESPACE, kind + canonical_json(logical_identity))`. The primary-key uniqueness
constraint then becomes the concurrency guard; the duplicate-key catch only makes the loser of
the race return the existing row cleanly.

**Schema posture:**
- Existing v2 rows may be dropped/recreated or migrated in one controlled reset. Do not preserve
  random UUIDs indefinitely.
- Selection helpers should ignore caller-supplied UUIDs by default; if a caller supplies one,
  verify it equals the deterministic ID and raise if not.
- Do not add an `identity_hash` column in the first pass. If later added, it is for auditability,
  not for correctness; the deterministic UUID PK is the core invariant.
- Direct raw `insert1` into selection masters should be treated as unsupported. The **helper is
  the validation boundary** — it alone holds the full logical payload, including the source-part
  contents (`recording_id` / `artifact_detection_id`) that the *master row does not carry*, so it is the
  only place that can compute and verify the deterministic ID for the part-bearing tables
  (`ArtifactDetectionSelection`, `SortingSelection`). An `insert`/`insert1` override, if added, should
  therefore **reject direct master inserts** rather than attempt to validate the ID from the
  master row alone (which is structurally impossible for the part-bearing tables). (`SpyglassMixin`
  does not currently override `insert`/`insert1`, so there is no existing override to compose
  with.) For source-part selections, the helper remains the only supported way to create
  master+part rows atomically.

**Steps:**
1. Add `src/spyglass/spikesorting/v2/_selection_identity.py` as a **DB-free** helper module:
   - `canonical_identity(payload: dict) -> str`: stable JSON (`sort_keys=True`, fixed
     separators) with explicit normalization.
   - `deterministic_id(kind: str, payload: dict) -> uuid.UUID`: UUIDv5 over a fixed v2 namespace.
   - Keep the helper small; do not import DataJoint or SpikeInterface here.
2. Canonicalize every type that has already proven dangerous:
   - `UUID` and UUID-ish strings → lowercase canonical UUID strings.
   - `sort_group_id` and integer-like IDs → `int`.
   - "no artifact" → one representation only; it must not alias with any real `artifact_detection_id`.
   - source kind is explicit in the payload, even when it is implicit in today's part-table
     topology.
3. Replace `uuid.uuid4()` in `RecordingSelection`, `ArtifactDetectionSelection`, and `SortingSelection`
   with deterministic IDs built from these payloads.
4. Simplify the find-existing flow:
   - compute deterministic ID first;
   - look up by PK/hash;
   - if found, verify all logical fields and required source/artifact parts match, then return;
   - if absent, insert master+parts in one transaction;
   - on duplicate PK/hash, refetch, verify, and return.
5. For `ArtifactDetectionSelection` and `SortingSelection`, collision verification must check source parts.
   If the deterministic master exists but the expected part row is missing, raise a clear
   integrity error; do not manufacture the missing part after the fact.
6. Add a one-time audit/reset helper for test and development schemas:
   - report non-deterministic selection IDs;
   - report logical duplicates;
   - optionally rebuild v2 selection/computed rows after a schema reset.

**Risk:** low–moderate. The canonicalization is the important part. If the payload normalization
is sloppy, the old `str` vs `UUID` `artifact_detection_id` class of bug comes back under a prettier name.

**Tests:**
- deterministic IDs are stable across `uuid.UUID` vs `str` inputs;
- helper ignores or rejects caller-supplied random IDs;
- repeated helper calls return the same PK;
- duplicate-PK collision path refetches and verifies the stored logical identity;
- true same-selection concurrency produces exactly one master row where practical;
- artifact-backed and artifact-free `SortingSelection` identities do not alias;
- source kind is part of the identity for artifact and sorting selections;
- existing `test_insert_selection_dedup_accepts_str_artifact_detection_id` still passes.

---

## Phase B — Analyzer cache contract

**Goal:** remove the absolute-path drift class entirely. `SortingAnalyzer` folders are large,
regeneratable cache; the database should not persist an absolute scratch path as if it were a
canonical artifact.

**Chosen shape:** remove `Sorting.analyzer_folder` from the durable schema, or replace it with a
short deterministic `analyzer_cache_key` only if operators need an inspectable label. The actual
filesystem path is resolved at runtime from:

```text
analyzer_root / f"{sorting_id}.analyzer"
```

where `analyzer_root` is a dedicated v2 setting, not an incidental read of `temp_dir` scattered
through the table class. Recommended resolution:

1. `dj.config["custom"]["spikesorting_v2_analyzer_dir"]` when set;
2. otherwise `Path(temp_dir) / "spikesorting_v2" / "analyzers"`.

Sites that want persistent analyzer caches can configure the root to shared storage. Sites that
want scratch semantics can leave it under `temp_dir`. Changing the root becomes an explicit cache
relocation choice: old folders are simply cache misses and can be cleaned by the operator.

**Steps:**
- Add a DB-free `_analyzer_cache.py` or move `_analyzer_path` into a clearer helper with:
  - `analyzer_cache_root() -> Path`;
  - `analyzer_path(sorting_id) -> Path`;
  - `remove_analyzer_cache(sorting_id, *, missing_ok=True)`.
- Update `Sorting.definition` to stop storing absolute `analyzer_folder`. Because this is
  pre-production, prefer a schema reset over transitional dual-read behavior.
- Update `Sorting.make_insert` to insert canonical NWB metadata and unit rows only (no
  `analyzer_folder` column); analyzer folder creation remains part of compute, but the path is
  not persisted as row state. **Blast radius beyond the table `definition`:** drop `analyzer_folder`
  from the `Sorting.definition` only. KEEP `analyzer_folder` on the `SortingComputed` NamedTuple as
  TRANSIENT in-memory state and thread it `make_compute → make_insert → _populate_unit_part`, so the
  folder `_build_analyzer` wrote is the exact one loaded/cleaned up (a recomputed path could be
  diverted by a mid-populate config change). The `make_compute → make_insert` positional contract
  therefore keeps the field (tri-part dispatch passes NamedTuple fields positionally). Update any
  test asserting the column is populated, plus the raw-insert/orphan fixtures and the schema-heading
  test (which now asserts the column is ABSENT).
  *(Decided during implementation, post-review: the original plan said to drop `analyzer_folder`
  from `SortingComputed`; that was changed to "keep it transient, drop only the DB column" so the
  build/load/cleanup all reference one resolved path.)*
- Update `get_analyzer` to resolve the canonical path, load it if present, and rebuild it into
  that same canonical path if missing.
- Update `_build_analyzer` and `_rebuild_analyzer_folder` to accept an explicit folder from the
  helper so path policy lives in one place.
- Update `delete` to remove the canonical analyzer cache for each deleted `sorting_id`.
- Simplify `find_orphaned_analyzer_folders` to scan the currently configured analyzer root. Do
  not try to infer every historical root from database rows once paths are no longer stored.
  DB-side orphan detection does **not** disappear — it becomes "for each `Sorting` row, compute
  the canonical `analyzer_path(sorting_id)` and check existence," rather than reading a stored
  column.

**Risk:** moderate, mostly because it is a schema change and touches populate/delete/rebuild
paths. Architecturally it is cleaner than path self-healing: the row no longer has a value that
can disagree with the accessor. **Contract to state explicitly:** with the default root under
`temp_dir`, analyzers are ephemeral by default, so the cache-miss path (`get_analyzer` rebuilds
into the canonical path) must always be recoverable — and it is, because a valid `Sorting` row
keeps its FK-guaranteed upstream `Recording`/NWB, so the analyzer can always be regenerated.

**Tests:**
- analyzer path helper honors `spikesorting_v2_analyzer_dir` and falls back to `temp_dir`;
- `Sorting.make()` creates the analyzer in the helper-resolved path but does not persist an
  absolute folder;
- `get_analyzer` loads from the canonical path when present;
- missing analyzer cache rebuilds into the canonical path;
- changing the configured root causes a cache miss + rebuild, not a stale-row inconsistency;
- delete removes the canonical cache folder;
- orphan audit reports folders under the current analyzer root;
- zero-unit carve-out remains unchanged.

---

## Phase C — Service extraction (pre-release architecture pass)

**Goal:** shrink the large table modules by moving logic behind the table classes into
dependency-light, hermetically-testable service modules — the same pattern as
`_artifact_compute.py`. Table classes become orchestration shells: fetch → call pure/IO service
→ insert.

Because v2 is not in production, this should happen before adding more user-facing workflows.
Keep it incremental, but treat it as part of making the pipeline maintainable rather than as
nice-to-have cleanup.

**Rules (non-negotiable):**
- Each extraction is a **pure move** in its **own commit** — never bundled with Phase A/B
  behavior changes (clean review + bisect).
- The v2 integration tests are the behavioral oracle; the suite must be green **between** each
  extraction.
- Prefer DB-free service modules so nothing connects at import (keeps spawn-worker and
  cold-import paths clean — see [the njobs lesson](AUDIT-REPORT.md)).

**Suggested order (safest first):**
1. `_selection_identity.py` — already created in Phase A; promote to shared infra.
2. `_analyzer_cache.py` — already created in Phase B; keep path policy out of `Sorting`.
3. `_curation_transforms.py` — label-validation normalization, merge-group validation,
   `kept_unit_to_contributors`, curated unit-row construction. Mostly pure → safest.
4. `_units_nwb.py` — read absolute unit spike times; write sorting-units NWB; write curated-units
   NWB.
5. `_sorting_compute.py` — artifact-mask application, sorter dispatch, analyzer build, zero-unit
   analyzer behavior. Sequence this after Phase B so the cache contract is settled first.
6. the split `_recording_*` service modules — preprocessing execution, NWB write/rebuild validation, and
   cache-hash verification. This is high value but should come after the narrower extractions
   establish the pattern.

**Acceptance:** public methods behave identically; existing docs examples still work; integration
tests unchanged and green.

---

## Sequencing & open decisions

**Recommended order:** Phase 0 (now, free) → Phase A (selection identity/schema reset) → Phase B
(analyzer cache contract/schema cleanup) → Phase C (service extraction, one module per commit).

**Execution checkpoints:**
1. Commit Phase 0 alone. No tests required beyond import/format sanity.
2. Commit Phase A with focused identity tests first, then run the full v2 suite.
   **Env note:** on a local Colima/Docker machine run
   `python -m pytest tests/spikesorting/v2/ --no-dlc -q` — the conftest manages the MySQL
   container. The `--no-docker` form is the **CI** path and expects an *external* MySQL on
   `:3308`; using it locally targets a dead port (see [spikesorting-v2 local-test-env memory]).
3. Commit Phase B with analyzer-cache tests, then run the same v2 suite command.
4. Commit each Phase C extraction separately. After each extraction, run at least the affected
   focused test file(s); before handing off, run the full v2 pytest command.

**Implementation defaults, not open questions:**
1. **Duplicate exception handling:** catch only the **duplicate-primary-key** case around the
   deterministic master/part insert path. Verified: `dj.errors.DuplicateError` is **not** a
   subclass of `dj.errors.IntegrityError` (separate branches off `DataJointError`), so catch
   `DuplicateError` and, for a true DB-level race, the duplicate-key `IntegrityError` (MySQL
   errno 1062). **Do not** blanket-catch `IntegrityError`: FK / missing-source-part violations
   are *also* `IntegrityError`, and Phase A step 5 requires those to raise. Match on the
   duplicate-key signature (errno 1062 / `DuplicateError`), not the base class.
2. **Analyzer root default:** use the dedicated config key with `temp_dir` fallback described
   above.
3. **Extraction boundary:** stop each extraction when the table class becomes a clear
   orchestrator. Do not introduce a framework or registry abstraction unless repeated service
   modules actually need it.

**Stop-and-ask conditions:**
- DataJoint cannot represent the revised `Sorting` schema without a larger migration than a v2
  reset/redeclare.
- Deterministic UUIDs cannot be round-tripped through the existing DataJoint `uuid` columns.
- Removing `analyzer_folder` would break a documented public v2 API outside this branch.
- The full v2 pytest command cannot run because fixtures/environment are unavailable; in that
  case run focused unit tests and report the blocked full-suite gate.
- The duplicate-key exception observed at runtime differs from the assumed `DuplicateError` /
  errno-1062 `IntegrityError` signature — confirm the actual exception before relying on the
  catch (see implementation default #1).

**Not in scope:** v0/v1 parity for any of these (legacy); the artifact-kernel DB-free split is
already done (commit `32ba091a`).
