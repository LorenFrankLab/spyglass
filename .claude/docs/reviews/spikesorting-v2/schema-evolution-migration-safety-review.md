# Spike Sorting V2 Schema Evolution and Migration Safety Review

Date: 2026-06-25

Scope: schema-versioned parameter rows, default-catalog evolution, deterministic
selection identities, recompute/version tables, old-v2 database behavior,
published migration guidance, and tests that should catch schema drift. This is
a different lens from scientific reproducibility, DataJoint/concurrency, test
coverage, NWB portability, operational recovery, performance/memory scaling, and
user/API ergonomics.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 has several good migration-safety building blocks: Pydantic-validated lookup
rows, content fingerprints that exclude row names, duplicate-content guards for
most parameter tables, deterministic selection IDs, and `ImmutableParamsLookup`
blocking same-name parameter mutation. Those are the right primitives for a
pipeline where row names become stable provenance.

The main remaining risk is that there is not yet a first-class "old v2 database"
contract. Some insert paths can still store mismatched schema-version metadata
when callers omit the outer version column; shipped defaults are idempotent but
do not audit same-name stale rows; old/interim row names can block default
seeding with an unfriendly duplicate-content error; and the table definitions are
labeled final-shape while only v0/v1 definitions are snapshotted.

The recording `content_hash` work is directionally good for recompute and
rebuild safety, but the docs still contain `cache_hash` guidance from the older
design. That should be cleaned up before users rely on recompute/deletion
instructions after an upgrade.

## Findings

### 1. High: omitted outer schema versions can be stored as current-version rows even when the inner blob says otherwise

`validate_lookup_rows()` validates the Pydantic `params` blob, then calls
`_assert_schema_version_matches()` (`src/spyglass/spikesorting/v2/_lookup_validation.py:96-139`).
That drift check returns immediately if `params_schema_version` is not present
in the incoming row (`src/spyglass/spikesorting/v2/_lookup_validation.py:83-85`).

For most lookup tables, the DataJoint definition supplies a current-version
default:

- `PreprocessingParameters`: `params_schema_version={PREPROCESSING_SCHEMA_VERSION}`
  (`src/spyglass/spikesorting/v2/recording.py:690-696`).
- `ArtifactDetectionParameters`: `params_schema_version={ARTIFACT_DETECTION_SCHEMA_VERSION}`
  (`src/spyglass/spikesorting/v2/artifact.py:143-149`).
- `AnalyzerWaveformParameters`: `params_schema_version={ANALYZER_WAVEFORM_SCHEMA_VERSION}`
  (`src/spyglass/spikesorting/v2/sorting.py:603-608`).
- `MotionCorrectionParameters`: `params_schema_version={MOTION_CORRECTION_SCHEMA_VERSION}`
  (`src/spyglass/spikesorting/v2/session_group.py:261-267`).
- `MatcherParameters` uses the same shared validation path
  (`src/spyglass/spikesorting/v2/unit_matching.py:152-166`).

The Pydantic schemas define `schema_version` as a plain `int` default, not a
`Literal` or version-dispatch boundary, e.g. preprocessing
(`src/spyglass/spikesorting/v2/_params/preprocessing.py:138-140`), artifact
(`src/spyglass/spikesorting/v2/_params/artifact_detection.py:45-47`), waveform
(`src/spyglass/spikesorting/v2/_params/analyzer_waveform.py:41-43`), and motion
(`src/spyglass/spikesorting/v2/_params/motion_correction.py:103-105`). A custom
row can therefore carry `params={"schema_version": 2, ...}` while omitting
`params_schema_version`. The validator sees no outer column to compare, and the
database can fill the current table default. The stored row then has an outer
version that does not describe the inner blob.

`SorterParameters` already handles this correctly because it backfills the outer
column from the validated inner blob when the outer value is omitted or left at
the sentinel (`src/spyglass/spikesorting/v2/sorting.py:320-323`). The generic
lookup helper should either do the same for all validated lookup tables, or
reject omitted outer versions whenever the inner version is not the current
supported schema. `QualityMetricParameters` and `AutoCurationRules` also avoid
this exact path by building/storing `params_schema_version` from the cleaned
payload (`src/spyglass/spikesorting/v2/metric_curation.py:291-315`,
`src/spyglass/spikesorting/v2/metric_curation.py:433-448`).

Impact: an old-version row can land with current-version metadata. That breaks
downstream routing, parameter fingerprints, recompute diagnosis, and migration
audits because the outer column lies about what schema interpreted the blob.

Fix direction:

- Make `validate_lookup_rows()` backfill `params_schema_version` from
  `row["params"]["schema_version"]` when omitted, before duplicate checks and
  before `super().insert()`.
- If old inner versions are unsupported for a table, add an explicit check that
  rejects non-current inner versions with a migration message.
- Add tests for every shared-validated lookup: omitted outer/current inner is
  accepted and stored correctly; omitted outer/old inner either stores the old
  version honestly or raises; explicit outer/inner mismatch raises.

### 2. High: same-name stale shipped defaults can survive initialization silently

Parameter fingerprints intentionally exclude row names
(`src/spyglass/spikesorting/v2/_parameter_identity.py:1-15`), while deterministic
selection IDs include parameter-row names rather than parameter content
(`src/spyglass/spikesorting/v2/_selection_identity.py:40-46`). That separation is
sound only if parameter-row names remain immutable pointers to stable content.

The duplicate-content guard protects against a second name for already-existing
content, but it explicitly skips incoming rows whose primary key already exists
(`src/spyglass/spikesorting/v2/_lookup_validation.py:264-270`). The default
seeders then call `insert(..., skip_duplicates=True)`, including preprocessing
(`src/spyglass/spikesorting/v2/recording.py:738-741`), artifact detection
(`src/spyglass/spikesorting/v2/artifact.py:189-192`), sorter params
(`src/spyglass/spikesorting/v2/sorting.py:402-410`), analyzer waveform params
(`src/spyglass/spikesorting/v2/sorting.py:644-647`), motion correction
(`src/spyglass/spikesorting/v2/session_group.py:323-326`), quality metrics
(`src/spyglass/spikesorting/v2/metric_curation.py:320-323`), auto-curation rules
(`src/spyglass/spikesorting/v2/metric_curation.py:592-596`), and matcher params
(`src/spyglass/spikesorting/v2/unit_matching.py:191-194`).

This makes default initialization idempotent, which is good, but it does not
detect a shipped row that already exists under the same name with stale content.
`AutoCurationRules.insert_rules()` is the model to copy: same-name/same-payload
is idempotent, but same-name/different-payload raises a clear error
(`src/spyglass/spikesorting/v2/metric_curation.py:455-470`).

Impact: if a database already contains `franklab_hippocampus_2026_06`,
`default_neuropixels`, `default`, or another shipped name with older content,
`initialize_v2_defaults()` can leave it untouched. A new `recording_id`,
`sorting_id`, or analyzer-curation ID then records the same name under different
science than the code/docs imply.

Fix direction:

- Add a shipped-catalog audit path, for example
  `verify_v2_default_catalog()` or `initialize_v2_defaults(validate_existing=True)`,
  that compares same-name stored fingerprints against the current shipped
  default payloads.
- Make stale same-name shipped rows raise by default, with an error that tells
  the user to keep the old row under an explicitly legacy name, delete it if
  unused, or install the new dated row.
- Add tests that seed same-name stale defaults, run `initialize_v2_defaults()`,
  and assert a clear stale-default report.

### 3. Medium-high: old/interim v2 row names can block default seeding without an upgrade path

The v1->v2 migration guide tells users that old/interim names have no
back-compat aliases and lists examples such as `default_franklab`,
`franklab_tetrode_hippocampus_30kHz_ms4`,
`franklab_probe_ctx_30kHz_ms4`, `default_clusterless`, and the KS4 `default`
row (`docs/src/Features/SpikeSortingV2_Migration.md:13-35`).

`initialize_v2_defaults()` unconditionally calls every default seeder
(`src/spyglass/spikesorting/v2/__init__.py:18-61`). If an upgraded v2 database
still has an interim row with content identical to a new shipped row but under a
different name, the duplicate-content guard raises
`DuplicateParameterContentError` rather than installing the new row
(`src/spyglass/spikesorting/v2/_lookup_validation.py:271-281`). That guard is
right for provenance, but the initialization experience is not yet an upgrade
workflow.

Impact: a user doing the recommended first step, `initialize_v2_defaults()`, may
hit an error caused by old v2 names. The error protects against aliasing, but it
does not tell the user which known old row to rename/delete, which downstream
rows reference it, or how to complete the upgrade safely.

Fix direction:

- Add an explicit old-v2 parameter-row audit/migration helper that recognizes
  known interim names and reports whether they are referenced.
- In the migration guide, document exact cleanup/rename choices for known old
  names, including when deleting an unreferenced lookup row is safe.
- Add an upgrade fixture/test that pre-inserts known interim rows, runs the audit
  and/or initialization helper, and verifies the result is actionable rather
  than a raw duplicate-content failure.

### 4. Medium-high: v2 tables are labeled final-shape, but v2 schema drift is not snapshotted or gated

The v2 table modules say their tables are "final-shape under the zero-migration
policy", for example recording (`src/spyglass/spikesorting/v2/recording.py:1-17`),
sorting (`src/spyglass/spikesorting/v2/sorting.py:1-18`), and curation
(`src/spyglass/spikesorting/v2/curation.py:1-17`). The existing schema-stability
test only snapshots v0/v1 table definitions
(`tests/spikesorting/v2/test_legacy_runtime_boundary.py:45-68`) and asserts
those legacy definitions match `legacy_schemas.json`
(`tests/spikesorting/v2/test_legacy_runtime_boundary.py:90-130`).

At the same time, v2 table definitions and secondary fields are still changing:
for example `Recording` now stores `content_hash`
(`src/spyglass/spikesorting/v2/recording.py:1016-1026`) and
`ConcatenatedRecording` also stores `content_hash`
(`src/spyglass/spikesorting/v2/session_group.py:580-590`). The current design
plan says this is pre-release and no data migration is needed, but the module
headers say final-shape. Those two policies should not coexist implicitly.

Impact: a table-definition change can land without a mechanical prompt to update
the migration plan, changelog, docs, old-v2 DB handling, or backfill script.
That is exactly how pre-release data silently becomes unqueryable after users
have already declared/populated v2 schemas.

Fix direction:

- Add a v2 schema snapshot or schema-migration registry covering public v2
  master and part definitions.
- Require any v2 definition drift to update one of: an explicit "pre-release,
  drop/redeclare allowed" note for that table, an idempotent alter/backfill
  script, or a documented incompatibility.
- Include `Recording`, `ConcatenatedRecording`, recompute tables, analyzer
  curation, UnitMatch, and their part tables in the snapshot, not just the
  initial single-session path.

### 5. Medium: published migration/storage docs still describe `cache_hash` where code now uses `content_hash`

The current code has moved recording artifact identity to a semantic
`content_hash`: `Recording` stores `content_hash`
(`src/spyglass/spikesorting/v2/recording.py:1016-1026`), `RecordingArtifactVersions`
copies that stored value (`src/spyglass/spikesorting/v2/recompute.py:214-238`),
and `_recording_fingerprint.py` explicitly says whole-file `NwbfileHasher`
digests are volatile and unsuitable for rebuild confirmation
(`src/spyglass/spikesorting/v2/_recording_fingerprint.py:1-25`).

The docs have not fully caught up:

- The migration guide still says the preprocessed `Recording` cache carries an
  `NwbfileHasher` `cache_hash`
  (`docs/src/Features/SpikeSortingV2_Migration.md:109-111`).
- The storage-management page says recording rebuild uses warn-and-rebuild and
  logs when regenerated `cache_hash` differs
  (`docs/src/Features/SpikeSortingV2StorageManagement.md:82-89`).
- The main v2 page says `DriftEstimate` leaves the upstream recording's
  `cache_hash` unchanged (`docs/src/Features/SpikeSortingV2.md:781`).
- The changelog still has current v2 sections referring to `cache_hash`
  (`CHANGELOG.md:226`, `CHANGELOG.md:294`, `CHANGELOG.md:1148`).

Impact: operators reading the docs will not know whether the canonical
recompute authority is the old whole-file hash or the new semantic content
hash. That confusion matters for destructive cleanup, rebuild refusal, and
upgrade diagnosis.

Fix direction:

- Update the user docs to describe `content_hash`, component fingerprints, and
  fail-closed rebuild behavior.
- Keep old `cache_hash` mentions only in historical changelog entries where they
  are explicitly labeled as superseded.
- Add a docs grep/link check that prevents new v2 user-facing docs from naming
  `cache_hash` as the current recording identity.

### 6. Medium: preprocessing runtime semantics changed without a durable compatibility boundary

`PreprocessingParamsSchema` documents several behavior changes within schema
version 3. It says the runtime order changed from reference->filter to
filter->reference, but that the params blob shape is unchanged and "dev rows are
regenerated, not migrated" (`src/spyglass/spikesorting/v2/_params/preprocessing.py:106-136`).
The schema version is currently `3`
(`src/spyglass/spikesorting/v2/_params/preprocessing.py:7`), and
`Recording.make_fetch()` validates whichever stored preprocessing blob the row
name points to under the current code
(`src/spyglass/spikesorting/v2/recording.py:1127-1133`).

For a pre-release branch, regenerating dev rows may be acceptable. For an
already-declared database, though, a row with the same name and same
`params_schema_version` can now be interpreted with different runtime semantics.
Because `recording_id` hashes `preprocessing_params_name`, not the parameter
content or runtime semantics (`src/spyglass/spikesorting/v2/_selection_identity.py:40-46`),
the identity does not move when the runtime interpretation moves.

Impact: old rows can produce different artifacts under the same apparent schema
version and row name. Recompute will detect content drift, but the schema
metadata does not explain why.

Fix direction:

- Treat behavior-changing preprocessing runtime changes as schema/versioned
  changes, even when the blob shape does not change, or add an explicit
  `runtime_semantics_version` to the row/fingerprint.
- For pre-release-only changes, document the drop/redeclare expectation and add
  a guard that refuses old known dev rows instead of silently reinterpreting
  them.
- Add a regression test with a legacy preprocessing row that verifies the
  intended outcome: explicit migration failure, new row name/version, or
  preserved legacy runtime.

### 7. Medium: recompute inventories detect drift but do not record enough input fingerprints to diagnose migration causes

`RecordingArtifactVersions` stores `nwb_deps` and `content_hash`
(`src/spyglass/spikesorting/v2/recompute.py:214-258`). `SortingAnalyzerVersions`
stores `si_deps`, an analyzer extension manifest, and an aggregate analyzer hash
(`src/spyglass/spikesorting/v2/recompute.py:517-645`). These tables are useful
for deciding whether bytes match after recompute.

They do not store the parameter-row fingerprints/schema versions that caused the
artifact: preprocessing params for recordings; artifact detection/sorter params
for sorting; waveform/metric/auto-curation params for analyzers; motion params
for concat; or the old-v2 catalog names involved. If a recompute diverges after
a default-row or schema evolution, the system can say "hash mismatch" but not
"this recording was built from stale `default_neuropixels` content" or "this
analyzer used the old hippocampus waveform recipe".

Impact: drift diagnosis after upgrades becomes manual. That slows down exactly
the repair workflow recompute is supposed to support.

Fix direction:

- Store input parameter fingerprints and schema versions in the version tables,
  or add a companion report that joins the computed artifact back to the current
  live parameter fingerprints.
- Include both row names and content fingerprints, because row names are the
  selection identity while fingerprints reveal stale content.
- Add a stale-default recompute test that reports the exact stale parameter row
  instead of only an aggregate hash mismatch.

### 8. Medium: runtime aliases and hardcoded presets are not independently versioned

`MotionCorrectionParameters` has a typed params row, but the Spyglass-only
`"auto"` alias is resolved in current code to `AUTO_SAME_DAY_PRESET =
"rigid_fast"` (`src/spyglass/spikesorting/v2/_concat_recording.py:19-22`,
`src/spyglass/spikesorting/v2/_concat_recording.py:93-103`). `DriftEstimate` is
not a parameter lookup at all; it stores a hardcoded `_DEFAULT_PRESET =
"dredge_fast"` and passes that to `compute_motion`
(`src/spyglass/spikesorting/v2/recording.py:2174-2214`).

Impact: if the alias target changes, or if SpikeInterface changes the internals
of a named preset across versions, old and new computations can share the same
apparent parameter value. `UserEnvironment` helps with recompute attempts, but
the logical row identity and parameter metadata do not carry the resolved preset
implementation.

Fix direction:

- Store resolved motion preset/kwargs alongside the logical alias, or version
  the alias mapping explicitly.
- Consider making `DriftEstimate` use a small parameter lookup if users will
  compare drift outputs across dependency upgrades.
- Add tests that monkeypatch the `"auto"` alias or drift preset and assert old
  rows cannot be silently recomputed under the same logical identity.

### 9. Medium-low: duplicate/alias policy is weaker for metric and auto-curation parameter tables

Most validated lookup tables call `reject_duplicate_parameter_content()`, which
raises when a second name points to identical content
(`src/spyglass/spikesorting/v2/_lookup_validation.py:165-282`). The metric and
auto-curation tables are different:

- `QualityMetricParameters._default_rows()` intentionally creates
  `franklab_default` and `neuropixels_default` from the same payload
  (`src/spyglass/spikesorting/v2/metric_curation.py:230-279`), and its insert
  path validates but does not reject duplicate content
  (`src/spyglass/spikesorting/v2/metric_curation.py:281-318`).
- `AutoCurationRules.insert_rules()` detects same-name/different-payload drift,
  but different names with identical rule content are allowed and later reported
  as duplicates by `describe_parameter_rows()`
  (`tests/spikesorting/v2/test_parameter_identity.py:535-577`).

This may be an intentional alias policy. If so, it should be explicit because
`AnalyzerCurationSelection` deterministic identity includes the metric and rule
names (`src/spyglass/spikesorting/v2/metric_curation.py:718-730`). Two aliases
with identical science therefore produce different analyzer-curation IDs.

Impact: alias rows make future renames/default migrations harder to reason
about. They are less dangerous than same-name drift, but they still fork
deterministic provenance for equivalent science.

Fix direction:

- Either apply the same duplicate-content guard with an escape hatch, or document
  these tables as intentionally alias-friendly.
- If aliases remain intentional, include alias provenance in
  `describe_parameter_rows()` and migration docs so users know which name is
  canonical.
- Add tests that distinguish intentional aliases from accidental duplicates.

### 10. Low: concat selection ignores a supplied deterministic ID while other selection helpers reject mismatches

`ConcatenatedRecordingSelection.insert_selection()` documents that a
caller-supplied `concat_recording_id` is ignored
(`src/spyglass/spikesorting/v2/session_group.py:377-383`). It then derives the
deterministic ID from the logical identity and inserts/fetches that ID
(`src/spyglass/spikesorting/v2/session_group.py:460-486`). Other selection-plan
builders use `assert_supplied_id_matches()` so a caller-supplied stale ID raises
instead of being silently ignored (`src/spyglass/spikesorting/v2/_selection_plan.py:99-127`,
`src/spyglass/spikesorting/v2/_selection_identity.py:351-394`).

Impact: this is mostly a migration/backfill footgun. A script can pass an old
concat ID and still get a different deterministic ID back, hiding the fact that
the stored identity changed.

Fix direction:

- Make concat selection validate a supplied `concat_recording_id` the same way
  recording, artifact, sorting, and analyzer-curation selection helpers do.
- Add a test that a mismatched supplied concat ID raises, while a matching one
  is accepted.

## Positive Coverage and Design Notes

- `ImmutableParamsLookup` blocks same-name in-place updates unless an explicit
  maintenance escape hatch is passed
  (`src/spyglass/spikesorting/v2/utils.py:258-310`).
- Parameter fingerprints include schema version, params, job kwargs, sorter
  context, matcher context, and sorter execution params where relevant
  (`src/spyglass/spikesorting/v2/_parameter_identity.py:78-162`).
- The shipped recipe catalog is centralized, and parity tests maintain an
  independent copy rather than deriving expected literals from the same module
  (`src/spyglass/spikesorting/v2/_recipe_catalog.py:1-21`).
- `SorterParameters` already has the correct outer-version backfill pattern
  (`src/spyglass/spikesorting/v2/sorting.py:267-276`,
  `src/spyglass/spikesorting/v2/sorting.py:320-323`).
- `AutoCurationRules.insert_rules()` is a good same-name drift model: existing
  same payload is idempotent, existing different payload raises
  (`src/spyglass/spikesorting/v2/metric_curation.py:455-470`).
- `describe_parameter_rows()` is a good operator-facing starting point: it lists
  fingerprints, shipped-default status, preset usage, duplicate content, and
  warnings across all seeded lookup tables
  (`src/spyglass/spikesorting/v2/_pipeline_reporting.py:42-73`,
  `src/spyglass/spikesorting/v2/_pipeline_reporting.py:412-445`).

## Suggested Fix Order

1. Fix the omitted-outer-version insert path in `validate_lookup_rows()` and add
   shared tests for every validated lookup table.
2. Add a shipped-default catalog audit that catches same-name stale rows during
   initialization.
3. Add an old-v2 parameter-row audit/migration helper for known interim names.
4. Decide and encode the v2 schema policy: snapshot/gate final-shape tables, or
   explicitly mark which pre-release tables can be dropped/redeclared.
5. Update docs from `cache_hash` to `content_hash` and make the storage guide
   match current rebuild behavior.
6. Add richer recompute diagnostics by storing or reporting input parameter
   fingerprints.
7. Version runtime aliases/hardcoded presets where users are likely to compare
   outputs across upgrades.
