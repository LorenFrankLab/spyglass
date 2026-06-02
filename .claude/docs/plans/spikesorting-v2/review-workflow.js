export const meta = {
  name: 'spikesorting-v2-review',
  description: 'Thorough review of spikesorting-v2: correctness, efficiency, and test coverage with adversarial verification',
  phases: [
    { title: 'Map', detail: 'test inventory, behavior contract, cross-cutting triage, module surface' },
    { title: 'Deep-read', detail: 'per-unit audit (all changed subsystems) then adversarially verify each finding' },
    { title: 'Coverage', detail: 'pipeline-behavior x test matrix, gaps and weak tests' },
    { title: 'Parity', detail: 'v1 vs v2 divergence inventory, each classified justified / undocumented / unjustified' },
    { title: 'Synthesize', detail: 'dedup, rank, completeness critic, final report' },
  ],
}

// ---------------------------------------------------------------------------
// Shared context every agent gets: scope, env, known-red tests, ground rules.
// ---------------------------------------------------------------------------
const REPO = '/cumulus/edeno/spyglass'
const BASE = `You are reviewing the spikesorting-v2 implementation in the Spyglass repo (${REPO}).
Branch under review: spikesorting-v2 (vs master). Core code: src/spyglass/spikesorting/v2/.
Tests: tests/spikesorting/v2/. The branch is merged work; you may read the diff with
\`git diff master...HEAD -- <path>\`. Use the 'spyglass' skill for DataJoint/pipeline conventions
and the 'scientific-code-change-audit' skill for silent-error / indexing / numerical lenses.
Test env (if you need to RUN anything): the conda env spyglass_spikesorting_v2
(/home/edeno/miniconda3/envs/spyglass_spikesorting_v2/bin/python; Python 3.11, SpikeInterface
0.104.3), NOT the conda spyglass env (SI 0.99.1, which is for v1).
RUNNING TESTS (only if a static check is insufficient — most of this review is static analysis):
the Docker daemon IS available here (datajoint/mysql:8.0 on port 3306; tests/container.py auto-spawns
a 'spyglass-pytest-<branch>' container when --no-docker is omitted). DB-tier v2 tests first need the
MEArec smoke fixture: \`python tests/spikesorting/v2/fixtures/generate_mearec.py --smoke\`. Canonical
run (mirrors CI): \`/home/edeno/miniconda3/envs/spyglass_spikesorting_v2/bin/python -m pytest
tests/spikesorting/v2/ --no-docker --no-dlc -q\` (CI uses --no-docker / null-server; omit --no-docker
to exercise a real DB locally). Pure/Pydantic tests need no DB. Do NOT launch the full slow suite for
this review — prefer targeted, tiny checks.
Many review agents run CONCURRENTLY: do NOT start the Docker MySQL container or (re)generate the
MEArec fixture yourself — you would race other agents on the shared daemon / fixture path. If a
check genuinely needs a live DB, report it as "unverified (needs DB)" rather than spawning one.
AVOID HANGS (this runs unattended, many agents in parallel — a blocked agent stalls a slot):
 - Run ONLY fast, non-interactive, read-only commands. NEVER run anything that waits on stdin, opens
   a pager or editor, or starts a server / \`tail -f\` / watch loop.
 - Force non-interactive output: prefer the Read/Grep/Glob tools over shell; when you must use git,
   use \`git --no-pager …\` and pipe long output through \`head\`. Never run a bare paging \`git log\`/\`git diff\`.
 - Do NOT run: any pytest that could prompt, \`populate()\`, ANY DataJoint write/delete (the
   "Commit deletes? [yes, No]" prompt will hang until timeout), docker/container startup, or
   conda/mamba/pip installs. Reading source is almost always enough — prefer it over executing.
 - If you must run Python, keep it to a few seconds, NEVER let it block on input or open a DB
   connection, and put a timeout on the Bash call.
 - If a check would take more than ~30s or needs a DB/network, STOP and record the item as
   "unverified (would require running X)" instead of waiting. An unverified item is fine; a hang is not.
 - context7 MCP (si-best-practices only) is OPTIONAL: at most one attempt; if it is slow, fall back
   to reading the installed SI source.
As of the current branch tip there are NO known pre-existing red tests — every prior
known-red was fixed — so any failure you observe IS worth investigating. Before attributing a
failure to a *v2 regression* specifically, confirm whether it also reproduces on master (git stash
/ worktree). A conditional skip (e.g. a sorter not installed) is not a failure.
INTENTIONAL STUBS: unit_matching.py, metric_curation.py, figpack_curation.py, matcher_protocol.py
are Phase 4/5 roadmap placeholders. Each raises a clear ImportError on attribute access (and
AttributeError on dunders so import/inspection machinery is unharmed). Do NOT flag their absent
functionality as a bug — only their stub CONTRACT (clean error, no eager import) is in scope.
Do NOT edit any files. This is a read/analyze-only review. Cite every claim as file:line.`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    unit: { type: 'string', description: 'review unit key' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          file: { type: 'string', description: 'path:line, e.g. src/.../sorting.py:412' },
          category: { type: 'string', enum: ['correctness', 'silent-error', 'efficiency', 'numerical-stability', 'robustness', 'test-integrity', 'api-misuse', 'documentation', 'build-ci', 'spikeinterface-best-practice', 'type-design'] },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          evidence: { type: 'string', description: 'concrete code-grounded reasoning, with the relevant snippet/line' },
          why_it_matters: { type: 'string', description: 'what scientific/data/pipeline outcome breaks if this is real' },
          suggested_fix: { type: 'string' },
          similar_locations_to_check: { type: 'string', description: 'pattern + other files/lines that may share the same bug (generalize-from-failure)' },
        },
        required: ['title', 'file', 'category', 'severity', 'evidence', 'why_it_matters', 'suggested_fix', 'similar_locations_to_check'],
      },
    },
  },
  required: ['unit', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    isReal: { type: 'boolean', description: 'true only if the finding survives scrutiny against the actual code/library' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'not-a-bug'] },
    reasoning: { type: 'string', description: 'what you checked in the real code/library and what you found' },
  },
  required: ['isReal', 'confidence', 'corrected_severity', 'reasoning'],
}

const COVERAGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    area: { type: 'string' },
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          behavior: { type: 'string', description: 'pipeline behavior/invariant that should be tested' },
          current_coverage: { type: 'string', enum: ['none', 'weak', 'partial', 'good'] },
          existing_tests: { type: 'string', description: 'test names that touch this, or "none"' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          why_it_matters: { type: 'string' },
          suggested_test: { type: 'string', description: 'concrete test to add and what it should assert' },
        },
        required: ['behavior', 'current_coverage', 'existing_tests', 'severity', 'why_it_matters', 'suggested_test'],
      },
    },
    weak_tests: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          test: { type: 'string', description: 'test name + file:line' },
          problem: { type: 'string', description: 'e.g. asserts implementation not contract; would pass with the bug present; no assertions; over-broad tolerance' },
          fix: { type: 'string' },
        },
        required: ['test', 'problem', 'fix'],
      },
    },
  },
  required: ['area', 'gaps', 'weak_tests'],
}

// ===========================================================================
// PHASE 1 — MAP (parallel, barrier: phases 2 and 3 both consume these maps)
// ===========================================================================
phase('Map')
const [testInventory, contract, crossCutting, moduleSurface] = await parallel([
  () => agent(`${BASE}

TASK: Inventory the v2 test suite. For each test file under tests/spikesorting/v2/, list the
test functions, their pytest markers (slow/etc.), the fixtures they use, and — crucially — WHICH
pipeline stage/behavior each exercises (recording, artifact, sorting, curation, merge, concat,
pipeline orchestrator, params, v1-parity, downstream/decoding). Note which tests are real
integration tests (drive populate end-to-end) vs unit/pure-helper tests. Flag any test files that
look like dead scripts. Return a concise but COMPLETE structured-text map; this is the ground
truth a coverage-gap analysis will build on.`, { label: 'map:test-inventory', phase: 'Map', agentType: 'Explore' }),

  () => agent(`${BASE}

TASK: Extract the intended-behavior CONTRACT for v2. Read the plan docs in
.claude/docs/plans/spikesorting-v2/ (especially feature-parity.md, parity-extensions.md,
shared-contracts.md, overview.md, and review-fixes/phase-4-audit-correctness-and-parity.md) plus
the docstrings in src/spyglass/spikesorting/v2/pipeline.py, curation.py, sorting.py, artifact.py.
Produce a concise list of the INVARIANTS and CONTRACTS v2 must satisfy: v1-parity guarantees,
idempotency, zero-unit handling, gap/abs-time dedup semantics, merge-keyability, schema/param
defaults, and any explicitly documented divergences from v1 (with their justification). This is
the yardstick the deep-read and coverage agents measure against.`, { label: 'map:contract', phase: 'Map', agentType: 'Explore' }),

  () => agent(`${BASE}

TASK: Triage the NON-v2 changes on this branch. Run \`git diff master...HEAD --stat\` and focus on
changed files OUTSIDE src/spyglass/spikesorting/v2/ and tests/spikesorting/v2/ — especially
src/spyglass/common, src/spyglass/utils, src/spyglass/position, src/spyglass/spikesorting/v1,
src/spyglass/spikesorting/v0, and spikesorting_merge. For each meaningful changed area, classify it
as (a) directly supporting v2, (b) incidental/unrelated refactor, or (c) risky change that could
affect existing pipelines. Report findings using the schema for anything in category (c) or any
v2-supporting change that looks incorrect. Keep (b) to a one-line list.`, { label: 'map:cross-cutting', phase: 'Map', schema: FINDINGS_SCHEMA }),

  () => agent(`${BASE}

TASK: Map the v2 module surface and HOT computational paths. For each file in
src/spyglass/spikesorting/v2/ (recording.py, artifact.py, sorting.py, curation.py,
session_group.py, _nwb_iterators.py, utils.py, pipeline.py, _params/*), give: the public
tables/classes/methods, and the 2-4 computationally or correctness-critical code paths (the places
where indexing, frame<->time conversion, masking across gaps, dedup, broadcasting, or DB
cardinality bugs would live), each with file:line. This routes the deep-read; be precise about
line ranges.`, { label: 'map:module-surface', phase: 'Map', agentType: 'Explore' }),
])

// A phase-1 map thunk that throws resolves to null; interpolating null would put
// the literal "null" into every downstream prompt. Fall back to an explicit
// instruction so the agent compensates by reading source directly.
const NA = '(phase-1 map unavailable — derive directly from the source / tests / plan docs)'
const contractText = contract || NA
const moduleSurfaceText = moduleSurface || NA
const testInventoryText = testInventory || NA

// ===========================================================================
// Review units for the deep-read (phase 2). Full-branch scope: v2 core +
// stub-contract + every non-v2 subsystem the branch touches. Grouped by
// subsystem (not one-agent-per-file) so cross-file context is preserved.
// ===========================================================================
const V2_UNITS = [
  { key: 'recording', isV2: true, files: 'src/spyglass/spikesorting/v2/recording.py', focus: 'recording build, preprocessing chain order, probe/geometry attach, valid-times interval handling, frame<->time conversion, brain-region attribution' },
  { key: 'artifact', isV2: true, files: 'src/spyglass/spikesorting/v2/artifact.py and _nwb_iterators.py', focus: 'artifact detection thresholds, masking, gap-aware chunk detection, valid_times across gaps, removal_window_ms spillover, frame-space-across-gap (a KNOWN prior bug area — scrutinize window/join/complement math)' },
  { key: 'sorting', isV2: true, files: 'src/spyglass/spikesorting/v2/sorting.py', focus: 'sorter invocation + gating, zero-unit handling, frame<->abs-time conversion, spike readback (searchsorted, affine/recording-extractor frame shift), n_units accounting, disjoint-interval readback' },
  { key: 'curation', isV2: true, files: 'src/spyglass/spikesorting/v2/curation.py', focus: 'curation_label semantics, merge insert, lazy get_merged_sorting, abs-time dedup across gaps, parent_curation lineage, merge_id derivation, pre-curation NWB behavior' },
  { key: 'concat', isV2: true, files: 'src/spyglass/spikesorting/v2/session_group.py', focus: 'multi-session concat, abs-time dedup, per-session interval bookkeeping, concat rebuild correctness' },
  { key: 'merge-dedup', isV2: true, files: 'merge integration: src/spyglass/spikesorting/spikesorting_merge.py and the CurationV2 part-table registration', focus: 'SpikeSortingOutput registration, dedup logic, merge_get_part / get_restricted_merge_ids parity with v1, cardinality' },
  { key: 'params', isV2: true, files: 'src/spyglass/spikesorting/v2/_params/*.py', focus: 'Pydantic validation, schema version bumps, field defaults (history: noise_levels default caused 1400x divergence), extra=forbid, per-row default injection' },
  { key: 'utils-pipeline', isV2: true, files: 'src/spyglass/spikesorting/v2/utils.py, pipeline.py, exceptions.py', focus: 'orchestrator idempotency, preset bundles, helper correctness, error taxonomy, populate reserve_jobs usage' },
  { key: 'v2-stubs', isV2: true, files: 'src/spyglass/spikesorting/v2/{unit_matching,metric_curation,figpack_curation,matcher_protocol}.py and v2/__init__.py', focus: 'STUB CONTRACT ONLY: attribute access raises a clear ImportError naming the roadmap phase; dunder/inspection access raises AttributeError; __init__.py does NOT eagerly import them. Do NOT flag absent functionality — these are intentional Phase 4/5 roadmap stubs.' },
  { key: 'si-best-practices', isV2: true, files: 'v2 SpikeInterface usage across recording.py, artifact.py, sorting.py, curation.py, _params/*', focus: `SPIKEINTERFACE BEST-PRACTICE AUDIT (category spikeinterface-best-practice).

METHOD — this is mandatory, do not skip it. You must GROUND every claim in the actual installed
SpikeInterface 0.104.3 source, not in training memory or doc prose:
 - The SI source is at: /home/edeno/miniconda3/envs/spyglass_spikesorting_v2/lib/python3.11/site-packages/spikeinterface/
   Read the relevant modules directly: core/sortinganalyzer.py, core/baserecording.py,
   preprocessing/ (filter.py, common_reference.py, whiten.py, detect_bad_channels.py),
   sorters/runsorter.py + sorters/external/*, qualitymetrics/, metrics/, postprocessing/.
 - For EVERY SI call v2 makes (run_sorter, create_sorting_analyzer, bandpass_filter/highpass_filter,
   common_reference, whiten, get_noise_levels, detect_bad_channels, analyzer.compute, quality-metric
   functions): open the real function definition, read its CURRENT signature, defaults, and docstring,
   and confirm v2 passes valid kwargs with sensible values. Verify in-env with
   \`inspect.signature(...)\` / reading source — cite SI file:line for each recommendation.
 - Read SI's own recommended-pipeline guidance where it exists in-repo (module docstrings, the
   'how to' notes in core/ and qualitymetrics/). The context7 MCP tools (resolve-library-id
   spikeinterface -> query-docs) are a SECONDARY cross-check; the installed SOURCE is the source of truth.
 - Understand the data flow first: what recording object v2 builds, how it is preprocessed, how it is
   handed to the sorter, how the SortingAnalyzer + extensions + metrics are computed. State that
   understanding before judging.

WHAT TO CHECK against the source you read:
 (1) preprocessing order/choice (filter/highpass -> reference -> whiten), bad-channel/noise handling,
     whiten default sanity, dtype/scaling (float32 vs int16, return_scaled);
 (2) sorter invocation — run_sorter/run_sorter_jobs usage, per-sorter recommended params for
     MS4/MS5/KS4/clusterless thresholder, docker/singularity image handling, deterministic seeds,
     how sorter_params are passed (v2 has comments that some params flow via SI global state, NOT
     splatted into run_sorter — verify that against the real run_sorter signature);
 (3) analyzer API — v2 uses create_sorting_analyzer(format='binary_folder', sparse=True) (modern,
     correct — WaveformExtractor is REMOVED in 0.104); verify sparsity estimation + extension compute
     order are valid per the real SortingAnalyzer.compute source;
 (4) quality metrics — which metrics, computed how, vs what SI's qualitymetrics module actually offers
     and recommends in 0.104;
 (5) SI 0.99->0.104 MIGRATION correctness — moved/renamed modules (toolkit split, metrics/qualitymetrics),
     removed/renamed kwargs, behavior changes; confirm v2 targets the 0.104 API, not stale 0.99 calls.
Report deviations as findings with BOTH the v2 file:line and the SI source file:line; flag
deprecated/removed-API use as high severity.` },
]

const NON_V2_UNITS = [
  { key: 'common', isV2: false, files: 'src/spyglass/common/** (~22 files)', focus: 'schema/table/FK/ingestion changes; anything that could alter existing data semantics or break a v2 upstream dependency. Diff each vs master.' },
  { key: 'utils-mixins', isV2: false, files: 'src/spyglass/utils/** including utils/mixins/** (~24 files)', focus: 'HIGH RISK — SpyglassMixin, merge methods, hashers, nwb_helper_fn affect EVERY pipeline: cautious_delete/team perms, fetch_nwb list-vs-1, merge_get_part/merge_restrict restriction handling, hashing determinism' },
  { key: 'position', isV2: false, files: 'src/spyglass/position/** (~16 files)', focus: 'Trodes/DLC/pose correctness, interval/timestamp handling, any regression vs master' },
  { key: 'legacy-spikesorting', isV2: false, files: 'src/spyglass/spikesorting/v0/** and v1/** and figurl_views (~22 files)', focus: 'changes to legacy v0/v1 that v2 reuses or that must keep working; ensure no behavioral regression to existing sorts' },
  { key: 'decoding', isV2: false, files: 'src/spyglass/decoding/** (~11 files)', focus: 'decoding is the key DOWNSTREAM CONSUMER of SpikeSortingOutput — verify the consumer contract (merge_id keying, spike retrieval) still holds with v2 rows' },
  { key: 'export-downstream', isV2: false, files: 'src/spyglass/common/common_usage.py (Export / ExportSelection), src/spyglass/spikesorting/analysis/v1/** (SortedSpikesGroup, UnitAnnotation), src/spyglass/sharing/** (Kachery), and the figurl/figpack export paths', focus: 'EXPORT + DOWNSTREAM IMPACT: does Export trace and snapshot the v2 tables AND the SpikeSortingOutput.CurationV2 part for DANDI/FigURL/Kachery provenance (no missing-table gaps in the captured graph)? Do SortedSpikesGroup, UnitAnnotation, and the decoding/ripple/mua spike-input paths correctly consume a v2 merge_id (spike-time retrieval, fetch_nwb, get_merged_sorting)? Flag any v1-only assumption that silently breaks or omits v2 rows from an export/snapshot.' },
  { key: 'other-pipelines', isV2: false, files: 'src/spyglass/{lfp,ripple,mua,linearization,behavior,sharing,data_import}/**, src/spyglass/spikesorting/analysis/**, src/spyglass/settings.py, src/spyglass/__init__.py, src/spyglass/directory_schema.json', focus: 'misc pipeline + infra source changes; behavior changes / regressions, settings/path handling' },
  { key: 'non-v2-tests', isV2: false, files: 'changed test files OUTSIDE tests/spikesorting/v2/ (tests/common, tests/spikesorting/v1, tests/position, tests/decoding, tests/utils, tests/setup, conftest.py, container.py)', focus: 'TEST-INTEGRITY: were existing tests weakened, skipped, deleted, xfailed, or tolerances loosened to accommodate this branch? Diff each vs master; flag any test that now tests less.' },
  { key: 'user-facing', isV2: false, files: 'notebooks/**, notebooks/py_scripts/**, examples/cli_examples/**', focus: 'do the notebooks/CLI examples reflect the REAL current v1/v2 API? broken imports, stale signatures, misleading output, examples that would error if run. Light scientific-UX lens.' },
  { key: 'infra-docs', isV2: false, files: 'docs/**, .github/workflows/**, environments/**, pyproject.toml, docker-compose.yml, maintenance_scripts/**, scripts/**, .pre-commit-config.yaml, CHANGELOG.md', focus: 'CI matrix correctness (does the v2 job run the right env/tests?), dependency/version pins, env file correctness, doc drift vs implemented API' },
]

// Specialist units delegate to the pr-review-toolkit agents (resolved via
// opts.agentType, composed with FINDINGS_SCHEMA). Each agent's own system
// prompt drives HOW it reviews; we scope WHAT (v2) and translate its web-isms
// to Python / Pydantic / DataJoint.
const SPECIALIST_UNITS = [
  { key: 'silent-failures', isV2: true, agentType: 'pr-review-toolkit:silent-failure-hunter', files: 'all of src/spyglass/spikesorting/v2/, plus changed error-handling in src/spyglass/spikesorting/spikesorting_merge.py and src/spyglass/utils/mixins/**', focus: 'Hunt silent failures / inadequate error handling in PYTHON: bare or overly-broad `except Exception` / `except:` that can swallow unrelated errors, except blocks that log-and-continue or return None/default on failure, unjustified fallbacks that mask the real problem, swallowed exceptions during delete/cleanup paths, retries that exhaust silently, and `.get()`/optional access that hides a missing key. CONTEXT: this codebase deliberately NARROWED some excepts (Phase-1 E3, e.g. artifact.py orphan-delete now uses `except SchemaBypassError`); verify those narrowings did not leave a silent path and that errors are raised or logged via spyglass.utils.logger with actionable context. Ignore the web-isms in your base instructions (errorIds.ts, logForDebugging, Sentry) — translate to Python/DataJoint idioms.' },
  { key: 'comments', isV2: true, agentType: 'pr-review-toolkit:comment-analyzer', files: 'src/spyglass/spikesorting/v2/** (comment-dense)', focus: 'Comment ACCURACY + rot. This code has dense WHY-comments and a 423-commit history. Flag: comments that describe OLD or transitional behavior rather than what the CURRENT code does (project rule: a comment explains why the current code is the way it is, never narrates the old/original code), stale references to refactored code, leftover "Phase N" label leakage in runtime comments, TODO/FIXME that were already addressed, and docstrings whose stated signature/behavior no longer matches the function. Cross-check every flagged comment against the actual code at that line.' },
  { key: 'type-design', isV2: true, agentType: 'pr-review-toolkit:type-design-analyzer', files: 'src/spyglass/spikesorting/v2/_params/*.py (Pydantic params models) and the DataJoint table definitions across v2', focus: 'Type/invariant design for Python Pydantic + DataJoint (translate compile-time / access-modifier language to Python idioms). For each Pydantic params model: are invariants enforced at construction (field types, validators, ranges, extra="forbid") so illegal states are unrepresentable? History: a permissive default (noise_levels=[1.0]) caused a 1400x divergence — missing/loose validation is the key risk class here. For DataJoint tables: do PK/FK definitions encode the intended cardinality and relationships (e.g. artifact_id optionality, zero-unit gating via require_units) or are invariants left to runtime/external code? Rate encapsulation / expression / usefulness / enforcement and flag any invariant enforced only by documentation.' },
]

const ALL_UNITS = [...V2_UNITS, ...NON_V2_UNITS, ...SPECIALIST_UNITS]

// ===========================================================================
// PHASE 2 (deep-read) and PHASE 3 (coverage) run concurrently — both depend
// only on phase-1 maps. Each is its own pipeline/parallel internally.
// ===========================================================================
const COVERAGE_AREAS = [
  'recording build + preprocessing chain + probe/valid-times',
  'artifact detection + masking + gap handling',
  'sorting + zero-unit + spike readback + disjoint intervals',
  'curation + merge insert + abs-time dedup',
  'concat / session group / cross-session',
  'pipeline orchestrator: idempotency + presets + error paths (require_units, unknown preset)',
  'params validation + schema versions + defaults',
  'v1 parity + downstream consumers (decoding/SortedSpikesGroup/UnitAnnotation)',
  'export (DANDI/FigURL/Kachery) snapshots v2 tables + SpikeSortingOutput.CurationV2 part',
  'SpikeInterface integration: sorter/preproc defaults pinned, analyzer + quality-metric outputs asserted',
]

// v1 vs v2 study (Parity phase). Areas mirror the pipeline stages.
const PARITY_AREAS = [
  'recording + preprocessing (v1 SpikeSortingRecording vs v2 Recording)',
  'artifact detection (v1 vs v2)',
  'sorting (v1 SpikeSorting vs v2 Sorting)',
  'curation + merge + SpikeSortingOutput (v1 CurationV1 vs v2 CurationV2)',
  'params + schema + sorter gating (v1 vs v2)',
  'orchestration + UX/ergonomics + downstream/export consumer contract (v1 vs v2)',
]
const RISKY = new Set(['unjustified', 'undocumented', 'unclear'])

const PARITY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    area: { type: 'string' },
    divergences: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          aspect: { type: 'string' },
          v1_behavior: { type: 'string' },
          v2_behavior: { type: 'string' },
          documented: { type: 'string', description: 'where it is documented (plan file:line / CHANGELOG), or the literal "UNDOCUMENTED"' },
          classification: { type: 'string', enum: ['justified-bugfix', 'justified-improvement', 'justified-design', 'undocumented', 'unjustified', 'unclear'] },
          assessment: { type: 'string', description: 'is the rationale sound? cite the evidence' },
          risk: { type: 'string', description: 'risk to correctness/parity/downstream if this divergence is unintended or wrong' },
          file: { type: 'string', description: 'v1 and v2 file:line' },
        },
        required: ['aspect', 'v1_behavior', 'v2_behavior', 'documented', 'classification', 'assessment', 'risk', 'file'],
      },
    },
    improvement_gaps: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          improvement: { type: 'string', description: 'a v1 limitation / pain point that v2 was meant to (or arguably should) improve' },
          source: { type: 'string', description: 'documented v2 goal (plan file:line) OR reviewer-identified' },
          status: { type: 'string', enum: ['delivered', 'partial', 'stubbed-roadmap', 'not-addressed'] },
          should_do_now: { type: 'boolean', description: 'should this be done before relying on v2 in production?' },
          assessment: { type: 'string' },
        },
        required: ['improvement', 'source', 'status', 'should_do_now', 'assessment'],
      },
    },
  },
  required: ['area', 'divergences', 'improvement_gaps'],
}

const [deepRead, coverage, parity] = await parallel([
  // ---- PHASE 2: deep-read pipeline (review -> verify each finding) ----
  () => pipeline(
    ALL_UNITS,
    (m) => agent(`${BASE}

INTENDED-BEHAVIOR CONTRACT (from plan docs):
${contractText}
${m.isV2 ? `\nV2 MODULE SURFACE MAP (routing aid for v2 units):\n${moduleSurfaceText}\n` : ''}
TASK: Audit review unit "${m.key}" (${m.files}). Focus areas: ${m.focus}.
${m.agentType ? 'You are a SPECIALIST reviewer: apply your built-in lens to exactly the files above; the generic checklist below is secondary. Emit findings in the required schema (map your severity to critical/high/medium/low).\n' : ''}Apply the scientific-code-change-audit checklist items RELEVANT to this unit's kind. For code:
implementation correctness (shapes, axes, indexing, frame<->time units, sorts/joins alignment),
domain invariants, numerical stability, missing-data/masking semantics, efficiency (accidental
O(n^2), densification, repeated DB fetches, per-row Python loops over arrays), robustness/edge
cases, and DataJoint API misuse (fetch1 cardinality, merge-master restriction discard, classmethod
restriction loss). For tests/docs/notebooks/CI units, focus on test-integrity / drift / build
correctness instead of numerics. Read the real code AND \`git diff master...HEAD -- <file>\` to see
what changed. Report ONLY evidence-grounded findings with file:line. Prefer fewer high-confidence
findings over speculation. If a focus area is clean, do not invent a finding.`, { label: `review:${m.key}`, phase: 'Deep-read', schema: FINDINGS_SCHEMA, agentType: m.agentType }),

    (review, m) => parallel(
      (review?.findings || []).map((f, i) => () =>
        agent(`${BASE}

A reviewer flagged this finding in ${m.files}. Treat it as a HYPOTHESIS, not fact (per project
policy: reviewer findings must be probed against the actual code/library before acceptance).

FINDING: ${f.title}
LOCATION: ${f.file}
CATEGORY/SEVERITY: ${f.category} / ${f.severity}
EVIDENCE CLAIMED: ${f.evidence}
WHY IT SUPPOSEDLY MATTERS: ${f.why_it_matters}

TASK: Try to REFUTE it. Open the cited file:line and surrounding context. Check whether the claimed
behavior actually holds — read the relevant SpikeInterface/DataJoint/numpy API if the claim depends
on library behavior (you may inspect installed source or run a tiny check in the
spyglass_spikesorting_v2 conda env). Consider whether a guard, an earlier transform, or a test
already neutralizes it. Default to isReal=false / not-a-bug if you cannot positively confirm it from
the code. Be specific about what you read.`, { label: `verify:${m.key}#${i}`, phase: 'Deep-read', schema: VERDICT_SCHEMA })
          .then((v) => ({ ...f, unit: m.key, verdict: v }))
      )
    )
  ),

  // ---- PHASE 3: coverage-gap analysis (parallel by area) ----
  () => parallel(
    COVERAGE_AREAS.map((area) => () =>
      agent(`${BASE}

INTENDED-BEHAVIOR CONTRACT (from plan docs):
${contractText}

TEST INVENTORY (ground truth of what exists):
${testInventoryText}

TASK: Coverage-gap analysis for the area: "${area}".
Build the behavior x test mapping for this area: enumerate the behaviors and invariants the
contract requires here, then map each to the test(s) that actually exercise it (from the inventory;
open the test files to confirm what they ASSERT, not just that they run). Per project policy:
"verify parity/correctness" claims are bounded by what tests exercise + what audits found — so
enumerate the UNCOVERED paths explicitly; do not trust a green suite. Tests must enforce the
docstring/contract claim, not the current (possibly buggy) implementation. Flag:
 (1) behaviors with none/weak/partial coverage, especially edge cases: zero-units, gaps in
     valid_times, empty intervals, multi-shank, concat across sessions, idempotency/re-run,
     disjoint-interval readback, schema-default injection;
 (2) weak tests: assertion-light tests, tests that would still pass with a plausible bug present,
     over-broad tolerances, or tests asserting implementation rather than contract.
Return the schema.`, { label: `coverage:${area.slice(0, 24)}`, phase: 'Coverage', schema: COVERAGE_SCHEMA })
    )
  ),

  // ---- PHASE 4: v1 vs v2 parity study (review -> verify the risky divergences) ----
  () => pipeline(
    PARITY_AREAS,
    (area) => agent(`${BASE}

CONTRACT / DOCUMENTED DIVERGENCES (from plan docs):
${contractText}

TASK: Study v1 vs v2 for the area: "${area}".
First read the documented parity analysis: .claude/docs/plans/spikesorting-v2/feature-parity.md,
divergence-investigation.md, parity-extensions.md, overview.md, and CHANGELOG.md. Then compare the
ACTUAL source — v1 under src/spyglass/spikesorting/v1/ vs v2 under src/spyglass/spikesorting/v2/
(use the spyglass skill's code_graph.py describe / find-method / v0-v1 comparison helpers; read the
make()/insert paths, not just signatures). Produce TWO things:
 (1) divergences: every behavioral difference in this area. Classify each as justified
     (documented bugfix/improvement/design with sound rationale) or undocumented/unjustified/unclear,
     with the risk if unintended. Hunt SPECIFICALLY for UNDOCUMENTED divergences the plan docs and
     CHANGELOG do not mention.
 (2) improvement_gaps: v1 limitations / pain points that v2 was meant to (or arguably should)
     improve, and their delivery status (delivered / partial / stubbed-roadmap / not-addressed).
     Set should_do_now=true for gaps that matter before relying on v2 in production. The four stub
     modules (unit_matching/metric_curation/figpack_curation/matcher_protocol) are legitimately
     "stubbed-roadmap", NOT "not-addressed".
Return the schema.`, { label: `parity:${area.slice(0, 20)}`, phase: 'Parity', schema: PARITY_SCHEMA }),

    async (res, area) => {
      if (!res) return null
      const divs = res.divergences || []
      const risky = divs.filter((d) => RISKY.has(d.classification))
      const safe = divs.filter((d) => !RISKY.has(d.classification))
      const verified = await parallel(
        risky.map((d, i) => () =>
          agent(`${BASE}

A parity reviewer flagged a v1/v2 divergence in area "${area}" as "${d.classification}". Verify it.

ASPECT: ${d.aspect}
V1: ${d.v1_behavior}
V2: ${d.v2_behavior}
CLAIMED DOCUMENTATION: ${d.documented}
RISK IF UNINTENDED: ${d.risk}
FIRST-PASS ASSESSMENT: ${d.assessment}

TASK: Confirm against the ACTUAL v1 and v2 source that (a) the divergence is real, and (b) it truly
is undocumented/unjustified — search the plan docs and CHANGELOG for a justification the first pass
may have missed (per policy: a 'divergence' claim is a hypothesis until probed). Set isReal=false if
the divergence is not real OR is in fact documented/justified somewhere. Be specific about what you
read (cite file:line in both source and docs).`, { label: `parity-verify:${area.slice(0, 14)}#${i}`, phase: 'Parity', schema: VERDICT_SCHEMA })
            .then((v) => ({ ...d, verdict: v }))
        )
      )
      return { area, divergences: [...safe, ...verified.filter(Boolean)], improvement_gaps: res.improvement_gaps || [] }
    }
  ),
])

// ===========================================================================
// PHASE 5 — SYNTHESIZE
// ===========================================================================
phase('Synthesize')

// A verify agent returns null only if the user SKIPS it mid-run. A skipped
// verification must not silently drop a real finding (it would vanish) nor
// launder a risky divergence into "justified" -- such items are surfaced in
// explicit "unverified" buckets so the synthesis lead treats them as open.
const allFindings = deepRead.filter(Boolean).flat().filter(Boolean)
const verifiedFindings = allFindings.filter(
  (f) => f.verdict && f.verdict.isReal && f.verdict.corrected_severity !== 'not-a-bug'
)
const unverifiedFindings = allFindings.filter((f) => !f.verdict)

const crossCuttingFindings = (crossCutting?.findings || [])
const coverageResults = coverage.filter(Boolean)
const parityResults = parity.filter(Boolean)

// Classify divergences by their ORIGINAL classification (RISKY membership),
// not by presence of a verdict -- a skipped verdict on a risky divergence is
// "unverified", NOT "justified".
const allDivergences = parityResults.flatMap((p) =>
  (p.divergences || []).map((d) => ({ ...d, area: p.area }))
)
const concerningDivergences = allDivergences.filter(
  (d) => RISKY.has(d.classification) && d.verdict && d.verdict.isReal
)
const unverifiedRiskyDivergences = allDivergences.filter(
  (d) => RISKY.has(d.classification) && !d.verdict
)
const justifiedDivergences = allDivergences.filter((d) => !RISKY.has(d.classification))
const improvementGaps = parityResults.flatMap((p) =>
  (p.improvement_gaps || []).filter((g) => g.status !== 'delivered')
)

const report = await agent(`${BASE}

You are the synthesis lead. Produce the final review report (markdown) from these inputs.

VERIFIED CORRECTNESS/EFFICIENCY FINDINGS (already adversarially confirmed; JSON):
${JSON.stringify(verifiedFindings, null, 2)}

UNVERIFIED FINDINGS (verification was skipped — treat as OPEN, list under their group with an
"(unverified)" tag; do NOT silently drop; JSON):
${JSON.stringify(unverifiedFindings, null, 2)}

CROSS-CUTTING (non-v2) FINDINGS (JSON):
${JSON.stringify(crossCuttingFindings, null, 2)}

COVERAGE-GAP RESULTS (JSON):
${JSON.stringify(coverageResults, null, 2)}

CONCERNING v1/v2 DIVERGENCES (verified unjustified/undocumented/unclear; JSON):
${JSON.stringify(concerningDivergences, null, 2)}

UNVERIFIED RISKY DIVERGENCES (flagged risky but verification was skipped — treat as OPEN/concerning,
NOT justified; JSON):
${JSON.stringify(unverifiedRiskyDivergences, null, 2)}

JUSTIFIED v1/v2 DIVERGENCES (for the record; JSON):
${JSON.stringify(justifiedDivergences, null, 2)}

IMPROVEMENT GAPS OVER v1 (not yet delivered; JSON):
${JSON.stringify(improvementGaps, null, 2)}

TASK:
1. Dedup findings that point at the same root cause (cite all file:lines).
2. Rank everything by severity x confidence. Group as: (A) Correctness & silent-error bugs,
   (B) Efficiency, (C) Test coverage gaps & weak tests, (D) Cross-cutting & export/downstream risks,
   (E) v1/v2 divergences — lead with the CONCERNING ones and any UNVERIFIED-RISKY ones (each: aspect,
   v1 vs v2, why it's a problem, fix; tag the unverified ones "(unverified)"); then a short table of
   justified divergences for the record, (F) Improvement gaps over v1
   (status + should_do_now + what to do), keeping the four stub modules labeled stubbed-roadmap.
3. For each item give: a one-line title, file:line, severity, the evidence, and the concrete fix
   (or test to add). Keep it skimmable — a table or tight bullets per group.
4. Generalize: where multiple findings share a pattern, name the pattern once and list all sites.
5. Write a short top-of-report VERDICT (is the v2 pipeline correct / efficient / well-tested / at
   parity / export-safe enough to rely on?) and the TOP 5 highest-value actions. Include a short
   "Strengths" subsection naming what is genuinely well-done (solid patterns, well-tested areas) so
   the report is balanced, not just a defect list.
6. End with a "What was NOT checked / remaining risks" section (completeness critic): name any
   pipeline behavior, library assumption, test path, or export/downstream consumer no agent actually
   verified, and the next highest-value check.
Return the full markdown report as your message.`, { label: 'synthesize', phase: 'Synthesize' })

return {
  verifiedFindingCount: verifiedFindings.length,
  unverifiedFindingCount: unverifiedFindings.length,
  coverageAreaCount: coverageResults.length,
  concerningDivergenceCount: concerningDivergences.length,
  unverifiedRiskyDivergenceCount: unverifiedRiskyDivergences.length,
  improvementGapCount: improvementGaps.length,
  report,
}
