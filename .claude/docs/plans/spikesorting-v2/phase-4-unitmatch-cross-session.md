# Phase 4 — UnitMatch cross-session tracking

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#matcherparameters--unitmatch--trackedunit)

Adds **sort-then-match** cross-session unit tracking via UnitMatchPy. The design is pluggable: a `MatcherProtocol` interface accepts swappable backends (UnitMatch in this phase, DeepUnitMatch as future work). **Validation gate is the polymer probe** (Frank-lab standard, Chung et al. 2019).

**Phase 4 is split into two sub-phases** so the matcher's actual API and data flow are pinned BEFORE the v2 schema is finalized:

- **Phase 4a (technical spike)**: install UnitMatchPy, run it end-to-end against an existing v2 `SortingAnalyzer` plus the associated v2 recording when split-half waveform extraction requires traces, document the actual API surface, the input data layout it expects, and the failure modes. No DataJoint tables. Writes findings into `appendix.md § UnitMatchPy integration notes` and replaces the `PHASE4A_CONTRACT_STUB` markers in `shared-contracts.md` and `designs.md`.
- **Phase 4b (schema + implementation)**: locks in `MatcherParameters`, `UnitMatchSelection` (+ `MemberCuration` part), `UnitMatch` (+ `Pair` part), and `TrackedUnit` (+ `Member` part) based on what 4a discovered. Includes the polymer validation gate.

**Matcher contract** (see updated `shared-contracts.md § MatcherProtocol`): a matcher consumes **pre-extracted per-unit waveform arrays + channel positions** that v2 derives from the `SortingAnalyzer` and, if needed, wrapper-owned reads of the v2 `Recording` artifact. It writes those arrays into a matcher-specific on-disk layout (the exact layout is pinned by the Phase 4a spike — do not encode UnitMatchPy-specific directory or file names in shared-contracts before 4a runs). The v2 wrapper extracts what the matcher needs and feeds the matcher a self-contained directory; the matcher never receives raw NWB paths, `si.SortingAnalyzer` objects, or Spyglass table keys.

## Executor Checklist

Phase 4a PR:

- Run UnitMatchPy end-to-end outside DataJoint and replace the appendix / shared-contracts / designs contract stubs with observed API details.
- Reconcile `shared-contracts.md` and `designs.md` with the actual matcher input/output layout.
- Do not add DataJoint tables or `unit_matching.py` in 4a.

Phase 4b PR:

- Do not start 4b while `PHASE4A_CONTRACT_STUB` remains; run the grep gate in "Commands to run" first.
- Implement `matcher_protocol.py`, `_unitmatch_backend.py`, `unit_matching.py`, and Phase 4 parameter schemas from the 4a findings.
- Enforce explicit per-member curation choices and make-time provenance guards against schema-bypassing inserts.
- Implement strict tracked-unit derivation with a bounded graph-size cap.
- Gate shipping on the polymer MEArec AUC test.
- Run the Phase 4 validation slice plus `code_graph.py describe/path` for new tables.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/session_group.py](../../../../src/spyglass/spikesorting/v2/session_group.py) (Phase 3) — the `SessionGroup` table is reused for per-session-then-match (not just concat).
- [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) (Phase 1) — `Sorting.get_analyzer(key)` feeds matcher inputs.
- [src/spyglass/spikesorting/v2/curation.py](../../../../src/spyglass/spikesorting/v2/curation.py) (Phase 1) — each session's curation is the input to matcher.
- [.claude/docs/plans/spikesorting-v2/appendix.md § UnitMatchPy integration notes](appendix.md#unitmatchpy-integration-notes) — API surface; the demo notebook in the upstream repo is the template.
- UnitMatchPy upstream demo: https://github.com/EnnyvanBeest/UnitMatch/blob/main/UnitMatchPy/Demo%20Notebooks/UMPy_spike_interface_demo.ipynb (commit hash to record at integration time).

**Contracts referenced:**

- [MatcherProtocol — cross-session unit matching plugin interface](shared-contracts.md#matcherprotocol--cross-session-unit-matching-plugin-interface) — Phase 4 implements both the registry and the first concrete backend.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `MatcherParameters` gets a per-matcher Pydantic dispatch.
- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — matcher reads `templates`, `waveforms`, `unit_locations` extensions.

**Designs referenced:** [MatcherParameters + UnitMatch + TrackedUnit](designs.md#matcherparameters--unitmatch--trackedunit).

## Tasks

### Phase 4a — Technical spike (no schema changes)

Output of this sub-phase is documentation + a working notebook, NOT new tables. The DataJoint surface waits until 4a's findings are written down.

- **Install UnitMatchPy in a v2 dev env**: `pip install UnitMatchPy>=3.3,<4` (verify exact extras incantation). Document resolver warts in `appendix.md § UnitMatchPy integration notes`, including the current PyPI constraints (`python>=3.9,<3.13`, `numpy<2.0`) against the v2 SpikeInterface environment.
- **Build a v2 SortingAnalyzer fixture for at least one Frank-lab session** (or use the synthetic Neuropixels fixture from Phase 4b's `conftest.py`). Sort completes; analyzer has `templates`, `waveforms`, `unit_locations` extensions.
- **Walk the actual UnitMatchPy API** end-to-end against that analyzer. Document:
  - The exact entry-point function name and signature (the current example `um.MakeMatchTable(um_config)` may be stale).
  - The exact input data layout: file names, dtypes, shapes, where `RawWaveforms`, channel positions, and per-unit metadata live on disk.
  - Whether v2 can produce UnitMatchPy's required split-half average waveforms from existing `SortingAnalyzer` extensions alone. If not, document the wrapper-owned trace extraction from `Sorting.get_recording()` / `ConcatenatedRecording.get_recording()` needed to create the two cross-validation halves. This is acceptable only if the matcher still consumes the prepared bundle and never receives raw NWB paths or Spyglass table keys.
  - The output format: pairwise probability matrix shape, FDR estimate column, drift-correction outputs.
  - Compute cost on the fixture (wall time, peak RSS).
- **Write the findings into `appendix.md § UnitMatchPy integration notes`** and replace the `PHASE4A_CONTRACT_STUB` markers in `shared-contracts.md` and `designs.md`. Include the exact import statements and a minimal working code snippet (the v2 wrapper will be derived from this).
- **Reconcile with `shared-contracts.md § MatcherProtocol`**. The current contract says "matcher MUST NOT depend on raw recording data". If 4a finds UnitMatchPy genuinely requires raw data, EITHER:
  - revise the contract to say "the v2 wrapper preextracts waveforms from the analyzer and writes them in matcher-expected layout; raw paths are never handed to the matcher" (preferred), OR
  - mark UnitMatchPy as not-usable and document an alternative.
- **Output deliverables of 4a**: a notebook `notebooks/14a_UnitMatch_Spike.ipynb` that loads a v2 analyzer and runs UnitMatchPy end-to-end, plus the updated appendix and shared-contracts sections. NO new DataJoint tables. NO `unit_matching.py` written yet.

### Phase 4b — Schema + implementation (after 4a lands)

- **Implement `matcher_protocol.py`** — the protocol + registry per [shared-contracts.md § MatcherProtocol](shared-contracts.md#matcherprotocol--cross-session-unit-matching-plugin-interface) (revised by 4a). This is pure Python with no DataJoint dependency; tests should be importable standalone.

- **Implement `_unitmatch_backend.py`** — the UnitMatch wrapper. **Follow the API discovered in Phase 4a**; do not implement against pre-4a example function names or input layouts. Reference: `appendix.md § UnitMatchPy integration notes` (rewritten at the end of Phase 4a to capture the actual API). Required behavior, independent of the specific API surface:
  - Subclass `MatcherProtocol`, `name = "unitmatch"`.
  - `match(session_inputs: list[SessionMatcherInput], params) -> list[MatchPair]`:
    1. The wrapper (in `UnitMatch.make()`, NOT the matcher) has already pre-extracted UnitMatch-compatible split-half waveforms + channel positions from each session's curated sorting/analyzer/recording and written them into `session_input.waveform_dir` / `session_input.channel_positions_path` in the matcher's expected layout. **The matcher does not touch raw NWB paths, Spyglass table keys, or `si.SortingAnalyzer` objects directly** (shared-contracts.md § MatcherProtocol invariant).
    2. Read the bundle dirs (exact files / dtypes / shapes are pinned by 4a).
    3. Call the actual UnitMatchPy entry point (name + signature pinned by 4a).
    4. Parse the result table into `MatchPair` objects, preserving session keys (`(sorting_id, curation_id, unit_id)` on both sides).
  - Handle the **degenerate single-session case** — return `[]` immediately, no UnitMatch call.

- **Implement `unit_matching.py`** per [designs.md § MatcherParameters + UnitMatch + TrackedUnit](designs.md#matcherparameters--unitmatch--trackedunit):
  - `MatcherParameters` Lookup (Pydantic-validated per-matcher; default row `unitmatch_default`). The Pydantic model includes:
    - `tracked_unit_threshold: float = 0.5` — pair-probability cutoff.
    - `max_strict_nodes: int = 2000` — hard upper bound on graph size submitted to strict `networkx.find_cliques`. Exceeding raises `TrackedUnitBudgetExceededError`. No fallback policy and no time budget in v1; future policy values are pure inserts into `TrackedUnit.policy_used: varchar(32)`.
    - **`MatcherParameters.insert1()` validates the `matcher` string against the registered matcher registry before inserting.** Unknown names raise `UnknownMatcherError`. The same registry dispatches the per-matcher Pydantic schema for `params` validation, so a typo is caught at insert time. See [designs.md § MatcherParameters](designs.md#matcherparameters--unitmatch--trackedunit).
  - `UnitMatchSelection` Manual **with a `MemberCuration` part table** that explicitly pins one `(sorting_id, curation_id)` per `SessionGroup.Member`. The master row stores `curation_set_hash`, a SHA-256 hash over the canonical ordered list of `(member_index, sorting_id, curation_id)` choices, so `insert_selection()` can be idempotent under the shared insert-selection contract. NO implicit "latest curation" lookup.
  - `UnitMatch` Computed with `Pair` part. `make()` dispatches via `get_matcher(matcher_name).match(...)` using the explicitly-pinned curations from `UnitMatchSelection.MemberCuration`. Before extracting matcher inputs, `make()` re-validates the direct-insert bypass invariant: the part rows must exactly cover the parent `SessionGroup.Member` set, and each pinned `CurationV2` row must belong to that member's session/recording path. A schema-valid but provenance-invalid row raises `UnitMatchSelectionIntegrityError` with a message pointing users back to `UnitMatchSelection.insert_selection()`.
  - `TrackedUnit` Computed with `Member` part. `make()` reads `tracked_unit_threshold` and `max_strict_nodes` from `MatcherParameters`, seeds the graph with the complete curated-unit universe, derives maximal cliques in strict mode only, and raises `TrackedUnitBudgetExceededError` if the graph exceeds the node cap. See [designs.md § Algorithm for `TrackedUnit.make()`](designs.md#matcherparameters--unitmatch--trackedunit) for the algorithm.

- **Helper for building `UnitMatchSelection.MemberCuration` rows**: `UnitMatchSelection.insert_selection(session_group_owner, session_group_name, matcher_params_name, curation_choices: dict[member_index, curation_key]) -> dict`. The `curation_choices` argument maps each member's `member_index` to an explicit `{"sorting_id": ..., "curation_id": ...}` key. Helper validates that every member has exactly one choice, raises clearly if any are missing or extra, and verifies each chosen `CurationV2` belongs to that member's session/recording path. A curation from member B must never be accepted for member A just because it satisfies the independent `CurationV2` FK. It computes `curation_set_hash = sha256(json.dumps(sorted_choices, sort_keys=True))`, finds or inserts the master row by `(session_group_owner, session_group_name, matcher_params_name, curation_set_hash)`, inserts the part rows inside the same transaction for new selections, and returns the `unitmatch_id` PK dict per the [shared-contracts insert_selection convention](shared-contracts.md#insert_selection-return-value-normalization).

- **Primary validation gate is polymer probes** (the Frank-lab standard, [Chung et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30502044/)) — not tetrode and not Neuropixels. The gating test is `test_v2_unitmatch_polymer_mearec_ground_truth`:
  - Uses a new Phase 4b fixture `mearec_polymer_2sessions.nwb` (4-shank polymer probe; same probe-interface JSON as `mearec_polymer_60s.nwb` from Phase 0; two sessions generated from the same MEArec template set with different `seeds.spiketrain` and a small inter-session drift).
  - Runs v2 sort + curation on both sessions, runs UnitMatch, computes ROC of match probability vs ground-truth template correspondence.
  - **Pass criterion**: AUC > 0.85.

- **Documentation update**:
  - New section in `docs/src/Features/SpikeSortingV2.md` titled "Cross-session unit tracking".
  - CHANGELOG.md: "v2 cross-session unit tracking via UnitMatch. Polymer probe is the validated path (Frank-lab standard). `TrackedUnit` derives biological-unit identities across sessions."

## Deliberately not in this phase

- **DeepUnitMatch**. The `MatcherProtocol` design accepts it as a future plugin; Phase 4.1+ implements `_deepunitmatch_backend.py` separately. Not in this PR.
- **Concat identity backend.** A concat-backed Phase 3 sort has one `CurationV2` row for the concatenated recording, while Phase 4's `UnitMatchSelection.MemberCuration` intentionally pins one curation per `SessionGroup.Member`. Supporting identity matches for concat sorts needs a distinct concat-backed selection path or helper surface; defer it rather than weakening the per-member curation invariant.
- **Drift correction inside the matcher**. UnitMatch does its own rigid-shift estimation; v2 does not pre-correct drift on inputs to UnitMatch.
- **Replacing multi-day concat with UnitMatch as a workaround**. Phase 3 supports multi-day concat behind `allow_multi_day=True` (with explicit motion preset). Phase 4 is the **recommended** cross-day workflow, but does not remove Phase 3's opt-in path — they coexist.
- **Cross-probe matching**. UnitMatch assumes one probe across sessions in a group. Multi-probe is out of scope.
- **Curation propagation across sessions**. Phase 4 produces match pairs; if a user manually curates session A's unit 5 as `noise`, that label does NOT propagate to session B's matched unit. Curation-propagation tooling is a future feature.
- **Supplementary Neuropixels, tetrode, and real-data validation.** These are useful follow-up PRs after the polymer-gated implementation lands. They should add their own fixtures, validation notes, and opt-in test markers without blocking the first UnitMatch implementation.

## Validation goals

Behaviors the Phase 4b validation slice must cover. Name and split tests as the implementer sees fit; each goal must have at least one assertion exercising it.

1. **Matcher registry typo-at-insert**: unknown matcher names raise `UnknownMatcherError` at `MatcherParameters.insert1`, before the row commits; the message names registered matchers and `register_matcher()`. Per-matcher Pydantic dispatch validates `params` against the registered model.
2. **Degenerate single-session matcher**: a `SessionGroup` with one Member produces zero `MatchPair`s; no UnitMatch backend call attempted.
3. **Synthetic two-session correctness** (slow): on a planted-correspondence fixture, true positives score above threshold; random pairs score below.
4. **`UnitMatchSelection.insert_selection` idempotency**: identical inputs return the same `unitmatch_id`; changing any member's curation produces a different `curation_set_hash` and `unitmatch_id`.
5. **MemberCuration ownership**: a curation belonging to a different SessionGroup.Member is rejected by `insert_selection` before any part row is inserted; the master + part inserts are atomic on failure.
6. **`UnitMatch.make()` integrity recheck**: direct-inserted Selection rows with member coverage gaps or wrong-member CurationV2 raise `UnitMatchSelectionIntegrityError` before matcher inputs are extracted; no `UnitMatch` or `Pair` rows are created.
7. **Pair FK integrity**: `UnitMatch.Pair` rows that reference a unit absent from the pinned `CurationV2.Unit` are rejected by DataJoint.
8. **TrackedUnit strict-clique basic**: a 3-session clique with all pairwise edges above threshold yields exactly 1 component containing all three; pairwise A↔B / B↔C high but A↔C low yields ≥2 components, and no component contains both A and C.
9. **TrackedUnit unmatched singleton**: a unit with only below-threshold pairs surfaces as a singleton row (`n_sessions_observed == 1`, `median_match_probability IS NULL`, `policy_used == "strict"`).
10. **TrackedUnit bounded-search cap**: a synthetic graph with `max_strict_nodes` smaller than the graph raises `TrackedUnitBudgetExceededError`; under-cap input succeeds and every row carries `policy_used = 'strict'`.

Polymer gating fixture: `test_v2_unitmatch_polymer_mearec_ground_truth` (described above) is the gate — AUC > 0.85 on `mearec_polymer_2sessions.nwb`.

## Commands to run

### Phase 4a commands

Run these in the v2 matching environment. Phase 4a is documentation plus a working notebook; it does not run `code_graph.py` on new tables because no tables are added.

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

python -m pip check
python -m pip show UnitMatchPy

jupyter nbconvert --to notebook --execute notebooks/14a_UnitMatch_Spike.ipynb --output /tmp/14a_UnitMatch_Spike.executed.ipynb
git diff --check -- .claude/docs/plans/spikesorting-v2/appendix.md .claude/docs/plans/spikesorting-v2/shared-contracts.md .claude/docs/plans/spikesorting-v2/designs.md notebooks/14a_UnitMatch_Spike.ipynb
```

### Phase 4b commands

Phase 4b is blocked until the 4a grep gate finds no remaining contract-stub marker.

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

! rg -n "PHASE4A_CONTRACT_STUB" \
  .claude/docs/plans/spikesorting-v2/shared-contracts.md \
  .claude/docs/plans/spikesorting-v2/appendix.md \
  .claude/docs/plans/spikesorting-v2/designs.md

pytest tests/spikesorting/v2/test_phase4_unitmatch.py -q

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe MatcherParameters --file spyglass/spikesorting/v2/unit_matching.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe UnitMatchSelection --file spyglass/spikesorting/v2/unit_matching.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe UnitMatch --file spyglass/spikesorting/v2/unit_matching.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe TrackedUnit --file spyglass/spikesorting/v2/unit_matching.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up UnitMatch --file spyglass/spikesorting/v2/unit_matching.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --down TrackedUnit --file spyglass/spikesorting/v2/unit_matching.py --json

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 docs/src/Features notebooks CHANGELOG.md
```

## Fixtures

- **`mearec_polymer_2sessions.nwb` pair** (gating fixture) — 4-shank polymer probe (same geometry as `mearec_polymer_60s.nwb` from Phase 0); two sessions generated from the same MEArec template set with different `seeds.spiketrain` and a small inter-session drift. Planted shared templates → known cross-session correspondences. Generated by extending `tests/spikesorting/v2/fixtures/generate_mearec.py` in Phase 4b.
- No user-provided gold-standard dataset is required to land Phase 4.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — DeepUnitMatch is not in this PR.
- The "Deliberately not in this phase" list is honored — concat identity matching is not in this PR.
- Validation slice tests pass; slow / integration tests are marked.
- The synthetic two-session test produces real UnitMatch output (not a mock).
- The MEArec-based **polymer** validation test (`test_v2_unitmatch_polymer_mearec_ground_truth`) runs in CI and passes its AUC > 0.85 criterion — this is the gate. If it fails, Phase 4 does not ship; the implementer escalates rather than relaxing the threshold.
- `MatcherProtocol` is implementable by external code without touching v2 internals (verify by writing a 10-line dummy matcher in the test suite).
- `TrackedUnit` graph algorithm matches the binding policy in `designs.md` — strict maximal cliques only, singleton units preserved, and `TrackedUnitBudgetExceededError` raised when the graph exceeds `max_strict_nodes`.
- `UnitMatchSelection.insert_selection()` is idempotent by `curation_set_hash`, validates member/curation ownership before insertion, and inserts master + part rows atomically.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- `unitmatchpy` is gated as an optional dependency (`pip install -e ".[spikesorting-v2-matching]"`). Import-time guard in `_unitmatch_backend.py` raises `ImportError` with the install command if missing.
