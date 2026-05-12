# Phase 4 — UnitMatch cross-session tracking

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#matcherparameters--unitmatch--trackedunit)

Adds **sort-then-match** cross-session unit tracking via UnitMatchPy. The design is pluggable: a `MatcherProtocol` interface accepts swappable backends (UnitMatch in this phase, DeepUnitMatch as future work). **Validation gate is the polymer probe** (Frank-lab standard, Chung et al. 2019); tetrode and Neuropixels validations run alongside but are informational, not gating.

**Phase 4 is split into two sub-phases** so the matcher's actual API and data flow are pinned BEFORE the v3 schema is finalized (per review feedback — UnitMatchPy's API surface is under-spiked for a zero-migration schema-final phase):

- **Phase 4a (technical spike)**: install UnitMatchPy, run it end-to-end against an existing v3 `SortingAnalyzer`, document the actual API surface, the input data layout it expects, and the failure modes. No DataJoint tables. Writes findings into `appendix.md § UnitMatchPy integration notes` (replacing the current speculative content). Confirms or revises the `MatcherProtocol` contract in `shared-contracts.md`.
- **Phase 4b (schema + implementation)**: locks in `MatcherParameters`, `UnitMatchSelection` (+ `MemberCuration` part), `UnitMatch` (+ `Pair` part), and `TrackedUnit` (+ `Member` part) based on what 4a discovered. Includes the polymer validation gate + informational tetrode/Neuropixels tests.

The matcher contract was overly strict in its earlier form. **Refined contract** (see updated `shared-contracts.md § MatcherProtocol`): a matcher consumes **pre-extracted per-unit waveform arrays + channel positions** that v3 derives from the `SortingAnalyzer` and writes into a matcher-specific on-disk layout (the exact layout is pinned by the Phase 4a spike — do not encode UnitMatchPy-specific directory or file names in shared-contracts before 4a runs). The "must not depend on raw recording data" invariant becomes: the v3 wrapper extracts what the matcher needs from the analyzer and feeds the matcher a self-contained directory; the wrapper never hands raw NWB paths to the matcher.

**Inputs to read first:**

- [src/spyglass/spikesorting/v3/session_group.py](src/spyglass/spikesorting/v3/session_group.py) (Phase 3) — the `SessionGroup` table is reused for per-session-then-match (not just concat).
- [src/spyglass/spikesorting/v3/sorting.py](src/spyglass/spikesorting/v3/sorting.py) (Phase 1) — `Sorting.get_analyzer(key)` feeds matcher inputs.
- [src/spyglass/spikesorting/v3/curation.py](src/spyglass/spikesorting/v3/curation.py) (Phase 1) — each session's curation is the input to matcher.
- [.claude/docs/plans/spikesorting-v3/appendix.md § UnitMatchPy integration notes](appendix.md#unitmatchpy-integration-notes) — API surface; the demo notebook in the upstream repo is the template.
- UnitMatchPy upstream demo: https://github.com/EnnyvanBeest/UnitMatch/blob/main/UMPy_spike_interface_demo.ipynb (commit hash to record at integration time).

**Contracts referenced:**

- [MatcherProtocol — cross-session unit matching plugin interface](shared-contracts.md#matcherprotocol--cross-session-unit-matching-plugin-interface) — Phase 4 implements both the registry and the first concrete backend.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `MatcherParameters` gets a per-matcher Pydantic dispatch.
- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — matcher reads `templates`, `waveforms`, `unit_locations` extensions.

**Designs referenced:** [MatcherParameters + UnitMatch + TrackedUnit](designs.md#matcherparameters--unitmatch--trackedunit).

## Tasks

### Phase 4a — Technical spike (no schema changes)

Output of this sub-phase is documentation + a working notebook, NOT new tables. The DataJoint surface waits until 4a's findings are written down.

- **Install UnitMatchPy in a v3 dev env**: `pip install UnitMatchPy>=3.3` (verify exact extras incantation). Document any install warts in `appendix.md § UnitMatchPy integration notes`.
- **Build a v3 SortingAnalyzer fixture for at least one Frank-lab session** (or use the synthetic Neuropixels fixture from Phase 4b's `conftest.py`). Sort completes; analyzer has `templates`, `waveforms`, `unit_locations` extensions.
- **Walk the actual UnitMatchPy API** end-to-end against that analyzer. Document:
  - The exact entry-point function name and signature (the current placeholder `um.MakeMatchTable(um_config)` may be stale).
  - The exact input data layout: file names, dtypes, shapes, where waveforms / channel positions / per-unit metadata live on disk.
  - Whether UnitMatchPy accepts pre-extracted waveforms (the contract we want) or whether it insists on re-reading raw data (the contract we cannot meet without breaking the matcher invariant).
  - The output format: pairwise probability matrix shape, FDR estimate column, drift-correction outputs.
  - Compute cost on the fixture (wall time, peak RSS).
- **Write the findings into `appendix.md § UnitMatchPy integration notes`**, REPLACING the current speculative content. Include the exact import statements and a minimal working code snippet (the v3 wrapper will be derived from this).
- **Reconcile with `shared-contracts.md § MatcherProtocol`**. The current contract says "matcher MUST NOT depend on raw recording data". If 4a finds UnitMatchPy genuinely requires raw data, EITHER:
  - revise the contract to say "the v3 wrapper preextracts waveforms from the analyzer and writes them in matcher-expected layout; raw paths are never handed to the matcher" (preferred), OR
  - mark UnitMatchPy as not-usable and document an alternative.
- **Output deliverables of 4a**: a notebook `notebooks/14a_UnitMatch_Spike.ipynb` that loads a v3 analyzer and runs UnitMatchPy end-to-end, plus the updated appendix and shared-contracts sections. NO new DataJoint tables. NO `unit_matching.py` written yet.

### Phase 4b — Schema + implementation (after 4a lands)

- **Implement `matcher_protocol.py`** — the protocol + registry per [shared-contracts.md § MatcherProtocol](shared-contracts.md#matcherprotocol--cross-session-unit-matching-plugin-interface) (revised by 4a). This is pure Python with no DataJoint dependency; tests should be importable standalone.

- **Implement `_unitmatch_backend.py`** — the UnitMatch wrapper. **Follow the API discovered in Phase 4a**; do not implement against speculative function names or input layouts. Reference: `appendix.md § UnitMatchPy integration notes` (rewritten at the end of Phase 4a to capture the actual API). Required behavior, independent of the specific API surface:
  - Subclass `MatcherProtocol`, `name = "unitmatch"`.
  - `match(session_analyzers, params) -> list[MatchPair]`:
    1. For each `SessionAnalyzer`, extract per-unit waveforms from `analyzer.get_extension("waveforms")` and the channel positions from `analyzer.get_channel_locations()` (or the equivalent surface 4a discovered).
    2. Write the matcher's expected input layout to a temp directory (the exact files / dtypes / shapes are pinned by 4a).
    3. Call the actual UnitMatchPy entry point (name + signature pinned by 4a).
    4. Parse the result table into `MatchPair` objects, preserving session keys.
  - Handle the **degenerate single-session case** — return `[]` immediately, no UnitMatch call.

- **Implement `_concat_identity_backend.py`** — for SessionGroups whose sorting came from `ConcatenatedRecording` (Phase 3). The "matches" are identity-mappings: every unit ID in the concat sort corresponds to itself across sessions. This is a degenerate but legitimate matcher case so users on the Phase 3 chronic path get the same `TrackedUnit` downstream interface.
  - `name = "concat_identity"`.
  - `match()` walks each session's sorting (which is a split of the concat sort via `ConcatenatedRecording.split_sorting_by_session()`) and emits one identity pair per unit per session pair.
  - Match probability = 1.0 by definition; FDR = 0.0; drift = 0.0.

- **Implement `unit_matching.py`** per [designs.md § MatcherParameters + UnitMatch + TrackedUnit](designs.md#matcherparameters--unitmatch--trackedunit):
  - `MatcherParameters` Lookup (Pydantic-validated per-matcher; default rows for `unitmatch_default`, `concat_identity_default`). The Pydantic model includes a `tracked_unit_policy: Literal["strict", "transitive"] = "strict"` field and a `tracked_unit_threshold: float = 0.5` field.
  - `UnitMatchSelection` Manual **with a `MemberCuration` part table** that explicitly pins one `(sorting_id, curation_id)` per `SessionGroup.Member`. NO implicit "latest curation" lookup; this is a load-bearing reproducibility decision. The plan-level rationale: if matching silently picked the latest curation, adding a new curation to one source session would invalidate any prior `UnitMatch` row that referenced it without making the change visible.
  - `UnitMatch` Computed with `Pair` part. `make()` dispatches via `get_matcher(matcher_name).match(...)` using the explicitly-pinned curations from `UnitMatchSelection.MemberCuration`.
  - `TrackedUnit` Computed with `Member` part. `make()` reads `tracked_unit_policy` and `tracked_unit_threshold` from `MatcherParameters` and dispatches to either `_derive_tracked_units_strict` (default: maximal cliques) or `_derive_tracked_units_transitive` (connected components, with `n_transitive_only_edges` reported per component). See [designs.md § Algorithm for `TrackedUnit.make()`](designs.md#matcherparameters--unitmatch--trackedunit) for the algorithm.

- **Helper for building `UnitMatchSelection.MemberCuration` rows**: `UnitMatchSelection.insert_selection(session_group_name, matcher_params_name, curation_choices: dict[member_index, curation_key]) -> dict`. The `curation_choices` argument maps each member's `member_index` to an explicit `{"sorting_id": ..., "curation_id": ...}` key. Helper validates that every member has a choice and raises clearly if any are missing. Returns the unitmatch_id PK dict per the [shared-contracts insert_selection convention](shared-contracts.md#insert_selection-return-value-normalization).

- **Primary validation gate is polymer probes** (the Frank-lab standard, [Chung et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30502044/)) — not tetrode and not Neuropixels. The gating test is `test_v3_unitmatch_polymer_mearec_ground_truth`:
  - Uses a new Phase 4b fixture `mearec_polymer_2sessions.nwb` (4-shank polymer probe; same probe-interface JSON as `mearec_polymer_60s.nwb` from Phase 0; two sessions generated from the same MEArec template set with different `seeds.spiketrain` and a small inter-session drift).
  - Runs v3 sort + curation on both sessions, runs UnitMatch, computes ROC of match probability vs ground-truth template correspondence.
  - **Pass criterion**: AUC > 0.85.

- **Neuropixels validation is supplementary** (not a gate). `test_v3_unitmatch_neuropixels_mearec_ground_truth` runs against `mearec_neuropixels_2sessions.nwb` and records AUC in `docs/src/Pipelines/SpikeSorting/v3-validation-notes.md`. Provides independent verification of UnitMatch's published Neuropixels validation against the v3 wrapper — useful confirmation, but not what gates Phase 4 shipping.

- **Tetrode validation is informational only** (not a gate). Same shape as Neuropixels supplementary — `test_v3_unitmatch_tetrode_mearec_ground_truth` runs against `mearec_tetrode_2sessions.nwb` and records AUC in the same notes doc. Multi-day tetrode is not a current Frank-lab use case, so this is purely documentation for any future evaluation.

- **Optional real-data supplementary check**: if `SPIKESORTING_V3_REAL_NWB_PATH` is set AND points to a multi-session dataset with manual cross-session correspondences, `validate_unitmatch.py` (standalone CLI script, not a pytest test) reports AUC on the real data alongside the MEArec AUC. Provides empirical real-world confirmation but is NOT what gates Phase 4 shipping.

- **Validation in code (`make()`)**: in `UnitMatch.make()`, log an INFO message naming the probe type detected (polymer / Neuropixels / tetrode) so users see which validation regime applies; link to `docs/src/Pipelines/SpikeSorting/v3-validation-notes.md` where per-probe AUC results are recorded.

- **Documentation update**:
  - New section in `docs/src/Pipelines/SpikeSorting/v3.md` titled "Cross-session unit tracking".
  - New `docs/src/Pipelines/SpikeSorting/v3-validation-notes.md` documenting the polymer / tetrode / Neuropixels AUC results from the three MEArec fixtures.
  - CHANGELOG.md: "v3 cross-session unit tracking via UnitMatch. Polymer probe is the validated path (Frank-lab standard); Neuropixels + tetrode AUC recorded as informational. `TrackedUnit` derives biological-unit identities across sessions."

## Deliberately not in this phase

- **DeepUnitMatch**. The `MatcherProtocol` design accepts it as a future plugin; Phase 4.1+ implements `_deepunitmatch_backend.py` separately. Not in this PR.
- **Drift correction inside the matcher**. UnitMatch does its own rigid-shift estimation; v3 does not pre-correct drift on inputs to UnitMatch.
- **Replacing multi-day concat with UnitMatch as a workaround**. Phase 3 supports multi-day concat behind `allow_multi_day=True` (with explicit motion preset). Phase 4 is the **recommended** cross-day workflow, but does not remove Phase 3's opt-in path — they coexist.
- **Cross-probe matching**. UnitMatch assumes one probe across sessions in a group. Multi-probe is out of scope.
- **Curation propagation across sessions**. Phase 4 produces match pairs; if a user manually curates session A's unit 5 as `noise`, that label does NOT propagate to session B's matched unit. Curation-propagation tooling is a future feature.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_matcher_protocol_registry` | `register_matcher`-decorated class is findable via `get_matcher(name)`; unknown name raises. |
| `test_concat_identity_backend_basic` | Synthesize a `SessionGroup` with 2 members backed by a `ConcatenatedRecording` Phase 3 sort; `concat_identity` matcher returns N pair rows (one per unit) with probability=1.0. |
| `test_unitmatch_backend_single_session_degenerate` | A `SessionGroup` with 1 Member produces 0 `MatchPair`s; no UnitMatch call attempted. |
| `test_unitmatch_backend_two_sessions_synthetic` (slow) | Synthetic 2-session Neuropixels-shaped fixture (16-channel sort group, 5 units per session, half are "same neuron"): UnitMatch returns match probabilities high (>0.7) for true positives and low (<0.3) for random pairs. |
| `test_unit_match_make_writes_part_rows` (slow) | After `UnitMatch.populate()`, `UnitMatch.Pair & key` has expected number of rows. |
| `test_tracked_unit_strict_clique_basic` | Synthetic pair list with three sessions where ALL pairwise edges between three units exceed threshold (true clique); `TrackedUnit.make()` in default strict mode produces exactly 1 component containing all three. |
| `test_tracked_unit_strict_default_rejects_transitive` | Pairs A↔B (high), B↔C (high), A↔C (low). Default `tracked_unit_policy="strict"` (maximal cliques) produces ≥2 components (NOT 1) — A and C cannot be lumped without a direct above-threshold edge. Test asserts >1 component and asserts no component contains both A and C. |
| `test_tracked_unit_transitive_opt_in_unifies` | Same input as above; with `tracked_unit_policy="transitive"`, connected-components yields 1 component with `n_transitive_only_edges == 1` (the A↔C edge missing). |
| `test_probe_type_logged` | `UnitMatch.make()` emits an INFO log naming the detected probe type (polymer / Neuropixels / tetrode) so users see which validation regime applies. |
| `test_v3_unitmatch_neuropixels_smoke` (slow, optional) | Skipped unless `--run-neuropixels` is passed; uses a small Neuropixels fixture if available. |

## Fixtures

- **`mearec_polymer_2sessions.nwb` pair** (gating fixture) — 4-shank polymer probe (same geometry as `mearec_polymer_60s.nwb` from Phase 0); two sessions generated from the same MEArec template set with different `seeds.spiketrain` and a small inter-session drift. Planted shared templates → known cross-session correspondences. Generated by extending `tests/spikesorting/v3/fixtures/generate_mearec.py` in Phase 4b.
- **`mearec_neuropixels_2sessions.nwb` pair** (informational fixture) — same shape as the polymer pair but for Neuropixels-128. Used by the supplementary Neuropixels test.
- **`mearec_tetrode_2sessions.nwb` pair** (informational fixture) — linear tetrode probe. Used by the informational tetrode test; results documented in `v3-validation-notes.md`.
- All three are MEArec-generated; no user-provided gold-standard dataset is required to land Phase 4. An optional real-data path remains available via `SPIKESORTING_V3_REAL_NWB_PATH` for supplementary validation.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — DeepUnitMatch is not in this PR.
- Validation slice tests pass; slow / integration tests are marked.
- The synthetic Neuropixels test produces real UnitMatch output (not a mock).
- The MEArec-based **polymer** validation test (`test_v3_unitmatch_polymer_mearec_ground_truth`) runs in CI and passes its AUC > 0.85 criterion — this is the gate. If it fails, Phase 4 does not ship; the implementer escalates rather than relaxing the threshold. The informational tetrode and Neuropixels tests (`test_v3_unitmatch_tetrode_mearec_ground_truth`, `test_v3_unitmatch_neuropixels_mearec_ground_truth`) run alongside and record their AUCs in `docs/src/Pipelines/SpikeSorting/v3-validation-notes.md` regardless of value.
- `MatcherProtocol` is implementable by external code without touching v3 internals (verify by writing a 10-line dummy matcher in the test suite).
- `TrackedUnit` graph algorithm matches the binding policy in `designs.md` — strict (maximal cliques) by default, transitive only with opt-in via `MatcherParameters.params["tracked_unit_policy"]`. Tests `test_tracked_unit_strict_clique_basic`, `test_tracked_unit_strict_default_rejects_transitive`, and `test_tracked_unit_transitive_opt_in_unifies` exercise all three branches.
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- `unitmatchpy` is gated as an optional dependency (`pip install -e ".[spikesorting-v3-matching]"`). Import-time guard in `_unitmatch_backend.py` raises `ImportError` with the install command if missing.
