# Phase 4 — UnitMatch cross-session tracking

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#matcherparameters--unitmatch--trackedunit)

Adds **sort-then-match** cross-session unit tracking via UnitMatchPy. The design is pluggable: a `MatcherProtocol` interface accepts swappable backends (UnitMatch in this phase, DeepUnitMatch as future work). Includes an **explicit tetrode validation gate** before declaring tetrode support production-ready.

**Phase 4 is split into two sub-phases** so the matcher's actual API and data flow are pinned BEFORE the v3 schema is finalized (per review feedback — UnitMatchPy's API surface is under-spiked for a zero-migration schema-final phase):

- **Phase 4a (technical spike)**: install UnitMatchPy, run it end-to-end against an existing v3 `SortingAnalyzer`, document the actual API surface, the input data layout it expects, and the failure modes. No DataJoint tables. Writes findings into `appendix.md § UnitMatchPy integration notes` (replacing the current speculative content). Confirms or revises the `MatcherProtocol` contract in `shared-contracts.md`.
- **Phase 4b (schema + implementation)**: locks in `MatcherParameters`, `UnitMatchSelection` (+ `MemberCuration` part), `UnitMatch` (+ `Pair` part), and `TrackedUnit` (+ `Member` part) based on what 4a discovered. Includes the tetrode validation gate.

The matcher contract was overly strict in its earlier form. **Refined contract** (see updated `shared-contracts.md § MatcherProtocol`): a matcher consumes **pre-extracted per-unit waveform arrays + channel positions** that v3 derives from the `SortingAnalyzer` and writes into a matcher-specific on-disk layout (this is what UnitMatchPy's `KSDir`/`RawDataDir` style inputs actually want — pre-extracted artifacts, not raw recordings). The "must not depend on raw recording data" invariant becomes: the v3 wrapper extracts what the matcher needs from the analyzer and feeds the matcher a self-contained directory; the wrapper never hands raw NWB paths to the matcher.

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

- **Implement `_unitmatch_backend.py`** — the UnitMatch wrapper.
  - Subclass `MatcherProtocol`, `name = "unitmatch"`.
  - `match(session_analyzers, params) -> list[MatchPair]`:
    1. For each `SessionAnalyzer`, extract template waveforms with two-halves split (UnitMatch needs this for cross-validation). Use `analyzer.get_extension("waveforms").get_unit_waveforms(unit_id)` and split each unit's waveforms into two halves by index.
    2. Build the input dict for UnitMatch's `MakeMatchTable()`:
       - `KSDir` — per-session output directories (one per session; UnitMatch reads these for cluster info).
       - `RawDataDir` — actually not strictly required if waveforms are pre-extracted.
       - `WaveformParameters` — derived from analyzer parameters.
    3. Pre-extract waveforms to a temp directory in the UnitMatch-expected layout (`channel_pos.npy`, `RawWaveforms/Unit{id}_RawSpikes.npy` per session).
    4. Call `um.MakeMatchTable(um_config)`.
    5. Parse the result table into `MatchPair` objects, preserving session keys.
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

- **Tetrode validation gate.** A standalone validation script `tests/spikesorting/v3/validate_tetrode_unitmatch.py` (not a pytest test — invoked manually with a user-provided tetrode dataset):
  - Input: path to a multi-day tetrode dataset with manually-curated cross-day unit correspondences (gold standard). Lab provides during Phase 4 implementation.
  - Runs v3 sort + curation on each day independently.
  - Runs UnitMatch via the Phase 4 backend.
  - Computes ROC of match probability vs gold-standard correspondence:
    - Self-matches (gold positive): within-day cross-validation positives.
    - Random pairs (gold negative): pairs that gold annotates as different neurons.
  - Reports AUC + suggested threshold.
  - **Pass criterion**: AUC > 0.85 on the user's gold-standard dataset, OR documented otherwise as "tetrode matcher unreliable; route via concat for tetrode chronic".

  If the script fails the pass criterion, Phase 4 STILL ships — but:
  - `MatcherParameters` documentation marks `unitmatch` as "validated on Neuropixels; tetrode users should prefer `concat_identity` via Phase 3".
  - The default preset bundle for tetrodes (in Phase 5 `PRESETS` dict) routes through Phase 3 concat, not UnitMatch.

- **Validation in code (`make()`)**: in `UnitMatch.make()`, log a warning if `session_analyzers[0].analyzer.get_extension("templates")` has fewer than 16 channels per unit (a heuristic that catches tetrode usage). The warning includes a pointer to the tetrode validation document.

- **Documentation update**:
  - New section in `docs/src/Pipelines/SpikeSorting/v3.md` titled "Cross-session unit tracking".
  - New `docs/src/Pipelines/SpikeSorting/v3-tetrode-notes.md` documenting the tetrode validation findings (link to the validation script).
  - CHANGELOG.md: "v3 cross-session unit tracking via UnitMatch (Neuropixels validated; tetrode use case documented). `TrackedUnit` derives biological-unit identities across sessions."

## Deliberately not in this phase

- **DeepUnitMatch**. The `MatcherProtocol` design accepts it as a future plugin; Phase 4.1+ implements `_deepunitmatch_backend.py` separately. Not in this PR.
- **Drift correction inside the matcher**. UnitMatch does its own rigid-shift estimation; v3 does not pre-correct drift on inputs to UnitMatch.
- **Across-multi-day concatenation**. Phase 3 multi-day path is documented as future; this phase does not enable it via UnitMatch as a workaround.
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
| `test_tracked_unit_connected_components` | Synthetic pair list with three sessions and known clusters; `TrackedUnit.make()` produces expected number of biological-unit components. |
| `test_tracked_unit_strict_default_rejects_transitive` | Pairs A↔B (high), B↔C (high), A↔C (low). Default `tracked_unit_policy="strict"` (maximal cliques) produces ≥2 components (NOT 1) — A and C cannot be lumped without a direct above-threshold edge. Test asserts >1 component and asserts no component contains both A and C. |
| `test_tracked_unit_transitive_opt_in_unifies` | Same input as above; with `tracked_unit_policy="transitive"`, connected-components yields 1 component with `n_transitive_only_edges == 1` (the A↔C edge missing). |
| `test_tetrode_warning_logged` | Running UnitMatch on a SessionGroup with ≤4 channels per unit logs a warning string containing "tetrode". |
| `test_v3_unitmatch_neuropixels_smoke` (slow, optional) | Skipped unless `--run-neuropixels` is passed; uses a small Neuropixels fixture if available. |

## Fixtures

- **`synthetic_neuropixels_2_session_fixture`** — new conftest fixture. Two 16-channel SI recordings + sortings with 5 units each; the first 3 units in session B are template-copies of the first 3 units in session A (so UnitMatch should match them with high probability); units 4–5 are random. Built deterministically with a seed.
- **Tetrode gold-standard dataset** — provided by user during Phase 4 implementation; path via env var `SPIKESORTING_V3_TETRODE_GOLD_PATH`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — DeepUnitMatch is not in this PR.
- Validation slice tests pass; slow / integration tests are marked.
- The synthetic Neuropixels test produces real UnitMatch output (not a mock).
- The tetrode validation script has been run by the implementer against the user-provided gold-standard dataset; results are documented in `docs/src/Pipelines/SpikeSorting/v3-tetrode-notes.md`. If AUC is below 0.85, the doc explicitly recommends concat for tetrodes.
- `MatcherProtocol` is implementable by external code without touching v3 internals (verify by writing a 10-line dummy matcher in the test suite).
- `TrackedUnit` graph algorithm is documented (transitive closure has known semantics; test `test_tracked_unit_handles_inconsistent_matches` documents the choice).
- Docstrings, test names, and module names don't reference this plan, phase numbers, or files inside `.claude/docs/plans/`.
- `unitmatchpy` is gated as an optional dependency (`pip install -e ".[spikesorting-v3-matching]"`). Import-time guard in `_unitmatch_backend.py` raises `ImportError` with the install command if missing.
