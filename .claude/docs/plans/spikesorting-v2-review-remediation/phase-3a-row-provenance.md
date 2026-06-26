# Phase 3a — Reproducibility provenance in computed rows

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Bake the producer provenance into computed rows while pre-production (R1, R33-matcher):
the effective random seed, the producing library/sorter versions, the UnitMatch
waveform-bundle params (identity-bearing), and the matcher backend version. Provenance
is **secondary, never identity** — a parity test confirms ids are unchanged.

**Inputs to read first:**

- [shared-contracts.md#producer-provenance-field-set](shared-contracts.md#producer-provenance-field-set) — the field names/semantics 3b also uses; the never-identity invariant.
- [src/spyglass/spikesorting/v2/utils.py:607-632](../../../../src/spyglass/spikesorting/v2/utils.py#L607-L632) — `_resolved_job_kwargs` (the resolution order seed-capture builds on).
- [src/spyglass/spikesorting/v2/sorting.py:1127-1135](../../../../src/spyglass/spikesorting/v2/sorting.py#L1127-L1135) — `Sorting` master definition (add provenance attrs before the closing `"""`).
- [src/spyglass/spikesorting/v2/unit_matching.py:103-110](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L103-L110) (`MatcherParameters`), [:450-457](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L450-L457) (`UnitMatch` master).
- [src/spyglass/spikesorting/v2/_unitmatch_backend.py:107-116](../../../../src/spyglass/spikesorting/v2/_unitmatch_backend.py#L107-L116) — `extract_unitmatch_bundle` defaults `ms_before/ms_after/max_spikes_per_unit/seed`.
- [src/spyglass/spikesorting/v2/matcher_protocol.py](../../../../src/spyglass/spikesorting/v2/matcher_protocol.py) — the `replace=True` registry to honest-up; [recompute.py](../../../../src/spyglass/spikesorting/v2/recompute.py) `si_deps` (where `si.__version__` is already captured) for the version-string convention.

**Contracts referenced:** [Producer provenance field set](shared-contracts.md#producer-provenance-field-set) (defined here, consumed by 3b) — do not weaken the never-identity invariant.

## Tasks

1. **Effective-seed capture helper.** Add to `utils.py`, next to `_resolved_job_kwargs`:

   ```python
   def resolve_effective_seed(*row_job_kwargs: dict | None) -> int:
       """Return the random_seed actually used (default 0) after resolving SI
       globals -> dj.config -> per-row blobs. Warn ONCE if a science-affecting
       seed arrives via the ambient (SI-global / dj.config) layer rather than a
       per-row blob, so a non-reproducible ambient seed is visible, not silent."""
   ```

   It reuses `_resolved_job_kwargs`' merge but additionally inspects whether `random_seed` came from the ambient layers (compare ambient-merged vs per-row) and `logger.warning`s once if so. Route the seed sites (`_sorting_dispatch.py:340,513`, `_sorting_analyzer.py:649`) and the value stored below through this helper so the stored seed equals the used seed.

   **This is a (defensible) behavior change, not pure capture — call it out.** The three seed sites currently read **only the per-row `job_kwargs` blob** (`.get("random_seed", 0)`); they ignore SI globals / `dj.config`. Routing them through `resolve_effective_seed` means an ambient `dj.config['custom']['spikesorting_v2_job_kwargs']['random_seed']` that is currently ignored would now **take effect** (per-row still wins — correct precedence). Document this in the CHANGELOG and pin a regression test that a per-row seed still overrides ambient. **Do not store-without-routing:** if you store `resolve_effective_seed(...)` but leave the sites reading the per-row blob, the stored value diverges from the used value whenever an ambient seed exists.

2. **`Sorting` provenance columns.** Add to the `Sorting` master definition (`sorting.py:1135`, before the closing `"""`), as secondary attributes:

   ```
   effective_random_seed=null: int        # seed actually used (resolve_effective_seed)
   spikeinterface_version: varchar(32)     # si.__version__ at sort time
   sorter_version=null: varchar(64)        # distribution version of the sorter package, NULL for in-process
   ```

   Populate from `resolve_effective_seed(...)`, `spikeinterface.__version__`, and `importlib.metadata.version(<sorter dist>)` (NULL for `clusterless_thresholder`; map sorter name → distribution with a small dict). **Carrier threading required:** the values are computed in `make_compute` but the master row is inserted in `make_insert`, which receives only the `SortingComputed` NamedTuple (`sorting.py:146`) — whose fields are explicitly *not* `Sorting` columns. Add the three provenance values as fields on `SortingComputed`, thread them from `make_compute`, and write them in `make_insert`'s `insert1` dict. **Update the positional-signature contract test** for `SortingComputed` (the tri-part tests pin the NamedTuple shape).

3. **`UnitMatch` provenance columns + `MatcherParameters` bundle params.**
   - Add to the `UnitMatch` master (`unit_matching.py:457`): `spikeinterface_version: varchar(32)`, `matcher_backend: varchar(64)`, `matcher_backend_version=null: varchar(64)`. Populate from the resolved registry entry's module path + `importlib.metadata.version("unitmatchpy")`. **Same carrier threading as task 2:** extend the `UnitMatchComputed` NamedTuple (`unit_matching.py:85`) with these fields, thread from `make_compute`, write in `make_insert`, and update its positional-signature contract test.
   - Surface the bundle params as **identity-bearing** fields on `MatcherParameters` (`unit_matching.py:103-110`): extend the matcher Pydantic schema (`_params/matcher.py`) with `ms_before`, `ms_after`, `max_spikes_per_unit`, `seed` (defaults matching `extract_unitmatch_bundle`'s current literals — 1.5/1.5/100/0). Thread them from the resolved `MatcherParameters` row into the `extract_unitmatch_bundle(...)` call in `make_compute` (currently called with the function defaults). Because these are in the named `MatcherParameters` blob, they flow into `matcher_params_name` → `UnitMatchSelection` → `unitmatch_id` identity automatically. Re-seed the shipped default matcher row at the new schema version (coordinate with phase-2's version-bump convention).

   **Close the seed-override hole.** `extract_unitmatch_bundle` currently does `random_seed = compute_job_kwargs.pop("random_seed", seed)` (`_unitmatch_backend.py:51`) — so a `random_seed` in `MatcherParameters.job_kwargs` **overrides** the identity-bearing `seed` field, making the stored identity disagree with the seed actually used. Make `seed` authoritative: in `MatcherParameters` validation, **reject `random_seed` in the bundle `job_kwargs`** (it duplicates the `seed` field), OR keep the override but fold the *effective* seed into identity. Prefer reject. Add a **disagreement regression test**: a row whose `job_kwargs` carries a `random_seed` different from `seed` is rejected at insert (or, if folded, produces a distinct `unitmatch_id`).

4. **`noise_levels` provenance.** The base analyzer extensions include `noise_levels` (`_sorting_analyzer.py` ~617-655) but `extension_params` pins only the seeded `random_spikes` and `waveforms` windows. Record in the analyzer manifest / a provenance note that `noise_levels` is computed without an explicit seed (it is effectively deterministic given the recording, but mark it so the recompute manifest doesn't silently imply a pinned seed). No new column on a public table — extend the existing analyzer-manifest provenance (`recompute.py` `analyzer_manifest`) to list each base extension with its effective params or an explicit "unseeded" marker.

5. **R33-matcher registry honesty.** In `matcher_protocol.py`, remove the misleading `replace=True` re-registration path (a name silently meaning different code is a liability) OR gate it behind an explicit maintenance flag, and ensure the resolved backend's module + version are what task 3 stores. Update the matcher docstrings to state UnitMatchPy is the single backend (the Phase-5 docs edit says the same — overview "Phase 5 adjustments").

6. **Parity test (the never-identity guard).** Assert that for a fixed input, `sorting_id` and `unitmatch_id` are **unchanged** after the provenance columns are added (provenance is secondary, not identity). See validation slice.

7. **Docs.** CHANGELOG: new provenance columns on `Sorting`/`UnitMatch`, the `MatcherParameters` bundle-param fields (note: changes the matcher params schema version and `unitmatch_id` for any non-default bundle settings — acceptable pre-production), the matcher-registry honesty change.

## Additional tasks (Round-3 reviews)

8. **ALSC-5 — bind AnalyzerCuration output to its source provenance.** `AnalyzerCuration` stores only `AnalysisNwbfile` + three object_ids (`metric_curation.py:838-845`); compute loads mutable cache folders at runtime. Add secondary provenance attrs to the `AnalyzerCuration` row: the source analyzer recipe name(s) + analyzer manifest/hash, the sorting/recording `content_hash`, and the SI version — and a stale-detection helper (compare stored vs current). Same never-identity rule as above.

9. **CLUST-3 — persist clusterless unit semantics.** `run_clusterless_thresholder` emits all peaks as one unit (`_sorting_dispatch.py:393-396`); nothing persisted distinguishes a threshold-crossing "unit" from a sorted neuron, so UnitMatch / merge / `get_sort_metadata` surfaces can mistake it. Persist a `unit_semantics` marker (e.g. `"clusterless_threshold_crossings"` vs `"sorted_units"`) on the sort row or expose it via `CurationV2.get_sort_metadata`, and have the consuming surfaces honor it. (Enhancement; small.)

## Deliberately not in this phase

- **Writing provenance into NWB** — that is phase-3b (reads these same field names).
- **A full producer-runtime manifest** (every transitive library) — out of scope; capture SI + sorter + matcher versions + effective seed, the high-value subset (overview Non-Goals).
- **Making the effective seed identity-bearing** — explicitly forbidden by the shared-contract invariant (would fork ids on every upgrade).
- **A matcher/sorter plugin API** (R33 decided non-goal) — only backend-version provenance + registry honesty here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_provenance.py::test_sorting_row_records_effective_seed_and_versions` (new) | after a populate, `Sorting` row has non-null `spikeinterface_version` and an `effective_random_seed` equal to the seed the dispatch used; a per-row `random_seed` overrides ambient and is the stored value. |
| `test_provenance.py::test_ambient_seed_warns` (new) | setting `dj.config['custom']['spikesorting_v2_job_kwargs'] = {"random_seed": 7}` with no per-row seed emits the ambient-seed warning and stores 7. |
| `test_provenance.py::test_unitmatch_records_backend_version` (new) | `UnitMatch` row has `matcher_backend` (module path) + `matcher_backend_version` == `importlib.metadata.version("unitmatchpy")`. |
| `test_matcher_params.py::test_bundle_params_in_identity` (new) | two `MatcherParameters` rows differing only in `ms_before` yield different `matcher_params_name`-derived `unitmatch_id`s; the bundle params reach `extract_unitmatch_bundle`. |
| `test_provenance.py::test_ids_unchanged_after_provenance_columns` (new) | `sorting_id`/`unitmatch_id` for a fixed selection equal the pre-change deterministic values (provenance is secondary, not identity). Pin the expected uuids. |
| (regression) `test_sorter_parameters.py`, `test_unitmatch.py`, `test_matcher_params.py`, `test_recompute.py` | existing identity/runtime tests pass; the matcher default re-seed is idempotent. |

## Fixtures

`populated_sorting` (`conftest.py:215`) for the `Sorting`-row tests; `two_session_curated_group` (`test_unitmatch.py:508`) for the UnitMatch tests; the matcher-params and parity tests are mostly DB-free (schema + identity functions) or use a single inserted selection. Pin expected uuids from a fixed fixture input for the never-identity test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The never-identity invariant holds: the parity test pins unchanged `sorting_id`/`unitmatch_id`; provenance columns are all secondary attributes.
- `resolve_effective_seed` returns the value actually consumed by the seed sites (not a parallel computation that could drift from what the sorter used).
- Sorter→distribution mapping is correct; in-process sorters store NULL, not a wrong version.
- Bundle params reach `extract_unitmatch_bundle` and enter `MatcherParameters` identity (no longer silent function defaults).
- The matcher registry no longer silently re-routes a name to different code; backend version is stored.
- CHANGELOG notes the `unitmatch_id` change for non-default bundle settings; no plan/phase references in code or tests.
