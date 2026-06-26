# Shared contracts

[← back to PLAN.md](PLAN.md)

Two contracts cross phase boundaries. Each lives here once; phases link in by anchor.

- [Producer provenance field set](#producer-provenance-field-set) — defined and stored by phase-3a; re-emitted into NWB by phase-3b.
- [Master-row identity immutability](#master-row-identity-immutability) — implemented in phase-2; a "do not weaken" invariant any later schema change must respect.

---

## Producer provenance field set

The set of provenance values that identify *what produced* a computed v2 output.
phase-3a persists these as **secondary attributes** on the relevant computed
tables; phase-3b writes the same values into the corresponding artifact NWB so the
file is self-describing. The two phases must use the **same field names and
semantics**.

**Invariant — provenance is never identity.** Every field below is a secondary
attribute (below the `---` in the DataJoint definition). None is ever added to a
primary key or folded into a `deterministic_id` / content hash. Capturing the
effective seed changes *stored data*, not *row identity*: a parity test in phase-3a
asserts the `sorting_id` / `unitmatch_id` for a fixed input is unchanged after the
columns are added. (Rationale: these record the runtime that produced the row;
making them identity would fork ids on every library upgrade.)

| Field | Type | Meaning | Source |
| --- | --- | --- | --- |
| `effective_random_seed` | `int` | The seed actually used, after resolving SI globals → `dj.config` → per-row `job_kwargs` (`utils._resolved_job_kwargs`). | computed at run time on the Sorting / analyzer / UnitMatch path |
| `spikeinterface_version` | `varchar(32)` | `spikeinterface.__version__` at production time. | `importlib.metadata` / `si.__version__` (already captured in `recompute.py` `si_deps`) |
| `sorter_version` | `varchar(64)` | Distribution version of the sorter package actually run (e.g. `mountainsort5`), or `NULL` for in-process sorters (clusterless). | `importlib.metadata.version(...)` |
| `matcher_backend` | `varchar(64)` | Matcher backend module path (e.g. `spyglass...._unitmatch_backend`). | the resolved registry entry |
| `matcher_backend_version` | `varchar(64)` | Distribution version of the matcher backend package (`unitmatchpy`). | `importlib.metadata.version(...)` |

**Effective-seed capture rule.** `utils._resolved_job_kwargs` (`utils.py:607-632`)
merges SI globals, `dj.config["custom"]["spikesorting_v2_job_kwargs"]`, and the
per-row blob. phase-3a adds a helper that returns the *resolved* `random_seed`
(default `0`) and **emits a `logger.warning` once** when a science-affecting key
(`random_seed`) is supplied via the ambient SI-global / `dj.config` layer rather
than the per-row blob — so a non-reproducible ambient seed is visible, not silent.
The resolved value is what gets stored in `effective_random_seed` and consumed by
the seed sites (`_sorting_dispatch.py:340,513`; `_sorting_analyzer.py:649`).

**UnitMatch bundle params** (`extract_unitmatch_bundle` defaults `ms_before`,
`ms_after`, `max_spikes_per_unit`, `seed` — `_unitmatch_backend.py:107-116`): these
ARE identity-bearing (they change the matched result), so phase-3a surfaces them as
fields on `MatcherParameters` (`unit_matching.py:103-110`) so they enter the
`matcher_params_name` → identity, *not* as secondary provenance. They are listed
here only to disambiguate them from the provenance set above. See phase-3a.

---

## Master-row identity immutability

The v2 deterministic-id design assumes a row's identity-bearing content cannot be
silently changed after dependents exist. The param **Lookups** already enforce this
via `ImmutableParamsLookup.update1` (`utils.py:282-309`, rejects `update1` unless
`allow_param_mutation=True`). The **selection masters** and the `CurationV2` /
`SessionGroup` masters do **not** — `SelectionMasterInsertGuard` (`utils.py:171-254`)
overrides only `insert`, and `CurationV2`/`SessionGroup` are plain `dj.Manual`.

**Contract (implemented in phase-2):**

- `SelectionMasterInsertGuard` gains an `update1(self, row, *, allow_master_mutation=False)` override that rejects in-place mutation by default, mirroring `ImmutableParamsLookup.update1`. This covers all six masters that mix it in: `RecordingSelection` (`recording.py:745`), `ArtifactDetectionSelection` (`artifact.py:441`), `SortingSelection` (`sorting.py:651`), `UnitMatchSelection` (`unit_matching.py:198`), `ConcatenatedRecordingSelection` (`session_group.py:330`), `AnalyzerCurationSelection` (`metric_curation.py:646`).
- `CurationV2` (`curation.py:65`) and `SessionGroup` (`session_group.py:65`) get the same insert + `update1` guard (they are not selection masters but are identity/provenance roots that dependents reference).

**Do not weaken.** Any later phase that adds an identity-bearing column to a master
must keep it inside this guard's protection. Provenance columns (phase-3a) are
exempt **because they are secondary, non-identity attributes** (see the invariant
above) — but they must not be added to a master in a way that bypasses the guard's
`insert` validation.
