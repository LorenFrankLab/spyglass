# Phase 7 — Migration docs & stub roadmap (+ one small back-compat helper)

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Mostly documentation. The one source change is **A34: a single new opt-in classmethod (`SorterParameters.insert_default_legacy_si_sorters`) plus the four `__getattr__` shims for the stub modules in A33** — both are user-visible-but-low-blast-radius additions that belong with the migration docs they support, not with the behavioral fixes in Phases 4–5. Everything else (CHANGELOG entries, roadmap docstrings, migration guide, stale-ref grep sweep) is doc-only.

**No behavioral changes to existing source paths. If a doc fix reveals a real bug in an existing path, file it against an earlier phase; do not fix in Phase 7.** A33's `__getattr__` and A34's classmethod are NEW additions — they do not modify existing behavior of populated tables, of the `Sorting.make` pipeline, or of any downstream consumer.

**Inputs to read first:**

- [.claude/audits/COMBINED_SYNTHESIS.md](../../../audits/COMBINED_SYNTHESIS.md) — the full audit catalog this phase documents.
- [src/spyglass/spikesorting/v2/metric_curation.py](../../../../../src/spyglass/spikesorting/v2/metric_curation.py) — 3-line stub.
- [src/spyglass/spikesorting/v2/figpack_curation.py](../../../../../src/spyglass/spikesorting/v2/figpack_curation.py) — 3-line stub.
- [src/spyglass/spikesorting/v2/unit_matching.py](../../../../../src/spyglass/spikesorting/v2/unit_matching.py) — 3-line stub.
- [src/spyglass/spikesorting/v2/matcher_protocol.py](../../../../../src/spyglass/spikesorting/v2/matcher_protocol.py) — 3-line stub.
- [src/spyglass/spikesorting/v1/metric_curation.py](../../../../../src/spyglass/spikesorting/v1/metric_curation.py) — the v1 fallback the stub doc points at.
- [src/spyglass/spikesorting/v1/figurl_curation.py](../../../../../src/spyglass/spikesorting/v1/figurl_curation.py) — same for FigURL.
- [src/spyglass/spikesorting/v1/burst_curation.py](../../../../../src/spyglass/spikesorting/v1/burst_curation.py) — same for BurstPair.
- [src/spyglass/spikesorting/v1/recompute.py](../../../../../src/spyglass/spikesorting/v1/recompute.py) — same for Recompute.
- [CHANGELOG.md](../../../../../CHANGELOG.md) — repository-wide changelog file (verify the exact filename / format used by the repo). Read existing v2-related entries before adding new ones; the parent plan's Phase 1b B1 already adds a CHANGELOG entry for the artifact unit-conversion bug — append to it, don't replace.

## Tasks

### A32 — DOC: CHANGELOG entries enumerating v1 → v2 breaking changes

Add a `## Spike Sorting v2 — Breaking Changes` section to [CHANGELOG.md](../../../../../CHANGELOG.md) (verify section heading convention before adding). One terse bullet per change with the relevant source location. Group:

**API renames**
- `SpikeSorterParameters.sorter_param_name` → `SorterParameters.sorter_params_name` (column rename, with `s`). Existing code restricting `{"sorter_param_name": "..."}` returns empty on v2.
- `apply_merges` kwarg → `apply_merge` on `CurationV2.insert_curation` (Phase 1b B2 already changed; document the user-visible kwarg rename).
- Preprocessing schema field renames: `frequency_min`/`frequency_max` → `freq_min`/`freq_max` ([_params/preprocessing.py:22-23](../../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L22-L23)).
- Franklab MS4 presets renamed: `franklab_tetrode_hippocampus_30KHz` → `franklab_tetrode_hippocampus_30kHz_ms4` and ctx equivalent. Phase 4 A2 ships v1-name aliases for one release; this entry documents the deprecation timeline.

**Dropped or relocated data**
- `IntervalList` row keyed by `recording_id` no longer inserted. Reconstruction recipe:
  ```python
  row = (Recording & {"recording_id": rid}).fetch1()
  valid_times = np.asarray([[row["saved_start"], row["saved_end"]]])
  ```
- Artifact `IntervalList.interval_list_name` now prefixed `artifact_{uuid}` (was bare `str(uuid)`). Use `parse_artifact_interval_list_name` ([utils.py:529-533](../../../../../src/spyglass/spikesorting/v2/utils.py#L529-L533)) for backward-compatible lookup.
- `Sorting.time_of_sort` is `datetime`, not Unix int seconds. External consumers comparing against `int(time.time())` must cast.
- Object-ID columns widened varchar(40) → varchar(72) on `Sorting` and `CurationV2` master rows.
- `description` column widened varchar(100) → varchar(255) on `CurationV2`.

**Schema-defaults flips (programmatic users only)**
- `ClusterlessThresholderSchema().noise_levels` default changed from `[1.0]` to `None` (means "let SI compute per-channel MAD"). The shipped `default` Lookup row still carries `noise_levels=[1.0]` for production v1 parity — only the schema-level field default flipped. Programmatic users constructing the schema without arguments now get MAD semantics; pass `noise_levels=[1.0]` to preserve v1 microvolt semantics. This fixed a real 1400× clusterless divergence (see core memory).
- `MountainSort4Schema().freq_min` / `freq_max` defaults = `600.0` / `6000.0` (Frank-lab tetrode preset), not SI's wrapper defaults (`300.0` / `6000.0`). Phase 6 A30 pins these.
- `WhitenParams` default flipped from "on" to `None`. Documented in existing Phase 2 T6.

**Boundary semantics — small spike-count delta near artifact edges**
- v1's `_consolidate_intervals` had an off-by-one bug (zeroed last valid sample, left final frame unmasked). v2 fixes both ([sorting.py:993-1055](../../../../../src/spyglass/spikesorting/v2/sorting.py#L993-L1055)). v1↔v2 spike counts can differ near artifact-mask boundaries. Spike-by-spike numerical comparison on the same input WILL differ at those edges; this is correct behavior, not regression.

**Multi-channel clusterless `noise_levels` broadcast fix**
- v1's `noise_levels=[1.0]` on a multi-channel recording would silently misread channels (singleton indexing in SI's `locally_exclusive` peak detection). v2 broadcasts to `n_channels` at runtime ([sorting.py:1168-1176](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1168-L1176)). v1↔v2 clusterless sorts on multi-channel recordings WILL show real, correct differences — v2 is the right answer.

**Determinism — random seeds pinned**
- v2 explicitly pins SI's `seed` arg for `sip.whiten` and `get_noise_levels` (after SI's PR #3359 changed those defaults from `seed=0` to `seed=None`). Per-row override via `SorterParameters.job_kwargs={'random_seed': N}`.

**Default thresholds**
- Artifact-detection ships `amplitude_thresh_uV=500` (was `3000` in v1). Documented in existing Phase 1b B1.

**Removed v1 features**
- `MetricCuration` chain (`metric_curation`, `MetricCurationParameters`, `WaveformParameters`, `MetricParameters`) — see A33 stub roadmap.
- `FigURLCuration` chain — see A33.
- `BurstPair` chain — see A33.
- `RecordingRecompute` chain — see A33.
- `recording_id`-keyed `IntervalList` row — see above.

**Tags**
- Artifact `IntervalList.pipeline` tag `spikesorting_artifact_v1` → `spikesorting_artifact_v2`.

**Production-scale (Phase 5: A17 / A21 / A22)**
- Artifact detection restored to a chunked `ChunkRecordingExecutor` pass ([artifact.py `_scan_artifact_frames` / `_compute_artifact_chunk`](../../../../../src/spyglass/spikesorting/v2/artifact.py)). The in-memory full-`get_traces` scan (peaked at ~4 × n_samples × n_channels × 4 bytes — ~27 GB for a 1-hour 64-channel 30 kHz recording) is removed. The `ArtifactDetectionParameters.job_kwargs` blob is now **functional**: it controls the scan's `n_jobs` / `chunk_duration` (default `chunk_duration='1s'`, `n_jobs=1`) — previously dead weight. Output is frame-identical to the old in-memory path; no scientific change.
- SpikeInterface pinned to `==0.104.3` in [pyproject.toml](../../../../../pyproject.toml). The KS4/MS5/SC2/TDC2/Generic schemas use `extra="allow"`, so untyped sorter fields fall through to SI's per-version defaults; the pin + snapshot tests (`test_kilosort4_si_defaults_unchanged`, `test_ms5_si_defaults_unchanged`) make a SI bump a deliberate, audited step. Loosen only after the parent plan's Phase 0c SI migration; move the pin and the snapshots together.
- New ops helper `Sorting.find_orphaned_analyzer_folders(*, dry_run=True)` ([sorting.py](../../../../../src/spyglass/spikesorting/v2/sorting.py)) surfaces 5–50 GB analyzer-folder disk leaks from delete-override bypass (raw SQL delete / scripted `connection.query`). Reports DB-side orphans (row present, folder gone — never auto-deleted) and disk-side orphans (folder present, no row); `dry_run=False` deletes only disk-side orphans after interactive confirmation.

**Per-finding citations**: each CHANGELOG bullet references a specific source location. Use the markdown link form `[file.py:LL-LL](path)` so a reader can click directly. Do not bury rationale in the bullet — keep bullets terse; the rationale lives in the audit JSON and in the stub-roadmap docstrings (next task).

### A33 — DOC: stub-module roadmap docstrings + informative `ImportError`

Each of the four 3-line stub modules currently re-exports nothing and silently imports as an empty module. A v1-workflow user porting a notebook gets `ImportError: cannot import name 'MetricCuration' from 'spyglass.spikesorting.v2.metric_curation'` — technically true but missing the actionable hint (the v1 fallback). Replace each stub with a roadmap docstring + a defensive `__getattr__` that raises a custom-message error preserved across BOTH attribute-access and `from ... import` forms.

**Python from-import semantics — why we raise `ImportError`, not `AttributeError`:** when Python executes `from m import X`, it calls `getattr(m, 'X')` and then **catches `AttributeError` specifically, replacing it with a generic `ImportError("cannot import name 'X' from 'm'")` that loses the original message**. Any OTHER exception (including `ImportError` itself) propagates with its message intact. So a stub `__getattr__` that raises `AttributeError` is silently flattened to the unhelpful default; raising `ImportError` keeps the actionable message visible whether the user wrote `from m import X` OR `import m; m.X`. Verified against CPython's `import.c` (the `LOAD_FROM` opcode's error path).

Template (apply to each of `metric_curation.py`, `figpack_curation.py`, `unit_matching.py`, `matcher_protocol.py`):

```python
"""Spike-sorting metric curation — not yet ported to v2.

The v1 equivalent (`spyglass.spikesorting.v1.metric_curation`) ships
the `MetricCuration`, `MetricCurationParameters`, `WaveformParameters`,
and `MetricParameters` tables. v2's port is on the roadmap; use the v1
chain in the interim:

    from spyglass.spikesorting.v1 import (
        MetricCuration,
        MetricCurationParameters,
        WaveformParameters,
        MetricParameters,
    )

For roadmap details, see the spike-sorting-v2 documentation in the
project repository.
"""


def __getattr__(name):
    # Dunder names (__path__, __all__, __spec__, __file__, ...) are
    # probed defensively by the import machinery, pickle, and inspection
    # tools. Raising the custom ImportError for those produces confusing
    # messages like "ImportError: ... '__path__'" during from-import,
    # because the import machinery's hasattr probe catches only
    # AttributeError. Always raise AttributeError for dunders so the
    # probes get the answer they expect; the custom message is reserved
    # for real public-API names.
    if name.startswith("__"):
        raise AttributeError(name)
    # ImportError (not AttributeError) for public names so the message
    # survives the `from m import X` flattening path -- CPython only
    # collapses AttributeError into the generic "cannot import name"
    # ImportError.
    raise ImportError(
        f"spyglass.spikesorting.v2.metric_curation.{name!r} is not yet "
        "implemented in v2. Use the v1 fallback "
        "`from spyglass.spikesorting.v1 import "
        f"{name}` (where available), or wait for the v2 port. "
        "See the module docstring for details."
    )
```

Same shape for the other three modules with module-specific text:

- `figpack_curation.py` → fallback `spyglass.spikesorting.v1.figurl_curation` (`FigURLCuration`, `FigURLCurationSelection`). v2's FigPack consumer is a separate roadmap item.
- `unit_matching.py` → no v1 fallback (UnitMatch is a v2-introduced feature). Stub doc says "not available yet; planned for a future v2 release. See the project repository's spike-sorting-v2 plan."
- `matcher_protocol.py` → same as unit_matching; both land together. Stub doc says "plugin interface for future cross-session matchers."

The `ImportError`-from-`__getattr__` shape works for BOTH usage forms:
- `from spyglass.spikesorting.v2.metric_curation import MetricCuration` — Python's `from`-machinery propagates `ImportError` as-is (it only collapses `AttributeError`), so the user sees the custom message.
- `import spyglass.spikesorting.v2.metric_curation as m; m.MetricCuration` — `__getattr__` fires on the attribute access; the `ImportError` is raised directly with the custom message.
- `import spyglass.spikesorting.v2.metric_curation` (top-level import only, no attribute access) — succeeds; test discovery and module-level import hooks are not broken.

**Do not implement the stub bodies** — that's downstream roadmap work (handled by the main-epic spike-sorting-v2 plan's later phases).

**Note for executor on docstring wording**: the docstring's reference to "the spike-sorting-v2 documentation in the project repository" is intentionally indirect. Per the planning-skill rule, docstrings shipping in user-facing modules must not reference internal plan paths or milestones by name. The vague reference lets a curious developer find the path without baking a specific phase number into shipped code.

### A34 — DOC: `insert_default_legacy_si_sorters()` opt-in helper

The audit's recommendation #3 — replicate v1's auto-insert of `('sorter_name','default')` rows for every `sis.available_sorters()` entry, but as a separate opt-in helper rather than baked into `insert_default()`. v1 callers porting workflows that name a non-curated sorter via `('kilosort2_5','default')` etc. can call this once to populate the back-compat rows.

Add to `SorterParameters` at [sorting.py](../../../../../src/spyglass/spikesorting/v2/sorting.py):

```python
@classmethod
def insert_default_legacy_si_sorters(cls):
    """Insert ('sorter','default') rows for every installed SI sorter
    NOT already curated by ``insert_default()``.

    Replicates v1's auto-insert behavior at
    ``v1/sorting.py:184-189`` for users porting workflows that name a
    non-curated sorter via ``('kilosort2_5','default')`` or similar.
    Calls SpikeInterface's ``sis.get_default_sorter_params(sorter)``
    for each entry of ``sis.available_sorters()`` and validates
    through ``GenericSorterParamsSchema`` (``extra='allow'``) so the
    row passes ``insert1``'s gate without typo-rejection.

    Curated sorters (mountainsort4, mountainsort5, kilosort4,
    spykingcircus2, tridesclous2, clusterless_thresholder) are
    SKIPPED -- they already have their own typed schemas with
    ``extra='forbid'``, and SI's defaults for those sorters include
    keys those schemas intentionally strip
    (e.g. ``MountainSort5Schema`` strips ``filter``/``freq_min`` etc.
    because the upstream recording is already filtered). Routing SI's
    full default dict through the typed schema would either fail
    validation or quietly drop keys -- neither is what a v1 caller
    expects. The opt-in is targeted at the NON-curated escape-hatch
    sorters that fall back to ``GenericSorterParamsSchema`` anyway.

    Idempotent via ``skip_duplicates=True``.

    Examples
    --------
    >>> from spyglass.spikesorting.v2.sorting import SorterParameters
    >>> SorterParameters.insert_default()
    >>> SorterParameters.insert_default_legacy_si_sorters()
    """
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import (
        GenericSorterParamsSchema,
        _SORTER_SCHEMAS,  # re-grep the actual name before editing
    )
    from spyglass.utils import logger

    curated = set(_SORTER_SCHEMAS)  # sorters with their own typed schemas
    rows = []
    for sorter in sis.available_sorters():
        if sorter in curated:
            continue  # see docstring: would fail or drop keys
        try:
            params = sis.get_default_sorter_params(sorter)
        except Exception as exc:  # SI sometimes raises on metadata fetch
            logger.warning(
                f"insert_default_legacy_si_sorters: skipping {sorter!r} "
                f"({exc!r})."
            )
            continue
        # Validate through the generic schema (extra='allow') so the
        # row passes insert1's gate without typo-rejection.
        try:
            validated = _validate_params(GenericSorterParamsSchema, params)
        except Exception as exc:
            logger.warning(
                f"insert_default_legacy_si_sorters: {sorter!r} did "
                f"not validate against GenericSorterParamsSchema "
                f"({exc!r})."
            )
            continue
        rows.append((sorter, "default", validated, 1, None))
    cls.insert(rows, skip_duplicates=True)
```

- Add a behavioral test in Phase 6 A30 (deferred — list under A34 here so executor knows the test pairs with this code change) asserting that after `insert_default_legacy_si_sorters()`, a v1-style key `{"sorter": "<installed-sorter>", "sorter_params_name": "default"}` resolves to exactly one row. Skip on environments without the relevant sorter installed.
- This is the only code change in Phase 7. It is small enough not to merit its own phase but large enough that adding it as a Phase 6 test-only entry would be a lie. Land it here with the doc PR.
- Phase 4 A2 (preset-name aliases) is a different shim — A2 aliases the renamed Franklab presets; A34 covers all OTHER non-curated v1 sorter rows. They compose.

### A35 — DOC: migration guide notebook section / docs page

Add a markdown section to the user-facing docs directory (verify the path before editing: likely `docs/migration/spikesorting-v1-to-v2.md` or similar). Cover, in order, the user-visible deltas grouped for a notebook user not an executor:

1. **What you call differently** — sorter-row naming, `apply_merge` (single), `freq_min`/`freq_max` (renamed), the `insert_default_legacy_si_sorters()` opt-in.
2. **What you query differently** — IntervalList lookups (the `artifact_{uuid}` prefix; the missing `recording_id` row + recipe).
3. **What's faster / safer / more reproducible** — chunked artifact detection, monotonicity-repair provenance, hash-verifiable Recording rebuild, pinned SI version + KS4 snapshot test, analyzer-folder disk-leak audit.
4. **What's not there yet** — MetricCuration, FigURLCuration, BurstPair, RecordingRecompute, ConcatenatedRecording, SessionGroup, UnitMatch. For each, the link to the v1 fallback (with snippets) and the parent-plan phase that delivers it.
5. **What v1↔v2 comparisons WILL show** — small spike-count delta near artifact edges (v2 correct); real differences on multi-channel clusterless (v2 correct); KS4 may differ on a SI version bump (caught by the pinned snapshot test); MS4 outputs identical when seeds match.

Keep it ~1-2 pages. Long migration guides go unread. Each "what's not there yet" entry MUST link to (a) the v1 fallback class, (b) the parent-plan phase. Section 5 references the CHANGELOG entries from A32.

### A36 — DOC: stale-reference / line-drift inventory

Cross-cuts the audit's untested-branch list with the `D` series of existing Phase 3. The audit found numerous stale `# v1/utils.py:185-198` comments that are wrong because v1's `utils.py` is now 109 lines and the logic moved to `spikesorting/utils.py:179-198`. Existing Phase 3 D4 covers the artifact.py refs; this task generalizes:

- Grep across `src/spyglass/spikesorting/v2/` for comments matching `v1/.*\.py:\d+` patterns. For each, verify the v1 line still contains the cited logic; if not, update to the correct path/line OR remove the cite if the logic moved into a v2-owned helper.
- Same grep across `tests/spikesorting/v2/` — the audit's D5 covers a partial list; extend to every stale ref.
- Same grep across `.claude/docs/plans/spikesorting-v2/` — existing Phase 3 D6 covers `parity-extensions.md:308-311`; extend to every other doc file referencing `v1/...` by line number.
- Output: a single PR that updates every ref. Do not change behavior; do not add new comments. The grep result is the PR scope.
- Marked test-doc-only because doc-line drift recurs; the next audit will re-discover it. The fix is mechanical.

## Deliberately not in this phase

- **Implementing any of the stub modules** — parent-plan Phases 2/3/4/5.
- **Changing the `_apply_artifact_mask` behavior or any source-side semantics** — Phase 4.
- **Restoring the chunked artifact path** — Phase 5.
- **Removing the v1-name preset aliases added in Phase 4 A2** — at least one release after this CHANGELOG entry lands.
- **Auto-running `insert_default_legacy_si_sorters()` from `initialize_v2_defaults()`** — keeping it opt-in is intentional. Users who don't need v1 names should not be paying for the inserts.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_insert_default_legacy_si_sorters_skip_on_missing_sorter` | A34: a row for an installed sorter exists after the call; skip-on-missing logged for absent sorters; no exception raised. |
| `test_stub_module_from_import_raises_with_custom_message` | A33: `from spyglass.spikesorting.v2.metric_curation import MetricCuration` raises `ImportError` whose message contains `MetricCuration` AND `spikesorting.v1.metric_curation` (the v1 fallback hint). Repeat for `figpack_curation`. For `unit_matching` and `matcher_protocol`: assert the message names the symbol but does NOT claim a v1 fallback. Verifies CPython does not flatten the `__getattr__`-raised `ImportError` to the generic "cannot import name" form. |
| `test_stub_module_attribute_access_raises_with_custom_message` | A33: `import spyglass.spikesorting.v2.metric_curation as m; m.MetricCuration` raises `ImportError` (same custom message). Confirms both usage forms share the wording. |
| `test_stub_module_bare_import_succeeds` | A33: `import spyglass.spikesorting.v2.metric_curation` (no attribute access) succeeds — module loads, docstring is readable, no exception raised. Defends test discovery and module-level import hooks. |
| `test_stub_module_from_import_does_not_leak_dunder_in_message` | A33: `from spyglass.spikesorting.v2.metric_curation import DoesNotExist` raises an error whose message contains `'DoesNotExist'` and does NOT contain `'__path__'`, `'__all__'`, `'__spec__'`, or any other dunder. Defends the `if name.startswith("__")` guard against regressions that would otherwise produce confusing "cannot import name '__path__'" messages during the import machinery's defensive dunder probes. |
| `test_changelog_contains_v2_breaking_section` | A32: `CHANGELOG.md` contains the v2-breaking section heading and at least one bullet per category (renames, dropped data, schema-defaults flips, boundary semantics, multi-channel fix, determinism, default thresholds, removed v1 features, tags). Loose substring match — do not pin exact bullet text. |

The stub `__getattr__` test is behavioral (asserts a specific error message); the CHANGELOG test is a doc-structure assertion (verifies the section exists and is non-empty). Both run fast.

## Fixtures

- A34 needs no fixture beyond an SI install with at least one sorter beyond v2's curated set (KS2.5 / IronClust / TridesClous via the parent-plan SI 0.104 pin).
- A33 test needs no fixture (pure import).
- A32 test reads `CHANGELOG.md` and asserts on its contents.
- A36 needs no test; the PR's grep diff is its own validation.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- CHANGELOG entries (A32) cite source locations as clickable markdown links and are grouped under sub-headings matching the categories listed.
- Stub modules (A33) raise on attribute access with messages naming v1 fallbacks where applicable; module-level docstrings reference the project repository's spike-sorting-v2 documentation indirectly (NOT by phase number) — see the next checklist item for the rationale.
- The `insert_default_legacy_si_sorters` helper (A34) is opt-in — `initialize_v2_defaults()` does NOT call it.
- The migration guide (A35) is concise (1-2 pages); section 5 (v1↔v2 comparison expectations) is present.
- A36's grep sweep found and fixed all stale v1 refs; no remaining `v1/utils.py:1?5` refs (the canonical staleness signal).
- Docstrings in shipped code (A33) reference the project repository's plan directory indirectly, NOT by phase number.
- Source changes are limited to: (a) A33's four `__getattr__` shims on the previously-empty stub modules, (b) A34's new opt-in classmethod on `SorterParameters`. If the executor's diff touches `make_compute`, `make_fetch`, existing schema definitions, runtime helpers, or any populated-table runtime path, it's out of scope for Phase 7 — file against an earlier phase and revert.
