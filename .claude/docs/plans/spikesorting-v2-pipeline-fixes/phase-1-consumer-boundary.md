# Phase 1 — Consumer-boundary silent bugs (A1, A2)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The two highest-risk findings: a v2 `merge_id` resolution that silently ignores
`artifact_id`, and a multi-source `fetch_nwb` that misaligns/over-accumulates
merge_ids. Both feed the wrong data into downstream decoding/grouping. This
phase fixes both and adds the regression tests that would have caught them.

**Inputs to read first:**

- [spikesorting_merge.py:160-248](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L160-L248) — `_get_restricted_merge_ids_v2`: the A1 site. Note `sort_master` (`:236`) lacks `artifact_id` in its heading.
- [sorting.py:424-490](../../../../src/spyglass/spikesorting/v2/sorting.py#L424-L490) — `SortingSelection` and its `RecordingSource` / `ConcatenatedRecordingSource` / `ArtifactSource` parts. `ArtifactSource` (`:466`, `-> ArtifactDetection`) is the **optional** part carrying `artifact_id` (a `uuid`).
- [utils.py:750-790](../../../../src/spyglass/spikesorting/v2/utils.py#L750-L790) — `parse_artifact_interval_list_name` (returns a **str** or None) and the public `get_spiking_sorting_v2_merge_ids` wrapper (`:764`).
- [dj_merge_tables.py:535-577](../../../../src/spyglass/utils/dj_merge_tables.py#L535-L577) — `fetch_nwb` multi-source loop: the A2 site.

## Tasks

### Task 1 — A1: make `artifact_id` (and rec/curation) restrictions actually filter

In `_get_restricted_merge_ids_v2` ([spikesorting_merge.py:203-248](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L203-L248)):

1. Where `restrict_by_artifact` maps the interval name to `artifact_id` (`:208-214`), the parsed value is a **str**; the column is `uuid`. Cast it so the later restriction matches. Accept a caller-supplied UUID too:

```python
if artifact_id is not None:
    # parse_artifact_interval_list_name returns a str; the artifact_id
    # column is a uuid, so cast or the downstream restriction silently
    # matches nothing.
    key["artifact_id"] = uuid.UUID(str(artifact_id))
    key.pop("interval_list_name", None)
```

(add `import uuid` at module top if absent.)

2. `artifact_id` is not in `sort_master`'s heading, so include it via the `ArtifactSource` part **only when present** (the part is optional — joining it unconditionally would drop no-artifact sorts):

```python
# artifact_id lives on the optional SortingSelection.ArtifactSource part,
# NOT on SortingSelection itself. Restricting sort_master (which joins only
# RecordingSource) by artifact_id silently drops the key. Resolve it through
# the part and intersect on sorting_id instead.
sort_master = SortingSelection * sort_rec_source.proj()
sort_restriction = {
    k: key[k] for k in sort_keys if k in key and k != "artifact_id"
}
sort_master = sort_master & sort_restriction
if "artifact_id" in key:
    art = SortingSelection.ArtifactSource & {"artifact_id": key["artifact_id"]}
    sort_master = sort_master & (SortingSelection & art.proj()).proj()
```

Verify the exact projection that intersects `ArtifactSource` back onto `sort_master` by `sorting_id` against the live schema before finalizing (the PK that both share is `sorting_id`). The intent: `sort_master` after this line contains only sortings whose `ArtifactSource` row has the requested `artifact_id`.

3. Confirm the rec/curation restrictions already work (they're real columns on `RecordingSelection` / `CurationV2`); the bug is `artifact_id`-specific.

**Schema-change option (now permitted — the v2 schema is not frozen).** The join above is the recommended fix: it respects the intentional source-part design (`artifact_id` lives on the optional `ArtifactSource` part) and needs no migration. *But* if the join proves awkward or query-fragile, a schema change is on the table — e.g. denormalizing `artifact_id` as a nullable indexed column on `SortingSelection`, or making the part lookup first-class. If you take that route, weigh it explicitly against the join (queryability + simpler restriction vs. partially undoing the source-part design + a migration), record the rationale + a migration note in this phase per the overview's schema-change policy, and prefer it only if it's clearly better. Default: the join.

### Task 2 — A2: align merge_ids with the current source's files

**Context (this is a branch-introduced regression, not pre-existing):** `git diff master...HEAD -- src/spyglass/utils/dj_merge_tables.py` shows this branch (commit #1320, "Returned merge_id consistency for Merge.fetch_nwb") rewrote the `return_merge_ids` block. Master was `merge_ids.extend([k[self._reserved_pk] for k in source_restr])` (one id per source key — could misalign with the files `merge_restrict_class().fetch_nwb()` returns). #1320 switched to resolving a merge_id **per returned file**, which is the right intent — but iterated the **cumulative** `nwb_list` instead of the current source's files, so on source ≥2 it reprocesses prior sources' files (over-accumulates, or `fetch1` raises on the 0-tuple when a prior-source file is AND-ed with the current `source_restr`).

**Alternative considered:** revert to master's `[k[pk] for k in source_restr]`. Rejected — it reintroduces the file/id misalignment #1320 set out to fix (consumers `zip(nwb_list, merge_ids)`). The correct fix completes #1320's intent.

In `fetch_nwb` ([dj_merge_tables.py:549-577](../../../../src/spyglass/utils/dj_merge_tables.py#L549-L577)), the merge_id list is built by iterating the cumulative `nwb_list`. Build it per-source and restrict by the **current** source only:

```python
for source in sources:
    source_restr = (
        self
        & dj.AndList([{self._reserved_sk: source}, merge_restriction])
    ).fetch("KEY", log_export=False)
    source_nwb = (
        (self & source_restr)
        .merge_restrict_class(
            restriction,
            permit_multiple_rows=True,
            add_invalid_restrict=False,
        )
        .fetch_nwb()
    )
    nwb_list.extend(source_nwb)
    if return_merge_ids:
        # Resolve each file's merge_id within THIS source's restriction only
        # (was: iterating the cumulative nwb_list, which re-processed prior
        # sources' files and either over-accumulated or raised on fetch1).
        merge_ids.extend(
            (
                self
                & dj.AndList([self._merge_restrict_parts(file), source_restr])
            ).fetch1(self._reserved_pk)
            for file in source_nwb
        )
```

Preserve single-source behavior exactly — the only change is scoping the merge_id resolution to `source_nwb` instead of `nwb_list`.

### Task 3 — A6: fix `deprecate_log(alternate=...)` TypeError (branch-introduced)

[dj_merge_tables.py:958-960](../../../../src/spyglass/utils/dj_merge_tables.py#L958-L960) calls `ActivityLog().deprecate_log(name="delete_downstream_merge", alternate="Table.delete_downstream_merge")`, but the signature is `deprecate_log(cls, name, alt=None, warning=True)` ([common/common_usage.py:70](../../../../src/spyglass/common/common_usage.py#L70)) — the kwarg is `alt`, not `alternate`. Introduced on this branch (commit #1293); the `delete_downstream_merge` shim raises `TypeError` when invoked. Fix: `alternate=` → `alt=`. (Same file as Task 2 — ship together.)

### Task 4 — P1 audit: sweep for other silent dict-restriction no-ops and missed migration callers

A1/A2/A6 are instances of broader patterns this branch introduced; "fix once, audit all sites" (project rule). Grep + fix:
- **Silent dict-restriction no-op (P1):** any `query & {k: v}` where `k` comes from a multi-table key but the query joins only some tables — audit `resolve_artifact` / `resolve_source` and any direct `& {'artifact_id': ...}` / `& {'sorting_id': ...}` on a master in `spikesorting_merge.py` and v2. (The A1 site is the known one; confirm no siblings.)
- **str-vs-UUID restriction:** the same UUID-cast class as A1 — review reported it at [sorting.py:541](../../../../src/spyglass/spikesorting/v2/sorting.py#L541) (idempotency dedup compares a parsed str against a uuid column). Confirm and cast if real.
- **Missed migration callers:** the `get_curated_sorting` static→instance migration missed `decoding/v0/clusterless.py:201` (fixed in [Phase 5](phase-5-ci-infra.md)). Grep for any other unbound `Curation.<method>(key)` / `SpikeSorting.<method>(` call sites that should be instance calls after this branch's method-signature changes.

Report findings; fix the confirmed ones here (the clusterless caller itself is Phase 5).

## Deliberately not in this phase

- The export-safety gap (D2/D3/D4) — Phase 3. (A1/A2 are about *which* ids/files come back; export is about whether they're *logged*.)
- Routing the v2 accessors through `fetch_nwb` — Phase 3.
- The public-API docstring for `get_spiking_sorting_v2_merge_ids` advertising `artifact_id` is now correct once A1 lands; no doc change needed beyond confirming it.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_merge_ids_artifact_restriction_is_exclusive` (slow/integration) | Two sortings on one recording differing **only** by `artifact_id`; `get_spiking_sorting_v2_merge_ids({..., 'artifact_id': A})` returns exactly the merge_id of sort A (len==1), not both, not all. |
| `test_merge_ids_restrict_by_artifact_interval_name` (slow/integration) | The `restrict_by_artifact=True` + `interval_list_name="artifact_{uuid}"` path resolves to the single matching merge_id (str→UUID cast exercised). |
| `test_merge_ids_no_artifact_sort_unaffected` (slow/integration) | A sort with **no** `ArtifactSource` still resolves by rec/curation keys (the optional-part join didn't drop it). |
| `test_fetch_nwb_return_merge_ids_multi_source_aligned` (slow/integration) | Restriction spanning ≥2 sources (e.g. v1 + v2): `len(merge_ids) == len(nwb_list)` and each `merge_id` is the owner of the paired file. |
| `test_fetch_nwb_return_merge_ids_single_source_unchanged` | Single-source restriction returns the same `(nwb_list, merge_ids)` as before the fix (guards the shared-method change). |
| `test_delete_downstream_merge_shim_logs_not_raises` | Calling the deprecated `delete_downstream_merge` shim emits the deprecation log and does **not** raise `TypeError` (A6: `alt=` kwarg). |
| (Task 4 audit) | No remaining silent dict-restriction no-op or str-vs-UUID restriction in the merge-id resolution path; any found is fixed + covered. |

## Fixtures

Reuse the existing MEArec smoke fixture and the single-session populate helpers in [tests/spikesorting/v2/conftest.py](../../../../tests/spikesorting/v2/conftest.py) and `_ingest_helpers.py`. For A1, populate two `Sorting`s on one recording with two different `ArtifactDetection` rows (or one with / one without artifact). For A2's multi-source test, register one v1 and one v2 curation into `SpikeSortingOutput` (the `test_downstream_consumers.py` / merge tests already build v2 merge rows; add a v1 row). Mark all but the single-source equivalence test `slow`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- A1 join correctly resolves `artifact_id` through `ArtifactSource` and leaves no-artifact sorts unaffected; the str→UUID cast accepts both str and UUID.
- A2 change scopes merge_id resolution to the current source only; single-source output is unchanged.
- Validation slice passes; slow/integration tests are marked.
- Tests would fail with the bug reintroduced (revert the prod change locally and confirm red).
- No docstring/test names reference this plan or "Phase 1".
