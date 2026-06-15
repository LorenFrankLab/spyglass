# Phase 4 — Friendly curation wrappers + `summarize_curation`

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

`insert_curation` is powerful but intimidating: ten parameters, `parent_curation_id=-1` sentinels, `apply_merge` vs proposed-merge semantics. Add three intent-first, discoverable classmethods on `CurationV2` that cover the common write intents, plus a `summarize_curation` read accessor. The expert `insert_curation` stays exactly as is.

**Inputs to read first:**

- [curation.py:197-268](../../../../src/spyglass/spikesorting/v2/curation.py#L197-L268) — `insert_curation` signature and the `apply_merge` / proposed-merge semantics the wrappers delegate to. **Do not reimplement merge logic** — wrappers only pre-fill arguments.
- [curation.py:1043-1080](../../../../src/spyglass/spikesorting/v2/curation.py#L1043) — `get_merge_groups()` (returns `dict[int, list[int]]`), used by `summarize_curation`.
- [curation.py:89](../../../../src/spyglass/spikesorting/v2/curation.py#L89) — the `merges_applied` field; `curation.py:156` — the `MergeGroup` part; the `Unit` and `UnitLabel` parts (read their exact names from the class body before writing `summarize_curation`).
- The `SpikeSortingOutput.CurationV2` merge registration used at [pipeline.py:281](../../../../src/spyglass/spikesorting/v2/pipeline.py#L281) to resolve `merge_id` from a curation key.

## Tasks

- **Add three write classmethods to `CurationV2`** (next to `insert_curation`). Each is a thin pass-through; the merge/label/DAG logic stays in `insert_curation`. These names are user-intent-first: avoid exposing "root" as the friendly method name, because root is DAG/internal language; avoid "preview" for a method that writes a persisted curation row. Do not add the rejected aliases `create_root_curation`, `preview_merge_curation`, or `apply_merge_curation`.
  ```python
  @classmethod
  def create_initial_curation(cls, sorting_key, labels=None, description=""):
      """Create the initial curation (no merges) over a sort. Sugar for
      insert_curation(parent_curation_id=-1, apply_merge=False)."""
      return cls.insert_curation(
          sorting_key=sorting_key, labels=labels,
          parent_curation_id=-1, description=description,
      )

  @classmethod
  def propose_merge_curation(cls, sorting_key, merge_groups, labels=None,
                             parent_curation_id=-1, description=""):
      """Record proposed merges WITHOUT applying them.
      Sugar for insert_curation(apply_merge=False, merge_groups=...).
      Every original unit keeps its id; merges live in CurationV2.MergeGroup
      for lazy application via get_merged_sorting()."""
      return cls.insert_curation(
          sorting_key=sorting_key, labels=labels, merge_groups=merge_groups,
          apply_merge=False, parent_curation_id=parent_curation_id,
          description=description,
      )

  @classmethod
  def create_merged_curation(cls, sorting_key, merge_groups, labels=None,
                             parent_curation_id=-1, description=""):
      """Create a new curation with merges applied (merged unit set is committed).
      Sugar for insert_curation(apply_merge=True, merge_groups=...)."""
      return cls.insert_curation(
          sorting_key=sorting_key, labels=labels, merge_groups=merge_groups,
          apply_merge=True, parent_curation_id=parent_curation_id,
          description=description,
      )
  ```
  - The `parent_curation_id` parameter (default `-1`) is the [overview.md risk-table](overview.md#risks-and-mitigations) mitigation: a real merge workflow branches proposed/applied merge curations off an existing initial curation, not always a second initial curation. `create_initial_curation` omits it (the initial/root curation is always `parent_curation_id=-1`).
  - Keep the existing merge-group validation in `insert_curation` (≥2 members per group) — the wrappers inherit it; do not duplicate or weaken it.
- **Add `summarize_curation(curation_key)`** as a `CurationV2` classmethod returning a plain dict (notebook-printable). It accepts either a minimal curation key (`{"sorting_id": ..., "curation_id": ...}`) or the full `run_v2_pipeline` manifest, normalizing internally to the curation PK before restricting `CurationV2`/parts. Do not pass a full manifest directly into DataJoint restrictions without normalization; manifest-only keys such as `pipeline_preset`, `recording_id`, `artifact_detection_id`, and `n_units` are not part of the curation primary key.
  ```python
  {
      "sorting_id": uuid.UUID,
      "curation_id": int,
      "n_units": int,                 # count of CurationV2.Unit rows for this curation
      "labels": dict[int, list[str]], # unit_id -> labels, from the UnitLabel part
      "merge_groups": dict,           # from get_merge_groups(key)
      "merges_applied": bool,         # the stored field
      "is_merge_preview": bool,       # has merge groups AND not merges_applied
      "merge_id": uuid.UUID | None,   # from SpikeSortingOutput.CurationV2 & key, None if unregistered
      "description": str,
  }
  ```
  Read each field from the existing parts/fields (cite them in the docstring); `is_merge_preview = bool(real_merge_groups) and not merges_applied`, where "real" means a group with >1 contributor (mirror the `merges_applied` logic at [curation.py:857-866](../../../../src/spyglass/spikesorting/v2/curation.py#L857-L866)). Resolve `merge_id` defensively (a curation may not be merge-registered in edge cases) — return `None` rather than letting a `fetch1` raise. Include both `sorting_id` and `curation_id` because `curation_id` is only unique within a sort.
- **Docs:** NumPy-style docstrings on all four methods, each cross-referencing `insert_curation` as the expert API and stating exactly which arguments it pre-fills. Add a short "Curation: quick path vs expert path" note to `docs/src/Features/SpikeSortingV2.md` (or the v2 feature doc) and a CHANGELOG entry. The Phase 6 notebook uses `create_initial_curation` + `summarize_curation`.

## Deliberately not in this phase

- **No FigPack / web curation.** Master-roadmap Phase 5. These are the programmatic curation surface; they also *de-risk* the FigPack work by giving v2 a complete non-UI curation path.
- **No new merge semantics or validation.** Wrappers delegate entirely to `insert_curation`; the merge-id assignment, lazy-vs-applied equivalence, and ≥2-member rule are unchanged.
- **No `apply_merge` default flip.** `insert_curation` keeps `apply_merge=False` default; the wrappers make the choice explicit by *name* instead.
- **No curation-JSON ingress** (`gh://.../curation.json`). That audit follow-up is tracked in master-roadmap [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) ("GitHub-hosted curation JSON ingress").

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_create_initial_curation_equiv` (db, slow) | `create_initial_curation(sort_key)` produces a curation identical (same `curation_id`, units, no merges) to `insert_curation(sort_key, parent_curation_id=-1)`. |
| `test_propose_merge_records_not_applies` (db, slow) | `propose_merge_curation(sort_key, [[a,b]])` → `merges_applied is False`, all original units retained in `CurationV2.Unit`, the merge present in `MergeGroup`; `summarize_curation(...)["is_merge_preview"] is True`. |
| `test_create_merged_curation_commits` (db, slow) | `create_merged_curation(sort_key, [[a,b]])` → `merges_applied is True`, unit count reduced by the merge, `is_merge_preview is False`. |
| `test_wrappers_inherit_singleton_rejection` (db) | `propose_merge_curation(sort_key, [[a]])` (singleton group) raises the same error `insert_curation` raises — validation not bypassed. |
| `test_propose_merge_off_existing_initial_curation` (db, slow) | `propose_merge_curation(sort_key, [[a,b]], parent_curation_id=<initial>)` creates a child whose `parent_curation_id` is the initial curation (DAG branch, not a second initial curation). |
| `test_summarize_curation_fields` (db, slow) | On a known curation, `summarize_curation` returns the right `sorting_id`, `curation_id`, `n_units`, `labels` (matching inserted `UnitLabel` rows), `merge_groups`, `merges_applied`, `is_merge_preview`, and a `merge_id` resolvable through `SpikeSortingOutput`. Assert both accepted inputs: a minimal curation key and a full pipeline manifest normalize to the same summary. |
| `test_summarize_unregistered_merge_id_none` (db) | A curation with no merge registration yields `merge_id is None` (no raise). |

Place in `tests/spikesorting/v2/test_curation_wrappers.py`. Reuse the `populated_sorting` / `populated_sorting_with_curation` fixtures ([conftest.py:181-302](../../../../tests/spikesorting/v2/conftest.py#L181-L302)); pick two real unit ids from the populated sort for the merge-group tests rather than hard-coding ids.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- The wrappers are pure pass-throughs — no merge/label logic reimplemented, validation inherited from `insert_curation` (the singleton-rejection test passes).
- `parent_curation_id` threads through proposed/applied merge wrappers so merges can branch off an initial curation.
- `summarize_curation` reads existing parts/fields (no new computation), resolves `merge_id` defensively, and `is_merge_preview` matches the real-merge-group definition.
- `insert_curation` is unchanged (the expert API is intact).
- The "Deliberately not in this phase" list is honored — no FigPack, no new semantics, no JSON ingress.
- Tests use real unit ids from fixtures, assert behavior (not the mock), and share setup via fixtures.
- Docstrings/test names don't reference this plan; CHANGELOG + feature-doc note landed.
