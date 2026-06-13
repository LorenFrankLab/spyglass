# Phase 1 — Reference defaults in the SortGroupV2 grouping helpers

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make `SortGroupV2.set_group_by_shank` and
`set_group_by_electrode_table_column` inherit each electrode group's configured
reference (`Electrode.original_reference_electrode`) by default, resolving it
per group to one of the existing `reference_mode` values. Add a v1-style
`references` dict to `set_group_by_shank`. Fail loud on ambiguity. No schema or
table-structure change — the persisted columns and their validator are
unchanged.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/recording.py:275](../../../../src/spyglass/spikesorting/v2/recording.py#L275) —
  `set_group_by_shank`: current signature (`reference_mode="none"`,
  `reference_electrode_id=None`, no `references`), the per-group build loop
  (`:340-383`), the `omit_ref_electrode_group` branch (`:360-382`), and the
  master-row write (`:420-448`).
- [src/spyglass/spikesorting/v2/recording.py:463](../../../../src/spyglass/spikesorting/v2/recording.py#L463) —
  `set_group_by_electrode_table_column`: same default fields (`:469-470`),
  build loop (`:545-565`), master-row write (`:582-608`).
- [src/spyglass/spikesorting/v2/utils.py:194](../../../../src/spyglass/spikesorting/v2/utils.py#L194) —
  `ReferenceMode` literal + `_validate_reference_fields` (`:201`) +
  `assert_reference_not_member` (`:329`). The resolver's output must satisfy
  `_validate_reference_fields`.
- [src/spyglass/common/common_ephys.py:81](../../../../src/spyglass/common/common_ephys.py#L81) —
  `original_reference_electrode = -1: int` (the configured-reference column;
  read-only).
- [src/spyglass/spikesorting/utils.py:13](../../../../src/spyglass/spikesorting/utils.py#L13) —
  v1 `get_group_by_shank`: the algorithm to mirror. Sentinel handling is at
  `:76-94`; `references`-dict keying (by `electrode_group`, missing key →
  raise) at `:76-82`; `omit_ref_electrode_group` (omit the group that *contains*
  the reference electrode) at `:97-116`. **The mixed-reference branch at `:87-93`
  builds `ValueError(...)` but never raises it — Phase 1 fixes that.**

## Tasks

- **Add a pure resolver `resolve_group_reference(...)` to
  [v2/utils.py](../../../../src/spyglass/spikesorting/v2/utils.py)** (next to
  `assert_reference_not_member`; DB-free, unit-testable). It maps a group's
  configured/explicit reference to a validated `(reference_mode,
  reference_electrode_id)` pair. Sentinels are v1-compatible. It does **not**
  emit `"auto"` — only the three real modes. Use a private sentinel object for
  "auto" so an explicit/configured `None` can still mean "no reference".

  ```python
  _AUTO_REFERENCE = object()


  def resolve_group_reference(
      configured_ref_ids,          # iterable[int | None]:
                                   #   original_reference_electrode of every
                                   #   member electrode in the group
      *,
      explicit_ref=_AUTO_REFERENCE,
                                   # int/None sentinel from a `references`
                                   #   mapping, or _AUTO_REFERENCE to derive
                                   #   from config
      group_label="",              # for error messages
  ) -> tuple[str, int | None]:
      """Resolve one electrode group's reference to (reference_mode, id).

      Sentinels (v1-compatible): ``None`` or ``-1`` -> ``("none", None)``;
      ``-2`` -> ``("global_median", None)``; ``>= 0`` -> ``("specific", id)``.

      Auto-derivation (``explicit_ref is _AUTO_REFERENCE``) reads the per-member
      ``original_reference_electrode`` values and requires them to agree;
      mixed values raise (v1 silently fell through here). Membership validation
      is intentionally outside this resolver: grouping helpers must first honor
      ``omit_ref_electrode_group`` skips, then call
      ``assert_reference_not_member`` for groups they actually insert.
      """
      if explicit_ref is not _AUTO_REFERENCE:
          ref = -1 if explicit_ref is None else int(explicit_ref)
      else:
          uniq = sorted(
              {-1 if r is None else int(r) for r in configured_ref_ids}
          )
          if len(uniq) != 1:
              raise ValueError(
                  "SortGroupV2: electrode group "
                  f"{group_label!r} has mixed original_reference_electrode "
                  f"values {uniq}; cannot auto-derive a single reference. "
                  "Pass an explicit `references` entry for this group, or fix "
                  "the Electrode config."
              )
          ref = uniq[0]

      if ref == -1:
          return "none", None
      if ref == -2:
          return "global_median", None
      if ref < 0:
          raise ValueError(
              f"SortGroupV2: electrode group {group_label!r} has unrecognized "
              f"reference sentinel {ref}; expected -1 (none), -2 "
              "(global_median), or a non-negative electrode id."
          )
      return "specific", ref
  ```

  Keep `assert_reference_not_member` as the membership check. The helper-level
  call fails at group-creation for inserted rows; the existing materialization
  call remains the backstop for direct inserts / legacy rows.

- **Rewrite `set_group_by_shank`
  ([v2/recording.py:275](../../../../src/spyglass/spikesorting/v2/recording.py#L275)):**
  - Replace the `reference_mode: str = "none"` / `reference_electrode_id: int |
    None = None` params with a v1-style `references: dict | None = None`
    (keyed by `electrode_group_name`, value = reference electrode id /
    sentinel) **plus** the explicit overrides kept as optional
    `reference_mode: str | None = None` / `reference_electrode_id: int | None =
    None` (when `reference_mode` is not `None`, it forces that mode for every
    group this call creates and bypasses auto-derivation). Default
    (`reference_mode is None` and no `references`) = auto-inherit from config
    per group. Reject calls that combine a `references` dict with an explicit
    `reference_mode`.
  - Inside the per-(electrode_group, shank) loop (`:340-383`), gather the
    group's member electrode ids and their `original_reference_electrode`
    values, then call `resolve_group_reference(...)` with `explicit_ref =
    references[e_group]` when `references` is supplied (raise a clear
    `ValueError` naming the missing key if `e_group` is absent — mirror v1
    `:78-82`), else omit `explicit_ref` so the private auto sentinel is used.
    When the caller passed an explicit `reference_mode`, skip the resolver and
    use the override pair directly after validating it with
    `_validate_reference_fields`.
  - After resolving a group's reference, apply `omit_ref_electrode_group` before
    membership validation. For groups that will actually be inserted, run
    `assert_reference_not_member(resolved_mode, resolved_ref_id, group_elecs)`
    so explicit and auto-specific references fail early if they are still inside
    the sort group.
  - Write the **per-group** resolved `(reference_mode, reference_electrode_id)`
    into each master row (`:420-429`), not one call-wide scalar.
  - Update the docstring (`:286-319`): defaults now auto-inherit; document
    `references`; document the explicit-override path; document the mixed /
    in-group / missing-key failures.

- **Rewrite the `omit_ref_electrode_group` branch
  ([v2/recording.py:360-382](../../../../src/spyglass/spikesorting/v2/recording.py#L360)):**
  it currently only fires when the call-wide `reference_mode == "specific"`.
  Make it operate on each group's *resolved* reference: if
  `omit_ref_electrode_group` and the group's resolved mode is `"specific"` and
  the resolved reference electrode lives in *this* electrode group, skip the
  group (mirror v1 `:97-116`, which finds the reference electrode's group and
  omits it). Look up the reference electrode's group from **all** session
  electrodes, not only the bad-channel-filtered set, so a configured reference
  marked `bad_channel=True` can still identify the group to omit. If the
  resolved specific reference id is absent from the session's `Electrode` rows,
  raise a clear `ValueError`. Keep the existing `skipped` summary bookkeeping.

- **Update `set_group_by_electrode_table_column`
  ([v2/recording.py:463](../../../../src/spyglass/spikesorting/v2/recording.py#L463)):**
  - Same default change: drop the `reference_mode="none"` default in favor of
    auto-inherit-from-config per group, computed from each group's member
    `original_reference_electrode` values via `resolve_group_reference(...)`
    (using the default auto sentinel).
  - Keep a single **global** override (`reference_mode` / `reference_electrode_id`
    forcing one mode for all groups); validate the override pair once with
    `_validate_reference_fields` and run `assert_reference_not_member` for every
    inserted group. **Do not** add a per-group `references` mapping (Non-Goal).
  - Write per-group resolved values into the master rows (`:582-591`); update
    the docstring (`:484-508`).

- **Dev-state invalidation (when the new defaults are adopted):** re-running
  `set_group_by_*` flips a row's `reference_mode` (e.g. `none`→`specific`/
  `global_median`) while its `sort_group_id` PK is unchanged — so any
  downstream `RecordingSelection` / `Recording` / sorting / curation rows keyed
  on that `sort_group_id` are now stale (built with the old reference) yet keep
  their keys and are silently reused. Delete those downstream rows + cached
  analysis files before repopulating; see the
  [overview Rollout Strategy](overview.md#rollout-strategy). Unlike the Phase 2
  reorder (a no-op for the common specific-reference path), this default flip
  changes downstream output, so the invalidation is numerically required here,
  not just hygiene.

- **Documentation (ships in this phase):**
  - `docs/src/Features/SpikeSortingV2_Migration.md` — add a note: v2 grouping
    helpers now inherit the configured reference by default (matching v1),
    replacing the earlier v2 `"none"` default; mixed configured references in
    one group now raise (v1 silently mis-referenced via the un-raised
    `ValueError`).
  - `CHANGELOG.md` — one entry for the default-reference change + the new
    fail-early validation.

## Deliberately not in this phase

- The preprocessing-order change, `filtering_description`, the
  `_params/preprocessing.py` docstrings, and baseline regeneration — all
  Phase 2.
- A per-group `references` mapping for `set_group_by_electrode_table_column`
  (Non-Goal; revisit on demand).
- Any change to the `reference_mode` column, `ReferenceMode` literal, or
  `_validate_reference_fields` — the resolver produces values they already
  accept.

## Validation slice

| Test | Asserts |
| --- | --- |
| `resolve_group_reference` auto specific | uniform positive `original_reference_electrode` (e.g. all `5`) → `("specific", 5)`. |
| `resolve_group_reference` auto none | uniform `-1` and uniform `None` → `("none", None)` without confusing `None` with the auto sentinel. |
| `resolve_group_reference` auto global median | uniform `-2` → `("global_median", None)`. |
| `resolve_group_reference` mixed raises | members with `{3, 7}` and no `explicit_ref` → `ValueError` naming the group + the mixed values. |
| `resolve_group_reference` explicit overrides config | mixed config but `explicit_ref=-1` → `("none", None)` (no raise — explicit wins). |
| in-group specific raises | after resolving to `"specific"`, `assert_reference_not_member(..., group_member_ids)` raises before insert when the ref id is in the inserted group. |
| `resolve_group_reference` bad sentinel raises | `explicit_ref=-3` → `ValueError` (unrecognized sentinel). |
| `set_group_by_shank` default inherits *(integration, DB)* | a fixture session whose electrodes carry `original_reference_electrode=<id>` → created master rows have `reference_mode="specific"`, `reference_electrode_id=<id>` (not `"none"`). |
| `set_group_by_shank` `references` dict *(integration, DB)* | passing `references={group: -2}` → that group's row is `global_median`; a group missing from the dict → `ValueError` naming the key. |
| `set_group_by_shank` explicit `reference_mode="none"` *(integration, DB)* | forces `"none"` rows even when config has a specific reference. |
| `set_group_by_shank` `omit_ref_electrode_group` *(integration, DB)* | the electrode group containing a group's resolved specific reference is skipped before in-group validation and reported in the returned `skipped` list; a missing reference electrode id raises. |
| `set_group_by_electrode_table_column` auto + global override *(integration, DB)* | default inherits per group; an explicit `reference_mode="global_median"` forces CMR for all groups. |

Mark the DB-backed `set_group_by_*` tests as integration (need a populated
`Electrode` table); the `resolve_group_reference` tests are pure and run
without a DB.

## Fixtures

- Pure resolver and membership tests: plain Python lists/ints — no fixture.
- Integration tests: reuse the existing v2 conftest session fixtures that
  populate `Electrode`. **Update those fixtures/expectations** where they
  currently assume default `set_group_by_shank` produces `reference_mode="none"`
  rows — after this phase the default is the inherited reference. Confirm the
  fixture electrodes have a sensible `original_reference_electrode` (set one in
  the fixture if the synthesized electrodes default to `-1`, so the "auto
  specific" path is actually exercised).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no preprocessing-order
  changes leak in.
- Validation slice tests pass; integration tests are marked.
- Tests aren't trivial — the resolver and membership tests exercise each
  sentinel/raise branch, not tautologies; shared setup is in fixtures, not
  copy-pasted.
- Docstrings, test names, and module names don't reference this plan or its
  phases.
- The old call-wide-scalar reference path in both helpers is fully replaced by
  the per-group resolution (no dead `reference_mode`-scalar branch left behind).
- The migration-doc + CHANGELOG entries are present, not deferred.
