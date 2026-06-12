# Label Repair (Issue #1513)

> **One-time bugfix tool.** This page documents a retroactive repair workflow
> for data affected by a specific, now-fixed bug. See [Fixing Bugs](./index.md)
> for context.

## Where

- `spyglass.spikesorting.v0.spikesorting_curation.Fix1513Status`

## Why

PR #1281 (2025-04-22) introduced an early-return bug in
`AutomaticCuration.get_labels`: only the first metric in `label_params` was
processed, so any `AutomaticCuration` entry whose `auto_curation_params_name`
lists more than one metric may have stored incomplete or incorrect unit labels
(and, in turn, an incorrect `accepted`/`rejected` status in
`Curation.curation_labels`, `CuratedSpikeSorting.Unit.label`, and the linked NWB
analysis file).

`Fix1513Status` is a `dj.Computed` table, populated from `AutomaticCuration`,
that recomputes the correct labels for each affected entry, classifies the scope
of the discrepancy, and lets the data owner choose how to repair it.

## How

### Scope

An entry is **out of scope** (no repair needed) if its `label_params` has at
most one metric with a non-empty label list — in that case the original (buggy)
and recomputed labels are identical, so `Fix1513Status` records
`action='none_needed'` automatically.

For **in-scope** entries (more than one such metric), `Fix1513Status` computes a
`label_diff` — the old and new label lists per unit — and classifies the change
into one of three cases:

| Case  | Meaning                                                              | Available actions            |
| ----- | -------------------------------------------------------------------- | ---------------------------- |
| **A** | No unit's accept/reject status changed (label text may still differ) | `keep`, `update`, `skip`     |
| **B** | Some units are newly **rejected**                                    | `keep`, `repopulate`, `skip` |
| **C** | Some units are newly **accepted**                                    | `keep`, `repopulate`, `skip` |

`update` only patches the label text in place (NWB units table +
`Curation.curation_labels`) and is safe **only for Case A** — it raises
`ValueError` for Cases B/C, because `CuratedSpikeSorting` excludes rejected
units, so a unit's presence/absence in `CuratedSpikeSorting` must change too.
Cases B and C require `repopulate`, which deletes and recomputes the affected
`CuratedSpikeSorting` entries.

### Recommended workflow

Step 1 — Admin fast pass: clear all out-of-scope entries as `none_needed`.
In-scope entries are silently skipped and remain for Step 2:

```python
from spyglass.spikesorting.v0 import Fix1513Status

Fix1513Status.populate(make_kwargs={"action": "none_needed_only"})
```

Step 2 — Each data owner reviews their own in-scope entries.
`pending_for_member` restricts `populate()` to sessions the member owns,
avoiding `PermissionError` on other members' curations:

```python
unreviewed, skipped = Fix1513Status.pending_for_member("Alice")
Fix1513Status.populate(unreviewed)  # interactive: prompts per entry
```

To batch-apply a single action without prompts (e.g. for Case A entries only),
pass `action` via `make_kwargs`:

```python
Fix1513Status.populate(unreviewed, make_kwargs={"action": "update"})
```

If running unrestricted and permission errors should not halt the run, use
DataJoint's `suppress_errors` (errors are still logged):

```python
Fix1513Status.populate(suppress_errors=True)
```

Step 3 — After **any** `populate()` call, finish the work that could not safely
run inside that call's transaction:

```python
# Activates NWB label edits staged by action="update"
Fix1513Status.activate_pending_nwb_repairs()

# Runs CuratedSpikeSorting.populate() for action="repopulate" entries
Fix1513Status.run_pending_repopulates()
```

### Checking outstanding work

```python
# Not yet reviewed (includes entries skipped by none_needed_only):
AutomaticCuration() - Fix1513Status()

# Deferred entries (explicitly skipped by a user):
Fix1513Status() & "action='skip'"

# Repopulate requested but not yet confirmed:
Fix1513Status() & "action='repopulate' AND repopulated=0"

# Per-member breakdown of unreviewed/skipped counts:
Fix1513Status.pending_summary()
```

### Audit trail

Every `Fix1513Status` row records `action`, `reviewed_by`, `owner_member`,
`reviewed_at`, and `notes`. The `label_diff` column stores the
`{"old_labels": ..., "new_labels": ...}` dict computed for the entry — for
`update`/`repopulate`, this is the only record of which labels were actually
rewritten, so it is populated for every action (not just `keep`/`skip`).

```python
(Fix1513Status & key).fetch1("label_diff")
```

### Verifying correctness

For a `repopulate`d entry, confirm the new `CuratedSpikeSorting.Unit` labels
match the `new_labels` recorded in `label_diff`:

```python
diff = (Fix1513Status & key).fetch1("label_diff")
units = (CuratedSpikeSorting.Unit & key).fetch("unit_id", "label")
```

For an `update`d entry, confirm the NWB units table's `label` column matches
`new_labels` (after `activate_pending_nwb_repairs()` has run).
