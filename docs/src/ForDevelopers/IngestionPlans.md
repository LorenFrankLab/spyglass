# Ingestion Plans

This article explains how Spyglass decides what an NWB file would insert
*before* inserting it, what the resulting report tells you, and how the
`IngestionPlanLog` tables replace `InsertError`.

## Why plans exist

`insert_sessions` used to stop at the first table that raised. Everything after
it went uningested, the tables before it were already written, and the only
record was one `InsertError` row per exception — with no memory of what had been
staged. Fixing the file meant starting over and paying for the whole parse
again.

A *plan* is the parse without the insert. Spyglass reads the file once, works
out every row each table would write, checks those rows for problems, and
reports all of them together. Nothing is written, so a file with five errors
tells you about five errors instead of the first one.

## Planning a file

```python
from spyglass.data_import.planner import plan_nwbfile

plan = plan_nwbfile("minirec20230622_.nwb")
print(plan.report())
```

A clean plan is one line:

```text
minirec20230622_.nwb: no_op — already ingested, nothing to do
```

A plan with problems leads with the same verdict, then groups what it found by
severity, worst first:

```text
broken_.nwb: conflict — conflicts with what is already stored

New entries:
      1  `common`.`session`

Hard (2):
  [hard] `common`.`session`: missing_attribute: institution_name is required
         and absent (object abc-123)
      -> The NWB file does not supply a required column. Add it to the file,
         or declare it in the file's _spyglass_config.yaml.
  [hard] `common`.`subject`: divergence: already stored with different values
      -> The file disagrees with a row already stored. Apply the revision
         below, or correct the file to match.

Suggested revisions, to apply as-is:
  # `common`.`subject`
  {'sex': 'M'}

Blocked by the above (1): `common`.`electrode`
```

Each problem names the NWB `object_id` where there is one, so you can find the
thing in the file. Each *code* gets a one-line remedy, printed once per group
rather than under every occurrence.

`report(verbose=True)` also shows advisory problems, which are hidden by default
so that what blocks you is not buried in what does not.

### Verdicts

| Verdict       | Meaning                                          |
| ------------- | ------------------------------------------------ |
| `no_op`       | Everything in the file is already stored         |
| `all_new`     | Nothing in the file is stored yet                |
| `partial_new` | Some of it is stored; the rest is new            |
| `conflict`    | The file disagrees with something already stored |

### Blocked tables

A table whose parent could not be planned is reported once, at the end, as
blocked — not as a failure of its own. One root cause produces one problem,
rather than one per table downstream of it.

## The plan log

`common_usage.IngestionPlanLog` records a file's plan so a later attempt can see
what the last one achieved. It is a staging area for *incomplete* ingestion,
never a second copy of your data.

| Table                      | Holds                                                              |
| -------------------------- | ------------------------------------------------------------------ |
| `IngestionPlanLog`         | one live plan per file: verdict, status, attempt count, provenance |
| `IngestionPlanLog.Entry`   | one row per prospective entry, with its state and payload          |
| `IngestionPlanLog.Problem` | problems belonging to the file rather than to one entry            |

```python
from spyglass.common.common_usage import IngestionPlanLog

IngestionPlanLog & {"nwb_file_name": "minirec20230622_.nwb"}
IngestionPlanLog.Entry & {"nwb_file_name": "minirec20230622_.nwb"}
```

Each staged entry carries two hashes, which answer different questions:

- `key_hash` — of the entry's **primary key**. Its identity. A second attempt
    updates the row it already has rather than appending a duplicate, which is
    what lets attempt two stage N+M where attempt one staged N.
- `blob_hash` — of the **whole entry**. Its content. Same key with a different
    blob is a *divergence*, not a new row, and one hash could not tell those
    apart.

Entries keep their payload only while they are unstored. On full success they
are migrated to their real tables and the payload is cleared, leaving the hashes
behind as provenance. Storing your data twice is the failure this design exists
to avoid.

!!! note "The file hash is provenance, not a cache key"

    `nwb_hash` is recorded but never gates freshness. The expected workflow is
    ingest → read the report → **edit the file** → retry, so the file hash differs
    on every attempt by construction. Gating on it would discard the whole plan
    exactly when it is most useful. Validity is per entry.

## Migrating from `InsertError`

`InsertError` still exists and is still written, so existing queries keep
working. It is deprecated: it records one row per exception with no memory of
what was already staged, which is what `IngestionPlanLog` replaces.

| Instead of                              | Use                                                  |
| --------------------------------------- | ---------------------------------------------------- |
| `InsertError & {"nwb_file_name": name}` | `IngestionPlanLog.Problem & {"nwb_file_name": name}` |
| reading `error_message` per row         | `plan.report()`                                      |
| checking whether the list is empty      | the returned result is falsy when clean              |

The value returned by ingestion is falsy when nothing blocked, iterates over the
blocking problems, and prints as the report, so code written against the old
error list keeps working:

```python
result = ...  # returned by ingestion
if result:  # same test as before
    print(result)  # now prints the full report
    for problem in result:  # iterates blocking problems
        print(problem.code, problem.table)
```
