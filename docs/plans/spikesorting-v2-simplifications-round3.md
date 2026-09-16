# Spike sorting v2: three small remaining simplifications

Audited at `64676490`, after items 1–4 in
`spikesorting-v2-simplifications-round2.md` were implemented. These are
maintenance suggestions. This audit changes no pipeline code or tests.

Items 1 and 2 are straightforward removal of unnecessary machinery. Item 3
clarifies a database factory's error handling and merits focused behavioral
coverage. None requires a schema change. Further broad cleanup has diminishing
returns before release; keep the previously deferred merge-ID and cache work
outside this patch.

## 1. Remove the unused review-profile alias mechanism

Evidence: `review_profile.py:40–55` declares an empty `_SHIPPED_ALIAS_GROUPS`
and a helper that searches it. The insertion path at line 157 consults that
helper before rejecting duplicate content under a different name. Repository
search finds no configured aliases or other callers. Only one review profile
is shipped today.

Delete the empty constant and helper, and reject a claimed profile hash
directly. Update the class documentation and error message to describe the
current rule: reuse an existing profile name when its content is identical;
use a new name for changed content. An actual shipped compatibility alias can
justify adding alias support later.

Preserve normalization, profile hashes, ordered display/label semantics,
same-name idempotence, changed-content rejection, and duplicate detection
within one insertion batch. No generic immutability framework or new database
constraint is needed.

Verification: the existing persisted-profile test checks same-name reuse,
immutability, and duplicate-content rejection. Run that test against MySQL
when implementing; the pure normalization tests alone do not exercise table
insertion. No new test of the removed private helper is useful.

## 2. Declare each public export name once

Evidence: `_pipeline_public.py:7–117` declares 59 facade names and repeats 38
of them in `PACKAGE_ROOT_REEXPORTS`. Lines 120–127 then check at import time
that the second tuple is a subset of the first.

Keep the 38 shared names in `PACKAGE_ROOT_REEXPORTS`, declare the 21
facade-only names separately, and compose:

```python
PIPELINE_FACADE_EXPORTS = (
    *PACKAGE_ROOT_REEXPORTS,
    *_FACADE_ONLY_EXPORTS,
)
```

The subset relationship then follows from construction, so delete the runtime
subset check. Keep this as two explicit tuples; a registry, decorators, or
import introspection would add complexity.

Preserve both public name sets, the objects they resolve to, and package-root
lazy imports. This composition changes facade `__all__` ordering; repository
callers do not depend on the old order. Keep the existing package-root order
and choose a clear grouping for the facade-only names. No public import name
needs to be added or removed.

Verification: a dependency-free catalog probe confirmed 38 shared names and
21 facade-only names, with no existing duplicates and identical name sets
after composition. Run `test_pipeline_facade.py` and the package-root optional
import test in `test_service_import_contracts.py` after implementation. Compare
the before/after export sets once; avoid duplicating the entire catalog in a
new test solely to mirror its implementation.

## 3. Make annotation insertion's transaction and recovery explicit

Evidence: `CurationUnitAnnotationSet.from_dataframe` rejects an open outer
transaction at `unit_annotation.py:314`, then uses `transaction_or_noop` at
line 396. Its `except Exception` at line 400 attempts to reuse an existing
annotation set after every insertion/commit error.

Use `with cls.connection.transaction` directly in this factory and remove its
unused `transaction_or_noop` import. Retain the early outer-transaction error,
which explains the public calling contract.

Catch `dj.errors.DuplicateError` for the concurrent-insertion recovery path.
The installed DataJoint translates duplicate-entry errors to this type;
foreign-key violations use a separate exception. Preserve `_reuse_existing`
and all its provenance and stored-value hash checks, including re-raising when
no winner exists. Let other failures propagate without an additional recovery
query. This is a deliberate tightening of error recovery, not solely a syntax
change: an unrelated failure after a commit may require a caller retry, which
the existing pre-insertion reuse check already supports.

Verification: a read-only probe of the actual insertion block, using stubbed
writes/transactions, compared current and proposed handling. Both created a
new result normally and reused a duplicate-insertion winner. The proposed
handler propagated foreign-key and value errors without the current extra
reuse query. This does not validate MySQL transaction behavior.

When implementing, add focused behavior coverage for duplicate-winner reuse
and propagation of an unrelated insert failure without recovery. Run the
existing annotation integration test for atomic insertion, idempotence,
provenance, exact curation namespace, and typed-value round trips. Preserve
special-value encoding, set identity, and all schema definitions.

## Audit validation

The existing database-free facade, annotation, and review-profile tests passed:
**19 passed, 2 database tests deselected**. MySQL integration tests were not
run. The probes above were performed without editing repository source or
tests; they are evidence for the recommendations, not validation of an
implemented patch.
