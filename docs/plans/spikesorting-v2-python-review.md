# Spike sorting v2: Python reviewer comprehension audit

Reviewed the current branch through `81d05a4f`, including the three completed
round-three simplifications. This audit changes no implementation or tests.
The question here is where an experienced Python reviewer would need to stop
and reconstruct an unexpected contract. Item 1 is a correctness issue; the
remaining items are maintenance and clarity opportunities with different
payoffs.

## 1. A reference's identity guarantee depends on its Python representation

Location: `curation_api.py:96–125`, `CurationRef.from_key`.

Passing a `CurationRef` revalidates its generation UUID. Passing a mapping
reads only `sorting_id` and `curation_id`, silently ignoring a supplied
`curation_uuid`. Consequently, after a numeric ID is deleted and reused:

```python
CurationRef.from_key(stale_ref)          # raises CurationNotFoundError
CurationRef.from_key(asdict(stale_ref))  # returns the replacement generation
```

An expert Python user would reasonably expect dataclass serialization to
preserve the identity fields. The neighboring `AnnotationSetRef.from_key`
explicitly checks a supplied curation UUID (`annotation_api.py:128–135`), so
there is also an inconsistent convention across the public reference APIs.

This reaches meaningful callers: `select_units_for_analysis` resolves both
references and mappings through `CurationRef.from_key`, as do visualization
entry points. The delete/recreate lifecycle is supported and already covered
in `test_curation_api.py`; this is not an invented impossible state.

Suggested change: preserve current lookup behavior for bare two-field keys,
but verify `curation_uuid` whenever the mapping supplies it. Raise the existing
`CurationNotFoundError` for a mismatch. Keep the distinction clear in the
docstring: a bare key resolves the current generation; a supplied UUID pins
the requested generation. Centralize this check and reconcile the annotation
wrapper's overlapping validation/error expectations when implementing.

Verification: a read-only probe ran the actual constructor and identity-check
methods with a stubbed relation and reproduced both outcomes above. Extend the
existing delete/recreate integration scenario to cover `asdict(stale_ref)`, a
matching serialized ref, and a bare key resolving the replacement. Do not add
another reference class or serialization framework.

## 2. CurationOperation mixes record and mapping protocols without a consumer

Location: `curation_api.py:40–56`.

The two-field frozen dataclass also implements `Mapping`, with handwritten
indexing, iteration, and length. The field names therefore appear in the
dataclass declaration and again in the mapping implementation. A reviewer
must decide whether this is a record, a dictionary abstraction, or a
compatibility adapter.

There is an observable semantic surprise: dataclass-generated equality takes
precedence over mapping equality, so `operation == dict(operation)` is false
even though `operation` is a `Mapping` containing exactly those entries. This
is valid Python, but the dual protocol needs a reason.

All repository consumers found by search use attribute access or print the
object; none needs its mapping protocol. Keep a plain frozen dataclass and
remove the mapping methods and unused `Iterator` import. Use `asdict` at an
actual serialization boundary if one later needs it. This removes a draft
public capability, so describe that deliberately rather than claiming all
possible external behavior is unchanged.

Verification: the equality behavior was reproduced using the current class.
Existing operation-type and lineage assertions exercise the attribute API.

## 3. The public result types do not clearly describe the runtime receipt

Locations: `curation_api.py:858`; `_pipeline_run.py:597`, `:1187`, `:1310`,
`:1356`, and `:1574`; `_pipeline_types.py`.

`RunResult` is an unparameterized `dict` subclass with curation/review methods.
The separate `RunV2*Summary` TypedDicts describe result keys, but there is no
typed connection between those contracts and `RunResult`. The session runner
returns `RunResult` objects for successful entries and ordinary dicts for
failures, then casts the entire list to a union of TypedDicts. The UnitMatch
runner similarly assembles `dict[str, Any]` and casts its completed result.

A reviewer has to trace construction to learn which returned values have
methods and which have only keys. A cast does not validate the assembled
keys, and the batch annotation does not expose the successful receipt's
convenience methods.

Small improvement now: document the actual runtime distinction, use explicit
TypedDict construction/annotations for complete failure and UnitMatch
summaries, and separate those complete results from the intentionally partial
dict accumulated for error reporting. Do not merely cast more intermediate
values. A full redesign connecting a typed payload to every `RunResult`
mapping operation is a larger API/typing task and can wait until after launch.

Preserve dictionary access used in notebooks, success/failure discrimination,
generation-pinned accessors, and partial failure receipts. Existing import
and TypedDict metadata tests check declarations, not the assembled runtime
values; use the existing orchestration behavior tests when changing assembly.

## 4. Status names conceal different meanings and different kinds of object

Locations: `curation_api.py:20`; `_pipeline_types.py:26`;
`review_api.py:36`, `:126`, `:851–868`.

There are two different aliases named `StageStatus`:

- Curation: `Literal["computed", "reused"]`.
- Pipeline: `Literal["computed", "reused", "skipped"]`.

Review adds `ReviewStageState = Literal["computed", "reused", "complete"]`
and a `ReviewStageStatus` dataclass containing a name and one of those strings.
For the same `evaluation_populated` review step, initial review/resume can
report computed/reused, while the post-merge commit path reports complete.

The distinction is defensible, but a reader must infer that some statuses
describe work performed by this invocation while others only attest that a
step is complete. Identical alias names also make imports harder to reason
about.

Suggested change: give the narrower curation alias an explicit name such as
`MaterializationStatus`, and document review `complete` as completion without
a computed/reused classification. Keep the existing public pipeline status
contract. This needs neither a new status engine nor a union allowing every
status everywhere.

## 5. Some comments describe history or stronger guarantees than the code

Concrete examples:

- `_pipeline_types.py:243` describes every UnitMatch choice as committed.
  `_unit_match_member_choices` intentionally returns all matching curations,
  including previews; acceptance is enforced later. Describe these as
  discovered candidates, not already validated choices. Preserve discovery
  behavior while correcting the documentation.
- `_reference_resolution.py:21–29` and `curation.py:160–166` justify schema
  choices through a forbidden-ALTER/zero-migration policy. Given the current
  pre-production scope, those comments can make reviewers mistake historical
  process constraints for current architectural requirements. Explain the
  durable reason instead: extensible reference modes and custom labels can be
  added without changing column definitions. Existing production schemas
  still require their normal care.
- `_curation_plan.py:23–24` explains that "plan" is not a project milestone.
  That distinction has no bearing on the function's inputs, outputs, or
  algorithm and adds review noise.

Correct these locally. Keep comments explaining actual scientific invariants,
resource ownership, and necessary upstream compatibility workarounds. A broad
comment/style rewrite would add more review work than value.

## Suggested scope

Fix item 1 now. Items 2 and 5 are small clarity improvements; item 4 is a
bounded naming/documentation cleanup. Limit item 3 to clarifying and improving
the existing return construction before release; defer an API redesign.

The two Python behavior probes used no live database. No MySQL integration or
full test-suite run was performed for this read-only audit.
