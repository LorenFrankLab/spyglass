# Spike sorting v2: additional Python contract issues

Audited at `be0676a9`, after the first Python-review changes and follow-up
corrections. No implementation or test files were changed by this audit.
These three findings concern observable behavior at existing public API
boundaries, rather than a proposal for another architectural refactor.

## 1. Validate identifiers without changing their meaning

Evidence: `curation_api.py:947–985`, `_validate_merge_groups`, normalizes each
group with `list(map(int, group))` before checking shape and membership.

Using the actual function with a stubbed parent containing units 1–4 produced:

```python
[[1.9, 2.9]]  -> [[1, 2]]
["12"]        -> [[1, 2]]
[[True, 2]]   -> [[1, 2]]
```

`int()` truncates fractional values and accepts booleans. Iterating a string
group turns its characters into separate IDs. All three malformed requests
can therefore pass membership checks and become a valid merge of units the
caller did not actually identify. Both scripted curation helpers and
`EvaluationResult` merge methods pass these normalized groups to the writers.

A related boundary is `CurationRef.from_key` at line 113, where `int()` also
truncates a user-supplied fractional curation ID.

Suggested scope:

- Reject strings/bytes as group containers before iterating them.
- Normalize user-supplied identifiers losslessly: accept the intended integer
  representations, including NumPy integers, and reject fractional values and
  booleans. Use the same rule for facade curation IDs and unit IDs in merge
  groups where appropriate. Do not change conversions of integers already fetched from
  database columns.
- Preserve supported transport representations deliberately. FigPack's
  existing test requires numeric string IDs, such as `[["8", "9"]]`, to
  decode to `[[8, 9]]`; that is different from accepting `"89"` as a group.
  If sharing normalization with transport code, retain those semantics.
- Keep existing empty/singleton/overlap/membership checks. This does not require
  changing merged-unit assignment, merge composition, or database schemas.

Verification: add behavior coverage for the three malformed examples and
valid integer/NumPy-integer groups. Preserve any deliberately supported
integral-float representation rather than silently changing that contract.
Check that invalid inputs fail before a curation writer is called.

An optional local cleanup while touching this function: the duplicate-ID
comprehension at lines 973–976 repeatedly calls `flattened.count`, giving
quadratic work. `Counter(flattened)` expresses that check directly in one
pass. It is not a reason to reopen the deferred merge-engine work.

## 2. Keep matcher lookup from overwriting registration

Evidence: `matcher_protocol.py:161–175` and `:193–205`. Every matcher/schema
lookup calls `register_default_matchers`, which unconditionally calls
`_unitmatch_backend.register()`. That registers a fresh backend and schema;
the backend also registers itself as an import side effect.

Two observed consequences:

1. Two `get_matcher("unitmatch")` calls return different backend instances.
2. After bootstrapping the built-in, `register_matcher(replacement, schema,
   replace=True)` succeeds, but the subsequent `get_matcher("unitmatch")`
   raises a name-collision error when bootstrap attempts to reinstall the
   built-in class. This contradicts the documented replacement option.

A reviewer expects a getter to retrieve the installed object. Here a getter
also decides which implementation should occupy the registry, duplicating the
registration function's responsibility.

Suggested change: make lazy default installation fill missing defaults while
preserving explicitly registered objects and their schemas. Keep one clear
owner for bootstrap and make its interaction with explicit replacement work
on both a cold import and an already-imported backend. Preserve rejection of
unapproved name collisions and rebuilding defaults after the registry is
cleared. A new plugin framework is unnecessary.

Verification: extend registry coverage to assert stable object identity across
reads and successful get/schema lookup after an explicit built-in replacement.
Cover first-import ordering as well as the existing bootstrap-after-clear
scenario. The current replacement test uses a custom name and misses the
built-in bootstrap interaction.

## 3. Give the preset registry ownership of its validated configuration

Evidence: `register_pipeline_preset` at `_pipeline_presets.py:728–735` stores
an incoming `_PipelinePreset` instance directly. The model is mutable and its
configuration does not validate assignment. Mapping inputs take a different
path and become a new validated model.

After registering a model, the caller can change `preset.sorter` and thereby
change the registered recipe under the same name, without another registration
or row validation. Assigning `None` to that required string field also works.
This undermines the function's refusal to overwrite existing names and makes
its behavior depend on whether the caller supplied a dict or a model.

Suggested change: always create a fresh validated model from the supplied
content, using `model_dump()` for an existing model, before the normal row
checks and registry insertion. That makes registration capture the current
configuration and avoids retaining a caller-owned mutable object. Merely
copying a possibly modified model without validation would leave the second
problem unresolved.

All current preset fields are scalar recipe names or metadata. This needs no
deep-immutability framework or change to the public function signature.

Verification: register a model, mutate the original, and confirm the registry
still holds the original configuration. Also reject a model whose fields were
made invalid before registration. Retain the existing mapping-input,
unknown-field, name-collision, and referenced-row tests.

## Audit evidence and scope

All three behaviors were reproduced in isolated Python processes. Merge
membership reads were stubbed; the registry and preset probes used the actual
implementations without a database. The preset probe used `validate_rows=False`
to isolate object ownership from database availability.

The selected existing matcher, default-bootstrap, and preset-registration
tests passed: **15 passed**. They do not cover the reproduced cases above.
No MySQL integration tests were run.

Prioritize lossless identifier handling and matcher registration behavior.
Preset ownership is also a small fix. Keep the next patch limited to these
boundaries and their behavior tests; no schema or broad API redesign is needed.
