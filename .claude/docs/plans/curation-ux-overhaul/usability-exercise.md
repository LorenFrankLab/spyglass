# Browser-review usability exercise

Status: **awaiting moderated sessions with three current v1 FigURL users**.
Code-level acceptance is complete, but this human gate cannot be represented by
automated tests or inferred from implementation behavior.

## Protocol

Each participant uses the canonical curation notebook and one representative
session. The facilitator records time, errors, and interventions while the
participant independently:

1. starts an initial profile-backed review and opens FigPack;
2. inspects metrics/suggestions, makes label and merge edits, and saves;
3. reads `preview_import()` and explains the label, merge, unit-count, conflict,
   and sibling fields before committing;
4. commits, resolves any merge-label conflict without facilitator direction,
   and identifies whether merge verification is required;
5. discards the Python handle, resumes from `review_id`, and opens the
   continuation review;
6. verifies an actual merged waveform/correlogram and locates the final
   `merge_id` (or concat `member_merge_ids`) without constructing a key.

## Recording template

| Participant | v1 baseline | v2 total | Errors | Facilitator interventions | Material regression / fix |
| --- | ---: | ---: | ---: | ---: | --- |
| User 1 | pending | pending | pending | pending | pending |
| User 2 | pending | pending | pending | pending | pending |
| User 3 | pending | pending | pending | pending | pending |

Release remains gated until all three rows are completed and material
regressions are fixed and re-tested.
