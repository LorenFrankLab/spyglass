# Phase 3 — Browser-first FigPack review and collaboration

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [design D4](designs.md#d4)

Makes the browser the **primary hands-on curation workflow**, matching how researchers actually use v1 FigURL while removing its key/URI/table plumbing. A researcher starts one persisted-profile review, edits in FigPack, previews the exact imported diff, commits an identity-verified child, and continues directly into a second review over the actual merged analyzer when scientific verification is required. Root, label-only, and merged curations work locally or hosted. The Phase-2 scripted facade remains the automation/expert alternative.

**Inputs to read first:**

- [figpack_curation.py:450-471](../../../../src/spyglass/spikesorting/v2/figpack_curation.py) — the `if upload: view.show(...)` / else `view.save(...)` branch; `:691-697` the `FigPackUploadError` refusing a hosted upload of a pre-seeded curation; `:717-719` `_write_seed_annotations` (offline only); `_existing_curation_state:413`.
- [figpack_curation.py:341-343](../../../../src/spyglass/spikesorting/v2/figpack_curation.py) `FigPackCurationNamespaceError` (the raw-namespace restriction to lift); `build_curation_view:735`, `fetch_curation_from_uri:774`, `save_curation_from_uri:787`; `figpack_config_hash` at `:42`.
- [_curation_analyzer.py](../../../../src/spyglass/spikesorting/v2/) (Phase 1) — `get_curation_analyzer`, the routing the view build now uses.
- FigPack offline/round-trip memory notes (figpack 0.3.x / SI 0.104; edited curation persists as `<figure>/annotations.json`).
- [v1 notebook workflow](../../../../notebooks/py_scripts/10_Spike_SortingV1.py) (`:228-290`) and [v1 FigURL implementation](../../../../src/spyglass/spikesorting/v1/figurl_curation.py) (`:21-231`) — the actual browser-first researcher journey this phase must preserve and simplify, not replace with a type-the-unit-IDs notebook flow.

**Contracts referenced:**

- [get_curation_analyzer](shared-contracts.md#get_curation_analyzer) — the view build reads the display-role analyzer through this (so merged curations render).
- [Figure identity](shared-contracts.md#figure-identity) — embed-and-verify; **do not weaken** the refuse-on-mismatch rule.
- [CurationReviewProfile](shared-contracts.md#curationreviewprofile) and [FigPackReview/import contracts](shared-contracts.md#figpackreview-and-import-contracts) — the primary UX surface.

**Designs referenced:** [D4](designs.md#d4), [D6](designs.md#d6).

## Tasks

- **Spike first (gates the rest), two independent halves:** (a) can a hosted figpack.org figure be **seeded** with prior curation state? (b) can a **remote** reviewer's edits **persist for read-back** (`preview_import`)? Record both in [Open Question 1](overview.md#open-questions). Fallbacks so the browser-first **primary** journey survives a failed spike: seed-fail → publish a prebuilt **seeded bundle** through controlled storage; **remote-persist-fail → read edits back from external controlled storage (a GitHub/kachery-style URI, mirroring v1's `FigURLCuration.get_labels/get_merge_groups(gh://…curation.json)` at `figurl_curation.py:214-231`), NOT figpack.org's own persistence.** The offline-local-bundle path is not a substitute for remote collaboration. Never emit a silently-empty hosted figure, and don't make `upload=True` the canonical primary path until (b) is proven.
- Implement `RunResult.start_review(source="analysis"|"root", profile, *, upload=False, ephemeral=False) -> FigPackReview` as the canonical entry point, with no silent analysis→root fallback. Add `CurationRef.start_review(...)` and `EvaluationResult.start_review(...)` as explicit lower-level forms. Resolve the immutable Phase-0 profile; insert/populate or reuse its evaluation; seed current committed labels; expose proposed labels/merges, selected evaluation metrics, and applied-merge provenance as non-binding display properties; embed the exact profile snapshot; build/reuse the FigPack selection; return a handle with `.open()` and `.preview_import()`. `FigPackReview.resume(review_id)` reconstructs it. Users never call FigPack selection/populate directly.
- Route `build_curation_view` (`figpack_curation.py:735`) through `get_curation_analyzer(role="display")` and **remove** `FigPackCurationNamespaceError` (`:341-343`) so merged and label-only curations open with real merged waveforms/correlograms/locations.
- Embed [figure identity](shared-contracts.md#figure-identity) (`sorting_id`, **`curation_uuid`**, `curation_id`, existing `figpack_config_hash`) into every built figure per [D4](designs.md#d4). Verify on `curation_uuid` (Phase 0) — **not** `curation_id`, which is reused after a delete (`curation.py:1062-1072`), so an old figure could otherwise pass against an unrelated replacement curation (B1). No `unit_namespace_hash`.
- Harden the verified direct-import primitive `save_curation_from_uri` (`:787`) to recompute `curation_uuid` from the target and refuse an embedded mismatch. The primary `FigPackReview` path does not ask users for a parent key; this primitive remains for expert/internal composition. Add the figure-config read (today only `annotations.json` is read). It gets no `allow_legacy`; unsafe identity-less import remains the separately named `import_legacy_figpack_curation(uri, asserted_parent=…, confirm_unverified_identity=True)`.
- Resolve the seed asymmetry per the spike outcome. Seed current committed labels on either hosted or controlled-bundle delivery. Show evaluation suggestions as read-only `proposed_*` properties until explicitly chosen; never make "open then import" accept all suggestions. In a merged view, show raw contributor groups as read-only `merged_from` provenance and seed no already-applied group as pending `mergeGroups`. `build_curation_view` displays selected evaluation/built-in properties here; Phase 4 extends the same explicit path with typed annotation sets.
- Keep review identity in the existing content-addressed FigPack selection plus embedded figure config rather than adding mutable workflow-state columns. The config contains `curation_uuid`, `review_profile_name`/`profile_hash`, exact `EvaluationSpec`, ordered display/label configuration, and delivery metadata; `resume(review_id)` verifies this snapshot against the DB. `curation_id` is informational and never the verification identity.
- Implement mutation-free `FigPackReview.preview_import() -> CurationChangeSet`: verify figure/parent identity, canonicalize and hash annotations, show label before/after, merge groups, unit-count delta, unresolved contributor-label conflicts, and newer sibling curations. No insert, file write, or populate occurs.
- Implement TOCTOU-safe `CurationChangeSet.commit() -> ReviewImportReceipt`: re-read and verify the same `curation_uuid` + `annotations_hash`; refuse changed-since-preview figures, unknown units, unresolved merge-label conflicts, and zero-diff commits unless `confirm_no_changes=True`; create/reuse a sibling child from the pinned parent; never overwrite or silently rebase. Apply the profile's explicit `replace`/`overlay` label mode.
- For a merge import, automatically evaluate the committed child using the profile's exact spec and return `needs_merge_verification=True`. Implement `receipt.continue_review()` to carry the profile, displayed properties, label palette, upload/local mode, and provenance into a seeded FigPack figure over the **actual merged analyzer**. Expose named, resumable stage status throughout. A label-only import returns a final child without forcing an analyzer rebuild.
- Show collaboration context in preview/receipt: reviewed parent identity, reviewer/created metadata when available, and newer sibling children. Multiple reviewers intentionally produce sibling curations; combining/choosing them is explicit.
- Docs: rewrite the canonical curation notebook around `start_review → browser → preview_import → commit → continue_review → final merge_id`. Put the scripted Phase-2 path and raw table methods in separate automation/expert appendices. Note the optional extra and local-vs-hosted choice.
- Run a moderated usability exercise with at least three current v1 FigURL users. Each must complete initial review, inspect the diff, commit, resume after losing the Python handle, verify an actual merged unit in the continuation review, and find the final `merge_id` without constructing a key or facilitator intervention. Record time/errors relative to v1 and fix material regressions before release.

## Deliberately not in this phase

- The resolver itself (Phase 1) and the facade value objects (Phase 2) — this phase consumes both.
- Typed annotation tables and curation-specific annotation-set selection (Phase 4); this phase leaves a clean extension point but does not pretend those tables already exist.
- Any change to metric computation.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_figpack_opens_merged_curation` | `build_curation_view` on a merged curation builds without `FigPackCurationNamespaceError` and includes the merged unit's waveform/correlogram. |
| `test_figure_identity_embedded` | A built figure carries `sorting_id`/`curation_uuid`/`curation_id`/`figpack_config_hash` (no `unit_namespace_hash`). |
| `test_save_refuses_reused_id` | A figure built on a curation that is deleted, its numeric `curation_id` reused by a new curation, is **refused** on import (the embedded `curation_uuid` no longer matches) — never attaches to the replacement (B1). |
| `test_save_accepts_matching_identity` | Importing a figure back onto its own curation creates the identity-verified child. |
| `test_start_review_uses_profile` | One profile name resolves the exact evaluation recipes/display/label configuration, evaluates idempotently, builds the seeded figure, and requires no selection/populate call from user code. |
| `test_profile_snapshot_is_figure_identity` | Two profiles with otherwise identical display settings but different evaluation semantics produce distinct config/review identities; resume returns the exact embedded snapshot. |
| `test_run_start_review_never_falls_back` | `source="analysis"` with no analysis curation raises an actionable error naming `source="root"`; it never silently reviews the root. |
| `test_suggestions_are_not_accepted_by_opening` | A no-edit import has zero label/merge diff even when evaluation proposed labels/merges; proposals are visible as read-only properties until the reviewer acts. |
| `test_applied_merge_is_read_only_provenance` | A merged review shows `merged_from` provenance but seeds no pending group containing raw contributor IDs; further edits use only the current merged namespace. |
| `test_preview_import_is_pure` | Preview reports exact label/merge/unit-count changes, conflicts, and newer siblings while leaving DB and files byte-for-byte unchanged. |
| `test_commit_rechecks_preview_hash` | Editing annotations after preview makes commit raise `ReviewChangedSincePreviewError`; a reused/deleted parent UUID is also refused. |
| `test_merge_label_conflict_requires_resolution` | Conflicting contributor labels never resolve by implicit precedence; commit requires an explicit resolution. |
| `test_merge_import_evaluates_and_continues` | Merge import reuses the profile spec, receipt requests verification, and `continue_review()` opens the actual merged waveform/correlogram with the same configuration. |
| `test_review_resume` | Dropping the Python handle and resuming by `review_id` reconstructs parent/profile/URI/stages and continues idempotently. |
| `test_collaborative_siblings_not_rebased` | Two reviews of one parent create sibling children; preview reports the newer sibling and neither review silently changes parent. |
| `test_explicit_no_change_review` | A zero-diff commit refuses by default and succeeds only with `confirm_no_changes=True`, recording an explicit reviewed/no-change result. |
| `test_save_refuses_legacy_by_default` | The verified `save_curation_from_uri` primitive refuses an identity-less figure (no `allow_legacy`); `import_legacy_figpack_curation(..., confirm_unverified_identity=True)` is the only escape hatch. |
| `test_seed_local_bundle` | An offline bundle of an already-labeled curation opens pre-seeded with those labels/merges. |
| `test_seed_hosted_or_fallback` | Per the spike: a hosted pre-seeded upload succeeds, OR the seeded-bundle fallback path is exercised — never a silently-empty hosted figure. |
| `test_figpack_extra_absent` | With the extra uninstalled, every FigPack entry point raises the actionable install-hint `ImportError`. |
| `test_remote_round_trip` *(slow, integration)* | Browser-first acceptance: start from a profile, import merge edits, re-evaluate, continue into the merged view, explicitly finalize, and obtain the correct child `merge_id`. |

## Fixtures

Reuse the Phase 1 merged-curation fixture. For the round-trip test, drive edits by writing an `annotations.json` into the bundle (the documented offline round-trip) rather than a live browser, so the test is deterministic and CI-runnable. Gate hosted-path tests on `FIGPACK_API_KEY` presence (skip cleanly when absent).

## Review

Dispatch `code-reviewer` against the diff. Confirm: the browser review is the documented default; profile resolution is persistent and exact; preview is pure; commit rechecks UUID+annotations hash; conflicts/no-change are explicit; merge import re-evaluates and continuation uses the actual merged analyzer; reviews resume and branch without silent rebasing; identity/seed/optional-extra guarantees remain intact; the usability exercise is recorded; no plan references in shipped code.
