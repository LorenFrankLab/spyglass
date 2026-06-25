# Spike Sorting V2 Docs Build and Link Integrity Review

Date: 2026-06-25

Scope: v2-facing Markdown docs, MkDocs navigation, docs build instructions,
relative links, install snippets, copy-paste examples, and stale prose that
contradicts current code or notebooks. This is a different lens from user/API
ergonomics, downstream consumer contracts, dependency/runtime safety, and
general test coverage.

Method: local static inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run a full MkDocs
build because neither the base shell Python nor the local v2 test environment
has `mkdocs`, `mike`, `mkdocs-jupyter`, or `mkdocstrings` installed.

## Executive Summary

The v2 docs are substantial and generally close to the implementation, but the
build/link layer has several paper cuts that will disproportionately affect new
users: one tracked storage-management page is not discoverable from the nav,
some status/migration prose still describes implemented features as roadmap, and
the Markdown/downstream examples disagree with the notebook about which
`merge_id` should be carried into later analysis after curation.

The most important fixes are small documentation edits, not code rewrites:
surface the storage-management page, remove stale "placeholder" text for
UnitMatch and concatenation, correct the downstream final-curation handoff, and
make docs build commands work from a clean checkout.

## What Looks Solid

- The main v2 page links to the v2 notebook through the generated
  `docs/src/notebooks` path that the build script populates
  (`docs/src/Features/SpikeSortingV2.md:135`,
  `docs/build-docs.sh:11-15`).
- The main v2 page includes concrete, current sections for chronic concatenation
  and cross-session UnitMatch workflows
  (`docs/src/Features/SpikeSortingV2.md:784-855`,
  `docs/src/Features/SpikeSortingV2.md:880-944`).
- The MkDocs nav includes the primary v2 overview and migration pages
  (`docs/mkdocs.yml:90-91`).
- The notebook-side downstream handoff is explicit about using the final
  curation `merge_id`, which gives the Markdown a good source of truth to copy
  (`notebooks/py_scripts/10_Spike_SortingV2.py:490-505`).

## Findings

### 1. High: Markdown downstream examples point users at the root curation merge id after further curation

The main docs say that for most downstream work users should carry
`merge_id = run_summary["merge_id"]` forward
(`docs/src/Features/SpikeSortingV2.md:976-999`). That `merge_id` is the root
curation produced by `run_v2_pipeline`: the pipeline inserts a root
`CurationV2` row with empty labels and `parent_curation_id=-1`
(`src/spyglass/spikesorting/v2/_pipeline_run.py:401-416`).

The notebook has the safer wording: after auto/manual curation, key off
`final_merge_id`, not `run_summary["merge_id"]`, because the latter is the
uncurated root curation
(`notebooks/py_scripts/10_Spike_SortingV2.py:490-505`).

Impact: users who follow the Markdown after doing the documented curation steps
can decode, export, or analyze the uncurated root units and miss labels or
committed merges from the final curation.

Recommended fix: update the Markdown downstream section to mirror the notebook.
Describe `run_summary["merge_id"]` as the root/pre-further-curation output, and
show `final_merge_id = CurationV2.summarize_curation(final_curation)["merge_id"]`
after `materialize_curation` or manual `insert_curation`. Add a docs/notebook
smoke check that asserts the Markdown handoff names the final curation path.

### 2. Medium-high: the v2 storage-management page is orphaned from the docs nav and feature index

`docs/src/Features/SpikeSortingV2StorageManagement.md` exists and documents the
recording/analyzer recompute workflow
(`docs/src/Features/SpikeSortingV2StorageManagement.md:1-112`). It is not listed
in the MkDocs Features nav, which currently includes only the v2 overview and
migration pages (`docs/mkdocs.yml:79-91`). It is also absent from the Features
index, which links only the main v2 page
(`docs/src/Features/index.md:6-20`).

Impact: the page can be present in the repository but invisible to operators
who need the safe artifact-deletion workflow. A nav-completeness or strict docs
build check could also flag a tracked page that is never surfaced.

Recommended fix: add "Spike Sorting v2 Storage Management" next to the v2
overview/migration nav entries, add it to `docs/src/Features/index.md`, and link
the recompute bullet in `SpikeSortingV2.md` to it
(`docs/src/Features/SpikeSortingV2.md:87-89`). Add a simple docs inventory check
that every `SpikeSortingV2*.md` file is either in nav or intentionally excluded.

### 3. Medium-high: v2 status and migration prose still describes shipped UnitMatch/concat APIs as roadmap or placeholders

The main v2 page documents live chronic concatenation and UnitMatch workflows
(`docs/src/Features/SpikeSortingV2.md:784-855`,
`docs/src/Features/SpikeSortingV2.md:880-944`), but its Status section still says
cross-session unit matching is not yet available and that Unit matching remains a
placeholder (`docs/src/Features/SpikeSortingV2.md:1055-1066`).

The migration guide has the same stale wording: it says `unit_matching` and
`matcher_protocol` are placeholder modules and that `ConcatenatedRecording` /
`SessionGroup` methods are unimplemented
(`docs/src/Features/SpikeSortingV2_Migration.md:153-160`). Its feature table
still marks session-group concatenation and UnitMatch as roadmap
(`docs/src/Features/SpikeSortingV2_Migration.md:162-169`).

Current code has real `SessionGroup`, `ConcatenatedRecordingSelection`,
`ConcatenatedRecording`, `UnitMatchSelection`, and `UnitMatch` implementations
(`src/spyglass/spikesorting/v2/session_group.py:51-93`,
`src/spyglass/spikesorting/v2/session_group.py:330-486`,
`src/spyglass/spikesorting/v2/session_group.py:568-590`,
`src/spyglass/spikesorting/v2/unit_matching.py:198-351`,
`src/spyglass/spikesorting/v2/unit_matching.py:438-480`).

Impact: users get mutually exclusive guidance on the same page and may avoid
supported APIs or expect imports to fail. Migration readers may also choose v1
fallbacks unnecessarily.

Recommended fix: update the Status and migration sections so only genuinely
pending items, such as FigPack web curation views, remain pending. Add a static
docs grep/lint check banning stale phrases like `unit_matching.*placeholder`,
`UnitMatch.*roadmap`, and `ConcatenatedRecording.*roadmap` unless explicitly
marked as historical.

### 4. Medium: docs use stale `cache_hash` terminology for the recording content fingerprint

The drift-QC section says populating `DriftEstimate` leaves the upstream
`Recording.cache_hash` unchanged
(`docs/src/Features/SpikeSortingV2.md:780-782`). The migration guide also says
the recording cache carries an `NwbfileHasher` `cache_hash`
(`docs/src/Features/SpikeSortingV2_Migration.md:102-111`).

The current `Recording` table stores `content_hash`, not `cache_hash`
(`src/spyglass/spikesorting/v2/recording.py:1016-1025`). The storage-management
page already uses the newer content-fingerprint framing
(`docs/src/Features/SpikeSortingV2StorageManagement.md:26-34`).

Impact: copied queries such as `fetch1("cache_hash")` fail, and the migration
guide points users back toward the old whole-file-hash mental model rather than
the new content fingerprint.

Recommended fix: replace v2-doc references to `cache_hash` with `content_hash`
and describe it as the recording content fingerprint. Add a docs grep test that
fails on `cache_hash` in v2 docs unless it is explicitly discussing old/v1
terminology.

### 5. Medium: clean-checkout direct MkDocs instructions cannot work as written

The build script prepares generated docs inputs by copying `CHANGELOG.md`,
`LICENSE`, `QUICKSTART.md`, and notebooks into `docs/src`
(`docs/build-docs.sh:6-17`). From a clean checkout before that prep step,
several v2-facing paths do not exist under `docs/src`, including
`docs/src/CHANGELOG.md` and `docs/src/notebooks/10_Spike_SortingV2.ipynb`.

The docs README nevertheless gives a direct auto-reload command, and the command
has a typo: `mkdocs serve -f ./docs/mkdosc.yaml`
(`docs/README.md:45-48`). The build script comment also says
`./docs/mkdocs.yaml`, while the actual config is `./docs/mkdocs.yml`
(`docs/build-docs.sh:1-5`, `docs/mkdocs.yml:43-91`).

Impact: contributors trying to validate a v2 docs edit can run a command that
points at a nonexistent config, or can run MkDocs without the generated files
that v2 links depend on. Link failures become environment/order dependent.

Recommended fix: correct both references to `./docs/mkdocs.yml`, and document a
copy/prep step for direct `mkdocs serve`. A lightweight `docs/prep-docs.sh` or a
`build-docs.sh prep` mode would make link checking from a clean tree
deterministic.

### 6. Medium: the generic export example is copy-paste broken

The export docs example defines `export_key` with bare identifiers:
`{paper_id: "my_paper_id", analysis_id: "my_analysis_id"}`
(`docs/src/Features/Export.md:119-134`). That parses, but it raises `NameError`
unless variables named `paper_id` and `analysis_id` already exist. The same
example calls `Export().populate_paper(**export_key)`, but `populate_paper`
accepts `paper_id` and not `analysis_id`
(`src/spyglass/common/common_usage.py:485-507`).

Impact: v2 paper-export users are routed through `Export`, and the general
example they will copy fails at runtime or passes an unexpected keyword argument.

Recommended fix: use quoted keys and call
`Export().populate_paper(paper_id=export_key["paper_id"])`. Add a tiny docs
snippet smoke test that parses and executes non-DB setup lines from scoped
examples.

### 7. Medium-low: the UnitMatch install snippet is checkout-only and omits the base v2 extra

The UnitMatch section shows only an editable checkout command:
`pip install -e ".[spikesorting-v2-matching]"`
(`docs/src/Features/SpikeSortingV2.md:888-894`). The main environment section
uses the PyPI package form for base v2:
`pip install "spyglass-neuro[spikesorting-v2]"`
(`docs/src/Features/SpikeSortingV2.md:1042-1049`). The matching extra is
separate from the base v2 extra (`pyproject.toml:112-145`).

Impact: PyPI users outside a repository checkout will get a pip error from the
editable command. Fresh environments can also install UnitMatch dependencies
without the base v2 sorter/runtime dependencies if they copy only that snippet.

Recommended fix: show the user-facing command first:
`pip install "spyglass-neuro[spikesorting-v2,spikesorting-v2-matching]"`, then
show the editable form as a developer alternative. If the intended workflow is
to keep matching in a separate throwaway environment, say that explicitly.

### 8. Low: docs build version fallback uses an unset variable

`docs/build-docs.sh` computes `full_version=$(git describe --tags --abbrev=0)`
but then slices `${version_string:0:3}` instead of `${full_version:0:3}`
(`docs/build-docs.sh:29-36`). If `MAJOR_VERSION` is unset, the script will fall
through to `dev` even when a valid tag exists.

Impact: local or forked docs builds can publish under `dev` unexpectedly, making
versioned validation confusing. This is shared docs infrastructure rather than a
v2-specific content bug, but it affects v2 docs release previews.

Recommended fix: slice `full_version`, or make the fallback explicit and tested.
Add a small shellcheck-style test for the no-`MAJOR_VERSION` path.

## Verification Gaps

- I could not run `mkdocs build --strict`; `mkdocs` is not installed in the base
  Python or in `/Users/edeno/miniconda3/envs/spyglass_spikesorting_v2/bin/python`.
- I did not execute DB-backed snippets. This pass used source signatures and
  static snippet inspection.

