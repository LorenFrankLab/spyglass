# PR #1574 — Combined Review

- **Date:** 2026-07-29
- **Branch:** `test-base-dir-safety`, local HEAD `1aaecaa2`
- **Scope:** `git diff origin/master...HEAD` — 59 commits, 15 files, +3719/−130
- **Method:** five specialist agents run in parallel (general code review, test
  coverage, silent-failure hunt, type design, comment accuracy), findings
  deduplicated and severity-ranked here. Claims marked **[verified]** were
  reproduced directly rather than accepted from an agent report.

> **Important context:** GitHub PR #1574 still points at `95da875`. None of the
> 59 commits reviewed here are visible to reviewers, and CI has never built
> them. The PR description still says symlinks are *skipped*, which is the
> opposite of current behavior.

---

## 1. The empirical headline

The test-coverage agent ran mutation testing: it disabled each safety guard in
turn and checked whether the suite noticed. **13 of 23 guards survived** — the
tests stayed green with the guard removed.

| Guard disabled | Location | Result |
|---|---|---|
| Age gate ignores alias timestamps | `common_nwbfile.py:191-193` | **survived** |
| Act-time tracked paths not `.resolve()`d | `:1118-1127` | **survived** |
| Target timestamp recheck removed | `:1261-1265` | **survived** |
| Target size / mode recheck removed | `:1259-1260`, `:1266-1267` | **survived** |
| Re-point (readlink) check removed | `:1226-1232` | **survived** |
| Non-regular target skip removed | `:991-995`, `:1249-1250` | **survived** |
| Candidate-keyset preflight removed | `:1347-1352` | **survived** |
| Target-snapshot-path check removed | `:1360-1367` | **survived** |
| stat/lstat disagreement check removed | `:1251-1255` | **survived** |
| "broken link now resolves" refusal removed | `:1268-1276` | **survived** |
| Unblock failure logged `debug` not `critical` | `:1699-1704` | **survived** |
| File deletion moved after DB cleanup | `:1655-1667` | **survived** |
| Vanished-mid-scan tolerance removed | `:874-877` | **survived** |
| Age gate always returns True | `:922-925` | caught (4 tests) |
| Alias timestamp check removed | `:1212-1218` | caught |
| Per-leaf recheck before unlink removed | `:1447-1454` | caught |
| Unlink failures swallowed | `:1471` | caught |
| `*.nwb` suffix invariant removed | `:1376-1380` | caught (2 tests) |
| Registry refresh fails open | `:1108` | caught |
| Walk errors skipped | `:848-849` | caught |
| Validation moved after `block_new_inserts` | `:1602-1612` | caught (incidentally) |

The tests that exist assert real on-disk state rather than mock calls — the
problem is coverage, not test quality.

---

## 2. Critical

### C1 — A stray symlink deletes a raw acquisition file  **[verified]**

`common_nwbfile.py:961-963` (tracked set), `:989` (tracked check), `:1437` (unlink)

`tracked` is built only from **analysis-store** externals: `_ext_tbl` returns
`schema.external["analysis"]` (`utils/mixins/analysis.py:192-198`). The raw,
recording, sorting, and video stores are never consulted.

Scenario: a user runs
`ln -s $SPYGLASS_RAW_DIR/sub-x.nwb $SPYGLASS_ANALYSIS_DIR/copy.nwb` to expose a
raw file to an analysis script. 24 h later the cron sweep sees the resolved raw
path as untracked, `_candidate_still_matches` approves it (valid in-root
symlink, target identity matches), and the **raw acquisition file is
unlinked**. Non-recomputable. The age gate cannot help — raw files are old by
definition — and one extra deletion never trips the fraction/ratio limits.

This is the concrete case that was not identified when the owner decided
"anywhere a link points is still pretty safe." The fix preserves that decision:
union the other stores into `tracked`, or refuse targets resolving inside
`raw_dir` / `recording_dir` / `sorting_dir` / `video_dir`. `raw_dir` is already
imported at `common_nwbfile.py:22`.

### C2 — Empty `accesses` deletes a valid file, then crashes uncaught  *(demonstrated by execution)*

`common_nwbfile.py:184-194`, preflight `:1347-1385`

A candidate with zero accesses passes the **entire** structural preflight — the
`.nwb` loop is vacuous over an empty tuple — the loop deletes a neighbouring
*valid* file, then `CleanupCandidate.newest_ns` raises
`ValueError: max() arg is an empty sequence` from `_is_old_enough`, which
`except OSError` does not catch. Exactly the "malformed plan partially
executed" outcome the preflight exists to prevent.

Fix: one line in `CleanupCandidate.__post_init__` rejecting empty `accesses`.
Strictly better than a preflight check, because the builder itself calls
`_is_old_enough` before a plan exists.

### C3 — `sys.exc_info()` in `finally` swallows an unblock failure  **[verified]**

`common_nwbfile.py:1692`, `:1708-1709`

`sys.exc_info()` returns the exception being handled **anywhere up the calling
stack**, not the one propagating from the enclosing `try`. Verified on 3.10.20:

| call context | body | `sys.exc_info()[1]` | unblock failure re-raised? |
|---|---|---|---|
| normal frame | ok | `None` | yes (correct) |
| normal frame | raises | body exc | no (correct) |
| inside an `except` handler | ok | **outer exception** | **NO — swallowed** |
| inside a `finally` during unwind | ok | **outer exception** | **NO — swallowed** |

Consequence: a caller shaped `except Exception: AnalysisNwbfile().cleanup()`
gets a clean return while `BEFORE INSERT` triggers stay installed. In the cron
driver that counts as success — `analysis_storage_failed` stays `False`, exit 0,
`CRON JOB END`, no email, no Slack. The single `critical` line is the only
trace, in a log truncated to 1000 lines.

Fix: an explicit `body_failed` flag set in `except BaseException: ...; raise`.

### C4 — `_teardown_test_data` has no tests at all

`tests/conftest.py:444` (definition), `:430` (only call site)

The **canonical-directory rule** (`:466-473`) and the **symlink refusal**
(`:474-479`) are the PR's headline test-safety guarantees and are entirely
unpinned. Delete `:466-473` and
`pytest --base-dir ~/spyglass_data/tests` — which passes the `UsageError` gate
at `:356`, since a `tests` component is present — would `rmtree` a developer's
real `export/`, `moseq/`, `recording/`, `spikesorting/`, `tmp/`. Nothing fails,
and because failures only print, not even a nonzero exit.

Testability blocker is one line: `data_root` is computed inside the function
(`:466`). Change the signature to `_teardown_test_data(base_dir, data_root=None)`
and all six branches become pure `tmp_path` unit tests — no DB, no Docker.

### C5 — The teardown `analysis` branch is a no-op  **[verified]**

`tests/conftest.py:509`

`child.glob("*.nwb")` is non-recursive, but every analysis file lives one level
deeper: `analysis/<session>/<file>.nwb` (`utils/mixins/analysis.py:364`).
Verified: **zero** flat `.nwb` files exist at the top level of
`tests/_data/analysis/`, while nested session directories hold files stamped
throughout the working day.

So the accumulation this branch set out to fix is **not fixed**, and
`tests/README.md` documents a manual `rm -rf` remedy for a backlog the code was
supposed to clear. See §5 for how this was introduced.

### C6 — Symlink loops raise uncaught `ELOOP` on Linux

`common_nwbfile.py:874-897`

A cycle of `*.nwb` symlinks under `analysis_dir` makes `os.stat` raise `OSError`
errno 62 (ELOOP), which `_snapshot_entry` does not catch — only
`FileNotFoundError`. One accidental loop wedges the entire weekly cleanup.
Platform-divergent: a 2-cycle returned ENOENT on darwin (treated as a broken
link and deleted); CI runs Linux.

---

## 3. Important

| # | Finding | Location |
|---|---|---|
| I1 | **Per-candidate refresh is O(N_candidates × N_tables) full-table fetches** with a `resolve()` per row. Master computed `tracked` once. Measured 0.32 s per 20k `resolve()` calls; ~20k external rows × 1000 candidates ≈ 30 min of path resolution plus 6000 round trips. | `:1395-1403` |
| I2 | **Cross-volume deletion audit log lost on abort.** `deleted_external` is logged only after the loop; a mid-loop connection drop leaves irreversible off-volume deletions with no record of which files. | `:1394-1469` |
| I3 | **Failure summary never reaches a notification channel** in Slack-only deployments — cleanup failures go to `on_fail`, which is email-only and returns silently when `SPYGLASS_EMAIL_SRC` is unset (a documented configuration). The `errors` list is only `print`ed, so even a working email says just "exit code 1". | `cleanup.py:193-197`, `run_jobs.sh:101-110` |
| I4 | **Unguarded `os.readlink`** — the same vanished-entry race is a skip or a hard abort depending which syscall loses, so the weekly job fails intermittently for a benign reason. | `:880` |
| I5 | **`import spyglass` breaks** for anyone with `test_mode: "true"` in a saved global config plus a production base dir. The guard raises at module scope (`settings.py:714`, `on_startup=True`), bypassing the `load_failed` degradation path built for bad configs, with an error telling them to "run pytest with `--base-dir`". `test_mode` ships in `dj_local_conf_example.json:33`. **[verified]** | `settings.py:290-315` |
| I6 | **Three inverted comments** in `_candidate_still_matches` / `_access_still_matches` claiming "nothing outside `analysis_root` is ever removed" — 235 lines above code that unconditionally unlinks out-of-root targets. A reader trusting them could remove the in-root gate at `:1235`. **[verified]** | `:1193-1195`, `:1137-1139`, `:1276-1277` |
| I7 | **Sweep fails closed on any per-entry OS error**, disabling weekly analysis maintenance until a human intervenes — and because the driver then sets `analysis_storage_failed`, it also skips `DecodingOutput.cleanup()` and the external analysis sweep, every week, silently. An entry that can't be stat'd never becomes a candidate, so a per-entry warn-and-skip is strictly safer operationally. | `:848-853`, `:874-897` |
| I8 | **`run_jobs.sh` exits before log truncation**, so a persistently failing cron grows the log unbounded. | `run_jobs.sh:107-110` |
| I9 | **Partial `block_new_inserts` failure** leaves triggers installed with an error message that never says so — unlike the unblock path, which gives explicit recovery instructions. | `:1623`, `:777-786` |
| I10 | **Driver cannot distinguish "refused before acting" from "failed mid-delete"**, so a pre-act refusal (validator, `_check_number`, malformed plan) needlessly gates unrelated phases for a week. No traceback is printed either. | `cleanup.py:65-72` |
| I11 | **`_teardown_test_data` pre-loop checks escape the aggregation** — `is_symlink()` / `exists()` sit outside the `try` and re-raise `EACCES`, stranding every later child. | `conftest.py:492-499` |
| I12 | **Vacuous tests.** `test_remove_untracked_files_refuses_path_outside_analysis_dir` passes on the 24 h age gate, not the guard it names. `test_delete_skips_relinked_symlink` refuses at the dev/ino check and never reaches the readlink comparison. `test_registry_knowledge_persists_across_candidates` asserts only `assert survivors`, which passes if nothing was deleted. | `test_nwbfile.py:225`, `:748`, `:1327` |
| I13 | **`settings.py` env-var tests only prove the negative** — five prove vars are ignored under `test_mode`, zero prove they are honored when `test_mode=False`. `env_or_none` could `return None` unconditionally and the suite passes. | `test_config_schema.py:790-796` |
| I14 | **No test drives `cleanup()` at production defaults on real data.** The three `cleanup()`-level tests patch `_remove_untracked_files` away; the one integration test disables all three guards via `no_limits`. | — |
| I15 | **`main()`'s analysis-storage suppression survives deletion** — the existing test proves only the function-level skip, and stubs `cleanup_external_files` with a non-analysis failure so the flag is never set. No success-path test exists, so `if errors:` → `if True:` passes. | `cleanup.py:152-157` |

---

## 4. Suggestions

- **Stale docstrings.** `cleanup()`'s `Process:` list still describes the
  pre-reorder sequence with filesystem deletion last (it is now first) and never
  mentions symlink following, off-volume deletion, the age gate, or broken-link
  removal — nor does it have a `Raises` section despite three raise paths
  (`:1548-1556`). `max_delete_fraction` says "fraction of **scanned** files"
  when the code divides by *eligible* (`:1579-1580`). `load_config` claims no
  env var is consulted under `test_mode`, but `SPYGLASS_BASE_DIR` is, ungated
  (`settings.py:155-157`, `:206-211`). `_snapshot_entry` has no return
  annotation (`:859`).
- **`CleanupPlan` 8 fields → 4 + derived.** `files_to_delete` is literally
  `set(candidates)`; `empty_files`, `untracked_files`, `broken_links` are read
  only by the dry-run count. `frozen=True` prevents rebinding, not mutation of
  the contained sets, and the dry-run path returns *before* the preflight — so a
  dry run can report N files for a plan a real run would reject.
- **Per-run summary.** A real run logs nothing about what it deleted; 100%
  refusal is indistinguishable from success. `N deleted, N skipped by reason,
  N deferred` would make a store misconfiguration visible.
- **Distinct `CleanupRefused(RuntimeError)`** so pre-act refusals don't gate
  unrelated maintenance phases.
- **Rot risk:** "every scanned file that is kept is tracked" appears in four
  places and is imprecise — it holds over the *eligible* set, not the scan.
  CHANGELOG still says "provenance recheck", vocabulary retired with the voucher
  model. Several comments reference "master" as a baseline, which becomes
  self-referential once merged.

---

## 5. Two corrections to earlier claims made during development

Both were stated confidently and were wrong. Recorded so they are not repeated.

1. **`sys.exc_info()` semantics.** Earlier verification covered only the
   body-raises branch and concluded the masking logic was sound. The
   body-succeeds-inside-a-caller's-`except` branch is the broken one (C3).
2. **The teardown glob.** Earlier analysis correctly identified that master's
   non-recursive `glob("*.nwb")` never matched the nested layout, and proposed
   `rmtree`. When a reviewer objected that recursive removal races concurrent
   sessions, the non-recursive glob was restored — reintroducing the diagnosed
   bug, and then documenting a manual `rm -rf` remedy for the accumulation the
   fix was meant to prevent (C5).

---

## 6. Strengths

Independently praised by multiple agents; worth preserving:

- The alias-timestamp comment naming the exact attack
  (`os.utime(link, follow_symlinks=False)` refreshes a link without changing
  dev/ino) — `:1208-1211`.
- `_current_custom_tables` stating plainly that refresh failure is fatal *and
  why* fallback is unsound — `:1093-1107`.
- `_check_number`'s NaN rationale ("under NaN every comparison is False and the
  guard silently vanishes") — `:76-80`.
- `settings.py`'s explanation of why non-strict `Path.resolve()` is the right
  tool — `:299-302`.
- `broken ⟺ target is None` — called the strongest design decision in the PR;
  a `broken: bool` beside a defaulted `size=0` would make "broken" and "empty"
  indistinguishable by field values.
- `test_delete_rechecks_each_alias_immediately_before_unlink` — wraps the real
  function, mutates the filesystem mid-pass as a racing writer would, asserts
  surviving file contents. No mock assertions.
- `test_validator_excludes_deferred_from_denominator` — pins a subtle reasoning
  bug a coverage-driven test would never find.
- The integration test snapshots files, rows, *and* externals across both tables
  before and after the dry run.
- `Management.md`'s "Deletion authority" section — the clearest statement of the
  security model, and the yardstick that proved I6 stale.

---

## 7. Open decisions

1. **Per-candidate refresh cost (I1)** — targeted per-candidate query
   (correct, more code), bounded cadence (cheap, small staleness window), or
   hoist back to once (fast, restores the stale-tracking bug)?
2. **Raw-store fix (C1)** — union the other stores into `tracked`, or refuse
   targets resolving inside them?
3. **Scope** — the full list is ~25 items. Suggested minimum: C1–C6 plus I6 and
   I12, with the rest filed as issues.

## 8. Still deferred by prior owner decision (not findings)

Cleanup lease / MySQL advisory lock; `fcntl.flock` pytest session lock;
recursive `analysis` teardown; full phase-health suppression;
`DecodingOutput.cleanup()` hardening; dry-run-by-default from #1573.

## 9. Known-good local noise

- `tests/_data/raw/mearec_obs_smoke_.nwb` is missing, so `test_nwbfile_cleanup`
  fails on the pre-work baseline too.
- `TestIntegration::test_custom_table_with_object_id_and_fetch_nwb` fails from a
  checksum collision caused by the accumulated `tests/_data/analysis` state
  described in C5.
- Ruff `F841` in `src/spyglass/utils/dj_graph.py` is pre-existing on master and
  not in this diff.
