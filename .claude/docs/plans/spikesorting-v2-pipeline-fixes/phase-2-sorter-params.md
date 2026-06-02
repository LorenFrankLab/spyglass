# Phase 2 — Sorter params: validation, reproducibility, MS5 toggles

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Five contained fixes to the v2 sort/param layer: close the bulk-`insert`
validation hole, add the missing MS5 preprocessing toggles, guard per-channel
`noise_levels` length, seed the analyzer's stochastic spike subsampling, and
preserve the exception cause on the curation enum re-raise.

**Inputs to read first:**

- [_params/sorter.py:36-110](../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L36-L110) — `MountainSort4Schema` (has `filter:bool=False`, `whiten:bool=True` at `:70-71`) vs `MountainSort5Schema` (`:90-110`, missing both).
- [recording.py:135-143](../../../../src/spyglass/spikesorting/v2/recording.py#L135-L143) — `SortGroupV2`: the exemplar overriding **both** `insert1` and `insert` with validation. Mirror this shape.
- [curation.py:169-189](../../../../src/spyglass/spikesorting/v2/curation.py#L169-L189) — `CurationV2.UnitLabel`: second exemplar of the both-override pattern.
- [sorting.py:96-120](../../../../src/spyglass/spikesorting/v2/sorting.py#L96-L120) — `SorterParameters.insert1` (validates via `_validate_params`); no `insert` override.
- [sorting.py:72-87](../../../../src/spyglass/spikesorting/v2/sorting.py#L72-L87) and [sorting.py:1728-1770](../../../../src/spyglass/spikesorting/v2/sorting.py#L1728-L1770) — `_clusterless_noise_levels` and the `locally_exclusive` consumption (`noise_levels[chan]`, the P3 site).
- [sorting.py:2069-2097](../../../../src/spyglass/spikesorting/v2/sorting.py#L2069-L2097) — analyzer build / `compute([... 'random_spikes' ...])`, the reproducibility site. Cross-ref the existing seed pins (`seed=0`, SI PR #3359) used for whitening/noise_levels.
- [curation.py:418-425](../../../../src/spyglass/spikesorting/v2/curation.py#L418-L425) — `metrics_source` enum re-raise without `from`.

## Tasks

### Task 1 — P2: override bulk `insert` on the four param Lookups

`SorterParameters` ([sorting.py:117](../../../../src/spyglass/spikesorting/v2/sorting.py#L117)), `ArtifactDetectionParameters` ([artifact.py:234](../../../../src/spyglass/spikesorting/v2/artifact.py#L234)), `MotionCorrectionParameters` ([session_group.py:143](../../../../src/spyglass/spikesorting/v2/session_group.py#L143)), and `PreprocessingParameters` ([recording.py:656](../../../../src/spyglass/spikesorting/v2/recording.py#L656)) override only `insert1`. A bulk `insert([...])` bypasses `_validate_params` and the schema-version check. Add an `insert` override on each that maps the same validation over rows, mirroring `SortGroupV2.insert` ([recording.py:139-143](../../../../src/spyglass/spikesorting/v2/recording.py#L139-L143)). E.g. for `SorterParameters`:

```python
def insert(self, rows, **kwargs):
    rows = [dict(r) for r in rows]
    for row in rows:
        schema_cls = _get_sorter_schema(row["sorter"])
        row["params"] = _validate_params(schema_cls, row["params"])
    super().insert(rows, **kwargs)
```

Replicate the per-table validation each `insert1` already performs (each has a slightly different `_validate_params(...)` call — copy that table's own logic into the loop). Keep `insert1` delegating to its existing logic (do not call `insert` from `insert1` unless the table already does).

### Task 2 — MS5: add `filter` / `whiten` toggles

In `MountainSort5Schema` ([_params/sorter.py:101-110](../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L101-L110)) add, mirroring MS4:

```python
    filter: bool = False
    whiten: bool = True
```

Defaults match current behavior (the recording is already filtered/whitened upstream, so MS5's internal stages stay off by `filter=False`; `whiten=True` matches MS5's own default — confirm against the SI MS5 wrapper default before committing the `whiten` default). Update the `MountainSort5Schema` docstring to state the toggles exist and why `filter=False` (avoid double-filtering the already-preprocessed recording).

### Task 3 — P3: guard `noise_levels` length

At the clusterless consumption site ([sorting.py:1728-1770](../../../../src/spyglass/spikesorting/v2/sorting.py#L1728-L1770)), once `n_channels` is known and `noise_levels` is resolved, reject a wrong-length explicit array (the broadcast `[1.0]` and `None` stay valid):

```python
if noise_levels is not None and len(noise_levels) not in (1, n_channels):
    raise ValueError(
        "clusterless noise_levels must have length 1 (broadcast) or "
        f"n_channels={n_channels}; got {len(noise_levels)}."
    )
```

Place it where `n_channels` is available (the recording is in scope at the `detect_peaks` call). Confirm `_clusterless_noise_levels` ([sorting.py:72](../../../../src/spyglass/spikesorting/v2/sorting.py#L72)) is the right home or do it at the call site.

### Task 4 — analyzer reproducibility: seed `random_spikes`

At the analyzer `compute` ([sorting.py:2069-2097](../../../../src/spyglass/spikesorting/v2/sorting.py#L2069-L2097)), the `random_spikes` extension subsamples spikes stochastically; without a seed the persisted `peak_amplitude_uv` and peak channel vary across rebuilds. Pass a fixed seed to the `random_spikes` extension params (match the existing `seed=0` convention used for whitening/noise_levels). Verify the exact param name against the installed SI `random_spikes` extension signature (`spikeinterface/postprocessing` or `core` — `inspect` it) before writing; it is typically `extension_params={"random_spikes": {"seed": 0, ...}}` or a `seed=` on the relevant `compute` call.

### Task 5 — enum re-raise preserves cause

[curation.py:420-421](../../../../src/spyglass/spikesorting/v2/curation.py#L420-L421):

```python
        except ValueError as exc:
            raise ValueError(
                ...
            ) from exc
```

(mirrors `sorting.py:2287`.)

## Deliberately not in this phase

- The clusterless `detect_threshold` "µV" docstring↔CHANGELOG reconciliation — Phase 4 (doc-only).
- Clusterless waveform-feature extraction being unavailable for v2 (`waveform_features.py:137` dead branch) — that is a real regression fixed in [Phase 6](phase-6-clusterless-waveform-features.md), not here.
- Any change to `MountainSort4Schema` (already correct).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sorter_params_bulk_insert_validates` | `SorterParameters.insert([{invalid params}, ...])` raises (same error as `insert1`); a valid bulk insert succeeds. Parametrize across all four param Lookups. |
| `test_ms5_schema_accepts_and_defaults_filter_whiten` | `MountainSort5Schema()` has `filter is False`, `whiten is True`; `MountainSort5Schema(filter=True)` round-trips; an unknown kwarg still raises (`extra="forbid"`). |
| `test_clusterless_noise_levels_length_guard` | explicit `noise_levels` of length `!= 1` and `!= n_channels` raises; length-1 broadcast and `None` are accepted. |
| `test_analyzer_rebuild_is_seeded_reproducible` (slow) | Build the analyzer twice on the same fixture; assert persisted `peak_amplitude_uv` / peak channel are **equal** across builds (strengthens `test_rebuild_analyzer_folder_recreates_on_missing`, which only checks unit_ids). **Use a fixture with >`max_spikes_per_unit` (>500) spikes/unit** so `random_spikes` actually subsamples — otherwise no subsampling occurs and the test passes regardless of the seed fix (review "not-checked" #5). If the smoke fixture is too small, assert subsampling fired (e.g. selected-spike count == 500 < total) as a guard. |
| `test_metrics_source_invalid_preserves_cause` | `insert_curation(metrics_source="bogus")` raises `ValueError` whose `__cause__` is the underlying enum `ValueError`. |

## Fixtures

Param-validation and schema tests are pure (no DB / SI run) — extend [test_params_validation.py](../../../../tests/spikesorting/v2/test_params_validation.py). The analyzer-reproducibility test needs a real sort+analyzer build on the MEArec smoke fixture (`slow`); reuse the analyzer fixtures already in [test_single_session_pipeline.py](../../../../tests/spikesorting/v2/test_single_session_pipeline.py).

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm:
- All four Lookups validate on bulk `insert`; the validation logic in each `insert` matches that table's `insert1`.
- MS5 defaults preserve current behavior (no double-filter introduced); SI MS5 `whiten` default confirmed.
- `noise_levels` guard rejects only genuinely-wrong lengths.
- Analyzer seed param name verified against installed SI; reproducibility test is exact-equality, not shape-only.
- Validation slice passes; slow tests marked; no plan/phase references in code.
