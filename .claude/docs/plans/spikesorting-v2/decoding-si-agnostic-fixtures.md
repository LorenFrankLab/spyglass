# Plan: SI-version-agnostic decoding test fixtures (fix run-tests)

## Goal
Make the `tests/decoding/` suite pass under **whatever SpikeInterface version
the job installs** — SI 0.104.3 in `run-tests`, SI 0.99.1 in the legacy job —
so decoding is not coupled to a SpikeInterface version. This is the root-cause
fix for the 46 `run-tests` setup ERRORs.

## Root cause (verified)
`run-tests` (SI 0.104.3) excludes `tests/spikesorting/v0|v1|v2` but **not** the
consumers of v1 spikesorting. `tests/decoding/conftest.py` builds its upstream
`SpikeSortingOutput` entry by running the **v1** pipeline
(`pop_rec → ArtifactDetection → SpikeSorting → CurationV1`), which calls
WaveformExtractor-era APIs removed in SI ≥0.101 (`recording.channel_slice`,
`si.extract_waveforms`). Master is green only because it runs everything on SI
0.99.1.

## Key architectural facts (evidence)
- **Production already dispatches.** `decoding/v1/waveform_features.py::UnitWaveformFeatures.make`
  branches on the merge source: `CurationV2` → `_fetch_waveform_v2`
  (`create_sorting_analyzer`, SI 0.104); `CurationV1`/v0 → `_fetch_waveform`
  (`si.extract_waveforms`, gated SI<0.101).
- **Clusterless needs a v2 source under SI 0.104.** The v2 branch fires **only**
  when the source is `CurationV2` (joins v2 `SortingSelection`/`RecordingSelection`).
  `ImportedSpikeSorting` and `CurationV1` both fall into the gated v1 branch →
  cannot produce features under SI 0.104. So clusterless fixtures need the real
  v2 spikesorting pipeline (MEArec → v2 `Recording`→`Sorting`→`CurationV2`).
- **Sorted-spikes is lighter.** `SortedSpikesGroup` only needs spike times, which
  `ImportedSpikeSorting` (already an `imported_spike` fixture exists) can supply
  SI-agnostically — but for uniformity it can also come from the same v2 source.
- **Secondary defect.** v1 `SpikeSortingRecording.make_*` and `SpikeSorting.make_*`
  lack the `_require_legacy_si_environment` gate the other v1 makes have, so SI
  0.104 yields a confusing `AttributeError` instead of a clean RuntimeError.

## Design
Introduce an **SI-version dispatch** in the decoding spike-data fixtures:

```
if SI < 0.101:  build upstream via v1 (current fixtures)  -> CurationV1 source
else:           build upstream via v2 (MEArec smoke)      -> CurationV2 source
```

Downstream decoding fixtures (`UnitWaveformFeaturesGroup`, `SortedSpikesGroup`,
position groups, decode selections) consume by `merge_id` and stay unchanged.
`UnitWaveformFeatures` already picks the right extraction path from the source.

## Phases

### Phase 0 — Resolve open questions (no code)
1. Confirm the v2 spikesorting fixture (`tests/spikesorting/v2/_ingest_helpers.py`
   + MEArec smoke) yields a `SpikeSortingOutput.CurationV2` merge entry whose
   `get_recording`/`get_sorting` work for `_fetch_waveform_v2`. Build one in the
   `spyglass_spikesorting_v2` env and run `UnitWaveformFeatures` over it
   end-to-end (amplitude feature). **Gate decision on this working.**
2. Confirm a v2 source also drives `SortedSpikesGroup` (spike times) for the
   sorted-spikes decoding tests.
3. Decide CI placement (see "CI" below).

### Phase 1 — Secondary gate fix (independent, low-risk, ship first)
- Add `_require_legacy_si_environment("v1 SpikeSortingRecording.make")` /
  `("v1 SpikeSorting.make")` as the first statement of the respective
  `make_compute` (or the tri-part entry). Verify v1 spikesorting tests still
  pass under SI 0.99 (legacy job) and that SI 0.104 now raises the clean
  RuntimeError. This makes the boundary explicit regardless of Phase 2.

### Phase 2 — v2 decoding fixtures behind a dispatch
- Add a `tests/decoding/_v2_spike_source.py` helper (and/or extend
  `tests/decoding/conftest.py`) that, under SI ≥0.104, ingests the MEArec smoke
  fixture and runs v2 `Recording → Sorting → CurationV2`, returning the
  `merge_id`(s) + group keys decoding needs.
- Refactor `recording_ids`/`clusterless_spikesort`/`clusterless_curate`/
  `clusterless_mergeids` (and the sorted equivalents) into a dispatch: keep the
  v1 path for SI<0.101; use the v2 source for SI≥0.104.
- Keep the public fixture names (`clusterless_mergeids`, `pop_spikes_group`,
  etc.) stable so the decoding tests are untouched.

### Phase 3 — CI
- `run-tests` (SI 0.104) must build the MEArec smoke fixture for the v2 source:
  add the `pip install "neuron<9" "LFPy<2.3.7"` + nwbinspector (already in the
  validation extra) + `generate_mearec.py --smoke` steps (mirroring `pytest-v2`),
  OR cache/reuse the fixture. Decide vs. a dedicated `decoding-v2` job to keep
  `run-tests` lean.
- The legacy job already runs decoding-capable SI 0.99; if decoding stays in
  `run-tests` only, the v1 path is exercised by the legacy job's own decoding
  run (if added) — confirm we still cover the v1 extraction branch somewhere.

### Phase 4 — Verify
- Local: both envs. `spyglass_spikesorting_v2` (SI 0.104) → clusterless+sorted
  decoding over the v2 source. `spyglass`/legacy (SI 0.99) → over the v1 source.
- CI: `run-tests` green; legacy green; pytest-v2 unaffected.

## Risks / trade-offs
- **Cost:** adds MEArec generation (~3 min + NEURON deps) to `run-tests` (or a
  new job). Heaviest part of the change.
- **Coverage of the v1 extraction branch:** if decoding only runs the v2 source
  under SI 0.104, the v1 `si.extract_waveforms` path loses CI coverage unless we
  also run decoding under SI 0.99 (legacy). Recommend running decoding in BOTH
  contexts so both `UnitWaveformFeatures` branches are exercised.
- **Fixture complexity:** the decoding conftest is 1034 lines; dispatch must keep
  fixture names/contracts stable to avoid touching ~10 decoding test modules.

## Smallest-correct alternative (if descoping)
Ship Phase 1 (gate) now, and **temporarily** `--ignore=tests/decoding` +
`tests/utils/test_merge.py::test_merge_get_class_invalid` in `run-tests` with a
tracked TODO, so `run-tests` goes green immediately while the SI-agnostic v2
decoding fixtures (Phases 2–3) land as a focused follow-up. (User preferred the
full SI-agnostic route over relocating to SI 0.99, so this is only a stopgap.)
