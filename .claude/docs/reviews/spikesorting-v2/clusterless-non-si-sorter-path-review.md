# Spike Sorting V2 clusterless / non-SI sorter path review

Date: 2026-06-25

Scope: the v2 `clusterless_thresholder` path as the primary non-SpikeInterface
sorter, including sorter parameter identity, dispatch/preflight semantics,
downstream `UnitWaveformFeatures`, docs/notebooks, and user-facing scientific
meaning of the one-unit thresholded output.

Method: local source/test/docs inspection plus one independent explorer-agent
pass. No tests were run for this review.

## Executive Summary

The clusterless path is much better than a thin fake sorter wrapper. It has a
dedicated parameter schema, bypasses installed-SI-sorter checks, handles
microvolt and MAD threshold modes, supports zero-unit outputs, and the decoding
waveform-feature path has a v2-specific `SortingAnalyzer` implementation that
preserves one feature row per spike.

The main bug is a provenance/dispatch contract mismatch: `SorterParameters`
stores `execution_params` for all sorter rows and preflight treats container
execution as real, but `clusterless_thresholder` always runs locally and ignores
`execution_params`. That can create distinct parameter rows for identical local
outputs, or make preflight block a valid local run because a container backend is
configured for a sorter that cannot use one.

## What Looks Solid

- `clusterless_thresholder` is explicitly marked as a non-SI sorter, so default
  insertion does not depend on `spikeinterface.sorters.installed_sorters`
  (`src/spyglass/spikesorting/v2/sorting.py:369`).
- The runtime dispatch routes clusterless to a separate helper rather than
  through SI sorter execution (`src/spyglass/spikesorting/v2/sorting.py:2429`).
- The clusterless schema validates `threshold_unit`, peak detection settings,
  explicit noise levels, and implausible MAD thresholds
  (`src/spyglass/spikesorting/v2/_params/sorter.py:176`).
- V2 `UnitWaveformFeatures` builds a fresh SI 0.104 analyzer for v2 sources and
  enforces `max_spikes_per_unit=None` for clusterless 1:1 spike/feature alignment
  (`src/spyglass/decoding/v1/waveform_features.py:331`,
  `src/spyglass/decoding/v1/waveform_features.py:362`).
- Clusterless feature amplitudes are explicitly documented as microvolts, not v1
  raw ADC counts (`src/spyglass/decoding/v1/waveform_features.py:324`).

## Findings

### 1. Medium-high: clusterless accepts container execution provenance but always runs local

`SorterParameters` stores `execution_params` for every sorter row
(`src/spyglass/spikesorting/v2/sorting.py:215`). That field is documented as the
execution backend and container-side provenance because backend choice can affect
sort output (`src/spyglass/spikesorting/v2/sorting.py:225`). Clusterless is
listed as a non-SI sorter (`src/spyglass/spikesorting/v2/sorting.py:369`), and
the dispatch docstring says it has no SI scratch directory, external whitening,
or container backend, so `execution_params` does not apply
(`src/spyglass/spikesorting/v2/sorting.py:2420`).

Impact:

- A clusterless row can claim Docker/Singularity execution provenance that the
  runtime never honors.
- Preflight can treat non-local execution as meaningful for clusterless and block
  a valid local run because Docker/Singularity is unavailable.
- Distinct `SorterParameters` rows can produce identical local clusterless output
  solely because ignored execution metadata differs.

Recommended fix:

- Reject non-local `execution_params.backend` for `_NON_SI_SORTERS` during
  `SorterParameters.insert`.
- For existing rows, either normalize clusterless execution params to local in a
  migration or fail preflight with a clear "clusterless supports local execution
  only" message.
- Add regression tests for insert rejection, preflight behavior, and dispatch
  behavior for clusterless with non-local execution params.

### 2. Medium: clusterless waveform-feature scratch is routed to bare system temp

The v2 clusterless waveform-feature path intentionally extracts every spike's
full multi-channel waveform into a disk-backed zarr analyzer because realistic
clusterless decoding can require millions of crossings
(`src/spyglass/decoding/v1/waveform_features.py:374`). The temporary directory is
created with `tempfile.TemporaryDirectory(prefix="v2_clusterless_wf_")`
(`src/spyglass/decoding/v1/waveform_features.py:388`).

Impact:

- Large clusterless feature jobs can spill several GB into the system temp
  location rather than the Spyglass-managed temp/cache area.
- Users and cluster workers have little control over disk placement, quotas, or
  cleanup monitoring for this expensive path.

Recommended fix:

- Route this temp zarr through the configured Spyglass temp directory or an
  explicit decoding scratch setting.
- Before extraction, estimate scratch size from spike count, channel count, and
  waveform window and fail with an actionable error if it exceeds the configured
  budget.

### 3. Medium: one-unit clusterless output can be mistaken for a sorted single neuron

The clusterless detector returns all detected threshold crossings as one
`NumpySorting` unit. That is correct for clusterless decoding marks, but it moves
through generic `Sorting`, `CurationV2`, analyzer, and merge-table pathways that
otherwise describe isolated units.

Impact:

- Generic downstream users can accidentally treat clusterless unit `0` as one
  curated biological unit.
- Unit-level quality metrics, region summaries, merge suggestions, or matching
  semantics can become scientifically misleading if applied without a clusterless
  guard.

Recommended fix:

- Persist or expose a sorter-family flag such as `clusterless=True` or
  `unit_semantics="threshold_crossings"` on sorting metadata/accessors.
- Add guardrails in user-facing methods that imply single-neuron units, especially
  UnitMatch, automatic merge suggestion, and unit-summary docs.
- Keep decoding-specific paths permissive, but make the unit semantics explicit
  in return values and docs.

### 4. Medium-low: feature extraction does not inherit the clusterless sorter peak-sign contract

For clusterless amplitude features, `_compute_waveform_features` only forces
`estimate_peak_time=False` (`src/spyglass/decoding/v1/waveform_features.py:423`).
It does not reconcile the feature row's `peak_sign` with the peak sign used by
the clusterless sorter.

Impact:

- A sorter row using positive or both-sign thresholding can be paired with a
  waveform-feature row that extracts negative-peak amplitudes.
- Encoding/decoding models can remain internally consistent within one run while
  silently diverging from the detection semantics that produced the spikes.

Recommended fix:

- For v2 clusterless sources, default the waveform feature `peak_sign` from the
  source sorter params when the feature row does not specify one.
- If both are explicit and disagree, warn or raise with a clear migration path.

### 5. Low: the public clusterless waveform-feature notebook is still v1-first

The v2 docs advertise the clusterless preset and v2 downstream support
(`docs/src/Features/SpikeSortingV2.md:327`,
`docs/src/Features/SpikeSortingV2.md:1002`). The docs nav still points users to
the existing clusterless waveform-feature notebook, but its script uses v1
tables such as `sgs.SpikeSorterParameters`, `sgs.SpikeSortingSelection`,
`sgs.CurationV1`, and `sources=["v1"]`
(`notebooks/py_scripts/40_Extracting_Clusterless_Waveform_Features.py:122`,
`notebooks/py_scripts/40_Extracting_Clusterless_Waveform_Features.py:139`,
`notebooks/py_scripts/40_Extracting_Clusterless_Waveform_Features.py:278`).

Impact:

- Users looking for the v2 clusterless path are routed into legacy tables.
- The documented workflow does not exercise v2 params, dispatch, `CurationV2`,
  zero-unit behavior, or v2 `UnitWaveformFeatures` compatibility.

Recommended fix:

- Add a v2-specific clusterless notebook or update the existing one with
  `franklab_clusterless_2026_06`, `CurationV2`, final v2 `merge_id`,
  `UnitWaveformFeaturesSelection`, `max_spikes_per_unit=None`, and microvolt
  amplitude notes.
- If the notebook remains v1-only, label it explicitly and link to the v2 path.

