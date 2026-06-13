# Phase 3 — Bad-channel handling: remove vs interpolate

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add a `bad_channel_handling = "remove" | "interpolate"` preprocessing
parameter and an opt-in `bad_channel_detection` config. `"remove"` (the
default, no detection) is **byte-identical to today**. `"interpolate"` fills
the sort group's shank bad channels from good neighbors so geometry-aware
sorters see a complete probe. The optional detector runs SpikeInterface
`detect_bad_channels` at materialization (consuming
[phase 2](phase-2-bad-channel-detection.md)'s `detect_bad_channels`) and feeds
its result into the same handling step.

See the [overview's bad-channel design decision](overview.md#the-bad-channel-handling-design-decision-chosen-option-a):
the sort group stays the **good-channel sort target**; the curated-bad channels
that grouping excluded are re-included **only** on the `interpolate` path so
they can be filled.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_params/preprocessing.py:114](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L114) —
  `PreprocessingParamsSchema` fields; `to_pre_motion_dict` (`:134`).
- [src/spyglass/spikesorting/v2/_recording_materialization.py:61](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L61) —
  `restrict_recording`: the channel-slice (`:184-203`). The `specific` branch
  already shows how to add channels to the slice; `interpolate` adds the
  group's curated-bad channels the same way.
- [src/spyglass/spikesorting/v2/_recording_materialization.py:342](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L342) —
  `apply_pre_motion_preprocessing`: bandpass at `:369`, reference at `:380`.
  Handling goes **between** them.
- [src/spyglass/spikesorting/v2/recording.py:1075](../../../../src/spyglass/spikesorting/v2/recording.py#L1075) —
  `RecordingFetched` NamedTuple; `make_fetch` (`:1159`) computes its fields;
  `make_compute` (`:1272`) unpacks them into `_compute_recording_artifact`
  (`:1639`), which calls `restrict_recording` then
  `apply_pre_motion_preprocessing`.
- [phase 2](phase-2-bad-channel-detection.md) — exports `detect_bad_channels`
  (in `v2/_bad_channels.py`) and `BAD_CHANNEL_DETECT_DEFAULTS`.
- **Upstream model:** [appendix B.4](appendix.md#b4-bad-channel-interpolation)
  (IBL — kriging interpolation, filled after the filter and before the spatial
  step, the order this phase mirrors) and
  [appendix A.2](appendix.md#a2-bad-channel-detection--removal) (AIND — the
  remove path).

## Tasks

- **Schema: `bad_channel_handling` + `bad_channel_detection`**
  ([_params/preprocessing.py](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py)).
  Add a sub-model and two fields on `PreprocessingParamsSchema`:

  ```python
  class BadChannelDetectionParams(BaseModel):
      """At-materialization automated bad-channel detection options.

      When set, ``detect_bad_channels`` (coherence+psd) runs on the filtered
      recording and its per-channel labels feed the label-aware handling step.
      ``None`` (default) uses only the curated ``Electrode.bad_channel`` flag.

      Every field defaults to ``None`` meaning "leave at SpikeInterface's
      signature default" -- the values are NOT hardcoded here (that would
      diverge from SI and masquerade as defaults). Forward via
      ``detect_bad_channels(recording, **params.model_dump())``; the shared
      wrapper DROPS ``None`` values, so an unset knob uses SI's default and
      ``None`` never reaches SI (where e.g. ``psd_hf_threshold=None`` is
      invalid -- SI expects ``0.02``). ``extra="allow"`` lets a knob NOT named
      here (or one a future SI release adds) pass through; the executor MUST
      validate the forwarded keys against the installed ``detect_bad_channels``
      signature so a typo raises rather than being silently ignored. Thresholds
      are Neuropixels-derived -- recalibrate for polymer (overview Non-Goals).
      """

      model_config = ConfigDict(extra="allow")
      # Commonly-tuned knobs surfaced for discoverability/typing; None = use
      # SI's signature default (names per SI 0.104.3 -- the installed signature
      # is the source of truth).
      dead_channel_threshold: float | None = None
      noisy_channel_threshold: float | None = None
      outside_channel_threshold: float | None = None
      psd_hf_threshold: float | None = None
      outside_channels_location: Literal["top", "bottom", "both"] | None = None
      n_neighbors: int | None = None
      num_random_chunks: int | None = None
      chunk_duration_s: float | None = None
      direction: Literal["x", "y", "z"] | None = None
      seed: int | None = None
  ```

  (`channel_filters` — which label classes to act on — is **not** a detection
  knob here; the label policy lives in the handling step: `out` is always
  removed, `dead`/`noise` follow `bad_channel_handling`.)

  On `PreprocessingParamsSchema`:
  `bad_channel_handling: Literal["remove", "interpolate"] = "remove"` and
  `bad_channel_detection: BadChannelDetectionParams | None = Field(default=None)`.
  Add both to `to_pre_motion_dict`. Extend the `schema_version` history
  docstring (added at v3, **no version bump**: `remove` + `None` keep existing
  blobs valid and output identical; dev rows regenerated).

- **`make_fetch`: fetch the group's *interior* curated-bad channel ids**
  ([recording.py:1159](../../../../src/spyglass/spikesorting/v2/recording.py#L1159)).
  Add `bad_channel_ids` to `RecordingFetched` (`:1075`). The candidate set is
  **not** "every bad channel on the shank" — `set_group_by_electrode_table_column`
  ([recording.py:627](../../../../src/spyglass/spikesorting/v2/recording.py#L627))
  builds **arbitrary-membership** groups and does not persist the original
  requested column values, so a shank-wide fetch can pull in bad electrodes
  that were never near a custom group. Instead, in `make_fetch`:
  - Fetch the candidate bad electrodes that share the sort group's
    `(electrode_group_name, probe_shank)` set **and** carry `bad_channel='True'`.
  - **Restrict to interior channels:** keep only those whose probe position
    (`Electrode.x/y/z`) lies within the `[min, max]` coordinate span of the sort
    group's good channels — i.e. bad channels genuinely surrounded by group
    members, the only ones interpolation can fill from neighbors on both sides.
    For a contiguous shank group this is the shank's interior bad channels; for
    an arbitrary-column group it is only the bad channels inside that group's
    footprint, not the whole shank.
  - **If electrode positions are not populated** (legacy NWBs without
    coordinates), interpolation has no geometry to scope or to fill from: raise
    a clear error from the materialization for `bad_channel_handling=
    "interpolate"` on such a session (point the user to `remove`), rather than
    guessing a neighborhood. See overview Open Question 4.
  - Return the result as a sorted tuple (DeepHash-stable, like the other
    `make_fetch` outputs) and thread it through `make_compute` →
    `_compute_recording_artifact` → `restrict_recording` and
    `apply_pre_motion_preprocessing`.

- **`restrict_recording`: include curated-bad channels on the interpolate
  path** ([_recording_materialization.py:184](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L184)).
  Add `bad_channel_handling` and `bad_channel_ids` params. When
  `bad_channel_handling == "interpolate"`, union `bad_channel_ids` (the
  *interior* curated-bad set computed in `make_fetch`) into `slice_ids`
  (alongside the existing `specific`-reference channel) so those bad channels
  are present in the sliced recording to be filled. `"remove"` leaves the
  slice as today (good channels only; the curated-bad are not re-added).

- **`apply_pre_motion_preprocessing`: the LABEL-AWARE handling step**
  ([_recording_materialization.py:342](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L342)).
  Add `bad_channel_handling`, `bad_channel_ids`, and `bad_channel_detection`
  params (it already receives `reference_mode` / `reference_electrode_id`).
  Insert **after** the bandpass (`:369`) and **before** the reference (`:380`).
  Two invariants: **(a)** out-of-brain channels are always removed, never
  interpolated (per IBL `voltage.py:539`/`:774` and the SI preprocessing guide —
  an `out` channel has no in-brain neighbors to fill from); only `dead`/`noise`
  follow `bad_channel_handling`. **(b)** the `specific` reference electrode is
  sliced into the recording for subtraction (`restrict_recording:187`) and is
  dropped only *after* `common_reference` — so the handling step must **not**
  remove or interpolate it. If detection flags the reference electrode itself
  bad, **raise** (referencing to a bad channel is invalid):

  ```python
  # Bad-channel handling: between filter and reference (matches IBL/AIND).
  from spyglass.spikesorting.v2._bad_channels import detect_bad_channels

  present = {int(c) for c in recording.get_channel_ids()}
  # Protect the 'specific' reference channel: it must survive for subtraction
  # and is removed later by the existing post-reference drop.
  ref_id = int(reference_electrode_id) if reference_mode == "specific" else None
  to_remove, to_interpolate = set(), set()
  # Curated Electrode.bad_channel flags carry no label -> quality-bad by the
  # phase-2 invariant, so follow the user's explicit bad_channel_handling
  # (they were re-included for interpolate; for remove they are already absent).
  curated = (present & {int(c) for c in bad_channel_ids}) - {ref_id}
  (to_interpolate if bad_channel_handling == "interpolate" else to_remove)\
      .update(curated)
  # Detected channels carry SI labels: 'out' is ALWAYS removed; 'dead'/'noise'
  # follow bad_channel_handling. BadChannelDetectionParams is a model -> dump it
  # (the wrapper drops None so unset knobs use SI defaults).
  if bad_channel_detection is not None:
      labels = detect_bad_channels(
          recording, **bad_channel_detection.model_dump()
      )["labels"]
      if ref_id is not None and labels.get(ref_id) in ("dead", "noise", "out"):
          raise ValueError(
              "apply_pre_motion_preprocessing: the 'specific' reference "
              f"electrode {ref_id} was flagged bad ({labels.get(ref_id)!r}); "
              "referencing to a bad channel is invalid. Pick a different "
              "reference_electrode_id (or clear the detection)."
          )
      for cid, label in labels.items():
          cid = int(cid)
          if cid == ref_id:
              continue  # protected; dropped after referencing
          if label == "out":
              to_remove.add(cid)
          elif label in ("dead", "noise"):
              (to_interpolate if bad_channel_handling == "interpolate"
               else to_remove).add(cid)
  to_interpolate &= present
  to_remove = (to_remove & present) - to_interpolate
  if to_interpolate:
      # interpolate needs channel locations; fail clearly if absent.
      if recording.get_channel_locations() is None:
          raise ValueError(
              "apply_pre_motion_preprocessing: interpolate requires channel "
              "locations on the recording, but none are set (no probe "
              "geometry). Use bad_channel_handling='remove', or ensure the "
              "NWB / probe carries electrode positions."
          )
      recording = sip.interpolate_bad_channels(recording, sorted(to_interpolate))
  if to_remove:
      recording = recording.remove_channels(sorted(to_remove))
  applied_steps["bad_channels"] = {
      "interpolated": sorted(to_interpolate),
      "removed": sorted(to_remove),
  }
  ```

  Note the asymmetry (documented in the overview): for `"remove"`,
  `bad_channel_ids` (curated) are **already absent** (restrict_recording did
  not re-add them); for `"interpolate"`, the *interior* curated-bad channels
  are present and get filled. `applied_steps` is the report introduced in
  [phase 1](phase-1-adc-phase-shift.md) so `filtering_description` can name what
  ran (see the next task).

- **`filtering_description`**
  ([_recording_materialization.py:435](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L435))
  — read the `applied_steps["bad_channels"]` report (not the params) and append
  `"interpolate N bad channels"` / `"remove N bad channels"` (with the actual
  counts that ran) so the NWB provenance reflects reality, including the case
  where `out` channels were removed even under `interpolate`.

- **Documentation (ships in this phase):** the schema docstrings; a
  `SpikeSortingV2.md` subsection on `bad_channel_handling` (when to interpolate
  vs remove — interpolate for geometry-aware sorters on dense shanks, remove for
  tetrodes / sparse groups), the `bad_channel_detection` opt-in, **and the
  curated-flag invariant**: `Electrode.bad_channel='True'` must mark a
  *quality-bad* (dead/noise-class) channel — because under `interpolate` a
  label-less curated flag is filled, an outside-brain channel must NOT be marked
  `bad_channel` (use `remove`, exclude it from the sort group, or rely on
  at-materialization detection, which keeps the `out` label and always removes
  it). Also document that the `specific` reference electrode is protected from
  handling and that a detection that flags the reference bad raises. A CHANGELOG
  entry. A `SpikeSortingV2_Migration.md` note is **not** needed (default is
  byte-identical to current v2; these are new opt-in v2 capabilities with no
  v1↔v2 parity change).

## Deliberately not in this phase

- Changing the grouping helpers to include bad channels (the rejected Option B;
  see overview Open Question 1).
- A spatial-frequency "destripe" reference (`highpass_spatial_filter`) — a
  possible future `reference_mode`, out of scope here.
- Phase-shift (phase 1) and drift estimation (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| handling between filter and reference *(stub/mock)* | `interpolate` with a non-empty `bad_channel_ids` invokes `sip.interpolate_bad_channels` **after** `bandpass_filter` and **before** `common_reference`; `remove` invokes `remove_channels` in the same slot; order recorded via monkeypatch (mirror `test_preprocessing_order.py`). |
| remove default is a no-op *(stub/mock)* | `bad_channel_handling="remove"`, `bad_channel_detection=None`, empty `bad_channel_ids` → neither `interpolate_bad_channels` nor `remove_channels` is called. |
| interpolate needs locations *(stub/mock)* | `interpolate` on a recording whose `get_channel_locations()` is `None` raises the clear ValueError. |
| `out` is removed even under interpolate *(stub/mock)* | `bad_channel_detection` labels a channel `"out"` and another `"dead"` with `bad_channel_handling="interpolate"` → the `"out"` id goes to `remove_channels` and the `"dead"` id to `interpolate_bad_channels` (label policy, not blanket interpolate). |
| dead/noise follow the param *(stub/mock)* | the same `"dead"`/`"noise"` channels go to `remove` when `bad_channel_handling="remove"`. |
| specific reference is protected *(stub/mock)* | `reference_mode="specific"` with the reference electrode present and `bad_channel_detection` flagging *other* channels → the reference id is in **neither** `to_remove` nor `to_interpolate`; `common_reference(reference="single")` is still called with it. |
| reference-flagged-bad raises *(stub/mock)* | `bad_channel_detection` labels the `specific` reference electrode `"dead"` → a clear ValueError ("referencing to a bad channel") is raised before referencing. |
| detection params are dumped, None dropped *(stub/mock)* | `bad_channel_detection=BadChannelDetectionParams(psd_hf_threshold=None)` → `detect_bad_channels` is called via `model_dump()` and the `None` is not forwarded (no SI crash). |
| interior re-inclusion only *(unit/DB)* | for an arbitrary-column group, `make_fetch` returns only bad electrodes within the good channels' coordinate span — a bad electrode on the same shank but outside the group's footprint is NOT re-included. |
| no positions → interpolate raises *(integration, DB)* | a session without electrode coordinates + `bad_channel_handling="interpolate"` raises the clear "needs positions" error rather than guessing. |
| default cache_hash unchanged *(integration, DB+SI, slow)* | smoke-fixture `Recording` with all defaults has the same `cache_hash` as pre-phase code (headline regression guard). |
| interpolate fills a bad channel *(integration, DB+SI, slow)* | mark one *interior* smoke-fixture electrode `bad_channel='True'`, materialize with `bad_channel_handling="interpolate"` → the cached recording retains that channel (count complete) and its trace differs from the raw (filled, not zero); restore the flag. |
| remove drops a detected channel *(integration, DB+SI, slow)* | with `bad_channel_detection` flagging a synthesized-bad channel and `remove`, the cached recording omits it. |

Stub tests are DB-free (fake recording + monkeypatched `sip.*`). Mark the three
integration tests slow; restore any mutated `Electrode.bad_channel`.

## Fixtures

- Stub: the fake-recording + call-order harness from `test_preprocessing_order.py`
  (extend it to expose `get_channel_locations` / `get_channel_ids`).
- Integration: `mearec_polymer_smoke` + `dj_conn`; mutate/restore one
  electrode's `bad_channel` to exercise interpolate.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- The default path (`remove`, no detection) is byte-identical — the cache_hash
  regression row passes.
- Handling sits **between** filter and reference; interpolate re-includes only
  the group's **interior** curated-bad channels via `restrict_recording`,
  remove does not.
- **Label policy is honored:** `out` channels are removed even under
  `interpolate`; only `dead`/`noise` follow `bad_channel_handling`. Curated
  (label-less) channels follow the user's chosen handling.
- The interior re-inclusion does **not** pull whole-shank bad channels into an
  arbitrary-column group (High finding); positions absent → interpolate raises.
- **The `specific` reference electrode is excluded from handling** (it is
  needed for subtraction and dropped after referencing); a detection that flags
  the reference bad **raises** rather than removing/interpolating it.
- `BadChannelDetectionParams` is **dumped** (`.model_dump()`, not `**model`)
  before forwarding; its fields default to `None` = "use SI default" and the
  wrapper drops `None`, so no threshold is hardcoded and `None` never reaches
  SI; `extra="allow"` keys are validated against the installed
  `detect_bad_channels` signature.
- **Curated-flag invariant** documented: `Electrode.bad_channel` is quality-bad
  only (never outside-brain), so the curated-flag interpolate path is safe.
- `apply_pre_motion_preprocessing` returns the `applied_steps` report and
  `filtering_description` reads it (honest counts, not param-derived).
- Interpolation guards on channel locations and raises clearly, never letting
  SI fail opaquely deep in the call stack.
- `bad_channel_ids` is DeepHash-stable across two `make_fetch` calls (sorted
  tuple), matching the other fetched fields.
- The Option-A asymmetry is documented where the handling code lives.
- Validation slice passes; integration tests are marked; mutated flags restored.
- Docstrings / test names / module names don't reference this plan or its
  phases.
- The CHANGELOG + `SpikeSortingV2.md` subsection are present, not deferred.
