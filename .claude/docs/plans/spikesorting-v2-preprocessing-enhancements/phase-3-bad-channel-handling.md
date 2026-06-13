# Phase 3 — Bad-channel handling: remove vs interpolate

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add a `bad_channel_handling = "remove" | "interpolate"` preprocessing
parameter and an opt-in `bad_channel_detection` config. `"remove"` (the
default, no detection) is **byte-identical to today**. `"interpolate"` fills the
sort group's interior bad channels (those embedded among its good channels) from
good neighbors so geometry-aware sorters see a complete probe. The optional
detector runs SpikeInterface `detect_bad_channels` at materialization (consuming
[phase 2](phase-2-bad-channel-detection.md)'s `detect_bad_channels`) on a
reference-excluded, per-shank surface and feeds its labels into the same
handling step.

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
    `(electrode_group_name, probe_shank)` set **and** carry `bad_channel='True'`,
    with their positions (`Electrode.x/y/z`).
  - **Compute the shank's physical pitch.** For each `(electrode_group_name,
    probe_shank)` the group touches, read **all** that shank's electrode
    positions (good + bad) from `Electrode` and take the median
    nearest-neighbor distance — the probe's nominal electrode spacing, a
    physical constant. This is the scale for the adjacency test below.
  - **Restrict by absolute spatial adjacency, not a bounding box:** keep a
    candidate only if at least `MIN_GOOD_NEIGHBORS` of the group's good
    electrodes lie within `RADIUS_FACTOR × pitch` of it (helper below). Two
    things this gets right that a `[min, max]` coordinate span does not: (1) a
    custom group spanning two separated good-channel clusters has a span that
    swallows the gap between them, wrongly re-including unrelated electrodes
    there; and (2) the radius MUST come from the **full-shank pitch**, not from
    the *group's own* nearest-neighbor spacing — with only two far-apart good
    channels the group's spacing degenerates to the gap itself, so a
    `radius = 1.5 × (group spacing)` would admit the very midpoint channel we
    must exclude. Anchoring the radius to the dense probe pitch keeps only bad
    channels physically embedded among good neighbors (the only ones kriging can
    fill); a channel many pitches from any good channel is dropped, contiguous or
    not.

    ```python
    # Module-level, near the other geometry helpers. Constants are dimensionless
    # multiples of the probe's own pitch, so one rule fits dense Neuropixels
    # shanks and sparse polymer groups alike.
    MIN_GOOD_NEIGHBORS = 2  # surrounded (>=2), not merely adjacent on one side
    RADIUS_FACTOR = 1.5     # one pitch away counts; a multi-pitch gap does not

    def _shank_pitch(shank_xyz):
        """Median nearest-neighbor distance over ALL electrodes on a shank.

        ``shank_xyz``: (M, 3) positions of every electrode on the shank (good
        AND bad), so the result is the probe's physical pitch, independent of
        which channels a sort group happens to keep. Returns ``None`` if the
        shank has < 2 positioned electrodes.
        """
        import numpy as np

        xyz = np.asarray(shank_xyz, dtype=float)
        if xyz.shape[0] < 2:
            return None
        dd = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        return float(np.median(dd.min(axis=1)))

    def _interior_bad_channel_ids(good_xyz, candidate_xyz, pitch):
        """Curated-bad ids physically embedded among a group's good channels.

        ``good_xyz``: (N, 3) positions of the group's good channels.
        ``candidate_xyz``: list of ``(electrode_id, (x, y, z))`` for the
        curated-bad electrodes on the group's shank(s). ``pitch``: the shank's
        physical electrode spacing from ``_shank_pitch`` (NOT derived from the
        possibly-sparse good set). A candidate is kept only when at least
        ``MIN_GOOD_NEIGHBORS`` good channels lie within ``RADIUS_FACTOR *
        pitch`` of it -- so a bad channel between two far-apart good channels
        (its nearest good channel many pitches away) is excluded, while a bad
        channel in a dense run is kept. Returns a sorted list.
        """
        import numpy as np

        good = np.asarray(good_xyz, dtype=float)
        if good.shape[0] < 2 or not pitch:
            return []  # need >=2 good channels and a defined pitch to fill from
        radius = RADIUS_FACTOR * float(pitch)
        return sorted(
            int(cid)
            for cid, pos in candidate_xyz
            if int((np.linalg.norm(good - np.asarray(pos, float), axis=1)
                    <= radius).sum()) >= MIN_GOOD_NEIGHBORS
        )
    ```

    (Apply per shank when the group spans more than one — each shank uses its
    own `pitch`.)
  - **If electrode positions are not populated** (legacy NWBs without
    coordinates, so `Electrode.x/y/z` are null, or a shank with `< 2` positioned
    electrodes so `pitch` is `None`), interpolation has no geometry to scope or
    to fill from: raise a clear error from the materialization for
    `bad_channel_handling="interpolate"` on such a session (point the user to
    `remove`), rather than guessing a neighborhood. See overview Open Question 4.
  - Return `bad_channel_ids` as a sorted tuple (DeepHash-stable, like the other
    `make_fetch` outputs). Also build and return a `channel_shank` map
    (`{electrode_id: (electrode_group_name, probe_shank)}` for every channel the
    compute step may see — good channels plus re-included curated-bad) so the
    handling step can split detection per shank **without** a DB read (next
    task). Thread both through `make_compute` → `_compute_recording_artifact` →
    `restrict_recording` and `apply_pre_motion_preprocessing`.

- **`make_fetch`: reject a curated-bad `specific` reference**
  ([recording.py:1187](../../../../src/spyglass/spikesorting/v2/recording.py#L1187),
  where `reference_mode` / `reference_electrode_id` are already fetched). When
  `reference_mode == "specific"`, read the reference electrode's
  `Electrode.bad_channel`; if it is `'True'`, **raise** a clear error pointing
  the user to pick a different `reference_electrode_id` or clear the flag.
  Referencing every channel against a quality-bad (dead/noisy) electrode injects
  its bad signal into the whole group, yet v2 deliberately looks the reference up
  in the **full, not-bad-filtered** electrode set
  ([recording.py:120-122](../../../../src/spyglass/spikesorting/v2/recording.py#L120)),
  so a curated-bad electrode can currently be configured as the reference with no
  complaint. This is the **detection-independent** half of "a reference flagged
  bad raises": the handling-step raise (below) only fires when at-materialization
  detection labels the reference bad; this `make_fetch` check covers the common
  `bad_channel_detection=None` path and a flag set *after* the sort group was
  created. Validate at `make_fetch` (not only at SortGroup insert) — the curated
  flag's state at materialization is what the cached recording actually reflects.

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
  Add `bad_channel_handling`, `bad_channel_ids`, `bad_channel_detection`, and
  `channel_shank` params (it already receives `reference_mode` /
  `reference_electrode_id`). Insert **after** the bandpass (`:369`) and
  **before** the reference (`:380`). Invariants:
  - **(a)** out-of-brain channels are always removed, never interpolated (per IBL
    `voltage.py:539`/`:774` and the SI preprocessing guide — an `out` channel has
    no in-brain neighbors to fill from); only `dead`/`noise` follow
    `bad_channel_handling`. **Removal runs before interpolation** so removed
    `out` channels cannot act as interpolation donors.
  - **(b)** the `specific` reference electrode is sliced in for subtraction
    (`restrict_recording:187`) and dropped only *after* `common_reference`, so it
    must survive handling. It is therefore **excluded from the detector surface
    entirely**, not merely skipped after the fact: coherence/PSD detection
    compares each channel to the cross-channel median, so a reference left in the
    pool would perturb the labels of the real sort channels. The reference's own
    quality is vetted separately by the curated-bad check in `make_fetch` (above).
  - **(c)** detection runs **per shank** — the coherence method is spatially
    local, so mixing shanks corrupts it (the same rule as phase 2) — using the
    `channel_shank` map threaded from `make_fetch`.
  - **(d)** because the boolean `Electrode.bad_channel` cannot encode an `out`
    label, a label-less curated flag selected for `interpolate` is **audited**: a
    coherence/PSD pass classifies it, and any channel that comes back
    outside-brain is removed, never filled — so a pre-existing / manual / config
    flag that happens to mark an out-of-brain channel can never invent signal.

  ```python
  # Bad-channel handling: between filter and reference (matches IBL/AIND).
  from spyglass.spikesorting.v2._bad_channels import detect_bad_channels
  from spyglass.utils import logger

  present = {int(c) for c in recording.get_channel_ids()}
  # The 'specific' reference is present only for subtraction (dropped after
  # common_reference); it is never handled and never fed to detection.
  ref_id = int(reference_electrode_id) if reference_mode == "specific" else None
  to_remove, to_interpolate = set(), set()

  # (1) Curated Electrode.bad_channel flags carry no label (quality-bad by the
  #     phase-2 convention) -> follow the user's bad_channel_handling. They were
  #     re-included for interpolate; for remove they are already absent.
  curated = (present & {int(c) for c in bad_channel_ids}) - {ref_id}
  (to_interpolate if bad_channel_handling == "interpolate" else to_remove)\
      .update(curated)

  # (2) Labels (coherence+psd) from a REFERENCE-EXCLUDED, PER-SHANK detector
  #     surface (invariants b, c). Needed when the user enabled detection, OR --
  #     even if not -- when interpolating label-less curated flags, which must be
  #     audited for outside-brain channels (invariant d). detect_bad_channels
  #     pins method; {} -> SI signature defaults, model_dump() drops None.
  need_labels = bad_channel_detection is not None or (
      bad_channel_handling == "interpolate" and bool(curated)
  )
  labels = {}
  if need_labels:
      params = (bad_channel_detection.model_dump()
                if bad_channel_detection is not None else {})
      detector = recording.channel_slice(
          [c for c in recording.get_channel_ids() if int(c) != ref_id]
      )
      for shank_rec in _split_by_shank(detector, channel_shank):
          labels.update(
              {int(k): v for k, v
               in detect_bad_channels(shank_rec, **params)["labels"].items()}
          )

  # (3) Act on detections only when the user opted IN (never auto-drop channels
  #     they did not flag): 'out' removed, 'dead'/'noise' follow handling.
  if bad_channel_detection is not None:
      for cid, label in labels.items():
          if label == "out":
              to_remove.add(cid)
          elif label in ("dead", "noise"):
              (to_interpolate if bad_channel_handling == "interpolate"
               else to_remove).add(cid)

  # (4) Audit (invariant d): never interpolate a channel the detector classifies
  #     outside-brain, even a label-less curated flag with detection off.
  out_flagged = {c for c in to_interpolate if labels.get(c) == "out"}
  if out_flagged:
      logger.warning(
          "apply_pre_motion_preprocessing: channels %s were flagged bad but "
          "classified outside-brain; removing instead of interpolating.",
          sorted(out_flagged),
      )
      to_interpolate -= out_flagged
      to_remove |= out_flagged

  # (5) Resolve overlaps, then REMOVE BEFORE INTERPOLATE so removed (incl. out)
  #     channels are never interpolation donors (IBL keeps out-of-brain out of
  #     the spatial computation, voltage.py:539/:774). to_remove/to_interpolate
  #     are disjoint and to_interpolate is &= present, so removing never drops a
  #     fill target.
  to_interpolate &= present
  to_remove = (to_remove & present) - to_interpolate
  if to_remove:
      recording = recording.remove_channels(sorted(to_remove))
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
  applied_steps["bad_channels"] = {
      "interpolated": sorted(to_interpolate),
      "removed": sorted(to_remove),
  }
  ```

  Module-level helper (alongside `_interior_bad_channel_ids`):

  ```python
  from collections import defaultdict

  def _split_by_shank(rec, channel_shank):
      """Yield per-shank channel-sliced sub-recordings for detection.

      ``channel_shank`` maps electrode id -> a hashable shank key
      ((electrode_group_name, probe_shank)). Coherence/PSD detection is
      spatially local, so it must run within a shank (phase 2). A single-shank
      group yields the whole recording unchanged.
      """
      groups = defaultdict(list)
      for cid in rec.get_channel_ids():
          groups[channel_shank[int(cid)]].append(cid)
      if len(groups) <= 1:
          yield rec
      else:
          for ids in groups.values():
              yield rec.channel_slice(ids)
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
  curated-flag convention**: `Electrode.bad_channel='True'` is intended to mark a
  *quality-bad* (dead/noise-class) channel — the boolean cannot encode an `out`
  label, so populate it via `suggest_bad_channels` (phase 2, which never writes
  `out`). Document that this convention is **not enforced by the schema**, so the
  `interpolate` path **audits** every label-less curated flag with a coherence/PSD
  pass and *removes* (never fills) any channel that comes back outside-brain
  (invariant d) — a pre-existing / manual / config flag on an out-of-brain
  channel is therefore handled safely, never interpolated. Also document that the
  `specific` reference electrode is excluded from both handling **and** the
  detector surface, and that its own quality is vetted by the `make_fetch`
  curated-bad check (a curated-bad reference raises). A CHANGELOG entry. A
  `SpikeSortingV2_Migration.md` note is **not** needed (default is byte-identical
  to current v2; these are new opt-in v2 capabilities with no v1↔v2 parity
  change).

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
| `out` removed before interpolate *(stub/mock)* | `bad_channel_detection` labels a channel `"out"` and another `"dead"` with `bad_channel_handling="interpolate"` → the `"out"` id goes to `remove_channels` and the `"dead"` id to `interpolate_bad_channels` (label policy, not blanket interpolate), **and `remove_channels` is invoked before `interpolate_bad_channels`** so the removed `out` channel cannot be an interpolation donor (assert recorded call order). |
| dead/noise follow the param *(stub/mock)* | the same `"dead"`/`"noise"` channels go to `remove` when `bad_channel_handling="remove"`. |
| reference excluded from detector surface *(stub/mock)* | `reference_mode="specific"` with the reference present and `bad_channel_detection` set → the channel id passed into `detect_bad_channels` (via the `channel_slice` building the detector) **omits** the reference; the reference is in **neither** `to_remove` nor `to_interpolate`; `common_reference(reference="single")` is still called with it. |
| detection runs per shank *(stub/mock)* | a two-shank `channel_shank` map → `detect_bad_channels` is invoked **once per shank** on each shank's `channel_slice`, never once on the mixed-shank recording (spy on call count + the ids each call sees). |
| curated `out` flag audited, not filled *(stub/mock)* | `bad_channel_handling="interpolate"`, `bad_channel_detection=None`, a curated `bad_channel_ids` channel that the audit pass classifies `"out"` → it is moved to `remove_channels` (a warning is logged), **not** `interpolate_bad_channels`; a curated channel classified `dead`/`good` is interpolated. |
| curated-bad reference raises with detection off *(integration, DB)* | a `specific` sort group whose reference electrode has `Electrode.bad_channel='True'`, materialized with `bad_channel_detection=None` → `make_fetch` raises the clear "reference is curated-bad" error (the detection-independent reference guard); restore the flag. |
| detection params are dumped, None dropped *(stub/mock)* | `bad_channel_detection=BadChannelDetectionParams(psd_hf_threshold=None)` → `detect_bad_channels` is called via `model_dump()` and the `None` is not forwarded (no SI crash). |
| `_interior_bad_channel_ids` pitch-anchored *(unit, no DB)* | radius is `RADIUS_FACTOR × pitch` (full-shank `_shank_pitch`), **not** the group's own spacing: with exactly two far-apart good channels and a bad channel at their midpoint, the candidate is **dropped** (the degenerate case a group-`d_nn` rule would wrongly keep); a bad channel within `pitch` of ≥2 good channels on a dense run is kept; `pitch=None` or `<2` good channels → `[]`. |
| `_shank_pitch` is full-shank *(unit, no DB)* | `_shank_pitch` over all electrodes on a shank returns the dense nominal spacing regardless of how few channels the sort group keeps (it is computed from the whole shank, not the group). |
| interior re-inclusion via adjacency *(integration, DB)* | for an arbitrary-column group whose good channels form two separated clusters, `make_fetch` re-includes a bad electrode embedded among good members but NOT one in the gap between clusters, nor one outside the group's footprint — pitch-anchored adjacency, not a bounding box. |
| no positions → interpolate raises *(integration, DB)* | a session without electrode coordinates + `bad_channel_handling="interpolate"` raises the clear "needs positions" error rather than guessing. |
| default cache_hash unchanged *(integration, DB+SI, slow)* | smoke-fixture `Recording` with all defaults has the same `cache_hash` as pre-phase code (headline regression guard). |
| interpolate fills a bad channel *(integration, DB+SI, slow)* | mark one *interior* smoke-fixture electrode `bad_channel='True'`, materialize with `bad_channel_handling="interpolate"` → the cached recording retains that channel (count complete) and its trace differs from the raw (filled, not zero); restore the flag. |
| remove drops a detected channel *(integration, DB+SI, slow)* | with `bad_channel_detection` flagging a synthesized-bad channel and `remove`, the cached recording omits it. |

Stub tests are DB-free (fake recording + monkeypatched `sip.*`). Mark the three
DB+SI integration tests (cache_hash, interpolate-fills, remove-drops) slow;
restore any mutated `Electrode.bad_channel` in every DB test.

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
  `interpolate`, and **removed before** `interpolate_bad_channels` runs so they
  are not used as interpolation donors; only `dead`/`noise` follow
  `bad_channel_handling`. Curated (label-less) channels follow the user's chosen
  handling.
- The interior re-inclusion uses the **pitch-anchored adjacency** rule
  (`≥MIN_GOOD_NEIGHBORS` good channels within `RADIUS_FACTOR × pitch`, where
  `pitch` is the **full-shank** `_shank_pitch`), **not** a `[min,max]` span and
  **not** the group's own spacing (which degenerates to the gap for a two-channel
  group and would wrongly keep the midpoint); so it pulls in neither whole-shank
  bad channels nor inter-cluster gap channels; positions absent / `pitch=None` →
  interpolate raises.
- **The `specific` reference is excluded from both handling and the detector
  surface** — it is sliced in only for subtraction and dropped after referencing,
  and leaving it in the coherence pool would perturb the other channels' labels.
  Its own quality is guarded by `make_fetch`, which **raises** when the reference
  is curated `bad_channel='True'` — so "a reference flagged bad raises" holds, via
  the curated check rather than a detection-time check.
- **Detection runs per shank** (via the threaded `channel_shank` map), never once
  on a mixed-shank recording; labels are int-keyed and merged across shanks.
- `BadChannelDetectionParams` is **dumped** (`.model_dump()`, not `**model`)
  before forwarding; its fields default to `None` = "use SI default" and the
  wrapper drops `None`, so no threshold is hardcoded and `None` never reaches
  SI; `extra="allow"` keys are validated against the installed
  `detect_bad_channels` signature. The empty-params audit pass (detection off,
  curated interpolate) forwards `{}` → method-only SI defaults.
- **Curated-flag convention documented as unenforced**, with the **audit** as the
  enforcement: the `interpolate` path runs a coherence/PSD pass over label-less
  curated flags and *removes* (never fills) any classified outside-brain, so a
  manual / pre-existing / config flag on an out-of-brain channel cannot invent
  signal. Detection acts on non-curated channels **only when the user opted in**.
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
