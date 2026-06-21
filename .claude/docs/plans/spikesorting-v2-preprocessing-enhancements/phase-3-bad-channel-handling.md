# Phase 3 — Bad-channel handling: remove vs interpolate

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add a `bad_channel_handling = "remove" | "interpolate"` preprocessing
parameter. `"remove"` (the default) is **byte-identical to today**: the sort
group is its declared `SortGroupElectrode` members, and the curated-bad channels
that grouping already excluded stay excluded. `"interpolate"` re-includes the
group's **pitch-adjacent interior** curated-bad channels (the ones grouping
excluded) and fills them from good neighbors (`interpolate_bad_channels`) so
geometry-aware sorters see a complete probe.

**This phase does not detect bad channels.** Detection is owned entirely by
[phase 2](phase-2-bad-channel-detection.md)'s `suggest_bad_channels`, which runs
the coherence/PSD method **on the full physical shank** (the correct surface) and
writes `Electrode.bad_channel` flags. Phase 3 only *consumes* those curated
flags. (An inline at-materialization detector was considered and rejected: it
would run on the restricted sort-group recording — not the physical shank — and
mis-label coherence/`out` for sparse custom groups; see overview Non-Goals.)

**Where the flags are consumed — the ordering contract.** A sort group fixes its
membership **at creation**: `set_group_by_shank`
([recording.py:443](../../../../src/spyglass/spikesorting/v2/recording.py#L443))
and `set_group_by_electrode_table_column`
([recording.py:733](../../../../src/spyglass/spikesorting/v2/recording.py#L733))
filter `bad_channel='True'` then, and `make_fetch` later reads that stored
`SortGroupElectrode` membership verbatim. So `Electrode.bad_channel` flags are
consumed **(a)** at group creation — flagged channels are excluded from the sort
target — and **(b)** by phase 3's `interpolate`, which re-includes those excluded
interior channels and fills them. `remove` re-reads **nothing** at
materialization; it honors the declared membership (which is why it is a true
no-op and byte-identical to today). **Consequence the executor must document:**
set / curate `bad_channel` flags (e.g. run `suggest_bad_channels`) **before**
creating the sort group. A channel flagged bad *after* a group already exists
stays a member, and `remove` will **not** drop it — recreate the sort group to
apply later flags. (This is the deliberate, DataJoint-idiomatic choice — declared
membership is authoritative — not a gap; making `remove` re-filter at
materialization was considered and rejected because it contradicts "`remove` =
today's behavior" and conflicts with
`set_group_by_electrode_table_column(remove_bad_channels=False)` groups whose
members are bad *by design*. See the validation slice and overview Open
Question 6.)

**Convention boundary this phase relies on (and documents):**
`Electrode.bad_channel='True'` means a **quality-bad** (dead/noise-class)
channel that is safe to interpolate or remove — it must **not** mark an
outside-brain (`out`) channel. The boolean column cannot carry the `out` label,
and under `interpolate` a curated flag is *filled*, so an `out` channel marked
`bad_channel` would invent signal. [Phase 2](phase-2-bad-channel-detection.md)
enforces this on the write side (`suggest_bad_channels` never writes `out`);
phase 3 documents that a manually/externally set flag on an out-of-brain channel
must use `remove`/exclusion, not `interpolate`.

See the [overview's bad-channel design decision](overview.md#the-bad-channel-handling-design-decision-chosen-option-a):
the sort group stays the **good-channel sort target**; the curated-bad channels
that grouping excluded are re-included **only** on the `interpolate` path so they
can be filled. The `specific` reference electrode is **not a sort target** — it
is sliced in only for subtraction and dropped after referencing, so it is never a
handling target (a reference flagged `bad_channel='True'`, e.g. a dedicated
ground, is an intentionally-supported config and is **preserved**, exactly as
today — phase 3 adds no reference-quality check).

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_params/preprocessing.py:114](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L114) —
  `PreprocessingParamsSchema` fields; `to_pre_motion_dict` (`:134`).
- [src/spyglass/spikesorting/v2/_recording_restriction.py:154](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py#L154) —
  `restrict_recording`: the channel-slice (`:184-203`). The `specific` branch
  already shows how to add channels to the slice; `interpolate` adds the group's
  curated-bad channels the same way. (`:101`/`:198`: this codebase slices via
  `ChannelSliceRecording`, since SI 0.104 dropped `Recording.channel_slice` —
  phase 3 only extends the existing `slice_ids`, it adds no new slicing.)
- [src/spyglass/spikesorting/v2/_recording_preprocessing.py:31](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L31) —
  `apply_pre_motion_preprocessing`: bandpass at `:369`, reference at `:380`.
  Handling goes **between** them.
- [src/spyglass/spikesorting/v2/recording.py:1075](../../../../src/spyglass/spikesorting/v2/recording.py#L1075) —
  `RecordingFetched` NamedTuple; `make_fetch` (`:1159`) computes its fields;
  `make_compute` (`:1272`) unpacks them into `_compute_recording_artifact`
  (`:1639`), which calls `restrict_recording` then
  `apply_pre_motion_preprocessing`.
- [src/spyglass/spikesorting/v2/recording.py:117](../../../../src/spyglass/spikesorting/v2/recording.py#L117) —
  `_reference_electrode_group`: documents that v2 looks the `specific` reference
  up in the **full, not-bad-filtered** electrode set, so a `bad_channel='True'`
  reference is intentionally supported. Phase 3 preserves this (no raise).
- [phase 2](phase-2-bad-channel-detection.md) — the single detection surface;
  populates the `Electrode.bad_channel` flags phase 3 consumes.
- **Upstream model:** [appendix B.4](appendix.md#b4-bad-channel-interpolation)
  (IBL — kriging interpolation, filled after the filter and before the spatial
  step, the order this phase mirrors) and
  [appendix A.2](appendix.md#a2-bad-channel-detection--removal) (AIND — the
  remove path).

## Tasks

- **Schema: `bad_channel_handling`**
  ([_params/preprocessing.py](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py)).
  Add one field on `PreprocessingParamsSchema`:
  `bad_channel_handling: Literal["remove", "interpolate"] = "remove"`. Add it to
  `to_pre_motion_dict`. Extend the `schema_version` history docstring (added at
  v3, **no version bump**: `remove` keeps existing blobs valid and output
  identical; dev rows regenerated). No detection sub-model is added — detection
  is phase 2's `suggest_bad_channels`, not a preprocessing parameter.

- **`make_fetch`: fetch the group's *interior* curated-bad channel ids (interpolate
  only)** ([recording.py:1159](../../../../src/spyglass/spikesorting/v2/recording.py#L1159)).
  Add `bad_channel_ids` to `RecordingFetched` (`:1075`). It is **empty for
  `remove`** (that path re-includes nothing — today's behavior) and computed
  only when `preproc_validated.bad_channel_handling == "interpolate"`. The
  candidate set is **not** "every bad channel on the shank" —
  `set_group_by_electrode_table_column`
  ([recording.py:627](../../../../src/spyglass/spikesorting/v2/recording.py#L627))
  builds **arbitrary-membership** groups and does not persist the original
  requested column values, so a shank-wide fetch can pull in bad electrodes that
  were never near a custom group. For `interpolate`:
  - Fetch the candidate bad electrodes that share the sort group's
    `(electrode_group_name, probe_shank)` set **and** carry `bad_channel='True'`,
    with their positions (`Electrode.x/y/z`).
  - **Compute the shank's physical pitch.** For each `(electrode_group_name,
    probe_shank)` the group touches, read **all** that shank's electrode
    positions (good + bad) from `Electrode` and take the median nearest-neighbor
    distance — the probe's nominal electrode spacing, a physical constant. This
    is the scale for the adjacency test.
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
        which channels a sort group happens to keep. Returns ``None`` when the
        shank has < 2 electrodes OR any coordinate is non-finite
        (``Electrode.x/y/z`` are nullable -> a NULL arrives as NaN); a ``None``
        pitch makes the caller raise the clear "needs positions" error rather
        than silently producing NaN distances.
        """
        import numpy as np

        xyz = np.asarray(shank_xyz, dtype=float)
        if xyz.shape[0] < 2 or not np.isfinite(xyz).all():
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
        channel in a dense run is kept. Returns a sorted list. Defensive on
        non-finite input (``make_fetch`` raises before calling): a non-finite
        ``pitch`` or good position -> ``[]``; a candidate with a non-finite
        position is skipped (never silently treated as adjacent).
        """
        import numpy as np

        good = np.asarray(good_xyz, dtype=float)
        if (good.shape[0] < 2 or not pitch or not np.isfinite(pitch)
                or not np.isfinite(good).all()):
            return []  # need >=2 finite good channels and a finite pitch
        radius = RADIUS_FACTOR * float(pitch)
        return sorted(
            int(cid)
            for cid, pos in candidate_xyz
            if np.isfinite(pos).all()
            and int((np.linalg.norm(good - np.asarray(pos, float), axis=1)
                     <= radius).sum()) >= MIN_GOOD_NEIGHBORS
        )
    ```

    (Apply per shank when the group spans more than one — each shank uses its
    own `pitch`.)
  - **Reject non-finite positions explicitly — do not rely on the helpers'
    empty return.** `Electrode.x/y/z` are **nullable** (a NULL arrives as
    `NaN`). For `interpolate`, verify that every position needed is finite: the
    group's good-channel positions, the full-shank positions (so `pitch` is
    defined), and the candidate positions. If any is null/NaN — or `_shank_pitch`
    returns `None` (legacy NWBs without coordinates, or a shank with `< 2`
    positioned electrodes) — **raise** the clear "interpolate needs positions"
    error (point the user to `remove`). This is mandatory: non-finite
    coordinates otherwise make `_interior_bad_channel_ids` return an **empty**
    set, which would silently look like "no bad channels to fill" instead of the
    promised hard error. See overview Open Question 4.
  - Return `bad_channel_ids` as a sorted tuple (DeepHash-stable, like the other
    `make_fetch` outputs) and thread it through `make_compute` →
    `_compute_recording_artifact` → `restrict_recording` and
    `apply_pre_motion_preprocessing`.

- **`restrict_recording`: include curated-bad channels on the interpolate path**
  ([_recording_restriction.py:154](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py#L154)).
  Add `bad_channel_handling` and `bad_channel_ids` params. When
  `bad_channel_handling == "interpolate"`, union `bad_channel_ids` (the *interior*
  curated-bad set computed in `make_fetch`) into `slice_ids` (alongside the
  existing `specific`-reference channel) so those bad channels are present in the
  sliced recording to be filled. `"remove"` leaves the slice as today (good
  channels only; the curated-bad are not re-added).

- **`apply_pre_motion_preprocessing`: the handling step**
  ([_recording_preprocessing.py:31](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L31)).
  Add `bad_channel_handling` and `bad_channel_ids` params (it already receives
  `reference_mode` / `reference_electrode_id`). Insert **after** the bandpass
  (`:369`) and **before** the reference (`:380`). Only the `interpolate` path
  does anything — the curated-bad channels it re-included are filled; the
  `specific` reference is excluded (it is sliced in only for subtraction and
  dropped after `common_reference`, never a handling target). On `remove` the
  curated channels were never re-included, so there is nothing to do (today's
  behavior).

  ```python
  # Bad-channel handling: between filter and reference (matches IBL/AIND order).
  present = {int(c) for c in recording.get_channel_ids()}
  # The 'specific' reference is present only for subtraction (dropped after
  # common_reference); it is never a handling target.
  ref_id = int(reference_electrode_id) if reference_mode == "specific" else None
  # Only "interpolate" re-includes curated-bad channels (restrict_recording);
  # on "remove" they are already absent, so this set is empty -> no-op.
  to_interpolate = sorted(
      (present & {int(c) for c in bad_channel_ids}) - {ref_id}
  )
  if bad_channel_handling == "interpolate" and to_interpolate:
      # interpolate needs channel locations; fail clearly if absent.
      if recording.get_channel_locations() is None:
          raise ValueError(
              "apply_pre_motion_preprocessing: interpolate requires channel "
              "locations on the recording, but none are set (no probe "
              "geometry). Use bad_channel_handling='remove', or ensure the "
              "NWB / probe carries electrode positions."
          )
      recording = sip.interpolate_bad_channels(recording, to_interpolate)
  else:
      to_interpolate = []
  applied_steps["bad_channels"] = {"interpolated": to_interpolate}
  ```

  `applied_steps` is the report introduced in
  [phase 1](phase-1-adc-phase-shift.md) so `filtering_description` can name what
  ran (next task). Note the Option-A asymmetry (documented in the overview): for
  `"remove"`, `bad_channel_ids` are **already absent** (restrict_recording did
  not re-add them); for `"interpolate"`, the interior curated-bad channels are
  present and get filled.

- **`filtering_description`**
  ([_recording_preprocessing.py:243](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L243))
  — read the `applied_steps["bad_channels"]` report (not the params) and append
  `"interpolate N bad channels"` **only when N > 0** (with the actual count that
  ran) so the NWB provenance reflects reality. On the default `remove` path the
  count is 0 and nothing is appended, preserving today's string byte-for-byte.

- **Documentation (ships in this phase):** the schema docstring; a
  `SpikeSortingV2.md` subsection on `bad_channel_handling` — when to interpolate
  vs remove (interpolate for geometry-aware sorters on dense shanks, remove for
  tetrodes / sparse groups), that flags come from `suggest_bad_channels` (phase
  2) or curation, and the **convention boundary**: `Electrode.bad_channel='True'`
  is *quality-bad* (dead/noise-class) only — a manually/externally set flag on an
  outside-brain channel must use `remove`/exclusion, never `interpolate` (which
  would invent signal). **Document the ordering contract prominently:** curate /
  set `bad_channel` flags **before** creating the sort group (membership is fixed
  at creation); a channel flagged bad after the group exists stays a member and
  `remove` will not drop it — recreate the sort group to apply later flags. Note
  that the `specific` reference electrode is never a handling target and a
  `bad_channel='True'` reference (e.g. a dedicated ground) is supported exactly as
  today. A CHANGELOG entry. A `SpikeSortingV2_Migration.md` note is **not** needed
  (default is byte-identical to current v2; these are new opt-in v2 capabilities
  with no v1↔v2 parity change).

## Deliberately not in this phase

- **Inline at-materialization bad-channel detection.** Detection is owned by
  [phase 2](phase-2-bad-channel-detection.md)'s `suggest_bad_channels`, which
  runs on the **full physical shank** — the correct surface. An inline detector
  in the materialization path would run on the *restricted sort-group recording*
  and mis-label coherence/`out` for sparse arbitrary-column groups; it is
  rejected (overview Non-Goals). If inline auto-detection is wanted later, it
  must be built on a full-shank surface as its own phase.
- Any **reference-quality** check / raise on a `bad_channel='True'` reference —
  v2 intentionally supports a bad-marked reference (overview Open Question 5);
  changing that needs a separate feature with its own migration note.
- **Re-filtering sort-group membership at materialization** (the rejected
  alternative to the ordering contract): having `remove` re-read current
  `bad_channel` flags and `remove_channels` the now-bad members. Rejected because
  it contradicts "`remove` = today's behavior", makes the recording diverge from
  the declared `SortGroupElectrode` membership, and conflicts with
  `set_group_by_electrode_table_column(remove_bad_channels=False)` groups whose
  members are bad by design. Membership stays authoritative; flags are applied at
  group creation (overview Open Question 6).
- Changing the grouping helpers to include bad channels (the rejected Option B;
  see overview Open Question 1).
- A spatial-frequency "destripe" reference (`highpass_spatial_filter`) — a
  possible future `reference_mode`, out of scope here.
- Phase-shift (phase 1) and drift estimation (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| interpolate between filter and reference *(stub/mock)* | `interpolate` with a non-empty `bad_channel_ids` invokes `sip.interpolate_bad_channels` **after** `bandpass_filter` and **before** `common_reference`; order recorded via monkeypatch (mirror `test_preprocessing_order.py`). |
| remove default is a no-op *(stub/mock)* | `bad_channel_handling="remove"`, empty `bad_channel_ids` → neither `interpolate_bad_channels` nor `remove_channels` is called; `applied_steps["bad_channels"]["interpolated"]` is `[]`. |
| reference is never interpolated *(stub/mock)* | `reference_mode="specific"`, `interpolate`, with the reference id among the present channels → the reference id is **not** passed to `interpolate_bad_channels`; `common_reference(reference="single")` is still called with it. |
| interpolate needs locations *(stub/mock)* | `interpolate` with a non-empty set on a recording whose `get_channel_locations()` is `None` raises the clear ValueError. |
| `_interior_bad_channel_ids` pitch-anchored *(unit, no DB)* | radius is `RADIUS_FACTOR × pitch` (full-shank `_shank_pitch`), **not** the group's own spacing: with exactly two far-apart good channels and a bad channel at their midpoint, the candidate is **dropped** (the degenerate case a group-`d_nn` rule would wrongly keep); a bad channel within `pitch` of ≥2 good channels on a dense run is kept; `pitch=None` or `<2` good channels → `[]`. |
| `_shank_pitch` is full-shank *(unit, no DB)* | `_shank_pitch` over all electrodes on a shank returns the dense nominal spacing regardless of how few channels the sort group keeps (computed from the whole shank, not the group). |
| non-finite positions guarded *(unit, no DB)* | `_shank_pitch` with any `NaN` coordinate returns `None`; `_interior_bad_channel_ids` with a non-finite `pitch` or good position returns `[]` and skips a candidate whose own position is `NaN` (never counts it as adjacent) — so a partial-null shank cannot silently yield an empty re-inclusion that masquerades as "no bad channels". |
| partial-null positions raise *(integration, DB)* | an `interpolate` group where one needed electrode has a `NULL` `Electrode.x/y/z` → `make_fetch` raises the clear "interpolate needs positions" error (not an empty, silent re-inclusion); restore the row. |
| interior re-inclusion via adjacency *(integration, DB)* | for an arbitrary-column group whose good channels form two separated clusters, `make_fetch` re-includes a bad electrode embedded among good members but NOT one in the gap between clusters, nor one outside the group's footprint — pitch-anchored adjacency, not a bounding box. |
| no positions → interpolate raises *(integration, DB)* | a session without electrode coordinates + `bad_channel_handling="interpolate"` raises the clear "needs positions" error rather than guessing. |
| bad-marked reference still materializes *(integration, DB)* | a `specific` sort group whose reference electrode has `Electrode.bad_channel='True'`, with default params → materializes (no raise), the reference is used for subtraction and dropped after referencing, `cache_hash` matches the same row pre-plan (the round-3 raise is gone; v2's intentional support for a bad-marked reference is preserved); restore the flag. |
| default cache_hash unchanged *(integration, DB+SI, slow)* | smoke-fixture `Recording` with all defaults (`bad_channel_handling="remove"`) has the same `cache_hash` as pre-phase code (headline regression guard). |
| interpolate fills a bad channel *(integration, DB+SI, slow)* | mark one *interior* smoke-fixture electrode `bad_channel='True'`, materialize with `bad_channel_handling="interpolate"` → the cached recording retains that channel (count complete) and its trace differs from the raw (filled, not zero); restore the flag. |
| flag-before-create workflow: `remove` omits a flagged channel *(integration, DB)* | flag an electrode `bad_channel='True'`, **then** create the sort group (`set_group_by_shank`) → the channel is excluded from `SortGroupElectrode` at creation → materialize with `remove` → the cached recording omits it. This is the documented ordering contract; restore the flag. |
| stale group: post-creation flag is **not** dropped by `remove` *(integration, DB)* | create the sort group **first**, then flag one of its members `bad_channel='True'` → `make_fetch` still returns it as a member and `remove` leaves it in the recording (the documented limitation — declared membership is authoritative; recreate the group to apply later flags). Pins the behavior so it is not a silent surprise; restore the flag. |

Stub tests are DB-free (fake recording + monkeypatched `sip.*`). Mark the two
DB+SI integration tests (cache_hash, interpolate-fills) slow; restore any mutated
`Electrode.bad_channel` in every DB test.

## Fixtures

- Stub: the fake-recording + call-order harness from `test_preprocessing_order.py`
  (extend it to expose `get_channel_locations` / `get_channel_ids`).
- Integration: `mearec_polymer_smoke` + `dj_conn`; mutate/restore one electrode's
  `bad_channel` to exercise interpolate.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- The default path (`remove`) is byte-identical — the cache_hash regression row
  passes and `filtering_description` is unchanged (nothing appended when N = 0).
- Handling sits **between** filter and reference; `interpolate` re-includes only
  the group's **interior** curated-bad channels via `restrict_recording`, and
  `remove` re-includes nothing (so its handling step is a genuine no-op).
- **Ordering contract honored and documented:** flags are consumed at group
  creation (exclusion) and by `interpolate` (re-inclusion); `remove` honors the
  declared `SortGroupElectrode` membership and does **not** re-filter at
  materialization. The docs tell users to set `bad_channel` flags before creating
  the sort group; the flag-before-create and stale-group regression rows pass.
- The interior re-inclusion uses the **pitch-anchored adjacency** rule
  (`≥MIN_GOOD_NEIGHBORS` good channels within `RADIUS_FACTOR × pitch`, where
  `pitch` is the **full-shank** `_shank_pitch`), **not** a `[min,max]` span and
  **not** the group's own spacing (which degenerates to the gap for a two-channel
  group and would wrongly keep the midpoint); positions absent / `pitch=None` →
  interpolate raises.
- **Non-finite positions raise, not silently drop.** `Electrode.x/y/z` are
  nullable; `make_fetch` explicitly checks finiteness of every needed position
  and raises the "needs positions" error for `interpolate` — it does **not** rely
  on `_interior_bad_channel_ids` returning an empty set. The helpers also guard
  non-finite input defensively.
- **No detection in this phase.** There is no `bad_channel_detection` field, no
  per-shank detector, no `channel_shank`, no audit pass — detection is phase 2's
  `suggest_bad_channels` (full-shank). The only channels handled are curated
  `Electrode.bad_channel` flags.
- **Convention boundary documented:** `Electrode.bad_channel` is quality-bad only
  (never outside-brain); a manual `out` flag must use `remove`/exclusion. Phase 2
  enforces it on the write side (never writes `out`).
- **The `specific` reference is excluded from handling** (it is sliced in only
  for subtraction and dropped after referencing) and a `bad_channel='True'`
  reference materializes exactly as today — **no reference-quality raise** (the
  round-3 raise is removed; the regression row proves the default row still
  materializes).
- `apply_pre_motion_preprocessing` returns the `applied_steps` report and
  `filtering_description` reads it (honest count, not param-derived).
- Interpolation guards on channel locations and raises clearly, never letting SI
  fail opaquely deep in the call stack.
- `bad_channel_ids` is DeepHash-stable across two `make_fetch` calls (sorted
  tuple), matching the other fetched fields.
- Validation slice passes; integration tests are marked; mutated flags restored.
- Docstrings / test names / module names don't reference this plan or its phases.
- The CHANGELOG + `SpikeSortingV2.md` subsection are present, not deferred.
