# Designs — Optional motion correction and matching concatenated sorts

[← back to PLAN.md](PLAN.md) · [phase 3c](phase-3c-motion-correction.md) · [phase 4c](phase-4c-concat-unitmatch.md)

Owner-requested scope addition, 2026-09-19. These are planned v2 features, not descriptions of existing support. They extend the original review-fix scope; they do not change phase 4a's algorithm or acceptance thresholds. v2 schema recreation is allowed; v0/v1 remain unaffected.

## Evidence and upstream behavior

- SpikeInterface separates estimation (`compute_motion`) from application (`interpolate_motion`); `correct_motion` combines them. Estimate on filtered, unwhitened traces. Saving an estimate does not apply it. See the [motion guide](https://spikeinterface.readthedocs.io/en/latest/how_to/handle_drift.html) and [preprocessing guidance](https://spikeinterface.readthedocs.io/en/latest/modules/motion_correction.html#preprocessing-details).
- **Correction to the discussion:** in [SpikeInterface 0.104.3](https://github.com/SpikeInterface/spikeinterface/blob/0.104.3/src/spikeinterface/preprocessing/motion.py), `rigid_fast` uses `method="dredge_ap"`, `rigid=True`, `bin_s=5.0`, center-of-mass localization, and channel removal at the borders. `dredge` / `dredge_fast` use different localization, nonrigid settings, and extrapolation at the borders. It is incorrect to describe `rigid_fast` as containing no DREDge. Its suitability, resolution, and interpolation behavior need validation; selecting the DREDge name alone does not establish quality.
- Our `DriftEstimate` is an on-demand, fixed-preset QC result. Our concat path applies motion before whitening but discards motion trajectories. The ordinary single-recording path cannot select that application step. `run_v2_pipeline` currently uses the presence of a motion recipe to distinguish concat from single-recording mode; that coupling must be removed.
- Our UnitMatch restriction was introduced in `9343433f`: `_curation_member_identity` accepts only a `Recording` source. The restriction protects single-member ownership assumptions; it is not a UnitMatch restriction on waveform bundles.
- `.claude/docs/plans/pr1609-remediation/phase-2-concat-member-curations.md` and its `designs.md` incorrectly say matching continues to consume concat curations. Treat those statements as superseded by this design, not as implementation evidence. The feature documentation must not claim this workflow works until phase 4c ships.

## Independent motion stage

### Data flow and ownership

```text
raw -> temporal preprocessing on continuous acquisition data
    -> recording selection / spatial preprocessing
    -> selected artifact masks
    -> one recording OR assembly of frozen recording members
    -> optional motion estimation -> optional interpolation
    -> unwhitened sort input -> sorter-specific whitening -> sorting / curation
```

Assembly decides membership and frame mapping. Motion decides a spatial transformation of those frames. Neither choice implies the other. A single recording need not be wrapped in a fake one-member concat to obtain correction.

Keep `Recording` as the reusable preprocessed source. Artifact detection already depends on it; putting artifact-aware correction inside `Recording.make` would introduce the wrong dependency direction. Place the new result after mask selection, before sorting. New concat assembly persists its masks and member boundaries but does not run a separate motion implementation.

Add a v2 source-resolution service that returns both **lineage** and **effective traces**. Lineage includes the original recording/member keys, absolute timestamps, ordered frame spans, and electrode identities. Effective traces include the selected mask/correction result, channel map, geometry, and unwhitened recording. The existing single/concat lineage distinction remains explicit even after correction. Do not make metadata consumers infer an original session from an arbitrary first member.

### Persisted stages

Names below are proposed table names; the ownership and identity contracts are required.

| Table/result | Required identity and output |
| --- | --- |
| `MotionEstimateSelection` | Exactly one single-recording or uncorrected-concat source; frozen source content, mask selection, observation/boundary spans, geometry, and immutable estimation recipe |
| `MotionEstimate` | Serialized displacement field with time/depth coordinates and reference frame; resolved estimation parameters and producing versions; useful diagnostics and the input fingerprint |
| `MotionCorrectedRecordingSelection` | One saved estimate plus an immutable interpolation recipe; verifies that estimate and trace source have identical frames, geometry, and masks |
| `MotionCorrectedRecording` | Corrected unwhitened traces, source/estimate references, output channel-to-electrode mapping, observation/boundary spans, interpolation parameters, and content hash |

Separate estimation and interpolation configuration in the named parameter model. `MotionCorrectionParameters` can compose the two recipes for the public API, but changing only interpolation must reuse the same saved estimate. Resolve upstream preset defaults into explicit persisted parameters; storing only `preset="auto"` is insufficient. Include the resolved scientific settings and algorithm/schema version in stage identity so an upstream default change cannot reuse an incompatible artifact.

An optional correction-reference part on `SortingSelection` can select the processed result while retaining the base recording/concat lineage. At selection and compute time, require that correction source and masks exactly match the selected base input. Applying correction participates in `sorting_id`; merely producing a QC estimate does not change the uncorrected sort identity. A correction failure raises and produces no successful corrected row or sorting receipt; never fall back silently to uncorrected data.

Use the existing staged NWB/file-write and fetch/compute/insert patterns. Long estimation and interpolation run outside DB transactions. Persist the estimate needed to rebuild corrected traces; rebuilding must apply that saved estimate, not silently estimate motion again. Verify rebuilt content before replacing a cache. Failed writes leave no registered partial outputs; foreign keys and cleanup inventories protect active estimates and corrected recordings.

Keep existing `DriftEstimate` rows QC-only. New configurable estimation may share serialization and computational helpers with that table, but old fixed-preset estimates are not reusable for correction without matching the complete new source/mask/recipe contract.

### Public modes and defaults

| Mode | Motion result | Recording used for sorting |
| --- | --- | --- |
| `off` | None required | Existing masked, uncorrected source |
| `estimate` | Saved estimate and QC | Same source and scientific sort identity as `off` |
| `apply` | Saved estimate and corrected recording | Explicitly selected corrected source |

Expose the mode and named motion recipe consistently in planning, preflight, execution, receipts, and `describe_run`. `estimate`/`apply` require a named recipe; contradictory mode/recipe combinations fail preflight. Correction stays off for existing single-recording defaults. New recipes are opt-in until probe-specific scientific validation passes. Do not relabel the current experimental `rigid_fast` concat recipe as validated or silently remap an immutable `auto_default` row to another algorithm.

Reuse one motion implementation for both input shapes. Version/recreate preproduction concat artifacts and recipes when moving motion out of `ConcatenatedRecording`; document the transition. Existing corrected caches must never be treated as uncorrected assembly or corrected a second time. This is a bounded v2 schema change, not a new generic processing framework for v0/v1.

### Scientific and coordinate invariants

- Estimate from valid, unwhitened signal. Reuse phase 3b's explicit statistics spans for noise estimates and peak sampling. Mask zeros, artifact edges, and artificial joins must not become evidence for the estimator. Reapply exclusions after spatial interpolation; retain original observation intervals.
- Preserve sample count, sample order, sampling frequency, and the complete original-time/member back-map. Motion correction changes spatial samples, not when a spike occurred.
- Keep acquisition-continuity boundaries distinct from artifact exclusions. No waveform/noise support window may bridge an acquisition gap or member join. Estimation's time-bin/window and gap policy must be explicit, persisted, and tested with known jumps. Do not compress unobserved intervals into a fictitious continuous clock or independently reset each span to zero motion and assume their reference frames agree. Reject unsupported discontinuous inputs clearly until the adapter can establish a common reference frame; the feature is not complete for concat until that route is tested.
- Validate finite, distinct effective contact positions, motion axis, shank partition, and sufficient spatial/time support. A geometry suitable for sorting is not automatically suitable for nonrigid motion estimation; the repaired tetrode layout is not evidence that DREDge should be enabled on tetrodes.
- Make border handling explicit. Channel removal and extrapolation have different consequences; record removed channels and output positions. No zero-channel output or unexplained renumbering. Validate common effective geometry before matching corrected recordings across sessions; do not silently pad, reorder, or discard channels to satisfy UnitMatch.
- Check the selected sorter's internal motion behavior. New supported recipes explicitly specify how external and internal correction interact; an unvalidated combination is a preflight error, not an undocumented double correction. Keep preprocessing/whitening ownership explicit as in the current sorter dispatcher.
- Route sorting, all analyzer builders/rebuilds, curation accessors, waveform extraction, UnitMatch bundles, and recomputation through the effective-trace contract. Keep access to original preprocessed data explicit. Matching must record whether its waveforms came from corrected or original traces; the new external-correction path uses the selected corrected, unwhitened source.

### Validation and promotion

Software correctness and scientific suitability are separate. Passing serialization and schema checks does not validate a motion recipe for the 128-channel polymer probe.

1. Structural tests: exact bypass behavior, saved-estimate reuse, scientific identity changes, masks, original clocks, channel/electrode maps, persistence, failure cleanup, and consistent consumer/rebuild resolution.
2. Known-answer fixtures: no motion, rigid motion, nonrigid motion, abrupt jumps, disjoint acquisition intervals, artifacts, uneven firing, sparse activity, and units entering/leaving view. Compare estimated displacement in the same reference frame (allow the known arbitrary global offset), then compare actual interpolated traces and sorting results, not only shifted peak coordinates.
3. Probe-specific evaluation: paired `off` / `rigid_fast` / `dredge` / `dredge_fast` runs on the same seeded simulations and available representative lab recordings. Report unit precision/recall, false merges, oversplits, runtime, memory, border-channel effects, and no-motion degradation. Do not infer success from more matches or fewer residual motion pixels alone.
4. Before running the held-out acceptance set, commit its fixture/seed manifest, metric definitions, and numeric tolerances justified on separate development data. Record failures rather than adjusting the gates to the acceptance output. Until these results exist, retain experimental/opt-in status and make missing real-data evidence visible.

## Matching independently sorted recording groups

### Matching input, rather than one original session

The supported extension is one input per independently curated sort: a single recording or a same-day concatenation. Example: Monday's three jointly sorted epochs supply one bundle; Tuesday's two jointly sorted epochs supply another. Within a concat, shared unit IDs already establish member identity; do not run UnitMatch between those exports or count the same parent unit more than once.

Replace `UnitMatchSelection.MemberCuration`'s mandatory one-`SessionGroup.Member` model with an explicit ordered `Input` selection plus an `InputRecording` snapshot. Each input pins:

- Its `(sorting_id, curation_id)` and immutable curation generation; one generation per sorting in a run.
- Its base single/concat source, exact correction/mask result, and effective channel geometry.
- Every constituent original recording/member, its content identity, frame spans, original-time mapping, and session metadata needed for chronological ordering and regional provenance.

The selected input list, matcher recipe, and frozen snapshots determine match-run identity. A `SessionGroup` remains a way to discover ordinary single-session candidates; it does not force a daily concatenation into one constituent member. Preserve the simple existing planner call as an adapter where possible; v2 receipt/schema changes are documented and recreated rather than silently misinterpreting old selections.

Reject duplicate inputs, two generations of the same sorting, and overlapping constituent sessions across different inputs before extraction. This includes matching a daily concat against one of its own member sorts. Keep within-one-original-session window matching outside this phase. Verify the same chronic electrode space and compatible effective geometry across the expanded input sets. Do not use geometry alone as proof of physical probe identity.

Order inputs by frozen original acquisition times (with a deterministic tie-break), never by synthetic concat seconds or editable live group membership. Each input preserves all its original spans; the first member may anchor an output file but is not a substitute for the input's complete provenance. Multi-day concatenation matching is deferred; this phase supports same-day assemblies matched across days plus the existing single-session route.

### Bundle extraction and tracking

The backend still consumes self-contained waveform directories and returns curated-unit pairs. It does not need to know whether an input was assembled. Reuse phase 4a's per-unit temporal split on each input's effective recording and parent curation; sample waveforms only where their full support is valid, with no snippets spanning member joins or gaps. Exclusions preserve the frozen unit universe and cannot silently shorten or reorder inputs.

Keep pair/node identities at `(sorting_id, curation_id, unit_id)` and pin the generation in run provenance. Add input IDs and constituent-member provenance to exports/receipts as needed. Update every ownership check, canonicalization path, direct-insert recheck, and graph input resolver; removing only `_curation_member_identity`'s rejection is insufficient.

Retain the both-directions probability rule and the strict clique partition. A parent concat unit appears once in this graph. Its known per-member projections are views of that identity, not independent matching evidence. Preserve the full matchable universe, including singletons and units excluded from bundle extraction.

Distinguish the number of **matching inputs** from original sessions with detected spikes. Expose `n_matching_inputs` and define original-session counts explicitly from frozen per-unit member spike counts; an empty member export does not prove the unit was detected there. Retain or rename `n_sessions_observed` with that documented meaning and update all readers. Count distinct original sessions, not intervals; return per-member regions/electrode provenance rather than copying the concat anchor's region to every member.

### Required end-to-end evidence

- Two independently sorted daily concatenations, each with at least two constituent recordings and wall-clock gaps, with planted corresponding units and distractors; run real UnitMatchPy on extracted bundles.
- A unit firing in only one member of a daily concat; no fake duplicate nodes or false original-session presence, and phase 4a still supplies valid halves when enough spikes exist.
- Mixed single-recording/concat inputs, source changes after selection, incompatible post-correction geometry, shuffled discovery order, and overlap/duplicate rejection.
- A three-day scenario with gradual waveform change, disappearance/reappearance, and conflicting pair evidence. Measure incorrect identities and recall; do not relax the strict tracking rule to improve recall in this phase.
- The combined opt-in path: mask -> assemble -> estimate/apply motion -> sort/curate each day -> match daily inputs -> recover original-member spike times and regions. Include no-motion controls. Synthetic dropout recovery alone is not evidence for hours-long tracking.
