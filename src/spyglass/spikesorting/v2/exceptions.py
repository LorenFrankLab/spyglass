"""v2-specific exception classes.

Every named v2 invariant gets a dedicated exception class so callers can
catch the specific failure mode rather than parsing a bare ``ValueError``
/ ``RuntimeError`` message. Each message names the failed invariant and
the next action the user can take.
"""

from __future__ import annotations


class DuplicateSelectionError(ValueError):
    """Raise when ``insert_selection()`` finds duplicate identity rows.

    ``insert_selection()`` finds more than one row for the same logical
    identity. Message names the table, the logical identity fields, and
    contains "duplicate selection rows".
    """


class DuplicateParameterContentError(ValueError):
    """Raise when a parameter row duplicates existing content under a new name.

    A validated ``PreprocessingParameters`` / ``ArtifactDetectionParameters``
    / ``SorterParameters`` insert carries a ``params`` blob whose content
    fingerprint (name excluded; ``SorterParameters`` scoped per sorter)
    already exists under a different row name. A second name for identical
    content forks provenance -- two ``recording_id`` / ``sorting_id`` families
    for the same science -- so it is rejected by default. Message names the
    incoming name, the existing name, and the shared fingerprint. Pass
    ``allow_duplicate_params=True`` to opt in (the row then shows a
    ``duplicate_of`` in ``describe_parameter_rows()``).
    """


class SchemaBypassError(RuntimeError):
    """Raise when a source-part master row lacks exactly one source part.

    A source-part master row has zero or multiple source part rows at
    populate time. Message names the table, the PK, "expected exactly one
    source", and points the caller at ``insert_selection()``.
    """


class RecordingTruncatedError(RuntimeError):
    """Raise when a saved ``Recording`` does not cover the requested range.

    Saved ``Recording`` timestamp range does not cover the requested
    ``IntervalList.valid_times`` range. Message names the requested
    interval, the saved interval, the missing seconds, and the source
    NWB so the caller can spot upstream truncation without diffing
    timestamps manually.
    """


class EmptyArtifactValidTimesError(RuntimeError):
    """Raise when artifact detection keeps zero seconds of the recording.

    ``_apply_artifact_mask`` received an empty ``valid_times`` array --
    the artifact-detection pass kept zero seconds of the recording.
    Masking would zero the entire recording and the sort would run over
    all-zeros, emitting a misleading "zero units" result. Message names the
    ``artifact_detection_id`` and ``recording_id`` and points the caller at
    re-running ``ArtifactDetection`` with looser thresholds or overriding
    the artifact selection.
    """


class ArtifactFractionExceededError(RuntimeError):
    """Raise when the per-frame artifact set would exceed a sane fraction of
    the recording.

    Both the detection scan (``scan_artifact_frames``: frames flagged above
    threshold) and the sort-time mask (``apply_artifact_mask``: the complement
    of the kept ``valid_times``) materialize one int64 frame index per artifact
    sample. Under a misconfigured (too-loose) threshold, or a ``valid_times``
    that keeps almost nothing, that array is O(n_samples) -- hundreds of MB to
    GB on a long, many-channel recording -- and the subsequent per-frame pass is
    correspondingly slow. Past ``_MAX_ARTIFACT_FRAME_FRACTION`` of the recording
    this is not artifact removal but a misconfiguration: it fails fast here with
    the realized fraction rather than allocating the array (and, downstream, an
    all-near-zero recording the sorter would choke on or that
    ``EmptyArtifactValidTimesError`` would reject three stages later). Message
    names the realized fraction and points the caller at the detection
    thresholds / ``removal_window_ms`` / the ``valid_times`` override.
    """


class InsufficientZScoreChannelsError(ValueError):
    """Raise when z-score artifact detection has too few channels to be useful.

    ``ArtifactDetection``'s z-score detector standardizes ACROSS channels
    within each frame, so it is amplitude-sensitive only with >= 3 channels:
    on 1 channel it is identically zero, and on 2 channels it is a constant
    +/-1 for any two distinct values (independent of amplitude). Raised when
    ``zscore_threshold`` is the only detector on a <3-channel recording (a
    guaranteed no-op or amplitude-blind detector). Message names the channel
    count and points the caller at ``amplitude_threshold_uv`` for 1-2 channel
    groups. When ``amplitude_threshold_uv`` is also set the z-score is merely
    inert (the amplitude detector still fires), so that case warns instead of
    raising.
    """


class NonIntegerUnitIDError(ValueError):
    """Raise when a sorting returns unit IDs that are not integers.

    SpikeInterface sorting returns non-integer unit IDs that cannot
    be written to ``Sorting.Unit.unit_id``. Message names the offending
    IDs and instructs the caller to remap before insertion.
    """


class SessionGroupDateError(ValueError):
    """Raise on a bad recording date in ``SessionGroup.create_group()``.

    ``SessionGroup.create_group()`` receives caller-supplied
    ``recording_date``, or members span multiple dates without
    ``allow_multi_day=True``. Message clarifies that dates are derived
    from ``Session.session_start_time``; multi-day groups list the
    distinct dates and point at sort-then-match (UnitMatch) as the
    recommended cross-day workflow.
    """


class SessionGroupInputError(ValueError):
    """Raise on malformed input to ``SessionGroup.create_group()``.

    Covers a member dict missing a required key (``nwb_file_name`` /
    ``sort_group_id`` / ``interval_list_name``), a member referencing a session
    that is not ingested, an empty / duplicate member set, or a member whose
    ``SortGroupV2`` / ``IntervalList`` / ``LabTeam`` foreign key does not exist
    -- surfaced as a clear, typed error instead of a bare ``KeyError``,
    ``fetch1`` failure, or DataJoint integrity error. A ``ValueError`` subclass
    so existing ``except ValueError`` callers still catch it.
    """


class ConcatBrainRegionAmbiguousError(RuntimeError):
    """Raise on ambiguous brain regions for concat-backed data.

    ``Sorting.get_unit_brain_regions()`` or
    ``CurationV2.get_unit_brain_regions()`` is called on concat-backed
    data without ``allow_anchor_member=True``. Message explains the
    anchor-member ambiguity and points the caller at
    ``allow_anchor_member=True`` for anchor-only regions, or
    ``TrackedUnit.get_unit_brain_regions`` for per-session regions across
    matched sessions.
    """


class MissingRecordingForConcatError(RuntimeError):
    """Raise when a concat member has no populated ``Recording`` row.

    ``ConcatenatedRecordingSelection.insert_selection()`` or
    ``ConcatenatedRecording.make()`` cannot find a populated per-member
    ``Recording`` row with the shared ``preprocessing_params_name``. Message
    lists the missing member keys and instructs the caller to populate
    ``Recording`` for those members first.
    """


class PipelineInputError(ValueError):
    """Raise when ``run_v2_pipeline()`` gets an unknown pipeline preset.

    ``run_v2_pipeline()`` was given an unknown ``pipeline_preset`` name.
    Message lists the available pipeline presets and points at
    ``describe_pipeline_presets()`` / ``list_pipeline_presets()``.
    """


class PreflightError(ValueError):
    """Raise when a preflight check blocks ``run_v2_pipeline``.

    ``run_v2_pipeline(..., preflight=True)`` found a blocking
    configuration problem before any populate. Message lists each failed
    check and the action to fix it. Bypass with ``preflight=False`` to
    attempt the run anyway.
    """


class PipelineStageError(RuntimeError):
    """A run_v2_pipeline stage failed during populate/insert.

    Names the failing stage and carries the partial run summary of stages
    that completed before the failure, so callers can resume/inspect without
    re-deriving intermediate PKs. The original exception is chained
    (``raise PipelineStageError(...) from exc``) so the underlying traceback
    is preserved; its class name is also recorded in ``original_type`` and
    surfaced in this error's message, so the failing stage's ORIGINAL error
    type survives even where only ``str(exc)`` is kept (the run summary's
    ``error`` field) and ``__cause__`` is not inspected.
    """

    def __init__(
        self,
        stage: str,
        partial_run_summary: dict,
        message: str = "",
        *,
        original_type: str | None = None,
    ):
        self.stage = stage
        self.partial_run_summary = partial_run_summary
        # Class name of the wrapped exception (e.g. ``"RuntimeError"``);
        # ``None`` when constructed directly without a cause.
        self.original_type = original_type
        super().__init__(
            f"run_v2_pipeline: stage {stage!r} failed"
            + (f" ({original_type})" if original_type else "")
            + (f": {message}" if message else "")
            + f". Completed stages: {sorted(partial_run_summary)}."
        )


class ZeroUnitSortError(RuntimeError):
    """Raise when a sort finds zero units and ``require_units=True``.

    A sort produced zero units and the caller opted into treating
    that as an error (``run_v2_pipeline(..., require_units=True)``). Zero
    units is a legitimate result on a quiet shank, so it is graceful by
    default: ``run_v2_pipeline`` writes an EMPTY (but real) curation +
    merge row and returns a full run summary with real ``curation_id`` /
    ``merge_id`` and ``n_units=0`` (plus a warning). This is raised only
    when the caller requires units. Message names the recording/sort and
    suggests checking ``detect_threshold`` / the artifact mask.
    """


class ZeroUnitAnalyzerError(RuntimeError):
    """Raise when ``get_analyzer()`` is called on a zero-unit sort.

    ``Sorting.get_analyzer()`` was called on a sort with zero units.
    SpikeInterface cannot build a ``SortingAnalyzer`` over zero units
    (``estimate_sparsity`` -> ``np.concatenate([])``), so no analyzer
    folder was ever written. Raised instead of loading a phantom folder.
    Message names the sort and points the caller at the zero-unit
    result. Use ``get_sorting()`` (returns an empty sorting) if only the
    unit list is needed.
    """


class FigPackUploadError(RuntimeError):
    """Raise when a FigPack hosted upload is requested without credentials.

    ``FigPackCuration`` was asked to publish a hosted figpack.org figure
    (``upload=True``) but ``FIGPACK_API_KEY`` is unset and the figure is not
    ``ephemeral`` -- OR a curation that already carries labels/merges was asked
    to upload, which does not seed that initial state into the hosted figure.
    Message points the caller at setting the API key, using ``ephemeral=True``
    for a temporary figure, or ``upload=False`` to save a seeded local bundle.
    """


class FigPackCurationNamespaceError(RuntimeError):
    """Raise when a FigPack view is requested for a non-raw-namespace curation.

    The FigPack view is built over the sort's display analyzer (the RAW
    ``Sorting.Unit`` namespace). A merged curation -- or a label-only child of a
    merged curation -- lives in a different unit namespace, so rendering the raw
    analyzer would show the wrong units. Raised by
    ``FigPackCurationSelection.insert_selection``; curate the root curation
    instead.
    """


class FigPackDisplayedUnitPropertyError(ValueError):
    """Raise when requested FigPack unit-table columns cannot be rendered.

    ``FigPackCuration`` passes ``displayed_unit_properties`` to SpikeInterface's
    FigPack summary view. SpikeInterface warns and silently drops unavailable
    properties; Spyglass treats an explicit request as a contract and raises
    this error instead. ``None`` still delegates to SpikeInterface defaults.
    """


class FigPackRetrievalError(RuntimeError):
    """Raise when a figure's ``annotations.json`` cannot be retrieved/parsed.

    ``FigPackCuration.fetch_curation_from_uri`` fails CLOSED: only a genuine
    404 / missing local file means "pristine, no edits" (``({}, [])``). An
    unreachable host, refused connection, non-404 HTTP error, or malformed JSON
    raises this instead of silently looking like no edits (which could commit an
    empty child curation over a real one).
    """


class AnalyzerFolderMissingError(RuntimeError):
    """Raise when a no-rebuild analyzer load finds the folder absent.

    ``Sorting.get_analyzer(..., rebuild=False)`` was asked to load a
    units-bearing sort's analyzer folder, but the folder is gone on disk
    (regeneratable scratch removed out of band, or reclaimed). The default
    ``rebuild=True`` path self-heals by rebuilding it; the no-rebuild path exists
    for the recompute AUDIT, which must OBSERVE the missing state rather than
    silently rebuild-then-hash (which would inventory a reconstructed artifact as
    if it were present and let the deletion gate compare a rebuild to itself).
    Message names the sort and the recipe. Distinct from
    ``ZeroUnitAnalyzerError``: that sort legitimately has no analyzer (zero
    units), whereas this one should have one but the folder is absent.
    """


class AnalyzerFolderInvalidError(AnalyzerFolderMissingError):
    """Raise when an existing analyzer folder cannot be loaded.

    ``Sorting.get_analyzer(..., rebuild=False)`` found a directory at the
    computed cache path, but SpikeInterface could not load it as a valid
    ``SortingAnalyzer``. This usually means a killed build, partial cleanup, or
    out-of-band corruption left a half-written zarr store. It subclasses
    ``AnalyzerFolderMissingError`` so no-rebuild audit paths can treat invalid
    regeneratable scratch the same as absent scratch, while preserving a more
    precise diagnostic for callers and logs.
    """


class UnsupportedDirectInsertError(RuntimeError):
    """Raise on a direct insert into a structured Lookup with part rows.

    A structured Lookup whose validity depends on ordered part rows
    (e.g. ``AutoCurationRules`` + ``AutoCurationRules.Rule``) was inserted
    through an unsupported direct master or part insert path that bypasses
    whole-payload validation. Message names the table, the unsupported
    insert path, and the supported helper (``insert_rules(row, rule_rows)``)
    that validates the master row and its rule rows together.
    """


class StaleEnvMatchedError(RuntimeError):
    """Raise when recompute deletion has no current-environment match.

    A recompute ``delete_files()`` gate found ``matched=1`` rows only in
    non-current ``UserEnvironment`` rows (e.g. a verification that
    succeeded under an older SpikeInterface pin). A stale-env match is not
    evidence that the *current* environment can regenerate the artifact, so
    deletion is refused. Message names the current ``env_id``, the stale
    ``env_id``(s) that did match, and instructs the caller to rerun the
    recompute under the current environment or pass ``force_stale_env=True``
    with an audit justification.
    """


class UnknownMatcherError(ValueError):
    """Raise when a matcher-parameters row names an unregistered matcher.

    ``MatcherParameters.insert1()`` (and any registry lookup) received a
    ``matcher`` string that is not in the cross-session matcher registry.
    An unknown matcher would otherwise sit in the database until
    ``UnitMatch.populate()`` failed hours later, so it is rejected at insert
    time. Message names the unknown matcher, the registered matcher names,
    and points the caller at ``register_matcher()``.
    """


class SameSessionMatchError(ValueError):
    """Raise when a UnitMatch selection has two members from one recording session.

    Cross-session matching tracks a unit ACROSS recording sessions, so every
    member must come from a distinct session (``nwb_file_name``). A
    ``SessionGroup`` may legitimately carry several members from one NWB for
    *concatenation* (different intervals / sort groups), but feeding such a
    group to ``UnitMatchSelection.insert_selection`` would match within a single
    session and report it as a multi-session identity. Rejected at selection
    time; message names the offending ``member_index`` / ``nwb_file_name`` pairs.
    """


class UnitMatchSelectionIntegrityError(RuntimeError):
    """Raise when pinned member curations do not match the session group.

    ``UnitMatch.make()`` re-validates that the
    ``UnitMatchSelection.MemberCuration`` part rows exactly cover the parent
    ``SessionGroup.Member`` set and that each pinned ``CurationV2`` belongs to
    that member's session/recording path. A schema-valid but
    provenance-invalid selection (a direct insert that bypassed
    ``insert_selection``) raises this rather than silently matching the wrong
    units. Message names the missing/extra/wrong ``member_index`` values and
    points the caller at ``UnitMatchSelection.insert_selection()``.
    """


class SharedArtifactGroupMemberDriftError(RuntimeError):
    """Raise when a shared artifact group's member set changed under a fixed id.

    The shared-group ``artifact_detection_id`` identity is ``{params,
    group_name}`` only, but ``ArtifactDetection.make`` scans the LIVE
    ``SharedArtifactGroup.Member`` set. ``insert_selection`` snapshots the ordered
    member ``recording_id`` set as ``SharedGroupSource.member_set_hash``;
    ``make_fetch`` re-derives it from the current members and raises this when
    they disagree -- a member was added or removed after the selection was
    created, so the scanned set no longer matches what the id was minted for.
    The ``artifact_detection_id`` identity is params + group name only (the
    member set is a snapshot, not identity), so ``insert_selection`` returns this
    same id with its stale snapshot; recovery is to DELETE this selection and
    re-run ``insert_selection`` (which re-snapshots the current members), or to
    restore the group's members -- not to silently scan a different set.
    """


class UnitMatchPairIntegrityError(RuntimeError):
    """Raise when a ``UnitMatch.Pair`` row is outside the pinned curation universe.

    ``Pair`` FKs each endpoint to ``CurationV2.Unit`` GLOBALLY, so the schema
    only guarantees the unit exists in SOME curation -- not in the pinned
    ``UnitMatchSelection.MemberCuration`` for this run. The canonical
    ``make_insert`` path is safe because ``canonicalize_match_pairs`` orients and
    dedupes within the pinned, matchable set, but a raw / maintenance
    ``Pair.insert`` bypasses that. The validated ``Pair.insert`` override raises
    this when an endpoint's ``(sorting_id, curation_id)`` is not a pinned member
    curation, when both endpoints share one member (a unit cannot match itself
    across sessions), when a reversed/duplicate undirected edge is re-inserted,
    or when ``match_probability`` is outside ``[0, 1]``.
    """


class TrackedUnitBudgetExceededError(RuntimeError):
    """Raise when the strict tracked-unit graph exceeds its node budget.

    ``TrackedUnit.make()`` seeds a graph from the full curated-unit universe
    and derives maximal cliques in strict mode. If the graph exceeds
    ``MatcherParameters.params["max_strict_nodes"]`` the clique search is
    refused before exponential blow-up. Message names the node count, the
    configured cap, and instructs the caller to shrink the session group or
    raise the cap intentionally.
    """


class ConcatMemberDriftError(RuntimeError):
    """Raise when a frozen concat member's recording content no longer matches.

    ``ConcatenatedRecordingSelection.insert_selection`` freezes each member's
    ordered logical identity AND its ``Recording.content_hash`` into a
    ``MemberSnapshot`` part, and folds the ordered LOGICAL member set into
    ``concat_recording_id`` (so a different member set is a different concat).
    ``ConcatenatedRecording.make_fetch`` (materialize / rebuild) reads the frozen
    snapshot -- never the live ``SessionGroup.Member`` set -- and re-checks each
    member's current ``Recording`` against it. This is raised specifically when a
    frozen member's ``Recording`` still exists but its current ``content_hash``
    DIVERGES from the frozen one: the concatenation would be built from different
    underlying data than the ``concat_recording_id`` was minted for. (A frozen
    member whose ``Recording`` row is GONE raises
    ``MissingRecordingForConcatError`` instead -- absence vs. divergence are
    distinct.) Editing ``SessionGroup.Member`` does NOT raise this (old concat
    rows read their frozen snapshot); only the underlying member recording
    content changing does. Message names the ``member_index`` / ``recording_id``
    and the stored-vs-current ``content_hash``, and points the caller at
    re-running ``insert_selection`` (which re-snapshots and mints a new
    ``concat_recording_id`` for the changed inputs) or restoring the member
    recording.
    """


class ConcatSplitError(RuntimeError):
    """Raise when splitting a concat sorting back to members would drop spikes.

    ``ConcatenatedRecording.split_sorting_by_session`` (and the pure
    ``split_unit_spike_trains``) back-map a concat-frame sorting into each
    member's local sample frame using the per-member boundaries. The boundaries
    must be one strictly-increasing value per member partitioning ``[0, total)``,
    and every input spike frame must land in exactly one member. This is raised
    when the boundaries are not strictly increasing, do not cover the recording
    (a final boundary short of the concat sample count), or when an input spike
    frame falls outside ``[0, final_boundary)`` -- which the previous silent slice
    dropped. Message names the offending boundary / spike count so the caller can
    see the conservation violation rather than getting a silently truncated split.
    """


class RecordingContentDriftError(RuntimeError):
    """Raise when a recording rebuild diverges from its stored ``content_hash``.

    ``Recording._rebuild_nwb_artifact`` regenerates the preprocessed artifact to
    a temp file and fingerprints it; if that fingerprint does not match the
    ``Recording`` row's stored ``content_hash`` the rebuild is refused -- the
    canonical slot is never written and the drifted bytes are never served. A
    mismatch means the current environment no longer reproduces the recording
    the row identifies (e.g. a SpikeInterface/BLAS upgrade, an edited raw NWB, or
    changed upstream construction inputs). Message names the
    ``analysis_file_name`` and lists the recovery options: restore a backup of
    the artifact, rerun the recompute under the original environment, or delete
    and repopulate the ``Recording`` row (and its downstream).
    """


class RecordingInputDriftError(RuntimeError):
    """Raise when a recording's resolved construction inputs drifted post-select.

    ``recording_id`` is content-addressed over the FK set PLUS
    ``recording_input_hash`` -- the sort group's electrode membership, reference,
    and (on the ``interpolate`` path) the resolved interior bad-channel set.
    ``insert_selection`` snapshots that hash onto ``RecordingSelection``;
    ``make_fetch`` re-derives it from the LIVE inputs and raises this when they
    disagree -- an electrode was added/removed from the sort group, the reference
    changed, or a channel was (un)flagged bad AFTER the selection was created, so
    the recording ``make`` would build content that no longer matches the id it
    was minted for. Because the inputs are IDENTITY (folded into ``recording_id``,
    not merely snapshotted), recovery is to re-run ``insert_selection`` for the
    current inputs -- which mints a NEW ``recording_id`` for the changed set --
    or to restore the sort group's membership / bad-channel flags, not to
    silently build a drifted recording under the old id.
    """


class MissingDisplayExtensionError(RuntimeError):
    """A richer SI widget needs display-safe analyzer extensions not present.

    Raised by the read-only default visualization path so a notebook user gets
    an actionable message (the exact ``add_extensions`` call, or the
    ``compute_missing=True`` opt-in) instead of a deep SpikeInterface
    ``check_extensions`` failure. The absent extensions are also exposed on the
    ``missing`` attribute so a caller can act on them programmatically without
    parsing the message string.
    """

    def __init__(self, message: str, *, missing=None):
        self.missing = list(missing or [])
        super().__init__(message)
