"""v2-specific exception classes.

Each named v2 invariant has its own exception class so callers can catch a
specific failure instead of parsing a ``ValueError`` / ``RuntimeError``
message. Each message names the failed invariant and the next action.
"""

from __future__ import annotations

# Raised by the shared table guards v2 uses; importable from here.
from spyglass.utils.mixins.immutable import (  # noqa: F401
    ImmutableRowError,
    UnsupportedDirectInsertError,
)


class SpikeSortingV2Error(Exception):
    """Base class of every exception v2 defines; catch it for a v2 failure.

    The shared table-guard errors ``ImmutableRowError`` and
    ``UnsupportedDirectInsertError`` are ``DataJointError`` subclasses, not
    subclasses of this.
    """


class DuplicateSelectionError(SpikeSortingV2Error, ValueError):
    """Raise when ``insert_selection()`` finds duplicate identity rows.

    Message names the table and the logical identity fields, and contains
    "duplicate selection rows".
    """


class DuplicateParameterContentError(SpikeSortingV2Error, ValueError):
    """Raise when a parameter row duplicates existing content under a new name.

    Applies to ``PreprocessingParameters``, ``ArtifactDetectionParameters`` and
    ``SorterParameters`` (scoped per sorter). A second name for identical
    content would split provenance into two ``recording_id`` / ``sorting_id``
    families. Message names the incoming name, the existing name, and the
    shared fingerprint. Pass ``allow_duplicate_params=True`` to opt in; the
    row then shows ``duplicate_of`` in ``describe_parameter_rows()``.
    """


class SchemaBypassError(SpikeSortingV2Error, RuntimeError):
    """Raise when a stored row was written or edited outside its create path.

    For example a selection row that names zero or two sources, or whose
    stored source does not give its deterministic id. Message names the
    table and the PK and points the caller at ``insert_selection()``.
    """


class RecordingTruncatedError(SpikeSortingV2Error, RuntimeError):
    """Raise when a saved ``Recording`` does not cover the requested range.

    The saved timestamps do not cover ``IntervalList.valid_times``. Message
    names the requested interval, the saved interval, the missing seconds,
    and the source NWB.
    """


class EmptyArtifactValidTimesError(SpikeSortingV2Error, RuntimeError):
    """Raise when artifact detection keeps zero seconds of the recording.

    ``apply_artifact_mask`` received empty ``valid_times``; masking would
    zero the whole recording and yield a misleading zero-unit sort. Message
    names the ``artifact_detection_id`` and ``recording_id`` and suggests
    looser thresholds or overriding the artifact selection.
    """


class ArtifactFractionExceededError(SpikeSortingV2Error, RuntimeError):
    """Raise when artifact frames exceed ``_MAX_ARTIFACT_FRAME_FRACTION``.

    ``scan_artifact_frames`` and ``apply_artifact_mask`` allocate one int64
    index per artifact frame, so a too-loose threshold or near-empty
    ``valid_times`` is a misconfiguration that would cost GBs. Message names
    the realized fraction and points at the detection thresholds,
    ``removal_window_ms`` and the ``valid_times`` override.
    """


class IneffectiveManualExclusionError(SpikeSortingV2Error, ValueError):
    """Raise when a manual exclusion overlaps none of the time a run sorts.

    Exclusions are absolute session timestamps, so seconds counted from the
    recording start, or an interval inside an acquisition gap, would mask
    nothing. Message names the intervals and the sorted time range.
    """


class UnmappedPreprocessingWaveformsError(SpikeSortingV2Error, ValueError):
    """Raise when a preprocessing row has no analyzer waveform windows.

    v2 picks the display and metric waveform windows by preprocessing row; a
    custom row needs a ``PreprocessingWaveformParameters`` row. Message
    names the row to insert and the shipped window pairs.
    """


class InsufficientZScoreChannelsError(SpikeSortingV2Error, ValueError):
    """Raise when ``zscore_threshold`` is the only detector on <3 channels.

    The z-score is computed across channels per frame: identically zero on
    1 channel and a constant +/-1 on 2. Message names the channel count and
    points at ``amplitude_threshold_uv``. If ``amplitude_threshold_uv`` is
    also set, a warning is issued instead.
    """


class NonIntegerUnitIDError(SpikeSortingV2Error, ValueError):
    """Raise when a sorting returns unit IDs that are not integers.

    ``Sorting.Unit.unit_id`` requires integers. Message names the offending
    IDs and asks the caller to remap before insertion.
    """


class SessionGroupDateError(SpikeSortingV2Error, ValueError):
    """Raise on a bad recording date in ``SessionGroup.create_group()``.

    Raised for a caller-supplied ``recording_date`` or for members spanning
    several dates without ``allow_multi_day=True``. Dates come from
    ``Session.session_start_time``. For multi-day groups the message lists
    the dates and recommends sort-then-match (UnitMatch).
    """


class SessionGroupInputError(SpikeSortingV2Error, ValueError):
    """Raise on malformed input to ``SessionGroup.create_group()``.

    Covers a member missing ``nwb_file_name`` / ``sort_group_id`` /
    ``interval_list_name``, a session that is not ingested, an empty or
    duplicate member set, and a missing ``SortGroupV2`` / ``IntervalList`` /
    ``LabTeam`` row.
    """


class ConcatBrainRegionAmbiguousError(SpikeSortingV2Error, RuntimeError):
    """Raise on ambiguous brain regions for concat-backed data.

    Raised by ``Sorting.get_unit_brain_regions()`` and
    ``CurationV2.get_unit_brain_regions()`` without
    ``allow_anchor_member=True``. Message points at that flag for
    anchor-only regions, or at UnitMatch and
    ``TrackedUnit.get_unit_brain_regions`` for per-member regions.
    """


class MissingRecordingForConcatError(SpikeSortingV2Error, RuntimeError):
    """Raise when a concat member has no populated ``Recording`` row.

    Raised by ``ConcatenatedRecordingSelection.insert_selection()`` and
    ``ConcatenatedRecording.make()`` when a member lacks a ``Recording`` with
    the shared ``preprocessing_params_name``. Message lists the missing
    members.
    """


class PipelineInputError(SpikeSortingV2Error, ValueError):
    """Raise when a pipeline entry point rejects its request before any DB work.

    Raised by ``run_v2_pipeline``, ``run_v2_concat_pipeline``,
    ``run_v2_pipeline_session``, ``estimate_motion`` and
    ``estimate_concat_motion`` for, e.g., an unknown ``pipeline_preset`` (see
    ``describe_pipeline_presets()``), a ``None`` source input, or a
    contradictory motion request. Also raised by the preset API
    (``describe_pipeline_preset``, ``register_pipeline_preset``,
    ``clone_pipeline_preset``) for bad input or a missing parameter row.
    Message names the entry point.
    """


class PreflightError(SpikeSortingV2Error, ValueError):
    """Raise when a preflight check blocks ``run_v2_pipeline``.

    Message lists each failed check and its fix. Pass ``preflight=False`` to
    attempt the run anyway.
    """


class PipelineStageError(SpikeSortingV2Error, RuntimeError):
    """A pipeline stage failed during populate/insert.

    Carries the failing ``stage``, the ``partial_run_summary`` of the stages
    that finished, the entry point (``caller``) and what it ran on
    (``target``, e.g. the session and sort group). The message lists the
    finished stages with their status and says that re-running the same call
    reuses them. The original exception is chained as ``__cause__``; its
    class name is stored in ``original_type`` and included in the message.
    """

    def __init__(
        self,
        stage: str,
        partial_run_summary: dict,
        message: str = "",
        *,
        original_type: str | None = None,
        caller: str = "run_v2_pipeline",
        target: str | None = None,
    ):
        self.stage = stage
        self.partial_run_summary = partial_run_summary
        # None when constructed without a cause.
        self.original_type = original_type
        self.caller = caller
        self.target = target
        finished = [
            f"{key.removesuffix('_status')} {value}"
            for key, value in partial_run_summary.items()
            if key.endswith("_status")
        ]
        cause = ": ".join(part for part in (original_type, message) if part)
        super().__init__(
            f"{caller} failed at stage {stage!r}"
            + (f" for {target}" if target else "")
            + (f" ({cause})" if cause else "")
            + ". "
            + (
                f"Stages finished before it: {', '.join(finished)}. "
                if finished
                else "No stage finished before it. "
            )
            + "Fix the cause, then re-run the same call: finished stages are "
            "reused, not recomputed. error.partial_run_summary holds the ids "
            "recorded so far."
        )


class ZeroUnitSortError(SpikeSortingV2Error, RuntimeError):
    """Raise when a sort finds zero units and ``require_units=True``.

    By default a zero-unit sort is a valid result: ``run_v2_pipeline`` writes
    an empty curation and merge row and warns. Message names the
    recording/sort and suggests checking ``detect_threshold`` and the
    artifact mask.
    """


class ZeroUnitAnalyzerError(SpikeSortingV2Error, RuntimeError):
    """Raise when ``get_analyzer()`` is called on a zero-unit sort.

    SpikeInterface cannot build a ``SortingAnalyzer`` over zero units, so no
    analyzer folder exists. Use ``get_sorting()`` for the (empty) unit list.
    """


class CurationNotFoundError(SpikeSortingV2Error, LookupError):
    """Raise when a curation handle no longer names the same row generation.

    Covers a deleted row and a numeric ``curation_id`` reused for a different
    ``curation_uuid``.
    """


class CurationLockTimeoutError(SpikeSortingV2Error, TimeoutError):
    """Raise when a curation insert cannot take its sort's lock in time.

    ``CurationV2.insert_curation`` serializes the curations of one sort with
    a MySQL named lock; another session held it past
    ``curation.CURATION_LOCK_TIMEOUT_S``.
    """


class UnsupportedMetricValueError(SpikeSortingV2Error, ValueError):
    """Raise when a metric write receives a non-scalar or non-numeric cell.

    Prevents silent NaN conversion that would change auto-curation decisions.
    """


class FigPackUploadError(SpikeSortingV2Error, RuntimeError):
    """Raise when a FigPack hosted upload is requested without credentials.

    Raised for ``upload=True`` with no FigPack API key and not ``ephemeral``.
    Message suggests setting the API key, ``ephemeral=True``, or
    ``upload=False`` for a local bundle.
    """


class FigPackIdentityError(SpikeSortingV2Error, ValueError):
    """Raise when a FigPack figure cannot prove its curation identity.

    Verified imports require a ``FigPackCuration`` row of the parent curation
    with the figure's URI.
    """


class ReviewChangedSincePreviewError(SpikeSortingV2Error, RuntimeError):
    """Raise when a review figure changes after its import preview.

    ``CurationChangeSet.commit`` re-reads the annotations before writing so
    the committed child matches the previewed change set.
    """


class UnresolvedMergeLabelConflictError(SpikeSortingV2Error, ValueError):
    """Raise when merged contributors have incompatible unresolved labels.

    Provide an explicit label tuple for each merged unit listed in the
    message.
    """


class FigPackDisplayedUnitPropertyError(SpikeSortingV2Error, ValueError):
    """Raise when requested FigPack unit-table columns cannot be rendered.

    SpikeInterface silently drops unavailable ``displayed_unit_properties``;
    an explicit request raises this instead. ``None`` uses SpikeInterface
    defaults.
    """


class FigPackRetrievalError(SpikeSortingV2Error, RuntimeError):
    """Raise when a figure's ``annotations.json`` cannot be retrieved/parsed.

    Only a 404 or missing local file means "no edits". Network errors,
    non-404 HTTP errors and malformed JSON raise, so an empty child curation
    is never committed over real edits.
    """


class AnalyzerFolderMissingError(SpikeSortingV2Error, RuntimeError):
    """Raise when a no-rebuild analyzer load finds the folder absent.

    Raised by ``Sorting.get_analyzer(..., rebuild=False)``, which the
    recompute audit uses to observe a missing folder instead of rebuilding
    it. Message names the sort and the recipe. Unlike
    ``ZeroUnitAnalyzerError``, this sort should have an analyzer.
    """


class AnalyzerFolderInvalidError(AnalyzerFolderMissingError):
    """Raise when an existing analyzer folder cannot be loaded.

    Usually a partial folder from a killed build or out-of-band corruption.
    Subclasses ``AnalyzerFolderMissingError`` so audit paths treat it as
    absent.
    """


class StaleEnvMatchedError(SpikeSortingV2Error, RuntimeError):
    """Raise when recompute deletion has no current-environment match.

    ``delete_files()`` found ``matched=1`` only under non-current
    ``UserEnvironment`` rows, which does not show the current environment
    can regenerate the artifact. Message names the current and stale
    ``env_id`` values; rerun the recompute or pass ``force_stale_env=True``.
    """


class UnknownMatcherError(SpikeSortingV2Error, ValueError):
    """Raise when a matcher-parameters row names an unregistered matcher.

    Raised at ``MatcherParameters.insert1()`` and registry lookup so the
    error surfaces before ``UnitMatch.populate()``. Message lists the
    registered matchers and points at ``register_matcher()``.
    """


class OverlappingMatchInputError(SpikeSortingV2Error, ValueError):
    """Matching inputs reuse original raw acquisition samples.

    Disjoint epochs of the same NWB are supported. Shared sample ranges are
    rejected regardless of artifact masks, preprocessing or interval names.
    """


class UnitMatchSelectionIntegrityError(SpikeSortingV2Error, RuntimeError):
    """Raise when a match selection's frozen inputs no longer hold.

    ``UnitMatch.make()`` re-validates ``UnitMatchSelection.Input`` /
    ``InputRecording`` before extraction:

    - parts are well formed and realize ``input_set_hash``
    - at least one input, no sorting pinned twice, no multi-day concat input
    - frozen start times and input numbering agree
    - each pinned ``CurationV2`` exists with its ``curation_uuid`` and has no
      unapplied proposed merges
    - each live source matches its frozen recordings, including concat
      member content hashes
    - each frozen session start time equals the live ``Session`` row

    ``TrackedUnit.get_member_spike_times`` and
    ``TrackedUnit.get_unit_brain_regions`` run the curation and source checks.
    A ``ConcatMemberDriftError`` / ``MissingRecordingForConcatError`` is
    chained as ``__cause__``. Overlapping inputs raise
    ``OverlappingMatchInputError`` instead.
    """


class UnitMatchPairIntegrityError(SpikeSortingV2Error, RuntimeError):
    """Raise when a match part row is outside the match's unit universe.

    ``Pair`` endpoints reference ``CurationV2.Unit`` globally, so the
    ``Pair.insert`` override validates that each endpoint is a pinned input
    curation and a ``MatchableUnit`` of the match, both endpoints differ in
    curation, the undirected edge is new, and ``match_probability`` is in
    ``[0, 1]``. ``MatchableUnit.insert`` refuses a unit outside its input's
    pinned curation, and ``TrackedUnit.Member.insert`` a unit that is not a
    ``MatchableUnit`` of the match.
    """


class TrackedUnitBudgetExceededError(SpikeSortingV2Error, RuntimeError):
    """Raise when the strict tracked-unit graph exceeds its node budget.

    Strict-mode clique search is refused above
    ``MatcherParameters.matcher_params["max_strict_nodes"]``. Message names
    the node count and cap; shrink the session group or raise the cap.
    """


class ConcatMemberDriftError(SpikeSortingV2Error, RuntimeError):
    """Raise when a frozen concat member's recording content no longer matches.

    ``ConcatenatedRecording.make_fetch`` checks each member's current
    ``Recording.content_hash`` against the ``MemberSnapshot`` frozen by
    ``insert_selection``. A missing ``Recording`` raises
    ``MissingRecordingForConcatError`` instead; editing ``SessionGroup.Member``
    does not raise. Message names the ``member_index`` / ``recording_id`` and
    both hashes. Re-run ``insert_selection`` or restore the member recording.
    """


class ConcatSplitError(SpikeSortingV2Error, RuntimeError):
    """Raise when splitting a concat sorting back to members would drop spikes.

    Raised by ``ConcatenatedRecording.split_sorting_by_session`` and
    ``split_unit_spike_trains`` when the member boundaries are not strictly
    increasing, do not reach the concat sample count, or a spike frame falls
    outside ``[0, final_boundary)``. Message names the offending boundary or
    spike count.
    """


class RecordingContentDriftError(SpikeSortingV2Error, RuntimeError):
    """Raise when a recording rebuild diverges from its stored ``content_hash``.

    ``Recording._rebuild_nwb_artifact`` never writes or serves drifted bytes.
    Typical causes: a SpikeInterface/BLAS upgrade, an edited raw NWB, or
    changed upstream inputs. Message names the ``analysis_file_name`` and the
    recovery options.
    """


class RecordingInputDriftError(SpikeSortingV2Error, RuntimeError):
    """Raise when a recording's resolved construction inputs drifted post-select.

    ``recording_id`` includes ``recording_input_hash`` (electrodes, reference,
    on the ``interpolate`` path the interior bad channels, and the saved time
    intervals).
    ``make_fetch`` raises when the live inputs no longer match. Re-run
    ``insert_selection`` to mint a new ``recording_id``, or restore the inputs.
    """


class MissingDisplayExtensionError(SpikeSortingV2Error, RuntimeError):
    """A SpikeInterface widget needs analyzer extensions that are not computed.

    The message gives the ``add_extensions`` call or the
    ``compute_missing=True`` opt-in. The absent extensions are listed in the
    ``missing`` attribute.
    """

    def __init__(self, message: str, *, missing=None):
        self.missing = list(missing or [])
        super().__init__(message)
