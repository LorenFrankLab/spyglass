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


class SingleChannelZScoreError(ValueError):
    """Raise when z-score artifact detection is configured on one channel.

    ``ArtifactDetection``'s z-score detector is computed ACROSS channels
    within each frame, so on a single-channel sort group it is identically
    zero and would silently flag nothing. Raised when ``zscore_threshold`` is
    the only detector on a <2-channel recording (a guaranteed no-op). Message
    names the channel count and points the caller at ``amplitude_threshold_uv``
    for single-channel groups. When ``amplitude_threshold_uv`` is also set the
    z-score is merely inert (the amplitude detector still fires), so that case
    warns instead of raising.
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


class ConcatBrainRegionAmbiguousError(RuntimeError):
    """Raise on ambiguous brain regions for concat-backed data.

    ``Sorting.get_unit_brain_regions()`` or
    ``CurationV2.get_unit_brain_regions()`` is called on concat-backed
    data without ``allow_anchor_member=True``. Message explains the
    anchor-member ambiguity and points the caller at
    ``allow_anchor_member=True`` for anchor-only regions; per-session
    regions require cross-session unit matching, not available in this
    build.
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
    is preserved.
    """

    def __init__(
        self, stage: str, partial_run_summary: dict, message: str = ""
    ):
        self.stage = stage
        self.partial_run_summary = partial_run_summary
        super().__init__(
            f"run_v2_pipeline: stage {stage!r} failed"
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
