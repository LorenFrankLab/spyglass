"""v2-specific exception classes.

Per shared-contracts § Custom Exception Classes, every named v2
invariant gets a dedicated exception class so callers can catch the
specific failure mode rather than parsing a bare ``ValueError`` /
``RuntimeError`` message. Each message names the failed invariant and
the next action the user can take.
"""

from __future__ import annotations


class DuplicateSelectionError(ValueError):
    """``insert_selection()`` finds more than one row for the same logical
    identity. Message names the table, the logical identity fields, and
    contains "duplicate selection rows"."""


class SchemaBypassError(RuntimeError):
    """A source-part master row has zero or multiple source part rows at
    populate time. Message names the table, the PK, "expected exactly one
    source", and points the caller at ``insert_selection()``."""


class RecordingTruncatedError(RuntimeError):
    """Saved ``Recording`` timestamp range does not cover the requested
    ``IntervalList.valid_times`` range. Message names the requested
    interval, the saved interval, the missing seconds, and the source
    NWB so the caller can spot upstream truncation without diffing
    timestamps manually."""


class EmptyArtifactValidTimesError(RuntimeError):
    """``_apply_artifact_mask`` received an empty ``valid_times`` array --
    the artifact pass kept zero seconds of the recording. Masking would
    zero the entire recording and the sort would run over all-zeros,
    emitting a misleading "zero units" result. Message names the
    ``artifact_id`` and ``recording_id`` and points the caller at
    re-running ``ArtifactDetection`` with looser thresholds or overriding
    the artifact selection."""


class NonIntegerUnitIDError(ValueError):
    """SpikeInterface sorting returns non-integer unit IDs that cannot
    be written to ``Sorting.Unit.unit_id``. Message names the offending
    IDs and instructs the caller to remap before insertion."""


class SessionGroupDateError(ValueError):
    """``SessionGroup.create_group()`` receives caller-supplied
    ``recording_date``, or members span multiple dates without
    ``allow_multi_day=True``. Message clarifies that dates are derived
    from ``Session.session_start_time``; multi-day groups list the
    distinct dates and point at sort-then-match (UnitMatch) as the
    recommended cross-day workflow."""


class ConcatBrainRegionAmbiguousError(RuntimeError):
    """``Sorting.get_unit_brain_regions()`` or
    ``CurationV2.get_unit_brain_regions()`` is called on concat-backed
    data without ``allow_anchor_member=True``. Message explains the
    anchor-member ambiguity and points the caller at
    ``allow_anchor_member=True`` for anchor-only regions; per-session
    regions require cross-session unit matching, not available in this
    build."""


class MissingRecordingForConcatError(RuntimeError):
    """``ConcatenatedRecordingSelection.insert_selection()`` or
    ``ConcatenatedRecording.make()`` cannot find a populated per-member
    ``Recording`` row with the shared ``preproc_params_name``. Message
    lists the missing member keys and instructs the caller to populate
    ``Recording`` for those members first."""


class PipelineInputError(ValueError):
    """``run_v2_pipeline()`` receives zero, partial, or mixed input
    modes. Message says exactly one input mode is required and lists
    the required fields for each mode."""


class ZeroUnitSortError(RuntimeError):
    """A sort produced zero units and the caller opted into treating
    that as an error (``run_v2_pipeline(..., require_units=True)``). Zero
    units is a legitimate result on a quiet shank, so it is graceful by
    default: ``run_v2_pipeline`` writes an EMPTY (but real) curation +
    merge row and returns a full manifest with real ``curation_id`` /
    ``merge_id`` and ``n_units=0`` (plus a warning). This is raised only
    when the caller requires units. Message names the recording/sort and
    suggests checking ``detect_threshold`` / the artifact mask."""


class ZeroUnitAnalyzerError(RuntimeError):
    """``Sorting.get_analyzer()`` was called on a sort with zero units.
    SpikeInterface cannot build a ``SortingAnalyzer`` over zero units
    (``estimate_sparsity`` -> ``np.concatenate([])``), so no analyzer
    folder was ever written. Raised instead of loading a phantom folder.
    Message names the sort and points the caller at the zero-unit
    result. Use ``get_sorting()`` (returns an empty sorting) if only the
    unit list is needed."""
