"""Spike sorting and per-unit brain-region metadata.

Tables (all final-shape under the zero-migration policy):
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source (default).
        .ConcatenatedRecordingSource -- concat source; the runtime helper
                                       rejects this source today, but the
                                       FK is real so the schema is stable.
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.

``SorterParameters.insert1`` dispatches to the per-sorter Pydantic
schema via ``_get_sorter_schema``. ``insert_selection`` resolves a
sorting request to a single ``sorting_id``, ``make`` runs the
sorter and writes the units NWB + analyzer, and the accessor methods
(``get_sorting``, ``get_analyzer``) read those back. Concatenated-
recording sources are not yet supported: ``insert_selection`` rejects a
``concat_recording_id`` source and the analyzer rebuild raises
``NotImplementedError`` for a concat source.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import IntervalList  # noqa: F401
from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.analyzer_waveform import (
    ANALYZER_WAVEFORM_SCHEMA_VERSION,
    AnalyzerWaveformParamsSchema,
)
from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
from spyglass.spikesorting.v2._recipe_catalog import (
    _params_schema_version,
    sorter_default_contents,
    waveform_params_default_contents,
    waveform_params_for_preprocessing,
)
from spyglass.spikesorting.v2._sorting_analyzer import (
    build_analyzer,
    fetch_waveform_params,
    load_or_rebuild_analyzer,
    rebuild_analyzer_folder,
)
from spyglass.spikesorting.v2._sorting_artifact_mask import apply_artifact_mask
from spyglass.spikesorting.v2._sorting_dispatch import (
    _clusterless_noise_levels,  # noqa: F401  re-exported for tests
    remove_excess_spikes,
    run_clusterless_thresholder,
    run_si_sorter,
)
from spyglass.spikesorting.v2._sorting_units import (
    _to_int_unit_id,  # noqa: F401  re-exported for tests
    build_sorting_unit_rows,
)
from spyglass.spikesorting.v2._units_nwb import (
    abs_spike_times_dataframe,
    empty_spike_times_dataframe,
    numpysorting_from_abs_times,
    read_units_abs_spike_times,
    recording_timestamps,
    write_sorting_units_nwb,
)
from spyglass.spikesorting.v2.artifact import ArtifactDetection  # noqa: F401
from spyglass.spikesorting.v2.recording import Recording  # noqa: F401
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import (
    SelectionMasterInsertGuard,
    SourceResolution,
    _assert_v2_db_safe,
    _validate_params,
    find_orphaned_masters,
    reject_duplicate_parameter_content,
    split_leading_restrictions,
    transaction_or_noop,
    unit_brain_region_df,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si


class SortingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Sorting.make_fetch`.

    Attributes
    ----------
    source : SourceResolution
        Resolved recording source for the selection (``kind == "recording"``;
        concat sources are rejected upstream).
    sel_row : dict
        The ``SortingSelection`` row, with ``artifact_detection_id`` resolved
        and stashed on it for downstream readers.
    sorter_row : dict
        The matching ``SorterParameters`` row (``sorter``, ``params``,
        ``job_kwargs``).
    nwb_file_name : str
        Source NWB file backing the recording selection.
    obs_intervals : numpy.ndarray or None
        Artifact-removed valid-times window, or ``None`` when no
        artifact-detection pass is configured.
    display_waveform_params_name : str
        The DISPLAY ``AnalyzerWaveformParameters`` recipe resolved from the
        source preprocessing recipe (region), stored on the ``Sorting`` row.
    display_waveform_params : dict
        That recipe's resolved params blob, threaded into ``_build_analyzer``
        so ``make_compute`` does no parameter DB I/O.
    """

    source: SourceResolution
    sel_row: dict
    sorter_row: dict
    nwb_file_name: str
    obs_intervals: np.ndarray | None
    display_waveform_params_name: str
    display_waveform_params: dict


class SortingComputed(NamedTuple):
    """Compute -> insert carrier for :meth:`Sorting.make_compute`.

    NONE of these fields are ``Sorting`` columns -- they are values threaded
    from ``make_compute`` into ``make_insert`` (NWB staging, lookups,
    unit-part inserts). ``analyzer_folder`` is special only in WHY it is
    threaded: it is the EXACT folder ``_build_analyzer`` wrote, so
    ``make_insert`` / ``_populate_unit_part`` load and clean up that folder
    rather than recomputing a path a mid-populate config / path-policy
    change could divert. The analyzer cache folder is deliberately NOT a DB
    column; every code path WITHOUT an in-memory folder
    (``get_analyzer``, ``delete``, ``find_orphaned_analyzer_folders``)
    resolves the canonical location from ``sorting_id`` via
    ``_analyzer_cache.analyzer_path``.

    NOTE: the field ORDER here is a positional wire contract -- the tri-part
    dispatch unpacks this tuple positionally into ``make_insert``
    (``make_insert(key, *make_compute_result)``). Keep it in sync with
    ``make_insert``'s parameter order;
    ``test_sorting_computed_matches_make_insert_signature`` pins the
    alignment.

    Attributes
    ----------
    sorting_obj : spikeinterface.BaseSorting
        The computed sorting (unit ids + spike trains).
    analysis_file_name : str
        Staged-but-unregistered AnalysisNwbfile holding the units.
    units_object_id : str
        NWB object id of the units table inside that file.
    analyzer_folder : pathlib.Path
        Exact on-disk analyzer folder ``_build_analyzer`` wrote; threaded so
        ``make_insert`` loads/cleans that folder rather than recomputing a
        path.
    recording_id : str
        Recording id the sort was run against.
    nwb_file_name : str
        Source NWB file backing the recording selection.
    display_waveform_params_name : str
        The resolved DISPLAY recipe name; stored on the ``Sorting`` row by
        ``make_insert`` so every later rebuild reads it back deterministically.
    """

    sorting_obj: "si.BaseSorting"
    analysis_file_name: str
    units_object_id: str
    analyzer_folder: Path
    recording_id: str
    nwb_file_name: str
    display_waveform_params_name: str


_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_sorting")


@schema
class SorterParameters(SpyglassMixin, dj.Lookup):
    """Per-sorter Pydantic-validated parameter blob.

    The ``params`` blob is validated by the per-sorter schema returned by
    ``_get_sorter_schema(sorter)``. ``insert_default`` ships explicit
    default rows for MS4, MS5, KS4, SC2, TDC2, and
    ``clusterless_thresholder``. Users can insert additional rows for any
    installed SI sorter; the generic ``extra="allow"`` schema is the
    fallback dispatch for non-default sorters, the "try any installed
    sorter" escape hatch.
    """

    definition = """
    sorter: varchar(64)
    sorter_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=0: int
    job_kwargs=null: blob
    """

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Validate and insert a single row via the ``insert`` path."""
        # Delegate to ``insert`` so one validated path serves both.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate every row against its per-sorter schema, then insert.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob, scoped per sorter); see
        ``reject_duplicate_parameter_content``.
        """
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) before it lands, dispatching the Pydantic
        # schema per ``sorter``. Two per-row guards run after that:
        #
        # 1. Sorter-name typo guard. ``_get_sorter_schema`` falls back to
        #    the permissive ``GenericSorterParamsSchema`` for any unknown
        #    sorter (the "try any installed SI sorter" escape hatch), so a
        #    typo like ``"mountainSort4"`` would otherwise validate cleanly
        #    here and fail only much later at ``Sorting.populate`` with an
        #    opaque SI "sorter not registered" error. Reject a name that is
        #    not in ``sis.available_sorters()``, the curated v2 schema set,
        #    nor the Spyglass ``clusterless_thresholder`` path -- the check
        #    ``_get_sorter_schema``'s docstring already delegates here. The
        #    gate is deliberately ``available_sorters()`` (a pure spelling
        #    check), NOT ``insert_default``'s stricter ``installed_sorters()``
        #    availability gate: a correctly-spelled sorter whose binary is
        #    absent on THIS machine may still be staged for a compute node.
        #
        # 2. ``params_schema_version`` backfill. This Lookup is multi-sorter
        #    (MS4/MS5/KS4/clusterless each carry their own schema_version),
        #    so the column default cannot be pinned to any one sorter's
        #    version -- it defaults to the sentinel 0 ("unspecified"). The
        #    validated ``params`` blob already carries the authoritative
        #    ``schema_version``, so backfill the outer column from it when
        #    the caller left it at 0 rather than making them copy the number
        #    by hand. An explicitly-passed NON-zero value is left untouched
        #    for ``_assert_schema_version_matches`` to cross-check, so a real
        #    outer-vs-blob mismatch still raises.
        import spikeinterface.sorters as sis

        from spyglass.spikesorting.v2._params.sorter import (
            _SORTER_SCHEMAS,
            reject_internal_whiten,
        )

        valid_sorters = (
            set(sis.available_sorters())
            | set(_SORTER_SCHEMAS)
            | self._NON_SI_SORTERS
        )

        def _check_sorter_and_backfill_version(row, _schema_cls):
            sorter = row["sorter"]
            if sorter not in valid_sorters:
                raise ValueError(
                    f"SorterParameters.insert: sorter {sorter!r} is not a "
                    "known SpikeInterface sorter or the Spyglass "
                    "'clusterless_thresholder' path -- check the spelling. "
                    f"Curated v2 sorters: {sorted(_SORTER_SCHEMAS)}; or any "
                    "sorter from spikeinterface.sorters.available_sorters()."
                )
            # Internally-whitening sorters that fall through to the permissive
            # generic schema (kilosort2_5 / kilosort3 / ironclust) must not
            # carry a truthy ``whiten`` -- it would double-whiten via the
            # runtime's external float64 whitening. (KS4 self-guards in its
            # typed schema; MS4/MS5 use whiten=True deliberately.)
            reject_internal_whiten(sorter, row["params"])
            if int(row.get("params_schema_version", 0)) == 0:
                row["params_schema_version"] = _params_schema_version(
                    row["params"]
                )

        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda row: _get_sorter_schema(row["sorter"]),
            table_name="SorterParameters",
            per_row_hook=_check_sorter_and_backfill_version,
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="SorterParameters",
            name_attr="sorter_params_name",
            sorter_keyed=True,
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    # The shipped rows are defined in
    # ``_recipe_catalog.sorter_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = sorter_default_contents()

    # Sorter names in ``_DEFAULT_CONTENTS`` that are NOT SpikeInterface
    # registered sorters and so must never be gated on
    # ``sis.installed_sorters()``. ``clusterless_thresholder`` is a
    # Spyglass-internal peak detector built on ``detect_peaks``.
    _NON_SI_SORTERS: frozenset[str] = frozenset({"clusterless_thresholder"})

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        The default-content catalog includes MS4, MS5, KS4, SC2,
        TDC2, and clusterless_thresholder. Rows whose SpikeInterface
        sorter is NOT in ``spikeinterface.sorters.installed_sorters()``
        are skipped (logged at INFO) -- otherwise a user who inserts an
        uninstalled sorter's default row and then populates ``Sorting``
        hits an unhelpful "sorter not registered" error from SI. MS4 and
        KS4 are the common uninstalled cases (their Python wrappers exist
        even when the runtime/binary is absent, so ``available_sorters``
        is too lax). ``installed_sorters`` gates the row INSERT here -- a
        mild check that only decides whether to ship the default params
        row. It is NOT proof a sort will run: the mountainsort4 wrapper
        imports even when its ``ml_ms4alg`` backend is absent/broken, so
        ``installed_sorters`` over-reports runnability.
        ``preflight_v2_pipeline``'s ``sorter_runtime_available`` check is
        the actual runtime gate.

        ``clusterless_thresholder`` is never gated (it is a Spyglass
        peak-detection special case, not an SI registered sorter). The
        full catalog stays in ``_DEFAULT_CONTENTS`` for introspection;
        only the insert is gated. See the per-sorter Pydantic schemas in
        ``spyglass.spikesorting.v2._params.sorter`` for the validated
        field surface.
        """
        insertable, skipped = cls._gated_default_rows()
        for row in skipped:
            logger.info(
                "SorterParameters.insert_default: skipping default "
                f"row {row[1]!r} -- sorter {row[0]!r} is not in "
                "spikeinterface.sorters.installed_sorters() on this "
                "platform."
            )
        cls.insert(insertable, skip_duplicates=True)

    @classmethod
    def _gated_default_rows(cls):
        """Split ``_DEFAULT_CONTENTS`` into (insertable, skipped) by install.

        A default row is insertable when its sorter is a Spyglass-internal
        sorter (``_NON_SI_SORTERS``, never gated) or is in
        ``spikeinterface.sorters.installed_sorters()``. Returns
        ``(insertable, skipped)`` so ``insert_default`` can log the skips
        and tests can assert the gating decision without depending on the
        live ``SorterParameters`` table state.
        """
        import spikeinterface.sorters as sis

        installed = set(sis.installed_sorters())
        insertable: list = []
        skipped: list = []
        for row in cls._DEFAULT_CONTENTS:
            sorter = row[0]
            if sorter in cls._NON_SI_SORTERS or sorter in installed:
                insertable.append(row)
            else:
                skipped.append(row)
        return insertable, skipped

    @classmethod
    def insert_default_legacy_si_sorters(cls):
        """Insert ('sorter','default') rows for installed non-curated sorters.

        Opt-in back-compat helper for users porting v1 workflows that name
        a non-curated sorter via ``('kilosort2_5','default')`` or similar.
        For each entry of ``sis.available_sorters()`` it calls
        SpikeInterface's ``sis.get_default_sorter_params(sorter)`` and
        validates the result through ``GenericSorterParamsSchema``
        (``extra='allow'``) so the row passes without typo-rejection.

        Two classes of sorter are skipped (logged at INFO):

        - **Not installed.** Gated on
          ``spikeinterface.sorters.installed_sorters()`` -- the SAME gate
          ``insert_default`` uses (see the install-gate rationale at
          :meth:`insert_default`). ``get_default_sorter_params`` succeeds
          for wrapper-only sorters whose binary is absent (e.g.
          ``kilosort2_5``, ``ironclust``), so enumerating
          ``available_sorters()`` alone would ship rows that fail at
          ``Sorting.populate`` time with an unhelpful "sorter not
          installed" error. Inserting only *installed* sorters keeps the
          back-compat value (rows a user can actually run) without that
          trap.
        - **Curated** (mountainsort4, mountainsort5, kilosort4,
          spykingcircus2, tridesclous2, clusterless_thresholder) -- they
          already have their own typed schemas with ``extra='forbid'``,
          and SI's defaults for those sorters include keys those schemas
          intentionally strip (e.g. ``MountainSort5Schema`` strips
          ``filter`` / ``freq_min`` because the upstream recording is
          already filtered). Routing SI's full default dict through the
          typed schema would either fail validation or quietly drop keys --
          neither is what a v1 caller expects. The opt-in targets the
          NON-curated escape-hatch sorters that fall back to
          ``GenericSorterParamsSchema`` anyway.

        This helper is deliberately NOT called by ``initialize_v2_defaults``
        -- users who do not need v1 sorter names should not pay for the
        inserts. Idempotent via ``skip_duplicates=True``.

        Examples
        --------
        >>> from spyglass.spikesorting.v2.sorting import SorterParameters
        >>> SorterParameters.insert_default()
        >>> SorterParameters.insert_default_legacy_si_sorters()
        """
        import spikeinterface.sorters as sis

        from spyglass.spikesorting.v2._params.sorter import (
            GenericSorterParamsSchema,
            _SORTER_SCHEMAS,
        )

        curated = set(_SORTER_SCHEMAS)  # sorters with their own schemas
        installed = set(sis.installed_sorters())
        rows = []
        skipped_not_installed = []
        for sorter in sis.available_sorters():
            if sorter in curated:
                continue  # see docstring: would fail or drop keys
            if sorter not in installed:
                # Gate on installed_sorters() to match insert_default's installed-sorters
                # gate -- get_default_sorter_params succeeds for
                # wrapper-only sorters, so an available-but-not-installed
                # row would only fail later at populate time.
                skipped_not_installed.append(sorter)
                continue
            try:
                params = sis.get_default_sorter_params(sorter)
            except Exception as exc:  # SI may raise on metadata fetch
                logger.warning(
                    "insert_default_legacy_si_sorters: skipping "
                    f"{sorter!r} ({exc!r})."
                )
                continue
            # Validate through the generic schema (extra='allow') so the
            # row passes the insert gate without typo-rejection.
            try:
                validated = _validate_params(GenericSorterParamsSchema, params)
            except Exception as exc:
                logger.warning(
                    f"insert_default_legacy_si_sorters: {sorter!r} did "
                    "not validate against GenericSorterParamsSchema "
                    f"({exc!r})."
                )
                continue
            rows.append(
                (
                    sorter,
                    "default",
                    validated,
                    _params_schema_version(validated),
                    None,
                )
            )
        if skipped_not_installed:
            logger.info(
                "insert_default_legacy_si_sorters: skipping "
                f"{sorted(skipped_not_installed)} -- available in "
                "SpikeInterface but not in installed_sorters() on this "
                "platform (a 'default' row would fail at populate time)."
            )
        cls.insert(rows, skip_duplicates=True)


_WAVEFORM_PARAMS_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _reject_unsafe_waveform_params_name(row, _schema_cls) -> None:
    """Reject a ``waveform_params_name`` that is not path-safe.

    The name is embedded in the analyzer cache folder
    ``{sorting_id}__{waveform_params_name}.zarr`` (see ``analyzer_path``), so it
    must match ``^[A-Za-z0-9_]+$`` -- no path separators, dots, or traversal.
    A ``per_row_hook`` for ``validate_lookup_rows``.
    """
    name = row["waveform_params_name"]
    if not _WAVEFORM_PARAMS_NAME_RE.match(name):
        raise ValueError(
            "AnalyzerWaveformParameters: waveform_params_name "
            f"{name!r} is not path-safe; it is embedded in the analyzer "
            "cache folder name, so it must match ^[A-Za-z0-9_]+$ (letters, "
            "digits, underscore)."
        )


@schema
class AnalyzerWaveformParameters(SpyglassMixin, dj.Lookup):
    """Tracked window / subsample / whitening for a sort's analyzer recipe.

    Mirrors v1's ``WaveformParameters`` so the settings that produced an
    analyzer (``ms_before`` / ``ms_after`` / ``max_spikes_per_unit`` / whitening)
    are recorded in the DB rather than hardcoded in the analyzer build. The
    ``params`` blob is validated by :class:`AnalyzerWaveformParamsSchema`;
    ``insert_default`` ships the region-specific display/metric rows
    (hippocampus 0.5/0.5 ms, cortex 1.0/2.0 ms; both 20000 spikes).

    A sort's DISPLAY recipe is resolved from its source preprocessing recipe
    (region) and persisted on ``Sorting.display_waveform_params_name`` -- the
    row name is not a free per-sort knob and is not part of ``sorting_id``
    identity. ``analyzer_path`` embeds ``waveform_params_name`` in the cache
    folder name, so the name is validated path-safe (``^[A-Za-z0-9_]+$``) at
    insert time.
    """

    definition = f"""
    waveform_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={ANALYZER_WAVEFORM_SCHEMA_VERSION}: int
    """

    # The shipped region rows are defined in
    # ``_recipe_catalog.waveform_params_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = waveform_params_default_contents()

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one row through the validated bulk ``insert`` path."""
        # Delegate to ``insert`` so one validated path serves both.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate each ``params`` blob + path-safe name, then insert.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: AnalyzerWaveformParamsSchema,
            table_name="AnalyzerWaveformParameters",
            per_row_hook=_reject_unsafe_waveform_params_name,
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="AnalyzerWaveformParameters",
            name_attr="waveform_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the region display/metric waveform recipes (idempotent)."""
        cls().insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SortingSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` or ``ConcatenatedRecordingSource`` exists for
    each selection row. The runtime helper today rejects the concat
    path with a clear "not implemented yet" error; the schema is final
    so the validator can be relaxed without a migration once the concat
    materializer lands.

    Whether an artifact-detection pass was applied is recorded by the
    presence or absence of an ``ArtifactDetectionSource`` part row
    (zero-or-one), NOT by a nullable FK on the master. A nullable FK
    conflates "no artifact-detection pass" with "match anything" in a
    restriction and forces every reader to special-case ``None``; the part
    row makes "no ``ArtifactDetectionSource`` row" queryable, joinable, and
    impossible to alias. ``ArtifactDetectionSource`` is independent of the
    recording-source parts -- it is NOT counted by ``resolve_source``
    (a sort still has exactly one *recording* source) nor by
    ``prune_orphaned_selections``.
    """

    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    """

    class RecordingSource(SpyglassMixinPart):
        """Single-session recording source for a sorting selection."""

        definition = """
        -> master
        ---
        -> Recording
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        """Concatenated-recording source for a sorting selection."""

        definition = """
        -> master
        ---
        -> ConcatenatedRecording
        """

    class ArtifactDetectionSource(SpyglassMixinPart):
        """Optional artifact-detection pass for a sorting selection.

        Present iff an artifact detection was configured for the sort;
        absent means "no artifact-detection pass." Deliberately separate from the
        recording-source parts so ``resolve_source``'s "exactly one
        recording source" invariant is unaffected -- read it through
        :meth:`SortingSelection.resolve_artifact_detection`.
        """

        definition = """
        -> master
        ---
        -> ArtifactDetection
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Reads exactly one of ``recording_id`` (single-session) or
        ``concat_recording_id`` (concat) from ``key``; raises ValueError
        on zero or two sources. The concat path is rejected today with
        ``NotImplementedError`` because the concat materializer is gated.
        ``artifact_detection_id`` is optional: when supplied (non-None), an
        ``ArtifactDetectionSource`` part row records the artifact-detection
        pass; when omitted/None, no ``ArtifactDetectionSource`` row is created.
        The find-existing path keys on the presence/absence and identity
        of that part row, so an artifact-backed and an artifact-free
        selection for the same ``(recording_id, sorter,
        sorter_params_name)`` are distinct, idempotent rows.

        Parameters
        ----------
        key : dict
            Selection request. Must carry exactly one of ``recording_id`` or
            ``concat_recording_id``, plus ``sorter`` and
            ``sorter_params_name``. ``artifact_detection_id`` is optional;
            an explicit ``sorting_id`` is cross-checked against the derived
            deterministic id.

        Returns
        -------
        dict
            Primary-key-only dict (``{"sorting_id": ...}``) for the
            inserted-or-existing master row.

        Raises
        ------
        ValueError
            If zero or both source keys are supplied.
        NotImplementedError
            If ``concat_recording_id`` is supplied -- concat-source
            sorting is not yet implemented.
        DuplicateSelectionError
            If any matching master has a non-deterministic ``sorting_id``
            (a raw ``insert`` bypass or a pre-determinism legacy row) --
            even a single one; an integrity bug, not user error.
        SchemaBypassError
            If a deterministic master exists but its recording/artifact-detection
            source parts are missing/mismatched (a raw-insert orphan).
        """
        from spyglass.spikesorting.v2._selection_plan import (
            build_sorting_selection_plan,
        )

        # Pure half: validate inputs, normalize the artifact id, derive the
        # deterministic sorting_id, and shape the master + source part rows.
        plan = build_sorting_selection_plan(key)

        existing = cls._find_existing_pk(
            plan.master_restriction,
            plan.source_restriction,
            plan.artifact_detection_id,
            plan.sorting_id,
        )
        if existing is not None:
            return existing

        # Translate the would-be DataJoint FK IntegrityError into a
        # clear "missing default row" message before the inserts attempt.
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError
        from spyglass.spikesorting.v2.utils import (
            _ensure_lookup_row_exists,
            _is_duplicate_key_error,
        )

        _ensure_lookup_row_exists(
            SorterParameters,
            plan.master_restriction,
            helper_name="SortingSelection.insert_selection",
            insert_default_path="SorterParameters.insert_default()",
        )
        cls._validate_artifact_detection_source_for_recording(
            recording_id=plan.source_restriction["recording_id"],
            artifact_detection_id=plan.artifact_detection_id,
        )

        try:
            with transaction_or_noop(cls.connection):
                # allow_direct_insert: this helper IS the validation boundary.
                cls.insert1(plan.master_row, allow_direct_insert=True)
                cls.RecordingSource.insert1(plan.recording_source_row)
                if plan.artifact_source_row is not None:
                    cls.ArtifactDetectionSource.insert1(
                        plan.artifact_source_row
                    )
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK
            if not _is_duplicate_key_error(exc):
                raise
            # Lost a concurrent race on the same deterministic sorting_id;
            # refetch and return the winner's master+source row.
            # (Top-level recovery only -- see transaction_or_noop.)
            logger.debug(
                "SortingSelection.insert_selection: lost deterministic-id "
                "race on %s; returning the existing row.",
                plan.sorting_id,
            )
            existing = cls._find_existing_pk(
                plan.master_restriction,
                plan.source_restriction,
                plan.artifact_detection_id,
                plan.sorting_id,
            )
            if existing is not None:
                return existing
            # The deterministic master exists (that is the duplicate key)
            # but its recording/artifact-detection source parts are missing or
            # do not match the requested selection -- a raw-insert orphan. Surface
            # a clear schema-bypass error instead of the opaque duplicate.
            raise SchemaBypassError(
                f"SortingSelection master {plan.sorting_id} exists but its "
                "recording/artifact-detection source parts are missing or do not match "
                "the requested selection: the master was inserted without "
                "insert_selection (raw-insert orphan). Use insert_selection() "
                "to create master+source atomically, or drop the orphan "
                "master."
            ) from exc
        return {k: plan.master_row[k] for k in cls.primary_key}

    @classmethod
    def _find_existing_pk(
        cls,
        master_restriction,
        source_restriction,
        artifact_detection_id,
        deterministic_id,
    ) -> dict | None:
        """Return the canonical master PK for this sort selection, or None.

        Matches a master with the same sorter + recording source AND the
        same artifact-detection-source state
        (present-with-this-``artifact_detection_id`` vs absent), so an
        artifact-detection-backed and an artifact-detection-free selection
        never alias. Splits the matches by primary key:

        * the master at ``deterministic_id`` is the canonical, content-
          addressed selection -> return ``{"sorting_id": ...}``;
        * ANY master with a different ``sorting_id`` is non-deterministic
          (a raw ``insert`` bypass or pre-determinism legacy row) and
          violates the content-addressed-identity invariant -> raise
          ``DuplicateSelectionError`` so it is reset rather than silently
          returned.

        Used by ``insert_selection`` for both the pre-insert lookup and
        the post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        candidates = (
            (cls * cls.RecordingSource)
            & master_restriction
            & source_restriction
        ).fetch("KEY", as_dict=True)
        master_ids = {
            cand["sorting_id"]
            for cand in candidates
            if cls.resolve_artifact_detection(
                {"sorting_id": cand["sorting_id"]}
            )
            == artifact_detection_id
        }
        bypassed = [sid for sid in master_ids if sid != deterministic_id]
        if bypassed:
            raise DuplicateSelectionError(
                f"SortingSelection has {len(master_ids)} master rows for "
                f"{master_restriction | source_restriction} with "
                f"artifact_detection_id={artifact_detection_id} whose sorting_id "
                "is not the "
                f"deterministic id {deterministic_id}: {bypassed}. This is a "
                "non-deterministic selection row (a raw insert or pre-"
                "determinism legacy row); drop it and re-insert via "
                "insert_selection."
            )
        return {"sorting_id": deterministic_id} if master_ids else None

    @classmethod
    def _validate_artifact_detection_source_for_recording(
        cls,
        *,
        recording_id,
        artifact_detection_id,
    ) -> None:
        """Ensure an artifact-detection pass is valid for a sorting recording.

        Single-recording artifact detections may only be linked to that exact
        ``recording_id``. Shared-group detections may be linked to any member
        recording in the group. This keeps artifact masks from one recording
        in a session from being silently applied to a different recording.
        """
        if artifact_detection_id is None:
            return

        from spyglass.spikesorting.v2.artifact import (
            ArtifactDetectionSelection,
            SharedArtifactGroup,
        )

        artifact_detection_key = {
            "artifact_detection_id": artifact_detection_id
        }
        if not (ArtifactDetection & artifact_detection_key):
            raise ValueError(
                "SortingSelection.insert_selection: artifact_detection_id "
                f"{artifact_detection_id!r} is not in ArtifactDetection. Populate "
                "ArtifactDetection before linking an artifact-detection pass "
                "to a sort."
            )

        source = ArtifactDetectionSelection.resolve_source(
            artifact_detection_key
        )
        target_recording_id = str(recording_id)
        if source.kind == "recording":
            artifact_recording_id = str(source.key["recording_id"])
            if artifact_recording_id != target_recording_id:
                raise ValueError(
                    "SortingSelection.insert_selection: artifact_detection_id "
                    f"{artifact_detection_id!r} belongs to recording_id="
                    f"{artifact_recording_id!r}, not the requested "
                    f"recording_id={target_recording_id!r}."
                )
            return

        if source.kind == "shared_artifact_group":
            group_name = source.key["shared_artifact_group_name"]
            if not (
                SharedArtifactGroup.Member
                & {
                    "shared_artifact_group_name": group_name,
                    "recording_id": recording_id,
                }
            ):
                raise ValueError(
                    "SortingSelection.insert_selection: artifact_detection_id "
                    f"{artifact_detection_id!r} belongs to shared artifact group "
                    f"{group_name!r}, which does not include requested "
                    f"recording_id={target_recording_id!r}."
                )
            return

        raise ValueError(
            "SortingSelection.insert_selection: artifact_detection_id "
            f"{artifact_detection_id!r} has unsupported source kind {source.kind!r}."
        )

    @classmethod
    def prune_orphaned_selections(cls, dry_run: bool = True) -> list[dict]:
        """Find or delete master rows that have no source-part row.

        See ``ArtifactDetectionSelection.prune_orphaned_selections`` for the full
        rationale. Same contract: dry-run by default; with
        ``dry_run=False`` runs cautious_delete on each orphan so the
        cascade preview shows downstream ``Sorting`` / ``CurationV2`` /
        ``SpikeSortingOutput.CurationV2`` impact.
        """
        orphans = find_orphaned_masters(
            cls,
            [cls.RecordingSource, cls.ConcatenatedRecordingSource],
        )
        if dry_run or not orphans:
            return orphans
        for orphan in orphans:
            (cls & orphan).cautious_delete()
        return orphans

    @classmethod
    def resolve_source(cls, key: dict) -> SourceResolution:
        """Return the source-resolution record for a sorting selection.

        Layer 2 of the source-part pattern: fetches source-part rows for
        the master key, asserts exactly one exists across both source
        parts, and returns a ``SourceResolution(kind, key)`` so
        ``Sorting.make`` can dispatch on source shape without inspecting
        the raw part tables.

        Raises
        ------
        SchemaBypassError
            If zero or multiple source part rows exist for ``key``.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rec_rows = (cls.RecordingSource & master_key).fetch(as_dict=True)
        concat_rows = (cls.ConcatenatedRecordingSource & master_key).fetch(
            as_dict=True
        )
        total = len(rec_rows) + len(concat_rows)
        if total != 1:
            raise SchemaBypassError(
                f"SortingSelection {master_key} has {total} source part "
                "rows; expected exactly one. Use "
                "SortingSelection.insert_selection() to add or remove "
                "this selection."
            )
        if rec_rows:
            return SourceResolution(
                kind="recording",
                key={"recording_id": rec_rows[0]["recording_id"]},
            )
        return SourceResolution(
            kind="concatenated_recording",
            key={"concat_recording_id": concat_rows[0]["concat_recording_id"]},
        )

    @classmethod
    def resolve_artifact_detection(cls, key: dict):
        """Return the ``artifact_detection_id`` for a selection, or ``None``.

        Reads the optional ``ArtifactDetectionSource`` part row. Returns the
        ``artifact_detection_id`` when an artifact-detection pass was
        configured, else ``None`` (no ``ArtifactDetectionSource`` row = no
        artifact-detection pass). This is the single accessor every reader uses
        instead of a nullable-FK column lookup.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactDetectionSource`` row exists for ``key``
            (the part is zero-or-one by construction).
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rows = (cls.ArtifactDetectionSource & master_key).fetch(
            "artifact_detection_id"
        )
        if len(rows) > 1:
            raise SchemaBypassError(
                f"SortingSelection {master_key} has {len(rows)} "
                "ArtifactDetectionSource rows; expected zero or one."
            )
        return rows[0] if len(rows) == 1 else None

    @classmethod
    def resolve_source_preprocessing_params_name(cls, key: dict) -> str:
        """Return the sort source's ``preprocessing_params_name``.

        Region resolution input: a sort's analyzer waveform window is keyed on
        the SAME signal that sets the region filter cutoff -- the source
        preprocessing recipe. Reuses :meth:`resolve_source` for source
        detection (the single source-part integrity check) rather than
        re-inspecting the source part tables, then reads the source's
        ``preprocessing_params_name``:

        - ``RecordingSource`` -> the upstream ``RecordingSelection`` row.
        - ``ConcatenatedRecordingSource`` -> the upstream
          ``ConcatenatedRecordingSelection`` row (its
          ``-> PreprocessingParameters`` FK's primary key, not a literal column
          in the class definition).

        Does NOT query ``RecordingSelection`` for a concat source: concat member
        recordings are provenance inputs, but the concatenated source row is the
        sort input.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )

        source = cls.resolve_source(key)
        if source.kind == "recording":
            return (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("preprocessing_params_name")
        return (
            ConcatenatedRecordingSelection
            & {"concat_recording_id": source.key["concat_recording_id"]}
        ).fetch1("preprocessing_params_name")


@schema
class Sorting(SpyglassMixin, dj.Computed):
    """Sorted units NWB + SortingAnalyzer folder.

    ``make()`` applies post-motion preprocessing, runs the sorter,
    removes excess spikes, builds a
    ``SortingAnalyzer(format="zarr", sparse=True)``, computes
    the base extensions (``random_spikes``, ``noise_levels``,
    ``templates``, ``waveforms``), writes a fresh/whitelisted
    ``AnalysisNwbfile`` containing only the v2 sorting Units (NOT the
    parent NWB's units), and populates ``Sorting.Unit``.
    """

    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(72)
    n_units: int
    time_of_sort: datetime    # wall-clock time the sort was populated
    -> AnalyzerWaveformParameters.proj(display_waveform_params_name="waveform_params_name")
    """
    # ``display_waveform_params_name`` is a secondary FK to
    # ``AnalyzerWaveformParameters``: it records which DISPLAY recipe produced
    # this sort's analyzer + peak_amplitude_uv, and the FK enforces the
    # provenance in the database -- a sort cannot be inserted referencing a
    # recipe that is not tracked, and a referenced recipe row cannot be deleted.
    # The name is resolved at sort time from the source preprocessing recipe
    # (region; see _recipe_catalog.waveform_params_for_preprocessing) and read
    # back -- never re-resolved -- on every later rebuild, so the analyzer is
    # deterministic for a sorting_id. It is NOT a free per-sort knob and is NOT
    # part of sorting_id identity. The metric recipe is carried on
    # AnalyzerCurationSelection (not yet implemented), not here.
    # The SortingAnalyzer cache folder is intentionally NOT a column: it is
    # large (5-50 GB) regeneratable scratch resolved at runtime from
    # (sorting_id, display_waveform_params_name) via _analyzer_cache.analyzer_path.
    # Persisting an absolute path here previously drifted from the
    # accessor-computed path whenever temp_dir changed between runs.

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        Brain region is reached through the ``Sorting.Unit * Electrode *
        BrainRegion`` join (``Electrode`` carries a NON-NULL ``BrainRegion``
        FK in Spyglass). For concat sorts the Electrode FK is anchored to
        the FIRST member's row.
        """

        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uv: float    # peak template amplitude in microvolts
        n_spikes: int
        """

    # Exclude concat-source selections from populate. ``make_fetch``
    # raises ``NotImplementedError`` for the concat path (the concat
    # materializer is gated), but the DEFAULT key_source (the full
    # ``SortingSelection``) would still hand concat rows to ``populate()``,
    # which then prints a confusing per-row error. The antijoin drops any
    # selection that has a ``ConcatenatedRecordingSource`` part. No-op
    # today (no concat rows exist). When ``ConcatenatedRecording.make`` is
    # implemented this antijoin is removed so concat rows populate normally.
    key_source = SortingSelection - SortingSelection.ConcatenatedRecordingSource

    # Tri-part dispatch + parallel populate. ``Sorting.make`` is the
    # longest of the three Computed stages (sorters routinely take
    # 5-20 minutes); moving the run outside the framework
    # transaction is the dominant motivation. Parallel populate via
    # the non-daemon process pool is the secondary benefit.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs.

        Layer-2 source re-check fires here; concat raises
        ``NotImplementedError`` until the schema's concat runtime
        lands. All returned values are deterministic bytes
        (DataJoint fetches inline dicts) so DataJoint's tri-part
        DeepHash integrity check across the two fetches stays
        stable.

        Parameters
        ----------
        key : dict
            Primary key restricting to one ``SortingSelection`` row.

        Returns
        -------
        SortingFetched
            DB inputs (source, ``sel_row``, ``sorter_row``,
            ``nwb_file_name``, ``obs_intervals``) for the compute step.

        Raises
        ------
        NotImplementedError
            If the resolved source is a concatenated recording.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = SortingSelection.resolve_source(key)
        if source.kind != "recording":
            raise NotImplementedError(
                "Sorting.make: concatenated_recording source is not yet "
                "implemented."
            )

        sel_row = (SortingSelection & key).fetch1()
        # The artifact-detection pass lives on the zero-or-one
        # ``ArtifactDetectionSource`` part, not a nullable
        # ``artifact_detection_id`` FK on the master, so ``sel_row`` does not
        # carry an ``artifact_detection_id`` key. Resolve it once here and stash it
        # on ``sel_row`` so the
        # downstream readers (obs_intervals derivation below,
        # make_compute's artifact-mask gate, _rebuild_analyzer_folder)
        # see the artifact-detection id without re-querying. Without this the
        # ``sel_row.get("artifact_detection_id")`` reads would always be None
        # and every artifact-backed sort would silently skip artifact masking.
        sel_row["artifact_detection_id"] = (
            SortingSelection.resolve_artifact_detection(key)
        )
        sorter_row = (
            SorterParameters
            & {
                "sorter": sel_row["sorter"],
                "sorter_params_name": sel_row["sorter_params_name"],
            }
        ).fetch1()
        # ``source`` is already resolved (above); read both the nwb file and
        # the preprocessing recipe from the one RecordingSelection row rather
        # than re-resolving the source / re-querying the row for each.
        nwb_file_name, preprocessing_params_name = (
            RecordingSelection & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name", "preprocessing_params_name")

        # Resolve the DISPLAY analyzer recipe from the source preprocessing
        # recipe (region) -- hippocampus -> the 0.5/0.5 row, cortex -> the
        # 1.0/2.0 row, any other recipe -> the wider cortex fallback. Resolve
        # the params blob HERE (make_fetch is the only stage allowed DB I/O);
        # make_compute builds with it and make_insert persists the name so every
        # later rebuild reads it back deterministically.
        display_waveform_params_name, _ = waveform_params_for_preprocessing(
            preprocessing_params_name
        )
        display_waveform_params = fetch_waveform_params(
            display_waveform_params_name
        )

        # Pre-fetch the observation-interval window so
        # ``_write_units_nwb`` can write ``obs_intervals=`` on every
        # ``add_unit`` call. Downstream firing-rate computations
        # need the artifact-removed valid_times to know which
        # segments of the recording the sort actually observed --
        # without this column the units NWB looks like the unit
        # was observed across the full session even where the
        # artifact mask blanked the signal. When
        # ``artifact_detection_id`` is unset the caller (make_compute)
        # falls back to the recording's full timestamp envelope.
        if sel_row.get("artifact_detection_id") is not None:
            from spyglass.spikesorting.v2.utils import (
                artifact_detection_interval_list_name,
            )

            obs_intervals = (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": artifact_detection_interval_list_name(
                        sel_row["artifact_detection_id"]
                    ),
                }
            ).fetch1("valid_times")
        else:
            obs_intervals = None
        return SortingFetched(
            source=source,
            sel_row=sel_row,
            sorter_row=sorter_row,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            display_waveform_params_name=display_waveform_params_name,
            display_waveform_params=display_waveform_params,
        )

    def make_compute(
        self,
        key,
        source,
        sel_row,
        sorter_row,
        nwb_file_name,
        obs_intervals,
        display_waveform_params_name,
        display_waveform_params,
    ):
        """Sort, build analyzer, stage Units NWB outside any DB transaction.

        The long-running steps run here:

        - load the cached preprocessed recording,
        - apply the artifact mask if ``artifact_detection_id`` is set,
        - dispatch ``_run_sorter`` (clusterless thresholder or SI
          sorter; tempdir + Singularity + container carve-outs apply),
        - ``_remove_excess_spikes`` (boundary safety),
        - ``_build_analyzer`` (writes the ``analyzer_folder`` on disk),
        - ``_write_units_nwb`` (stages the AnalysisNwbfile on disk
          without registering it).

        Cleanup contract (failure-mode A): if anything raises between
        ``_build_analyzer`` and the end of this method, the analyzer
        folder and any staged units NWB are removed before the
        exception propagates. DataJoint will not call ``make_insert``
        once this raises, so cleanup has to happen here.

        Parameters
        ----------
        key : dict
            Primary key of the sorting being populated.
        source : SourceResolution
            Resolved recording source from ``make_fetch``.
        sel_row : dict
            The ``SortingSelection`` row (with ``artifact_detection_id``).
        sorter_row : dict
            The ``SorterParameters`` row (``sorter``, ``params``,
            ``job_kwargs``).
        nwb_file_name : str
            Source NWB file backing the recording selection.
        obs_intervals : numpy.ndarray or None
            Artifact-removed valid-times window, or ``None`` when no
            artifact-detection pass is configured.
        display_waveform_params_name : str
            The DISPLAY recipe resolved in ``make_fetch``; selects the analyzer
            cache folder and is persisted by ``make_insert``.
        display_waveform_params : dict
            That recipe's resolved params blob, fed to ``_build_analyzer`` so
            the window / subsample are not hardcoded (no DB I/O here).

        Returns
        -------
        SortingComputed
            Carrier of the computed sorting, staged units NWB, analyzer
            folder, and lookups threaded into ``make_insert``.
        """
        recording_id = source.key["recording_id"]
        recording = Recording().get_recording({"recording_id": recording_id})

        if (
            sel_row.get("artifact_detection_id") is not None
            and obs_intervals is not None
        ):
            recording = self._apply_artifact_mask(
                recording=recording,
                valid_times=obs_intervals,
                artifact_detection_id=sel_row.get("artifact_detection_id"),
                recording_id=recording_id,
            )

        sorter = sorter_row["sorter"]
        sorter_params = dict(sorter_row["params"])
        # ``schema_version`` is Pydantic bookkeeping; the SI sorter
        # wrapper does not accept it.
        sorter_params.pop("schema_version", None)

        # Resolve job_kwargs ONCE per compute stage so the resolved dict
        # flows into BOTH ``sis.run_sorter`` AND
        # ``analyzer.compute`` so a user's ``n_jobs=N`` override --
        # via ``dj.config['custom']['spikesorting_v2_job_kwargs']``
        # or via the per-row ``job_kwargs`` blob -- propagates to
        # both stages from one resolution. The sort itself is the
        # longer of the two; honoring job_kwargs only in
        # ``_build_analyzer`` would leave the sorter ignoring them.
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])

        sorting_obj = self._run_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=key["sorting_id"],
            job_kwargs=job_kwargs,
        )
        sorting_obj = self._remove_excess_spikes(sorting_obj, recording)

        # ``_build_analyzer`` creates the analyzer folder on disk.
        # From this point on a failure must clean BOTH the analyzer
        # folder AND the staged units NWB (if it was created). Pass
        # ``sorter_row`` so ``_build_analyzer`` does not re-issue the
        # ``SortingSelection`` + ``SorterParameters`` reads we
        # already did in ``make_fetch``; pass ``job_kwargs`` so the
        # analyzer uses the same resolved value as the sorter; pass the
        # resolved DISPLAY recipe + its cache folder (the folder carries the
        # recipe identity) so the analyzer window / subsample are not
        # hardcoded.
        from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

        analyzer_folder = analyzer_path(
            key["sorting_id"], display_waveform_params_name
        )
        self._build_analyzer(
            sorting=sorting_obj,
            recording=recording,
            key=key,
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
            analyzer_folder=analyzer_folder,
            waveform_params=display_waveform_params,
        )
        analysis_file_name, units_object_id = self._stage_sorting_artifact(
            sorting=sorting_obj,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            analyzer_folder=analyzer_folder,
        )

        return SortingComputed(
            sorting_obj=sorting_obj,
            analysis_file_name=analysis_file_name,
            units_object_id=units_object_id,
            analyzer_folder=analyzer_folder,
            recording_id=recording_id,
            nwb_file_name=nwb_file_name,
            display_waveform_params_name=display_waveform_params_name,
        )

    def make_insert(
        self,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        analyzer_folder,
        recording_id,
        nwb_file_name,
        display_waveform_params_name,
    ):
        """Atomic registration of the AnalysisNwbfile + master + Unit rows.

        DataJoint's tri-part dispatch wraps this method in the
        framework transaction; the inner ``transaction_or_noop`` is
        a no-op there (kept defensively). ``_populate_unit_part``
        runs INSIDE the transaction so its unit-part rows commit
        atomically with the master row; splitting it across stages
        is explicitly forbidden.

        Failure-mode B: if the registration raises after a
        successful ``make_compute``, the staged units NWB AND the
        analyzer folder are removed before propagating.

        Parameters
        ----------
        key : dict
            Primary key of the sorting being populated.
        sorting_obj : spikeinterface.BaseSorting
            The computed sorting (unit ids + spike trains).
        analysis_file_name : str
            Staged AnalysisNwbfile registered here.
        units_object_id : str
            NWB object id of the units table inside that file.
        analyzer_folder : pathlib.Path
            On-disk analyzer folder used to load peak channels and for
            Mode-B rollback cleanup.
        recording_id : str
            Recording id the sort was run against.
        nwb_file_name : str
            Source NWB file backing the recording selection.
        display_waveform_params_name : str
            The resolved DISPLAY recipe, persisted on the master row so every
            later rebuild reads it back deterministically.

        Returns
        -------
        None
        """
        try:
            self._insert_sorting_rows_transaction(
                key=key,
                sorting_obj=sorting_obj,
                analysis_file_name=analysis_file_name,
                units_object_id=units_object_id,
                analyzer_folder=analyzer_folder,
                recording_id=recording_id,
                nwb_file_name=nwb_file_name,
                display_waveform_params_name=display_waveform_params_name,
            )
        except Exception:
            # Failure-mode B: registration failed after a successful
            # make_compute -- roll back BOTH on-disk side effects (the staged
            # units NWB and the analyzer folder) before propagating.
            self._cleanup_staged_sorting_artifacts(
                analyzer_folder=analyzer_folder,
                analysis_file_name=analysis_file_name,
            )
            raise

    def _stage_sorting_artifact(
        self,
        *,
        sorting,
        recording,
        nwb_file_name,
        obs_intervals,
        analyzer_folder,
    ):
        """Stage the units NWB; clean the built analyzer if staging fails.

        ``_build_analyzer`` has already written ``analyzer_folder`` on disk,
        so a failure in the units-NWB write must remove it (failure-mode A)
        before propagating -- DataJoint does not call ``make_insert`` once
        ``make_compute`` raises. Returns ``(analysis_file_name,
        units_object_id)``.
        """
        try:
            return self._write_units_nwb(
                sorting=sorting,
                recording=recording,
                nwb_file_name=nwb_file_name,
                obs_intervals=obs_intervals,
            )
        except Exception:
            # Mode A cleanup: the analyzer folder exists but the units NWB
            # write failed; remove it so a half-built scratch dir doesn't leak.
            self._cleanup_staged_sorting_artifacts(
                analyzer_folder=analyzer_folder
            )
            raise

    def _insert_sorting_rows_transaction(
        self,
        *,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        analyzer_folder,
        recording_id,
        nwb_file_name,
        display_waveform_params_name,
    ):
        """Register the AnalysisNwbfile + master + Unit rows atomically.

        Runs the ``transaction_or_noop`` block: registers the staged
        AnalysisNwbfile, inserts the Sorting master, and populates the Unit
        part rows INSIDE the transaction so they commit atomically with the
        master (splitting them across stages is forbidden).
        ``transaction_or_noop`` is a no-op when the framework transaction is
        already active (the tri-part dispatch path); it is kept so an
        out-of-populate caller still gets atomic registration.
        ``analyzer_folder`` is the transient folder ``_build_analyzer``
        wrote (threaded from make_compute) to load each unit's peak channel;
        it is NOT a stored column -- the canonical path is resolved from
        sorting_id everywhere outside this single populate() invocation.
        """
        import datetime as dt

        with transaction_or_noop(self.connection):
            AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
            self.insert1(
                {
                    **key,
                    "analysis_file_name": analysis_file_name,
                    "object_id": units_object_id,
                    "n_units": len(sorting_obj.unit_ids),
                    "time_of_sort": dt.datetime.now(),
                    "display_waveform_params_name": (
                        display_waveform_params_name
                    ),
                }
            )
            self._populate_unit_part(
                sorting=sorting_obj,
                recording_id=recording_id,
                nwb_file_name=nwb_file_name,
                key=key,
                analyzer_folder=analyzer_folder,
            )

    @staticmethod
    def _cleanup_staged_sorting_artifacts(
        *, analyzer_folder, analysis_file_name=None
    ):
        """Best-effort removal of a sort's on-disk side effects after failure.

        Removes the staged units NWB (when ``analysis_file_name`` is given --
        failure-mode B, registration failed after staging) and the analyzer
        folder (failure-mode A, the units-NWB write failed after the analyzer
        was built; or B). Each unlink/rmtree is best-effort: a cleanup failure
        is logged, never raised, so it cannot mask the original error.
        DataJoint cannot roll back these filesystem side effects, so the
        caller invokes this in its ``except`` before re-raising.
        """
        import pathlib
        import shutil

        if analysis_file_name is not None:
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                pathlib.Path(abs_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting._cleanup_staged_sorting_artifacts: failed to "
                    f"clean up staged units NWB {analysis_file_name!r}: "
                    f"{cleanup_exc!r}"
                )
        try:
            if analyzer_folder.exists():
                shutil.rmtree(analyzer_folder, ignore_errors=False)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._cleanup_staged_sorting_artifacts: failed to remove "
                f"analyzer folder {analyzer_folder!r}: {cleanup_exc!r}"
            )

    # ---- Accessors -------------------------------------------------------

    def get_sorting(
        self, key: dict, as_dataframe: bool = False
    ) -> "si.BaseSorting | pd.DataFrame":
        """Return the SpikeInterface BaseSorting backed by the units NWB.

        Spike times are persisted by ``_write_units_nwb`` in the
        recording's ABSOLUTE wall-clock timeline
        (``spike_times = recording.get_times()[frame]``). For disjoint
        sort intervals that timeline is non-uniform -- it carries the
        wall-clock gaps that ``concatenate_recordings(ignore_times=True)``
        drops -- so the inverse map (absolute time -> recording frame)
        must use ``np.searchsorted`` against the actual recording
        timestamps. An affine
        ``round((t - t_start) * fs)`` inverse (SI's ``NwbSortingExtractor``)
        is correct only on a uniform grid and shifts every frame after a
        gap by the accumulated gap (and can push frames past the
        gap-excluded sample count), so it is NOT used here.

        Returns a ``NumpySorting`` (segment frame indices, ``t_start=0``),
        so ``get_unit_spike_train(uid)`` yields the original recording
        frames and a downstream ``extract_waveforms`` / analyzer build
        aligns to the right samples.

        ``as_dataframe=True`` returns a pandas DataFrame whose
        **index is the unit_id** and which carries a ``spike_times``
        column (the stored ABSOLUTE seconds, read straight from the
        units NWB). The ``CurationV2.get_sorting``
        accessor uses the same flag + index and adds a
        ``curation_label`` column joined from ``CurationV2.UnitLabel``.

        A zero-unit sort returns an empty sorting (with a warning);
        ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` instead.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        as_dataframe : bool, optional
            If ``True``, return a per-unit DataFrame instead of an SI
            sorting object. Defaults to ``False``.

        Returns
        -------
        si.BaseSorting or pd.DataFrame
            The sorting (a ``NumpySorting``) when ``as_dataframe`` is
            ``False``; otherwise a DataFrame indexed by ``unit_id`` with
            a ``spike_times`` column.
        """
        import spikeinterface as si

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        source = SortingSelection.resolve_source(key)
        recording_id = source.key["recording_id"]
        rec_row = (Recording & {"recording_id": recording_id}).fetch1()
        fs = float(rec_row["sampling_frequency"])

        if int(row["n_units"]) == 0:
            # Zero units is a valid result; return an empty sorting /
            # frame rather than crashing. (``get_analyzer`` on a
            # zero-unit sort raises instead, since an analyzer is not
            # representable over zero units.)
            logger.warning(
                f"Sorting.get_sorting: sorting_id={row['sorting_id']!r} "
                "has zero units; returning an empty sorting."
            )
            if as_dataframe:
                return empty_spike_times_dataframe()
            return si.NumpySorting.from_unit_dict({}, sampling_frequency=fs)

        abs_times = read_units_abs_spike_times(abs_path)
        if as_dataframe:
            return abs_spike_times_dataframe(abs_times)
        return numpysorting_from_abs_times(abs_times, rec_row, fs)

    @staticmethod
    def _recording_timestamps(recording_row):
        """Return the full timestamp vector of the upstream Recording.

        Thin delegator to :func:`._units_nwb.recording_timestamps`; kept
        as a ``Sorting`` staticmethod because
        ``test_recording_timestamps_reads_persisted_vector`` calls
        ``Sorting._recording_timestamps`` directly. The IO (reading only
        the persisted, gap-preserving ``ElectricalSeries`` timestamps for
        the ``np.searchsorted`` readback) lives in the service
        module.
        """
        return recording_timestamps(recording_row)

    def get_analyzer(
        self, key: dict, waveform_params_name: str | None = None
    ) -> "si.SortingAnalyzer":
        """Return the SortingAnalyzer; rebuild on missing folder.

        Recompute is in-place; the DataJoint row is not deleted on a
        missing analyzer folder.

        Raises ``ZeroUnitAnalyzerError`` for a zero-unit sort: SI cannot
        build a ``SortingAnalyzer`` over zero units, so no folder was
        written by ``_build_analyzer`` (it returns the would-be path).
        Loading it would surface a confusing SI/file error; this raises
        a clear signal instead. For a zero-unit sort this raises; use
        ``get_sorting`` (which returns an empty sorting, with a warning,
        rather than raising) if only the unit list is needed.

        This raise-vs-degrade split with ``get_sorting`` is intentional,
        not an inconsistency: a zero-unit *sorting* is a valid (empty)
        object SI represents natively, so ``get_sorting`` returns it and
        downstream consumers handle a quiet shank uniformly; a zero-unit
        *analyzer* has no valid representation, so there is nothing to
        return and a typed error is the honest contract (returning ``None``
        or a phantom folder would only defer the failure). Callers that
        want to branch without catching can precheck
        ``(Sorting & key).fetch1("n_units") > 0``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        waveform_params_name : str, optional
            The analyzer recipe to load. ``None`` (default) loads the sort's
            stored DISPLAY recipe (``display_waveform_params_name``); a caller
            needing the whitened metric recipe (e.g. the PC/NN cluster-
            separation metrics in ``AnalyzerCuration``) passes it explicitly.
            A missing folder is rebuilt for whichever recipe is requested.

        Returns
        -------
        si.SortingAnalyzer
            The loaded ``SortingAnalyzer`` for the sort, rebuilt in
            place if its folder was missing.
        """
        return load_or_rebuild_analyzer(
            self, key, waveform_params_name=waveform_params_name
        )

    def add_extensions(
        self, key: dict, extensions: list[str], **kwargs
    ) -> list[str]:
        """Add SortingAnalyzer extensions in place; return the ones computed.

        Convenience for callers (and ``AnalyzerCuration``) that need
        extensions beyond the sort-time base set. Only extensions NOT already
        present are computed, so the call is idempotent and never recomputes
        ``waveforms`` / ``templates`` (recomputing a parent cascade-deletes its
        derived extensions and rewrites the committed ``peak_amplitude_uv``). A
        different waveform window is a sort-time (``SorterParameters``)
        decision, not an analyzer-curation recompute.

        Job kwargs are resolved from this sort's ``SorterParameters`` row
        (per the Job-Kwargs Resolution convention); explicit ``kwargs`` win on
        conflict. The computed extensions persist to the on-disk analyzer
        folder (SI's ``zarr`` format saves them automatically).

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        extensions : list of str
            SortingAnalyzer extension names to add.
        **kwargs
            Job kwargs that override the resolved per-row defaults.

        Returns
        -------
        list of str
            The extensions actually computed (already-present ones are
            skipped); empty when every requested extension already exists.
        """
        from spyglass.spikesorting.v2._sorting_analyzer import (
            ensure_extensions,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        analyzer = self.get_analyzer(key)
        sorter_job_kwargs = (
            SorterParameters & (SortingSelection & key)
        ).fetch1("job_kwargs")
        resolved = _resolved_job_kwargs(sorter_job_kwargs)
        resolved.update(kwargs)
        return ensure_extensions(analyzer, extensions, job_kwargs=resolved)

    def _rebuild_analyzer_folder(self, key) -> None:
        """Rebuild the analyzer folder for an existing Sorting row.

        Reloads the canonical sorting from the units NWB so the
        rebuilt analyzer is bit-equivalent to the one Sorting.make
        wrote -- not a fresh, possibly nondeterministic, sort.

        ``key`` must carry a literal ``sorting_id`` (it is used directly
        for ``analyzer_path`` and the ``SortingSelection`` fetches). The
        public ``get_analyzer`` resolves the canonical id from a general
        restriction and hands this private helper a normalized
        ``{"sorting_id": ...}``; callers should do the same.
        """
        return rebuild_analyzer_folder(self, key)

    def delete(self, *args, safemode=None, **kwargs):
        """Cascade-delete + analyzer-cache cleanup on disk.

        The analyzer cache folder is regeneratable scratch resolved from
        ``sorting_id`` (not a DataJoint-tracked column), so a plain
        ``.delete()`` would leave the 5-50 GB folder on disk per row.
        Mirrors ``ArtifactDetection.delete``'s IntervalList cleanup pattern:
        snapshot every ``sorting_id`` BEFORE the cascade delete (it can no
        longer be fetched once the row is gone), call ``super().delete()``,
        then ``remove_analyzer_cache`` each (which resolves the path from
        ``sorting_id``, no-ops a missing folder, and surfaces a permission
        error loudly rather than swallowing it).

        Two cleanup paths already cover other points in the analyzer-cache
        lifecycle: ``_run_sorter`` cleans the
        sorter scratch ``TemporaryDirectory`` on successful sort,
        and the make_compute / make_insert except blocks clean the
        folder on populate failure. This override closes the third
        lifecycle event: row deletion.

        A leading positional restriction is accepted as a compatibility guard
        for the easy-to-mistype ``Sorting().delete(restriction)`` form. DataJoint's
        own ``delete`` does not take restrictions positionally, and Spyglass's
        cautious-delete layer would otherwise read that dict as a truthy
        ``force_permission`` (``cautious_delete(self, force_permission=False,
        ...)``) and cascade-delete EVERY row of the unrestricted instance --
        destroying each row's 5-50 GB analyzer folder. Mirrors
        ``ArtifactDetection.delete``'s guard.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``super().delete``.
        safemode : bool or None, optional
            Forwarded to ``super().delete`` to control the confirmation
            prompt. ``None`` (default) omits the argument so DataJoint's
            own default applies.
        **kwargs
            Keyword arguments forwarded to ``super().delete``.
        """
        restriction_args, args = split_leading_restrictions(args)
        if restriction_args:
            target = self
            for restriction in restriction_args:
                target = target & restriction
            return target.delete(*args, safemode=safemode, **kwargs)

        from spyglass.spikesorting.v2._analyzer_cache import (
            remove_analyzer_cache,
        )

        # Snapshot the PKs BEFORE the cascade -- after deletion the row is
        # gone, but the cache path is a pure function of sorting_id, so this
        # delete is one of the sites that resolves it from sorting_id (vs the
        # populate/rebuild cleanups, which rmtree the EXACT transient folder
        # they built to avoid a recompute).
        rows = self.fetch("KEY", as_dict=True)
        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)
        # Only remove a folder whose DB row was ACTUALLY deleted. A cancelled
        # confirmation prompt (user answers "no") or an empty restriction
        # leaves the rows in place and returns normally -- removing their
        # 5-50 GB analyzer scratch then would destroy data for a row the user
        # chose to keep. ``remove_analyzer_cache`` no-ops a missing folder and
        # propagates a real removal error (``ignore_errors=False``).
        for row in rows:
            if not (Sorting & row):
                remove_analyzer_cache(row["sorting_id"], missing_ok=True)

    @classmethod
    def find_orphaned_analyzer_folders(cls, *, dry_run: bool = True) -> dict:
        """Audit 5-50 GB analyzer-folder disk leaks; never auto-delete DB rows.

        Each populated sort writes a 5-50 GB ``analyzer_folder`` of
        regeneratable scratch outside the DataJoint-tracked store. The
        ``Sorting.delete`` override cleans it up on row delete, but an external
        path that bypasses the override (raw SQL delete, scripted
        ``dj.Table.connection.query``) leaks the folder. This periodic audit
        mirrors ``prune_orphaned_selections`` and reports two leak classes:

        - **DB-side orphan**: a ``Sorting`` row whose computed analyzer cache
          folder (``analyzer_path(sorting_id, display_waveform_params_name)``)
          no longer exists on disk (the
          regeneratable scratch was removed out of band). Reported only --
          deleting the *row* is a destructive DB operation the human decides on
          (per the Spyglass destructive-op contract); this method NEVER
          auto-deletes a row.
        - **Disk-side orphan**: an on-disk folder under the analyzer root that
          no ``Sorting`` row references (the row was deleted via a path that
          bypassed the ``delete`` override). Safe to delete after inspection.

        **Zero-unit carve-out.** Rows with ``n_units == 0`` are NOT DB-side
        orphans: ``_build_analyzer`` short-circuits before writing a folder and
        ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` before reading the
        path, so an absent folder is expected. The cache path is COMPUTED from
        ``sorting_id`` (not a stored column), so the carve-out is
        keyed on ``(Sorting & {"n_units": 0})``, NOT on any column value.

        Parameters
        ----------
        dry_run : bool, optional
            When True (default) only report the two orphan lists. When False,
            after interactive confirmation (``dj.utils.user_choice``), delete
            the **disk-side** orphan folders only; DB-side rows are never
            deleted.

        Returns
        -------
        dict
            ``{"db_side": [{"sorting_id", "computed_analyzer_path"}, ...],
            "disk_side": [folder_path_str, ...]}``. ``computed_analyzer_path``
            is resolved from ``sorting_id`` (there is no stored
            ``analyzer_folder`` column).
        """
        import shutil

        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_root,
            analyzer_path,
            classify_orphaned_analyzer_folders,
        )

        # Gather the DB / filesystem facts, then classify (the orphan set logic
        # is pure and DB-free in ``classify_orphaned_analyzer_folders``). Each
        # sort has one DISPLAY analyzer folder, named
        # {sorting_id}__{display_waveform_params_name}.zarr; the n_units==0
        # carve-out excludes legitimately folder-less rows from the DB-side set.
        # A sort's stored display recipe is the folder we expect on disk; a
        # {sid}__{other}.zarr folder (e.g. a stale recipe) is therefore a
        # disk-side orphan. (Retaining the whitened metric recipe's folder via
        # AnalyzerCurationSelection is not yet implemented.)
        units_bearing = []
        for r in (cls & "n_units > 0").fetch(
            "sorting_id", "display_waveform_params_name", as_dict=True
        ):
            path = analyzer_path(
                r["sorting_id"], r["display_waveform_params_name"]
            )
            units_bearing.append((r["sorting_id"], str(path), path.exists()))
        referenced_paths = {
            str(
                analyzer_path(
                    r["sorting_id"], r["display_waveform_params_name"]
                )
            )
            for r in cls.fetch(
                "sorting_id", "display_waveform_params_name", as_dict=True
            )
        }
        analyzer_root = analyzer_cache_root()
        disk_dir_paths = (
            [str(c) for c in sorted(analyzer_root.iterdir()) if c.is_dir()]
            if analyzer_root.exists()
            else []
        )
        classification = classify_orphaned_analyzer_folders(
            units_bearing, referenced_paths, disk_dir_paths
        )
        db_side = classification["db_side"]
        disk_side = classification["disk_side"]

        logger.info(
            "Sorting.find_orphaned_analyzer_folders: "
            f"{len(db_side)} DB-side orphan(s) (row present, folder missing), "
            f"{len(disk_side)} disk-side orphan(s) (folder present, no row)."
        )
        for row in db_side:
            logger.info(
                "  DB-side orphan: sorting_id=%s computed_analyzer_path=%s",
                row["sorting_id"],
                row["computed_analyzer_path"],
            )
        for folder in disk_side:
            logger.info("  disk-side orphan: %s", folder)

        if dry_run or not disk_side:
            return {"db_side": db_side, "disk_side": disk_side}

        # dry_run=False: delete ONLY the disk-side orphan folders, and only
        # after explicit interactive confirmation. DB-side rows are never
        # auto-deleted -- removing a row is the human's decision.
        msg = (
            f"Delete {len(disk_side)} orphaned analyzer folder(s) on disk "
            "(no referencing Sorting row)? This frees 5-50 GB each and cannot "
            "be undone [yes/no]: "
        )
        if dj.utils.user_choice(msg).lower() in ("y", "yes"):
            for folder in disk_side:
                shutil.rmtree(folder, ignore_errors=False)
            logger.info(
                "Sorting.find_orphaned_analyzer_folders: deleted "
                f"{len(disk_side)} disk-side orphan folder(s)."
            )
        else:
            logger.info(
                "Sorting.find_orphaned_analyzer_folders: aborted; nothing "
                "deleted."
            )
        return {"db_side": db_side, "disk_side": disk_side}

    def get_unit_brain_regions(
        self, key: dict, *, allow_anchor_member: bool = False
    ) -> "pd.DataFrame":
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts raise ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; the anchor-member output is
        labeled ``region_resolution='anchor_member'``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Sorting`` row.
        allow_anchor_member : bool, optional
            If ``True``, return anchor-member regions for concat-backed
            sortings instead of raising. Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            One row per (unit, electrode) with the brain-region columns
            and a ``region_resolution`` label.
        """
        from spyglass.spikesorting.v2.exceptions import (
            ConcatBrainRegionAmbiguousError,
        )

        source = SortingSelection.resolve_source(key)
        if source.kind == "concatenated_recording":
            if not allow_anchor_member:
                raise ConcatBrainRegionAmbiguousError(
                    f"Sorting.get_unit_brain_regions: sorting_id "
                    f"{key['sorting_id']} is concat-backed; the unit peak "
                    "channel maps to multiple Electrode rows (one per "
                    "SessionGroup.Member). Pass allow_anchor_member=True "
                    "to return anchor-member regions. Per-session regions "
                    "require cross-session unit matching, which is not "
                    "available in this build."
                )
            resolution = "anchor_member"
        else:
            resolution = "single_session"
        return unit_brain_region_df(self.Unit & key, resolution)

    # ---- Implementation helpers -----------------------------------------

    @staticmethod
    def _apply_artifact_mask(
        recording, valid_times, *, artifact_detection_id=None, recording_id=None
    ):
        """Zero out the complement of ``valid_times`` on the recording.

        Thin delegator to :func:`._sorting_artifact_mask.apply_artifact_mask`;
        kept as a ``Sorting`` staticmethod because ``make_compute`` calls
        ``self._apply_artifact_mask(...)`` and the v2 tests call
        ``Sorting._apply_artifact_mask`` directly. The complement-walk
        masking + input validation (empty/shape/order checks, the
        disjoint-gap boundary carve-out) live in the service module.
        """
        return apply_artifact_mask(
            recording,
            valid_times,
            artifact_detection_id=artifact_detection_id,
            recording_id=recording_id,
        )

    @staticmethod
    def _run_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
    ):
        """Dispatch sort execution; clusterless_thresholder vs SI sorters.

        Clusterless is a Spyglass-specific peak-detection path with no
        SI scratch directory or external whitening; SI sorters get
        per-sort scratch, external whitening, and a small MATLAB-
        sorter carve-out. The two paths share nothing but the
        signature; dispatch routes each to its own helper.
        """
        if sorter == "clusterless_thresholder":
            return Sorting._run_clusterless_thresholder(
                sorter_params=sorter_params,
                recording=recording,
                job_kwargs=job_kwargs,
            )
        return Sorting._run_si_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=sorting_id,
            job_kwargs=job_kwargs,
        )

    @staticmethod
    def _run_clusterless_thresholder(
        sorter_params,
        recording,
        job_kwargs,
    ):
        """Run Spyglass's clusterless-thresholder peak-detection path.

        Thin delegator to
        :func:`._sorting_dispatch.run_clusterless_thresholder`; kept as a
        ``Sorting`` staticmethod because ``_run_sorter`` dispatches to
        ``Sorting._run_clusterless_thresholder`` and the v2 tests call it
        directly. The detect_peaks pipeline -- noise_levels/threshold_unit
        precedence and validation, the uV ``scale_to_uV`` carve-out, and
        the deterministic seeding -- lives in the service module.
        """
        return run_clusterless_thresholder(
            sorter_params=sorter_params,
            recording=recording,
            job_kwargs=job_kwargs,
        )

    @staticmethod
    def _run_si_sorter(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        job_kwargs,
    ):
        """Run an SI registered sorter under a managed scratch dir.

        Thin delegator to :func:`._sorting_dispatch.run_si_sorter`; kept as
        a ``Sorting`` staticmethod because ``_run_sorter`` dispatches to
        ``Sorting._run_si_sorter`` and the v2 tests call it directly. The
        managed ``TemporaryDirectory`` scratch, external float64 whitening,
        scoped ``np.Inf`` patch, MATLAB-sorter carve-out, and the
        global-job-kwargs save/restore live in the service module.
        """
        return run_si_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=sorting_id,
            job_kwargs=job_kwargs,
        )

    @staticmethod
    def _remove_excess_spikes(sorting, recording):
        """Drop spikes whose sample index is outside the recording window.

        Thin delegator to :func:`._sorting_dispatch.remove_excess_spikes`;
        kept as a ``Sorting`` staticmethod because ``make_compute`` calls
        ``self._remove_excess_spikes(...)``.
        """
        return remove_excess_spikes(sorting, recording)

    @staticmethod
    def _build_analyzer(
        sorting,
        recording,
        key,
        *,
        sorter_row=None,
        job_kwargs=None,
        analyzer_folder=None,
        waveform_params=None,
    ):
        """Build the binary-folder SortingAnalyzer + base extensions.

        Thin delegator to :func:`._sorting_analyzer.build_analyzer`; kept as
        a ``Sorting`` staticmethod because ``make_compute`` /
        ``_rebuild_analyzer_folder`` call ``self._build_analyzer(...)`` and
        the v2 tests call it directly. The analyzer creation, seeded
        extension compute, zero-unit short-circuit, and partial-folder
        cleanup -- plus the rebuild-only ``SorterParameters`` fallback
        fetch -- live in the service module. ``waveform_params`` is the
        resolved analyzer-waveform params blob (window / subsample); ``None``
        is invalid and the service raises ``ValueError``.
        """
        return build_analyzer(
            sorting,
            recording,
            key,
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
            analyzer_folder=analyzer_folder,
            waveform_params=waveform_params,
        )

    @staticmethod
    def _write_units_nwb(sorting, recording, nwb_file_name, obs_intervals=None):
        """Write a fresh AnalysisNwbfile containing only the v2 Units table.

        Thin delegator to :func:`._units_nwb.write_sorting_units_nwb`;
        kept as a ``Sorting`` staticmethod because ``make_insert`` calls
        ``self._write_units_nwb(...)`` and the v2 tests both monkeypatch
        ``Sorting._write_units_nwb`` (the Mode-A analyzer-cleanup audit)
        and call it directly (the zero-unit guard test). The NWB staging
        IO -- absolute-timeline spike times, the ``obs_intervals`` +
        ``curation_label`` columns, and the zero-unit empty-Units guard --
        lives in the service module.
        """
        return write_sorting_units_nwb(
            sorting=sorting,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
        )

    @staticmethod
    def _populate_unit_part(
        sorting,
        recording_id,
        nwb_file_name,
        key,
        analyzer_folder,
    ):
        """Insert one ``Sorting.Unit`` row per sorted unit.

        Each row carries the full ``Electrode`` FK for the peak-amplitude
        channel (resolved through the sort group's
        ``SortGroupV2.SortGroupElectrode``) plus the peak template
        amplitude in microvolts and the spike count.

        ``analyzer_folder`` is the transient folder ``_build_analyzer``
        wrote (threaded from make_compute), NOT a stored column -- so the
        EXACT folder built is loaded here, not a recomputed path.

        Zero-unit early-return: ``_build_analyzer`` skips the
        ``create_sorting_analyzer`` call when ``sorting.get_num_units
        () == 0`` (SI's ``estimate_sparsity`` crashes on empty
        sortings), so the analyzer folder does not exist. There are
        no units to insert in that case; return early so
        ``si.load_sorting_analyzer`` is not called against a
        non-existent folder.
        """
        if sorting.get_num_units() == 0:
            return

        import numpy as np
        import spikeinterface as si
        from spikeinterface.core import template_tools

        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )
        from spyglass.spikesorting.v2.utils import resolve_peak_sign

        analyzer = si.load_sorting_analyzer(analyzer_folder)
        # Honor the sorter's configured detection polarity (clusterless
        # ``peak_sign`` / MountainSort ``detect_sign``) rather than
        # SpikeInterface's ``"neg"`` default, so a positive-going detection
        # attributes each unit to its true peak channel instead of the
        # most-negative one. The polarity is threaded here at sort time.
        sorter_params = (
            SortingSelection * SorterParameters
            & {"sorting_id": key["sorting_id"]}
        ).fetch1("params")
        peak_sign = resolve_peak_sign(sorter_params)
        peak_channels = template_tools.get_template_extremum_channel(
            analyzer, peak_sign=peak_sign, outputs="id"
        )
        # ``mode="extremum"`` measures the amplitude at the template PEAK
        # (matching ``get_template_extremum_channel`` above), not SI's
        # ``mode="at_index"`` default which reads the alignment-sample value
        # -- that under-reports the peak AND can land on a different channel
        # than the attributed electrode. ``abs_value=True`` (SI default)
        # returns the non-negative magnitude ``peak_amplitude_uv`` stores.
        peak_amplitudes = template_tools.get_template_extremum_amplitude(
            analyzer, peak_sign=peak_sign, mode="extremum"
        )

        sort_group_id = int(
            (RecordingSelection & {"recording_id": recording_id}).fetch1(
                "sort_group_id"
            )
        )
        sg_electrodes = (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch(as_dict=True)
        electrode_by_id = {
            int(row["electrode_id"]): row for row in sg_electrodes
        }

        # Spike counts come from the in-memory sorting; the remaining row
        # construction (Electrode FK resolution, peak-amplitude/n_spikes
        # assembly, channel-mismatch guard) is pure and lives in
        # ``build_sorting_unit_rows``.
        n_spikes_by_unit = {
            unit_id: int(len(sorting.get_unit_spike_train(unit_id=unit_id)))
            for unit_id in sorting.unit_ids
        }
        rows = build_sorting_unit_rows(
            unit_ids=sorting.unit_ids,
            peak_channels=peak_channels,
            peak_amplitudes=peak_amplitudes,
            n_spikes_by_unit=n_spikes_by_unit,
            electrode_by_id=electrode_by_id,
            key=key,
            sort_group_id=sort_group_id,
            nwb_file_name=nwb_file_name,
        )
        Sorting.Unit.insert(rows)
