"""Spike sorting and per-unit brain-region metadata.

Tables (all final-shape under the zero-migration policy):
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source.
        .ConcatenatedRecordingSource -- concat source (same-day chronic).
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.

``SorterParameters.insert1`` dispatches to the per-sorter Pydantic
schema via ``_get_sorter_schema``. ``insert_selection`` resolves a
sorting request to a single ``sorting_id``, ``make`` runs the
sorter and writes the units NWB + analyzer, and the accessor methods
(``get_sorting``, ``get_analyzer``) read those back. Both source kinds
populate: a recording source loads ``Recording``, a concatenated-recording
source loads ``ConcatenatedRecording`` and anchors the per-unit Electrode FK
and the analysis-NWB parent to the first frozen
``ConcatenatedRecordingSelection.MemberSnapshot`` member.
"""

from __future__ import annotations

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
    sorter_distribution_version,
)
from spyglass.spikesorting.v2._sorting_units import (
    _to_int_unit_id,  # noqa: F401  re-exported for tests
    build_sorting_unit_rows,
)
from spyglass.spikesorting.v2._units_nwb import (
    abs_spike_times_dataframe,
    empty_spike_times_dataframe,
    numpysorting_from_abs_times,
    numpysorting_from_sample_indices,
    read_units_abs_spike_times,
    read_units_spike_sample_indices,
    recording_timestamps,
    write_sorting_units_nwb,
)
from spyglass.spikesorting.v2.artifact import ArtifactDetection  # noqa: F401
from spyglass.spikesorting.v2.recording import Recording  # noqa: F401
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    SourceResolution,
    _validate_params,
    find_orphaned_masters,
    reject_duplicate_parameter_content,
    resolve_effective_seed,
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
        Resolved sort input source (``kind == "recording"`` or
        ``"concatenated_recording"``).
    recording_id : str
        The anchor ``recording_id``: the sort's own recording for a
        single-recording source, or the FIRST frozen ``MemberSnapshot``
        member's recording for a concat source (the deterministic parent anchor
        for the per-unit ``Electrode`` FK and the analysis-NWB parent).
    sel_row : dict
        The ``SortingSelection`` row, with ``artifact_detection_id`` resolved
        and stashed on it for downstream readers.
    sorter_row : dict
        The matching ``SorterParameters`` row (``sorter``, ``params``,
        ``job_kwargs``).
    nwb_file_name : str
        Anchor NWB file (the single-recording session, or the first concat
        member's session).
    obs_intervals : numpy.ndarray or None
        Artifact-removed valid-times window, or ``None`` when no
        artifact-detection pass is configured (always ``None`` for a concat
        source).
    display_waveform_params_name : str
        The DISPLAY ``AnalyzerWaveformParameters`` recipe resolved from the
        source preprocessing recipe (region), stored on the ``Sorting`` row.
    display_waveform_params : dict
        That recipe's resolved params blob, threaded into ``_build_analyzer``
        so ``make_compute`` does no parameter DB I/O.
    execution_params : dict
        The validated ``SorterParameters.execution_params`` blob (sorter
        execution backend + container provenance), resolved here so
        ``make_compute`` does no parameter DB I/O before passing it to the
        DB-free sorter dispatch.
    """

    source: SourceResolution
    recording_id: str
    sel_row: dict
    sorter_row: dict
    nwb_file_name: str
    obs_intervals: np.ndarray | None
    display_waveform_params_name: str
    display_waveform_params: dict
    execution_params: dict
    # Per-unit Electrode FK resolution, threaded so make_compute builds the
    # Sorting.Unit rows DB-free (anchored to the recording / first concat
    # member). ``region_by_electrode`` maps electrode_id -> brain region for the
    # NWB ``brain_region`` column.
    sort_group_id: int
    electrode_by_id: dict
    region_by_electrode: dict


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
    nwb_file_name : str
        Source NWB file backing the recording selection.
    display_waveform_params_name : str
        The resolved DISPLAY recipe name; stored on the ``Sorting`` row by
        ``make_insert`` so every later rebuild reads it back deterministically.
    effective_random_seed : int
        The random seed the sort actually used (``resolve_effective_seed``),
        recorded as secondary provenance -- NOT identity.
    spikeinterface_version : str
        ``spikeinterface.__version__`` at sort time (secondary provenance).
    sorter_version : str or None
        Installed distribution version of the sorter package, or ``None`` for
        in-process / SI-internal sorters (secondary provenance).
    unit_rows : list of dict
        The ``Sorting.Unit`` rows built ONCE in ``make_compute`` (peak channel /
        amplitude / spike count), reused by ``make_insert`` for the part insert
        so the DB and the NWB cannot drift. Empty for a zero-unit sort.
    """

    sorting_obj: "si.BaseSorting"
    analysis_file_name: str
    units_object_id: str
    analyzer_folder: Path
    nwb_file_name: str
    display_waveform_params_name: str
    effective_random_seed: int
    spikeinterface_version: str
    sorter_version: str | None
    unit_rows: list[dict]


schema = dj.schema("spikesorting_v2_sorting")


@schema
class SorterParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
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
    execution_params: blob               # validated SorterExecutionParamsSchema dump
    execution_params_schema_version=1: int
    """
    # ``execution_params`` records the sorter EXECUTION backend (local vs
    # Docker/Singularity) + container-side install provenance, validated by
    # ``SorterExecutionParamsSchema``. It is tracked here -- not in the scientific
    # ``params`` blob and not in ``job_kwargs`` -- because the backend can change
    # the sort output and ``sorter_params_name`` already flows into ``sorting_id``.
    # A row that omits it backfills the schema default (``backend="local"``).
    # Switching backend for an existing logical row therefore requires a NEW
    # ``sorter_params_name`` (a local MS4 row and a containerized MS4 row are
    # distinct named rows), not an in-place mutation.

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
            reject_reserved_execution_keys,
            validate_execution_params,
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
            # Container backend / install provenance is tracked ONLY on
            # ``execution_params`` -- reject the reserved execution keys from the
            # scientific ``params`` blob (the permissive ``extra="allow"`` sorter
            # schemas would otherwise pass them straight through) AND from
            # ``job_kwargs``. The strict schemas already reject the same keys via
            # ``extra="forbid"``.
            reject_reserved_execution_keys(
                row["params"], context="SorterParameters params blob"
            )
            reject_reserved_execution_keys(
                row.get("job_kwargs"), context="SorterParameters job_kwargs"
            )
            # ``params_schema_version`` is backfilled from the validated blob by
            # ``validate_lookup_rows`` (the shared path, for every Lookup), so it
            # is NOT re-done here. The execution-params version below is
            # SorterParameters-specific and stays in this hook.
            # Validate + backfill the execution backend provenance. A row that
            # omits ``execution_params`` defaults to local execution; the outer
            # ``execution_params_schema_version`` is backfilled from the
            # validated blob when omitted and cross-checked when supplied.
            validated_execution = validate_execution_params(
                row.get("execution_params")
            )
            row["execution_params"] = validated_execution
            # In-process sorters (clusterless) run in the host process; the
            # runtime ignores any container backend, so a row that claims one
            # would make preflight falsely require a container image. Reject it
            # at insert -- these must run with backend='local' (or omit it).
            if (
                sorter in self._NON_SI_SORTERS
                and validated_execution["backend"] != "local"
            ):
                raise ValueError(
                    f"SorterParameters.insert: sorter {sorter!r} runs "
                    "in-process (clusterless) and is executed locally; it "
                    "cannot use execution backend "
                    f"{validated_execution['backend']!r}. Set backend='local' "
                    "or omit execution_params."
                )
            inner_execution_version = int(validated_execution["schema_version"])
            if "execution_params_schema_version" not in row:
                row["execution_params_schema_version"] = inner_execution_version
            elif (
                int(row["execution_params_schema_version"])
                != inner_execution_version
            ):
                raise ValueError(
                    "SorterParameters.insert: execution_params_schema_version="
                    f"{row['execution_params_schema_version']} does not match "
                    "the inner SorterExecutionParamsSchema schema_version="
                    f"{inner_execution_version} on the validated execution_params "
                    "blob. Drop the column or align it with the blob's "
                    "schema_version."
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
    def _is_seed_dependent_sort(cls, sorter: str, sorter_params: dict) -> bool:
        """Whether the sort's output depends on ``random_seed``.

        A seed-dependent sort must reject an ambient-only seed (present in the
        SI-global / ``dj.config`` layer but not in the ``SorterParameters`` row):
        the seed changes the output but is not folded into ``sorting_id``, so a
        later ambient-seed change would silently reuse this sort.

        SI sorters cluster stochastically, so they are always seed-dependent.
        The ``clusterless_thresholder`` peak detector is deterministic EXCEPT on
        its per-channel MAD noise path (``threshold_unit='mad'`` with no explicit
        ``noise_levels``), which estimates noise from randomly sampled chunks;
        the ``'uv'`` path and an explicit ``noise_levels`` are deterministic.
        """
        if sorter not in cls._NON_SI_SORTERS:
            return True
        return (
            sorter_params.get("threshold_unit") == "mad"
            and sorter_params.get("noise_levels") is None
        )

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
        sorter (``_NON_SI_SORTERS``, never gated), is in
        ``spikeinterface.sorters.installed_sorters()``, OR runs on a tracked
        container backend (``execution_params.backend`` in
        ``{"docker", "singularity"}``). Container rows ship even when the local
        sorter runtime is unavailable -- their runtime lives in the image, so a
        selected container row is gated by preflight (the container runtime
        check), not by local-install at default-row insertion. Returns
        ``(insertable, skipped)`` so ``insert_default`` can log the skips and
        tests can assert the gating decision without depending on the live
        ``SorterParameters`` table state.
        """
        import spikeinterface.sorters as sis

        from spyglass.spikesorting.v2._sorting_dispatch import (
            is_container_backend,
        )

        installed = set(sis.installed_sorters())
        insertable: list = []
        skipped: list = []
        for row in cls._DEFAULT_CONTENTS:
            sorter = row[0]
            # row = (sorter, name, params, params_sv, job_kwargs,
            #        execution_params, execution_params_sv)
            if (
                sorter in cls._NON_SI_SORTERS
                or sorter in installed
                or is_container_backend(row[5])
            ):
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
        from spyglass.spikesorting.v2._sorting_dispatch import MATLAB_SORTERS

        curated = set(_SORTER_SCHEMAS)  # sorters with their own schemas
        installed = set(sis.installed_sorters())
        rows = []
        skipped_not_installed = []
        skipped_matlab = []
        for sorter in sis.available_sorters():
            if sorter in curated:
                continue  # see docstring: would fail or drop keys
            if sorter.lower() in MATLAB_SORTERS:
                # MATLAB sorters require a container backend; a local 'default'
                # row is rejected by the dispatcher at populate time. The lab
                # inserts MATLAB rows explicitly with a container
                # execution_params, so do not seed an unrunnable local default.
                skipped_matlab.append(sorter)
                continue
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
            # Append a MAPPING row (not a positional tuple) so the insert hook
            # backfills the omitted execution_params (default local execution) +
            # its schema version -- a positional row would have to enumerate all
            # SorterParameters columns and would break the moment the table gains
            # one (as it did with execution_params).
            rows.append(
                {
                    "sorter": sorter,
                    "sorter_params_name": "default",
                    "params": validated,
                    "params_schema_version": _params_schema_version(validated),
                    "job_kwargs": None,
                }
            )
        if skipped_not_installed:
            logger.info(
                "insert_default_legacy_si_sorters: skipping "
                f"{sorted(skipped_not_installed)} -- available in "
                "SpikeInterface but not in installed_sorters() on this "
                "platform (a 'default' row would fail at populate time)."
            )
        cls.insert(rows, skip_duplicates=True)


def _reject_unsafe_waveform_params_name(row, _schema_cls) -> None:
    """Reject a non-path-safe ``waveform_params_name`` at insert.

    A ``per_row_hook`` for ``validate_lookup_rows``; delegates to the DB-free
    :func:`._analyzer_cache.assert_path_safe_waveform_params_name` (the owner of
    the analyzer-path policy), which the load path reuses.
    """
    from spyglass.spikesorting.v2._analyzer_cache import (
        assert_path_safe_waveform_params_name,
    )

    assert_path_safe_waveform_params_name(row["waveform_params_name"])


@schema
class AnalyzerWaveformParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
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
    ``RecordingSource`` (single-session) or ``ConcatenatedRecordingSource``
    (same-day chronic concatenation) exists for each selection row.
    ``insert_selection`` dispatches on the requested source and both kinds
    populate through ``Sorting.make``.

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
        ``concat_recording_id`` (same-day chronic concatenation) from ``key``;
        raises ValueError on zero or two sources, and inserts the matching
        source part (``RecordingSource`` or ``ConcatenatedRecordingSource``).
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
            If zero or both source keys are supplied, or if a concat source
            also supplies an ``artifact_detection_id`` (concat sorts have no
            artifact pass).
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

        # Dispatch on the input source: a recording-backed sort inserts a
        # RecordingSource part; a concat-backed sort inserts a
        # ConcatenatedRecordingSource part. find-existing keys on the same
        # source part so the two source families never alias.
        if plan.source_kind == "concat":
            source_part = cls.ConcatenatedRecordingSource
            source_row = plan.concat_source_row
        else:
            source_part = cls.RecordingSource
            source_row = plan.recording_source_row

        existing = cls._find_existing_pk(
            plan.master_restriction,
            plan.source_restriction,
            plan.artifact_detection_id,
            plan.sorting_id,
            source_part,
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
        # Artifact-detection passes apply only to a single-recording source;
        # the plan already rejects a concat source that supplies an artifact,
        # so this validation runs for the recording path only.
        if plan.source_kind == "recording":
            cls._validate_artifact_detection_source_for_recording(
                recording_id=plan.source_restriction["recording_id"],
                artifact_detection_id=plan.artifact_detection_id,
            )

        try:
            with transaction_or_noop(cls.connection):
                # allow_direct_insert: this helper IS the validation boundary.
                cls.insert1(plan.master_row, allow_direct_insert=True)
                source_part.insert1(source_row)
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
                source_part,
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
        source_part,
    ) -> dict | None:
        """Return the canonical master PK for this sort selection, or None.

        ``source_part`` is the source part table for the requested input
        (``RecordingSource`` for a recording source,
        ``ConcatenatedRecordingSource`` for a concat source); the find-existing
        join keys on that part so a recording-backed and a concat-backed sort
        never alias even if their source-id strings collide.

        Matches a master with the same sorter + source AND the same
        artifact-detection-source state
        (present-with-this-``artifact_detection_id`` vs absent -- a concat
        source always has no artifact pass), so an artifact-detection-backed
        and an artifact-detection-free selection never alias. Splits the
        matches by primary key:

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
            (cls * source_part) & master_restriction & source_restriction
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

    ``make()`` resolves the source recording, applies sorter-owned
    preprocessing such as external MS4/MS5 whitening when requested,
    runs the sorter, removes excess spikes, builds a
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
    effective_random_seed=null: int     # seed actually used (resolve_effective_seed); provenance, NOT identity
    spikeinterface_version: varchar(32) # spikeinterface.__version__ at sort time
    sorter_version=null: varchar(64)    # sorter package distribution version, NULL for in-process sorters
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
    # part of sorting_id identity. The whitened METRIC recipe is carried on
    # CurationEvaluationSelection.metric_waveform_params_name, not here.
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

    # Both source kinds populate: ``make_fetch`` resolves the source (recording
    # vs concatenated_recording) and dispatches accordingly, so the default
    # ``key_source`` (the full ``SortingSelection``) is correct -- no antijoin.

    # Tri-part dispatch + parallel populate. ``Sorting.make`` is the
    # longest of the three Computed stages (sorters routinely take
    # 5-20 minutes); moving the run outside the framework
    # transaction is the dominant motivation. Parallel populate via
    # the non-daemon process pool is the secondary benefit.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs.

        Layer-2 source re-check fires here (``resolve_source`` asserts exactly
        one input source). Dispatches on the source: a single-recording source
        anchors to its own ``RecordingSelection``; a concat source anchors
        deterministically to the FIRST frozen ``MemberSnapshot`` member (so the
        per-unit ``Electrode`` FK and the analysis-NWB parent both resolve to the
        anchor member) and observes no artifact pass. All returned values are
        deterministic bytes (DataJoint fetches inline dicts) so DataJoint's
        tri-part DeepHash integrity check across the two fetches stays stable.

        Parameters
        ----------
        key : dict
            Primary key restricting to one ``SortingSelection`` row.

        Returns
        -------
        SortingFetched
            DB inputs (source, anchor ``recording_id``, ``sel_row``,
            ``sorter_row``, anchor ``nwb_file_name``, ``obs_intervals``,
            display recipe, execution params) for the compute step.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = SortingSelection.resolve_source(key)

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
        # (A concat source never has an artifact pass, so this is None there.)
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

        # Resolve the anchor recording_id / nwb file / preprocessing recipe.
        # Single-recording: the sort's own RecordingSelection. Concat: the
        # first member (deterministic parent anchor). Concat sorts observe the
        # full recording (no artifact pass).
        if source.kind == "recording":
            recording_id = source.key["recording_id"]
            nwb_file_name, preprocessing_params_name = (
                RecordingSelection & {"recording_id": recording_id}
            ).fetch1("nwb_file_name", "preprocessing_params_name")
            # Pre-fetch the observation-interval window so ``_write_units_nwb``
            # can write ``obs_intervals=`` on every ``add_unit`` call.
            # Downstream firing-rate computations need the artifact-removed
            # valid_times to know which segments of the recording the sort
            # actually observed -- without it the units NWB looks like the unit
            # was observed across the full session even where the artifact mask
            # blanked the signal. When ``artifact_detection_id`` is unset,
            # make_compute falls back to the recording's full envelope.
            if sel_row.get("artifact_detection_id") is not None:
                # Route through the strict ownership helper instead of
                # fetching the IntervalList directly by reconstructed name. The
                # direct fetch would accept a partially-deleted artifact (no
                # ownership part rows) or a hand-inserted same-name IntervalList;
                # read_artifact_removed_intervals validates the
                # ArtifactRemovedInterval part rows own the IntervalList and
                # raises otherwise. as_dict=True so a shared-group source returns
                # the per-nwb dict uniformly; select this recording's array (and
                # raise clearly if absent rather than feeding a dict to the mask).
                from spyglass.spikesorting.v2._artifact_intervals import (
                    read_artifact_removed_intervals,
                )

                intervals_by_nwb = read_artifact_removed_intervals(
                    {"artifact_detection_id": sel_row["artifact_detection_id"]},
                    as_dict=True,
                )
                if nwb_file_name not in intervals_by_nwb:
                    raise ValueError(
                        "Sorting.make_fetch: artifact-removed intervals for "
                        f"nwb_file_name={nwb_file_name!r} not found among "
                        f"{sorted(intervals_by_nwb)} for artifact_detection_id="
                        f"{sel_row['artifact_detection_id']!r}; the "
                        "ArtifactDetection may be partially deleted."
                    )
                obs_intervals = intervals_by_nwb[nwb_file_name]
            else:
                obs_intervals = None
        else:  # concatenated_recording
            # A concat source has no artifact pass: concat sorts reuse the
            # per-member Recording artifacts and observe the full recording.
            # ``insert_selection`` rejects a concat source carrying an
            # artifact_detection_id, but a direct insert of a
            # ConcatenatedRecordingSource + ArtifactDetectionSource pair can
            # bypass that. Re-assert here (the compute boundary) so the sort is
            # never run UNMASKED while a stray ArtifactDetectionSource row claims
            # an artifact pass -- raise rather than silently dropping the mask.
            if sel_row.get("artifact_detection_id") is not None:
                from spyglass.spikesorting.v2.exceptions import (
                    SchemaBypassError,
                )

                raise SchemaBypassError(
                    "Sorting.make_fetch: concat source for sorting_id="
                    f"{key.get('sorting_id')!r} carries artifact_detection_id="
                    f"{sel_row['artifact_detection_id']!r}, but a concat sort "
                    "has no artifact-detection pass (concat sorts reuse "
                    "per-member Recording artifacts). The ArtifactDetectionSource "
                    "part was inserted without SortingSelection.insert_selection "
                    "(schema bypass); the sort would otherwise run unmasked "
                    "while claiming artifact metadata. Re-create the selection "
                    "via insert_selection, or drop the stray "
                    "ArtifactDetectionSource row."
                )
            recording_id, nwb_file_name, preprocessing_params_name = (
                self._resolve_concat_anchor(source.key)
            )
            obs_intervals = None

        # Resolve the DISPLAY analyzer recipe from the source preprocessing
        # recipe (region) -- hippocampus -> the 0.5/0.5 row, cortex -> the
        # 1.0/2.0 row, any other recipe -> the wider cortex fallback. The
        # concat source resolves the SAME recipe from its single shared
        # preprocessing recipe. Resolve the params blob HERE (make_fetch is the
        # only stage allowed DB I/O); make_compute builds with it and
        # make_insert persists the name so every later rebuild reads it back
        # deterministically.
        display_waveform_params_name, _ = waveform_params_for_preprocessing(
            preprocessing_params_name
        )
        display_waveform_params = fetch_waveform_params(
            display_waveform_params_name
        )

        # Resolve + validate the sorter execution backend (local vs container)
        # here -- make_fetch is the only stage allowed DB I/O. make_compute
        # passes the resolved dict to the DB-free sorter dispatch.
        from spyglass.spikesorting.v2._params.sorter import (
            validate_execution_params,
        )

        execution_params = validate_execution_params(
            sorter_row.get("execution_params")
        )

        sort_group_id, electrode_by_id, region_by_electrode = (
            self._fetch_unit_electrode_metadata(recording_id, nwb_file_name)
        )

        return SortingFetched(
            source=source,
            recording_id=recording_id,
            sel_row=sel_row,
            sorter_row=sorter_row,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            display_waveform_params_name=display_waveform_params_name,
            display_waveform_params=display_waveform_params,
            execution_params=execution_params,
            sort_group_id=sort_group_id,
            electrode_by_id=electrode_by_id,
            region_by_electrode=region_by_electrode,
        )

    @staticmethod
    def _fetch_unit_electrode_metadata(recording_id, nwb_file_name):
        """DB reads for the per-unit Electrode FK + brain region (fetch stage).

        Resolved once here so ``make_compute`` builds the ``Sorting.Unit`` rows
        (and the matching NWB unit columns) with no DB I/O. ``electrode_by_id``
        comes from the unjoined ``SortGroupElectrode`` so it stays complete (the
        row-construction key set is unchanged); ``region_by_electrode`` is a
        best-effort ``electrode_id -> brain region`` map (an electrode without a
        region simply has no entry). Anchored to ``recording_id`` /
        ``nwb_file_name`` (the sort's own recording, or the first concat member).
        """
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )

        sort_group_id = int(
            (RecordingSelection & {"recording_id": recording_id}).fetch1(
                "sort_group_id"
            )
        )
        restriction = {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
        electrode_by_id = {
            int(row["electrode_id"]): row
            for row in (SortGroupV2.SortGroupElectrode & restriction).fetch(
                as_dict=True
            )
        }
        region_by_electrode = {
            int(row["electrode_id"]): str(row["region_name"])
            for row in (
                (SortGroupV2.SortGroupElectrode & restriction)
                * Electrode
                * BrainRegion
            ).fetch("electrode_id", "region_name", as_dict=True)
        }
        return sort_group_id, electrode_by_id, region_by_electrode

    @staticmethod
    def _first_concat_member(source_key):
        """Return the concat anchor member row and its preprocessing recipe.

        The anchor is the FIRST frozen member (by ``member_index``) from the
        ``ConcatenatedRecordingSelection.MemberSnapshot`` -- never the live
        ``SessionGroup.Member`` set, so a later group edit cannot re-point an
        existing concat sort's anchor. The frozen row carries the anchor's
        ``nwb_file_name`` (the analysis parent) and ``recording_id`` (the
        per-unit ``Electrode`` FK); the preprocessing recipe is the concat
        selection's single shared one.

        Parameters
        ----------
        source_key : dict
            ``{"concat_recording_id": ...}`` from ``resolve_source``.

        Returns
        -------
        tuple[dict, str]
            ``(first_member_snapshot_row, preprocessing_params_name)``.
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )

        concat_sel = (ConcatenatedRecordingSelection & source_key).fetch1()
        snapshot = (
            ConcatenatedRecordingSelection.MemberSnapshot & source_key
        ).fetch(as_dict=True, order_by="member_index", limit=1)
        if not snapshot:
            raise SchemaBypassError(
                "Sorting._first_concat_member: concat selection "
                f"{dict(source_key)} has no MemberSnapshot rows; the frozen "
                "member set is written by insert_selection. Drop the selection "
                "and re-insert via insert_selection."
            )
        return snapshot[0], concat_sel["preprocessing_params_name"]

    @staticmethod
    def _resolve_concat_anchor(source_key):
        """Resolve a concat source to its anchor recording / NWB / preprocessing.

        Reads the anchor ``recording_id`` straight from the frozen snapshot (no
        ``RecordingSelection`` join needed -- the snapshot froze the resolved
        ``recording_id`` when the concat id was minted). The full multi-session
        provenance stays queryable through
        ``ConcatenatedRecordingSelection.MemberSnapshot``.

        Parameters
        ----------
        source_key : dict
            ``{"concat_recording_id": ...}`` from ``resolve_source``.

        Returns
        -------
        tuple[uuid.UUID, str, str]
            ``(anchor_recording_id, anchor_nwb_file_name,
            preprocessing_params_name)``.
        """
        first_member, preprocessing_params_name = Sorting._first_concat_member(
            source_key
        )
        return (
            first_member["recording_id"],
            first_member["nwb_file_name"],
            preprocessing_params_name,
        )

    @staticmethod
    def resolve_anchor_nwb_file_name(key) -> str:
        """Return the analysis-NWB parent ``nwb_file_name`` for a sort.

        Source-agnostic: a single-recording sort anchors to its own
        ``RecordingSelection``; a concat sort anchors to the FIRST frozen
        ``MemberSnapshot`` member (the deterministic parent the per-unit Electrode
        FK and the curated/analyzer NWBs all use). Centralizes the
        unwrap-to-nwb dispatch that several reporting / curation accessors need,
        so the "which member is the anchor" decision lives in exactly one place.

        Parameters
        ----------
        key : dict
            Restriction carrying ``sorting_id``.

        Returns
        -------
        str
            The anchor session's ``nwb_file_name``.
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = SortingSelection.resolve_source(key)
        if source.kind == "recording":
            return (RecordingSelection & source.key).fetch1("nwb_file_name")
        # NWB only -- read it off the anchor member row; no recording_id join.
        return Sorting._first_concat_member(source.key)[0]["nwb_file_name"]

    def make_compute(
        self,
        key,
        source,
        recording_id,
        sel_row,
        sorter_row,
        nwb_file_name,
        obs_intervals,
        display_waveform_params_name,
        display_waveform_params,
        execution_params,
        sort_group_id,
        electrode_by_id,
        region_by_electrode,
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
            Resolved sort input source from ``make_fetch`` (selects whether the
            recording is loaded from ``Recording`` or ``ConcatenatedRecording``).
        recording_id : str
            The anchor ``recording_id`` from ``make_fetch`` (the sort's own
            recording, or the first concat member's), threaded forward so the
            ``Sorting.Unit`` Electrode FK resolves against the anchor member.
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
        execution_params : dict
            The validated sorter execution backend / container provenance from
            ``make_fetch``, passed to the DB-free sorter dispatch.

        Returns
        -------
        SortingComputed
            Carrier of the computed sorting, staged units NWB, analyzer
            folder, and lookups threaded into ``make_insert``.
        """
        # Load the sort input: a single-recording source reads the cached
        # Recording; a concat source reads the materialized ConcatenatedRecording
        # cache. ``recording_id`` is the anchor (threaded from make_fetch) used
        # for the per-unit Electrode FK, NOT necessarily the loaded recording's
        # own id. Concat sorts have no artifact pass, so the mask block is
        # skipped (obs_intervals is None there).
        if source.kind == "recording":
            recording = Recording().get_recording(
                {"recording_id": recording_id}
            )
        else:  # concatenated_recording
            recording = ConcatenatedRecording().get_recording(source.key)

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
        # Producer provenance (secondary, never identity). The effective seed
        # is resolved from the SAME per-row blob ``job_kwargs`` was, so it bottoms
        # out on ``_resolved_job_kwargs`` and equals the value the seed sites
        # consume below (``job_kwargs['random_seed']``) -- stored == used, not a
        # parallel computation that could drift.
        import spikeinterface as si

        # reject_ambient_seed: a seed-dependent sort's seed MUST live in the
        # SorterParameters row (part of sorter_params_name -> sorting_id), never
        # the ambient dj.config layer, or a later ambient-seed change would
        # silently reuse this sort under an unchanged id. The clusterless
        # thresholder is exempt ONLY on its deterministic paths -- its per-channel
        # MAD noise estimate (threshold_unit='mad', no explicit noise_levels)
        # samples random chunks and IS seed-dependent (see
        # _is_seed_dependent_sort), so that path is NOT exempt.
        effective_random_seed = resolve_effective_seed(
            sorter_row["job_kwargs"],
            reject_ambient_seed=SorterParameters._is_seed_dependent_sort(
                sorter, sorter_params
            ),
        )
        spikeinterface_version = si.__version__
        sorter_version = sorter_distribution_version(sorter)

        sorting_obj = self._run_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=key["sorting_id"],
            job_kwargs=job_kwargs,
            execution_params=execution_params,
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
        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
            analyzer_path,
            publish_analyzer_atomically,
        )

        analyzer_folder = analyzer_path(
            key["sorting_id"], display_waveform_params_name
        )
        # Publish the analyzer atomically: build into a private temp folder,
        # then move it into the canonical slot under the per-sort lock, rather
        # than writing ``overwrite=True`` straight into the live folder. The
        # low-level ``_build_analyzer`` still builds wherever it is told (the
        # temp here); only the canonical install goes through the publisher.
        with analyzer_cache_lock(key["sorting_id"]):
            publish_analyzer_atomically(
                analyzer_folder,
                lambda temp_folder: self._build_analyzer(
                    sorting=sorting_obj,
                    recording=recording,
                    key=key,
                    sorter_row=sorter_row,
                    job_kwargs=job_kwargs,
                    analyzer_folder=temp_folder,
                    waveform_params=display_waveform_params,
                ),
            )
        # Compute the per-unit rows ONCE here (from the analyzer just built) and
        # reuse them for BOTH the NWB unit columns and the Sorting.Unit insert in
        # make_insert -- so the file and the DB cannot drift, and the peak
        # channel/amplitude are not computed twice.
        unit_rows = self._build_unit_rows_from_analyzer(
            sorting=sorting_obj,
            analyzer_folder=analyzer_folder,
            sorter_row=sorter_row,
            electrode_by_id=electrode_by_id,
            sort_group_id=sort_group_id,
            nwb_file_name=nwb_file_name,
            key=key,
        )
        unit_metadata = {
            int(row["unit_id"]): {
                "peak_amplitude_uv": row["peak_amplitude_uv"],
                "peak_electrode_id": int(row["electrode_id"]),
                "n_spikes": int(row["n_spikes"]),
                "brain_region": region_by_electrode.get(
                    int(row["electrode_id"])
                ),
            }
            for row in unit_rows
        }
        concat_recording_id = (
            source.key["concat_recording_id"]
            if source.kind == "concatenated_recording"
            else None
        )
        source_provenance = {
            "recording_id": recording_id,
            "concat_recording_id": concat_recording_id,
            "sorter": sorter_row["sorter"],
            "sorter_params_name": sorter_row["sorter_params_name"],
            "sorter_params": sorter_row["params"],
            # The execution backend (container / engine) can change the sorter
            # output, so it is part of the named-parameter-row provenance.
            "execution_params": execution_params,
            "artifact_detection_id": sel_row.get("artifact_detection_id"),
            "display_waveform_params_name": display_waveform_params_name,
            "effective_random_seed": effective_random_seed,
            "spikeinterface_version": spikeinterface_version,
            "sorter_version": sorter_version,
        }
        analysis_file_name, units_object_id = self._stage_sorting_artifact(
            sorting=sorting_obj,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )

        return SortingComputed(
            sorting_obj=sorting_obj,
            analysis_file_name=analysis_file_name,
            units_object_id=units_object_id,
            analyzer_folder=analyzer_folder,
            nwb_file_name=nwb_file_name,
            display_waveform_params_name=display_waveform_params_name,
            effective_random_seed=effective_random_seed,
            spikeinterface_version=spikeinterface_version,
            sorter_version=sorter_version,
            unit_rows=unit_rows,
        )

    def make_insert(
        self,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        analyzer_folder,
        nwb_file_name,
        display_waveform_params_name,
        effective_random_seed,
        spikeinterface_version,
        sorter_version,
        unit_rows,
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
        nwb_file_name : str
            Source NWB file backing the recording selection.
        display_waveform_params_name : str
            The resolved DISPLAY recipe, persisted on the master row so every
            later rebuild reads it back deterministically.
        effective_random_seed : int
            Seed the sort actually used; secondary provenance, not identity.
        spikeinterface_version : str
            ``spikeinterface.__version__`` at sort time.
        sorter_version : str or None
            Sorter package distribution version, ``None`` for in-process sorters.

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
                nwb_file_name=nwb_file_name,
                display_waveform_params_name=display_waveform_params_name,
                effective_random_seed=effective_random_seed,
                spikeinterface_version=spikeinterface_version,
                sorter_version=sorter_version,
                unit_rows=unit_rows,
            )
        except Exception:
            # Failure-mode B: registration failed after a successful
            # make_compute -- remove the attempt-owned, unregistered units NWB
            # before propagating. The analyzer folder is shared by sorting_id and
            # regeneratable, and a concurrent worker may have just published it
            # for the same sorting_id (without yet committing its row), so it is
            # NEVER deleted here -- an analyzer with no committed row is a
            # disk-side orphan reclaimed by find_orphaned_analyzer_folders, and
            # get_analyzer self-heals a missing one.
            self._cleanup_staged_units_nwb(
                analysis_file_name=analysis_file_name
            )
            raise

    def _stage_sorting_artifact(
        self,
        *,
        sorting,
        recording,
        nwb_file_name,
        obs_intervals,
        unit_metadata=None,
        source_provenance=None,
    ):
        """Stage the units NWB; return ``(analysis_file_name, units_object_id)``.

        ``_write_units_nwb`` self-cleans its own staged file on a write failure
        (mode A) before propagating, and the analyzer folder is deliberately left
        for orphan-sweep / ``get_analyzer`` self-heal rather than eagerly deleted
        (it is shared by ``sorting_id`` and may belong to a concurrent worker),
        so no extra failure cleanup is needed here. ``unit_metadata`` /
        ``source_provenance`` are the compute-once per-unit columns + source
        header embedded in the NWB.
        """
        return self._write_units_nwb(
            sorting=sorting,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )

    def _insert_sorting_rows_transaction(
        self,
        *,
        key,
        sorting_obj,
        analysis_file_name,
        units_object_id,
        nwb_file_name,
        display_waveform_params_name,
        effective_random_seed,
        spikeinterface_version,
        sorter_version,
        unit_rows,
    ):
        """Register the AnalysisNwbfile + master + Unit rows atomically.

        Runs the ``transaction_or_noop`` block: registers the staged
        AnalysisNwbfile, inserts the Sorting master, and populates the Unit
        part rows (built once in ``make_compute``) INSIDE the transaction so
        they commit atomically with the master (splitting them across stages is
        forbidden). ``transaction_or_noop`` is a no-op when the framework
        transaction is already active (the tri-part dispatch path); it is kept
        so an out-of-populate caller still gets atomic registration.
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
                    "effective_random_seed": effective_random_seed,
                    "spikeinterface_version": spikeinterface_version,
                    "sorter_version": sorter_version,
                }
            )
            self._populate_unit_part(unit_rows)

    @staticmethod
    def _cleanup_staged_units_nwb(*, analysis_file_name):
        """Best-effort removal of the attempt-owned units NWB after failure.

        Removes ONLY the staged units NWB (failure-mode B: registration failed
        after ``make_compute`` staged the file) -- it is per-attempt and
        content-addressed, owned by this populate attempt. The analyzer folder is
        deliberately NOT touched: it is shared by ``sorting_id`` and
        regeneratable, and a concurrent worker may have just published it for the
        same ``sorting_id`` (without yet committing its row), so eagerly deleting
        it could destroy a peer's analyzer. An analyzer with no committed
        ``Sorting`` row is a disk-side orphan reclaimed by
        ``find_orphaned_analyzer_folders``; ``get_analyzer`` self-heals a missing
        one. Best-effort: a cleanup failure is logged, never raised, so it cannot
        mask the original error. DataJoint cannot roll back this filesystem side
        effect, so the caller invokes this in its ``except`` before re-raising.
        """
        import pathlib

        try:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            pathlib.Path(abs_path).unlink(missing_ok=True)
        except Exception as cleanup_exc:  # pragma: no cover -- defensive
            logger.error(
                "Sorting._cleanup_staged_units_nwb: failed to clean up staged "
                f"units NWB {analysis_file_name!r}: {cleanup_exc!r}"
            )

    # ---- Accessors -------------------------------------------------------

    def get_sorting(
        self, key: dict, as_dataframe: bool = False
    ) -> "si.BaseSorting | pd.DataFrame":
        """Return the SpikeInterface BaseSorting backed by the units NWB.

        Spike times are persisted by ``_write_units_nwb`` in two forms:
        absolute ``spike_times`` for NWB interoperability, and Spyglass's
        ``spike_sample_index`` sidecar for efficient frame-based readback. New
        files reconstruct directly from ``spike_sample_index``; older/manual
        files without that column fall back to the previous absolute-time
        search against the recording timestamps.

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
        # Resolve the recording row backing this sort's absolute-time readback:
        # a single-recording sort reads its Recording row, a concat-backed sort
        # reads its ConcatenatedRecording row. Both carry analysis_file_name /
        # electrical_series_path / sampling_frequency, which is all the
        # absolute-time -> frame mapping below needs.
        source = SortingSelection.resolve_source(key)
        if source.kind == "recording":
            rec_row = (
                Recording & {"recording_id": source.key["recording_id"]}
            ).fetch1()
        else:  # concatenated_recording
            rec_row = (ConcatenatedRecording & source.key).fetch1()
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

        if as_dataframe:
            abs_times = read_units_abs_spike_times(abs_path)
            return abs_spike_times_dataframe(abs_times)
        sample_indices = read_units_spike_sample_indices(abs_path)
        if sample_indices is not None:
            return numpysorting_from_sample_indices(sample_indices, fs)
        abs_times = read_units_abs_spike_times(abs_path)
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
        self,
        key: dict,
        waveform_params_name: str | None = None,
        *,
        rebuild: bool = True,
    ) -> "si.SortingAnalyzer":
        """Return the SortingAnalyzer; rebuild on missing or invalid folder.

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
            separation metrics in ``CurationEvaluation``) passes it explicitly.
            A missing or invalid folder is rebuilt for whichever recipe is
            requested (unless ``rebuild=False``).
        rebuild : bool, optional
            If ``True`` (default), a missing or invalid analyzer folder is
            rebuilt in place (the self-healing cache). If ``False``, a missing
            folder raises ``AnalyzerFolderMissingError`` and an unloadable folder
            raises ``AnalyzerFolderInvalidError`` instead -- the recompute audit
            uses this to OBSERVE a missing/reclaimed/corrupt analyzer rather
            than silently rebuild-then-hash it.

        Returns
        -------
        si.SortingAnalyzer
            The loaded ``SortingAnalyzer`` for the sort, rebuilt in
            place if its folder was missing or invalid (when ``rebuild=True``).

        Raises
        ------
        AnalyzerFolderMissingError
            If ``rebuild=False`` and the analyzer folder is absent on disk.
        AnalyzerFolderInvalidError
            If ``rebuild=False`` and the analyzer folder exists but cannot be
            loaded.
        """
        return load_or_rebuild_analyzer(
            self,
            key,
            waveform_params_name=waveform_params_name,
            rebuild=rebuild,
        )

    def add_extensions(
        self, key: dict, extensions: list[str], **kwargs
    ) -> list[str]:
        """Add SortingAnalyzer extensions in place; return the ones computed.

        Convenience for callers (and ``CurationEvaluation``) that need
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
        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_lock,
        )
        from spyglass.spikesorting.v2._sorting_analyzer import (
            ensure_extensions,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        sorting_id = (self & key).fetch1("sorting_id")
        sorter_job_kwargs = (
            SorterParameters & (SortingSelection & key)
        ).fetch1("job_kwargs")
        resolved = _resolved_job_kwargs(sorter_job_kwargs)
        resolved.update(kwargs)
        # Hold the per-sort lock across BOTH the load and the in-place extension
        # compute: ``ensure_extensions`` persists into the canonical analyzer
        # folder, so loading under the lock and releasing it before the compute
        # would leave the mutation unguarded. The lock is reentrant, so the
        # nested ``get_analyzer`` load does not self-deadlock.
        with analyzer_cache_lock(sorting_id):
            analyzer = self.get_analyzer(key)
            return ensure_extensions(analyzer, extensions, job_kwargs=resolved)

    # ---- visualization / export delegates (see v2.visualization facade) ---

    def plot_summary(
        self, key, *, compute_missing=False, backend=None, **kwargs
    ):
        """Delegate to ``visualization.plot_sorting_summary`` for this sort.

        A local-discoverability one-liner; the display-analyzer routing and
        extension policy live in the ``v2.visualization`` facade, which the
        notebook/docs teach as the primary surface. ``backend`` is required (SI's
        ``SortingSummaryWidget`` has no matplotlib backend); see the facade.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_sorting_summary(
            key, compute_missing=compute_missing, backend=backend, **kwargs
        )

    def plot_unit_summary(
        self,
        key,
        unit_id,
        *,
        compute_missing=False,
        backend="matplotlib",
        **kwargs,
    ):
        """Delegate to ``visualization.plot_unit_summary`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_unit_summary(
            key,
            unit_id,
            compute_missing=compute_missing,
            backend=backend,
            **kwargs,
        )

    def plot_waveforms(
        self, key, unit_ids=None, *, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_waveforms`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_waveforms(
            key, unit_ids=unit_ids, backend=backend, **kwargs
        )

    def plot_spikes_on_traces(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_spikes_on_traces`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_spikes_on_traces(
            key, compute_missing=compute_missing, backend=backend, **kwargs
        )

    def plot_unit_locations(
        self, key, *, compute_missing=False, backend="matplotlib", **kwargs
    ):
        """Delegate to ``visualization.plot_unit_locations`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_unit_locations(
            key, compute_missing=compute_missing, backend=backend, **kwargs
        )

    def export_si_report(
        self, key, output_folder, *, compute_missing=False, **kwargs
    ):
        """Delegate to ``visualization.export_si_report`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.export_si_report(
            key, output_folder, compute_missing=compute_missing, **kwargs
        )

    def export_to_phy(self, key, output_folder, **kwargs):
        """Delegate to ``visualization.export_to_phy`` for this sort."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.export_to_phy(key, output_folder, **kwargs)

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
            analyzer_cache_lock,
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
                # Remove the regeneratable cache UNDER the per-sort lock so a
                # concurrent reader / rebuild never races the rmtree.
                with analyzer_cache_lock(row["sorting_id"]):
                    remove_analyzer_cache(row["sorting_id"], missing_ok=True)

    @classmethod
    def find_orphaned_analyzer_folders(cls, *, dry_run: bool = True) -> dict:
        """Audit 5-50 GB analyzer-folder disk leaks; never auto-delete DB rows.

        Each populated sort writes a 5-50 GB ``analyzer_folder`` of
        regeneratable scratch outside the DataJoint-tracked store. The
        ``Sorting.delete`` override cleans it up on row delete, but an external
        path that bypasses the override (raw SQL delete, scripted
        ``dj.Table.connection.query``) leaks the folder. This periodic audit
        mirrors ``prune_orphaned_selections`` and reports three classes:

        - **DB-side orphan**: a ``Sorting`` row whose computed analyzer cache
          folder (``analyzer_path(sorting_id, display_waveform_params_name)``)
          no longer exists on disk (the
          regeneratable scratch was removed out of band). Reported only --
          deleting the *row* is a destructive DB operation the human decides on
          (per the Spyglass destructive-op contract); this method NEVER
          auto-deletes a row.
        - **Reclaimed**: a missing analyzer folder with a
          ``SortingAnalyzerRecompute.deleted=1`` audit trail. This is expected
          storage reclamation, not an unexpected DB-side orphan.
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
            "disk_side": [folder_path_str, ...],
            "reclaimed": [{"sorting_id", "computed_analyzer_path"}, ...]}``.
            ``computed_analyzer_path`` is resolved from ``sorting_id`` (there is
            no stored ``analyzer_folder`` column).
        """
        import shutil

        import datajoint as dj

        from spyglass.spikesorting.v2._analyzer_cache import (
            analyzer_cache_root,
            analyzer_path,
            classify_orphaned_analyzer_folders,
            is_canonical_analyzer_folder_name,
        )

        # Gather the DB / filesystem facts, then classify (the orphan set logic
        # is pure and DB-free in ``classify_orphaned_analyzer_folders``). Each
        # sort has one DISPLAY analyzer folder, named
        # {sorting_id}__{display_waveform_params_name}.zarr; the n_units==0
        # carve-out excludes legitimately folder-less rows from the DB-side set.
        # A sort's stored display recipe is the folder we expect on disk; a
        # {sid}__{other}.zarr folder (e.g. a stale recipe) is therefore a
        # disk-side orphan -- UNLESS it is a whitened metric recipe referenced
        # by a CurationEvaluationSelection (built on demand for PC/NN metrics),
        # which is retained below.
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
        # A metric (whitened) analyzer folder referenced by a PC-requesting
        # curation selection is in active use (its PC/NN metrics were computed
        # from it), so it is NOT a disk-side orphan even though it is not a
        # sort's display recipe. Only selections that actually build the metric
        # analyzer (CurationEvaluationSelection.pc_requesting() -- the same
        # source the recompute key_source uses) are retained, so a skip-PC
        # selection's recipe is not (a stale folder for it stays a cleanable
        # orphan). Lazily imported to avoid a metric_curation <-> sorting cycle.
        from spyglass.spikesorting.v2.metric_curation import (
            CurationEvaluationSelection,
        )

        referenced_paths.update(
            str(
                analyzer_path(r["sorting_id"], r["metric_waveform_params_name"])
            )
            for r in CurationEvaluationSelection.pc_requesting().fetch(
                "sorting_id", "metric_waveform_params_name", as_dict=True
            )
        )
        # A missing display analyzer with a recompute deleted=1 row is expected
        # storage reclamation, not an unexpected DB-side orphan. Import lazily to
        # avoid a sorting <-> recompute import cycle at module load.
        from spyglass.spikesorting.v2.recompute import SortingAnalyzerRecompute

        reclaimed_paths = {
            str(analyzer_path(r["sorting_id"], r["waveform_params_name"]))
            for r in (SortingAnalyzerRecompute & "deleted=1").fetch(
                "sorting_id", "waveform_params_name", as_dict=True
            )
        }
        analyzer_root = analyzer_cache_root()
        # Only canonical ``{sorting_id}__{recipe}.zarr`` directories are deletion
        # candidates. Skipping hidden directories excludes the atomic publisher's
        # ``.publish`` staging scratch; the canonical-name filter additionally
        # refuses to treat any unrelated subdirectory as an orphan, so a
        # misconfigured (e.g. shared, non-dedicated) analyzer root cannot lead
        # the sweep to delete non-analyzer folders.
        disk_dir_paths = (
            [
                str(c)
                for c in sorted(analyzer_root.iterdir())
                if c.is_dir()
                and not c.name.startswith(".")
                and is_canonical_analyzer_folder_name(c.name)
            ]
            if analyzer_root.exists()
            else []
        )
        classification = classify_orphaned_analyzer_folders(
            units_bearing,
            referenced_paths,
            disk_dir_paths,
            reclaimed_paths=reclaimed_paths,
        )
        db_side = classification["db_side"]
        disk_side = classification["disk_side"]
        reclaimed = classification["reclaimed"]

        logger.info(
            "Sorting.find_orphaned_analyzer_folders: "
            f"{len(db_side)} DB-side orphan(s) (row present, folder missing), "
            f"{len(disk_side)} disk-side orphan(s) (folder present, no row), "
            f"{len(reclaimed)} reclaimed folder(s) (missing with deleted=1)."
        )
        for row in db_side:
            logger.info(
                "  DB-side orphan: sorting_id=%s computed_analyzer_path=%s",
                row["sorting_id"],
                row["computed_analyzer_path"],
            )
        for row in reclaimed:
            logger.info(
                "  reclaimed analyzer: sorting_id=%s computed_analyzer_path=%s",
                row["sorting_id"],
                row["computed_analyzer_path"],
            )
        for folder in disk_side:
            logger.info("  disk-side orphan: %s", folder)

        if dry_run or not disk_side:
            return {
                "db_side": db_side,
                "disk_side": disk_side,
                "reclaimed": reclaimed,
            }

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
        return {
            "db_side": db_side,
            "disk_side": disk_side,
            "reclaimed": reclaimed,
        }

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
                    "to return anchor-member regions, or use "
                    "TrackedUnit.get_unit_brain_regions for per-session "
                    "regions across matched sessions."
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
        execution_params=None,
    ):
        """Dispatch sort execution; clusterless_thresholder vs SI sorters.

        Clusterless is a Spyglass-specific peak-detection path with no
        SI scratch directory, external whitening, or container backend, so
        ``execution_params`` does not apply to it. SI sorters get per-sort
        scratch, external whitening, and the tracked container-execution
        backend. The two paths share nothing but the signature; dispatch routes
        each to its own helper.
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
            execution_params=execution_params,
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
        execution_params=None,
    ):
        """Run an SI registered sorter under a managed scratch dir.

        Thin delegator to :func:`._sorting_dispatch.run_si_sorter`; kept as
        a ``Sorting`` staticmethod because ``_run_sorter`` dispatches to
        ``Sorting._run_si_sorter`` and the v2 tests call it directly. The
        managed ``TemporaryDirectory`` scratch, external float64 whitening,
        scoped ``np.Inf`` patch, container-execution backend (local vs
        Docker/Singularity + the MATLAB-sorter container policy), and the
        global-job-kwargs save/restore live in the service module.
        ``execution_params`` defaults to ``None`` (resolved to local) so direct
        test callers and the clusterless path stay unchanged.
        """
        return run_si_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=sorting_id,
            job_kwargs=job_kwargs,
            execution_params=execution_params,
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
    def _write_units_nwb(
        sorting,
        recording,
        nwb_file_name,
        obs_intervals=None,
        *,
        unit_metadata=None,
        source_provenance=None,
    ):
        """Write a fresh AnalysisNwbfile containing only the v2 Units table.

        Thin delegator to :func:`._units_nwb.write_sorting_units_nwb`;
        kept as a ``Sorting`` staticmethod because ``make_insert`` calls
        ``self._write_units_nwb(...)`` and the v2 tests both monkeypatch
        ``Sorting._write_units_nwb`` (the Mode-A analyzer-cleanup audit)
        and call it directly (the zero-unit guard test). The NWB staging
        IO -- absolute-timeline spike times, the ``obs_intervals`` +
        ``curation_label`` columns, the per-unit metadata + source-provenance
        scratch, and the zero-unit empty-Units guard -- lives in the service
        module.
        """
        return write_sorting_units_nwb(
            sorting=sorting,
            recording=recording,
            nwb_file_name=nwb_file_name,
            obs_intervals=obs_intervals,
            unit_metadata=unit_metadata,
            source_provenance=source_provenance,
        )

    @staticmethod
    def _build_unit_rows_from_analyzer(
        *,
        sorting,
        analyzer_folder,
        sorter_row,
        electrode_by_id,
        sort_group_id,
        nwb_file_name,
        key,
    ):
        """Build the ``Sorting.Unit`` rows from the freshly built analyzer.

        Run ONCE in ``make_compute`` (DB-free): the analyzer folder
        ``_build_analyzer`` just wrote is loaded here, each unit's peak channel
        + amplitude is resolved under the sorter's configured detection polarity
        (clusterless ``peak_sign`` / MountainSort ``detect_sign``, not SI's
        ``"neg"`` default, so a positive-going detection attributes each unit to
        its true peak channel), and the rows are assembled by
        ``build_sorting_unit_rows`` (which raises on a sort-group/recording
        channel-id mismatch). The resulting rows are reused for BOTH the NWB
        unit columns and the ``Sorting.Unit`` insert, so the file and the DB
        cannot drift.

        Empty for a zero-unit sort: ``_build_analyzer`` skips the
        ``create_sorting_analyzer`` call when ``sorting.get_num_units() == 0``
        (SI's ``estimate_sparsity`` crashes on empty sortings), so the analyzer
        folder does not exist; there is nothing to load or insert.
        """
        if sorting.get_num_units() == 0:
            return []

        import spikeinterface as si
        from spikeinterface.core import template_tools

        from spyglass.spikesorting.v2.utils import resolve_peak_sign

        analyzer = si.load_sorting_analyzer(analyzer_folder)
        peak_sign = resolve_peak_sign(sorter_row["params"])
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
        n_spikes_by_unit = {
            unit_id: int(len(sorting.get_unit_spike_train(unit_id=unit_id)))
            for unit_id in sorting.unit_ids
        }
        return build_sorting_unit_rows(
            unit_ids=sorting.unit_ids,
            peak_channels=peak_channels,
            peak_amplitudes=peak_amplitudes,
            n_spikes_by_unit=n_spikes_by_unit,
            electrode_by_id=electrode_by_id,
            key=key,
            sort_group_id=sort_group_id,
            nwb_file_name=nwb_file_name,
        )

    @staticmethod
    def _populate_unit_part(unit_rows):
        """Insert the pre-built ``Sorting.Unit`` rows.

        The rows are built ONCE in ``make_compute``
        (:meth:`_build_unit_rows_from_analyzer`) and threaded through, so this
        is a pure insert with no analyzer load or DB read. An empty list (a
        zero-unit sort) is a no-op.
        """
        Sorting.Unit.insert(unit_rows)
