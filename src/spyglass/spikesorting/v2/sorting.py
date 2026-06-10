"""Spike sorting and per-unit brain-region metadata.

Tables (all final-shape under the zero-migration policy):
    SorterParameters          -- Per-sorter Pydantic-validated params.
    SortingSelection          -- Source-polymorphic sorting request.
        .RecordingSource          -- single-session source (default).
        .ConcatenatedRecordingSource -- concat source; the runtime helper
                                       rejects this source today, but the
                                       FK is real so the schema is stable.
    Sorting (+ Unit)          -- Sorted units NWB + SortingAnalyzer folder.

``insert1`` on ``SorterParameters`` is live and dispatches to the
per-sorter Pydantic schema via ``_get_sorter_schema``; ``make`` /
``insert_selection`` / accessor methods are forward-declared stubs that
raise ``NotImplementedError`` until the matching runtime change lands.
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
from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
from spyglass.spikesorting.v2.artifact import ArtifactDetection  # noqa: F401
from spyglass.spikesorting.v2.recording import Recording  # noqa: F401
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,  # noqa: F401
)
from spyglass.spikesorting.v2.utils import (
    SourceResolution,
    _assert_noise_levels_length,
    _assert_v2_db_safe,
    _validate_params,
    find_orphaned_masters,
    transaction_or_noop,
    unit_brain_region_df,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si


class SortingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Sorting.make_fetch`."""

    source: SourceResolution
    sel_row: dict
    sorter_row: dict
    nwb_file_name: str
    obs_intervals: np.ndarray | None


class SortingComputed(NamedTuple):
    """Outputs of :meth:`Sorting.make_compute`."""

    sorting_obj: "si.BaseSorting"
    analysis_file_name: str
    units_object_id: str
    analyzer_folder: Path
    recording_id: str
    nwb_file_name: str


def _clusterless_noise_levels(
    noise_levels: list[float] | None, threshold_unit: str
) -> list[float] | None:
    """Resolve clusterless-thresholder ``noise_levels`` precedence.

    An explicit ``noise_levels`` always wins. Otherwise ``threshold_unit``
    governs how ``detect_threshold`` is interpreted:

    * ``"uv"`` -> ``[1.0]``; the caller (``_run_clusterless_thresholder``)
      scales the recording to microvolts (``scale_to_uV``) before
      ``detect_peaks``, so ``detect_threshold`` is a genuine microvolt
      threshold. (For Frank-lab data gain==1 uV/count, so this matches the
      old raw-count behavior; for non-unity-gain rigs it is the fix.)
    * ``"mad"`` -> ``None`` so SpikeInterface estimates per-channel MAD
      and ``detect_threshold`` is a MAD multiplier (scale-relative, so the
      recording is NOT uV-scaled on this path).
    """
    if noise_levels is not None:
        return noise_levels
    return [1.0] if threshold_unit == "uv" else None


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
    fallback dispatch for non-default sorters, preserving v1's "try any
    installed sorter" escape hatch.
    """

    definition = """
    sorter: varchar(64)
    sorter_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=0: int
    job_kwargs=null: blob
    """

    def insert1(self, row, **kwargs):
        # Delegate to ``insert`` so one validated path serves both.
        self.insert([row], **kwargs)

    def insert(self, rows, **kwargs):
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) before it lands, dispatching the Pydantic
        # schema per ``sorter``. Two per-row guards run after that:
        #
        # 1. Sorter-name typo guard. ``_get_sorter_schema`` falls back to
        #    the permissive ``GenericSorterParamsSchema`` for any unknown
        #    sorter (the v1 "try any installed SI sorter" escape hatch), so a
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

        from spyglass.spikesorting.v2._params.sorter import _SORTER_SCHEMAS

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
            if int(row.get("params_schema_version", 0)) == 0:
                row["params_schema_version"] = int(
                    row["params"]["schema_version"]
                )

        super().insert(
            validate_lookup_rows(
                rows,
                self.heading.names,
                schema_for=lambda row: _get_sorter_schema(row["sorter"]),
                table_name="SorterParameters",
                per_row_hook=_check_sorter_and_backfill_version,
            ),
            **kwargs,
        )

    _DEFAULT_CONTENTS: tuple = (
        (
            "mountainsort4",
            "franklab_tetrode_hippocampus_30kHz_ms4",
            _validate_params(
                _get_sorter_schema("mountainsort4"),
                {"freq_min": 600.0, "freq_max": 6000.0},
            ),
            1,
            None,
        ),
        (
            # v1-parity port of the ``franklab_probe_ctx_30KHz`` row at
            # ``v1/sorting.py:158-159`` (cortex-probe MS4 preset).
            # Cortex probes have lower spike-band content than
            # tetrodes; the ``freq_min=300`` floor catches more of
            # the spectrum than the tetrode preset's ``freq_min=600``.
            # Name normalized to v2's lowercase-k convention.
            "mountainsort4",
            "franklab_probe_ctx_30kHz_ms4",
            _validate_params(
                _get_sorter_schema("mountainsort4"),
                {"freq_min": 300.0, "freq_max": 6000.0},
            ),
            1,
            None,
        ),
        (
            # Back-compat alias for v1's bare ``franklab_tetrode_hippocampus
            # _30KHz`` row name at ``v1/sorting.py:158`` (uppercase K, no
            # ``_ms4`` suffix). v2 normalized the name to lowercase-k +
            # ``_ms4`` (to disambiguate the MS5 sibling), which silently
            # broke ``(SorterParameters & {"sorter_params_name":
            # "franklab_tetrode_hippocampus_30KHz"})`` for ported v1 code.
            # Carries the IDENTICAL params blob as the ``_ms4`` row above
            # so both names resolve to the same validated parameters. This
            # is a one-release shim: a future v2.x release drops it after
            # the CHANGELOG migration window passes.
            "mountainsort4",
            "franklab_tetrode_hippocampus_30KHz",
            _validate_params(
                _get_sorter_schema("mountainsort4"),
                {"freq_min": 600.0, "freq_max": 6000.0},
            ),
            1,
            None,
        ),
        (
            # Back-compat alias for v1's bare ``franklab_probe_ctx_30KHz``
            # row name at ``v1/sorting.py:163``. Same identical-params,
            # one-release-shim rationale as the tetrode alias above.
            "mountainsort4",
            "franklab_probe_ctx_30KHz",
            _validate_params(
                _get_sorter_schema("mountainsort4"),
                {"freq_min": 300.0, "freq_max": 6000.0},
            ),
            1,
            None,
        ),
        (
            "mountainsort5",
            "franklab_tetrode_hippocampus_30kHz_ms5",
            _validate_params(_get_sorter_schema("mountainsort5"), {}),
            1,
            None,
        ),
        (
            "kilosort4",
            "franklab_neuropixels_default",
            _validate_params(_get_sorter_schema("kilosort4"), {}),
            1,
            None,
        ),
        (
            "spykingcircus2",
            "default",
            _validate_params(_get_sorter_schema("spykingcircus2"), {}),
            1,
            None,
        ),
        (
            "tridesclous2",
            "default",
            _validate_params(_get_sorter_schema("tridesclous2"), {}),
            1,
            None,
        ),
        (
            "clusterless_thresholder",
            "default",
            _validate_params(
                _get_sorter_schema("clusterless_thresholder"),
                # ``threshold_unit="uv"`` makes the shipped
                # ``detect_threshold=100`` a TRUE 100 microvolt threshold:
                # the runtime scales the recording to uV (via the stored
                # NWB gain) before detection, rather than treating it as a
                # MAD multiplier. (For Frank-lab data gain==1 uV/count so
                # 100 "uv" == 100 counts == 100 uV either way.) This
                # honors the label v1 used at
                # ``src/spyglass/spikesorting/v1/sorting.py:177`` but never
                # enforced (v1 thresholded in raw counts). The
                # explicit ``noise_levels=[1.0]`` is the equivalent
                # advanced override and is kept as a belt-and-suspenders
                # regression guard against the 1,400x noise_levels
                # divergence; the runtime uses it verbatim (explicit
                # noise_levels take precedence over ``threshold_unit``).
                # The smoke / synthetic-fixture rows leave both at the
                # defaults (``threshold_unit="mad"``, no noise_levels)
                # so SI computes per-channel MAD and the threshold
                # tracks the recording's noise floor.
                {"threshold_unit": "uv", "noise_levels": [1.0]},
            ),
            # ClusterlessThresholderSchema is at schema_version=4:
            # v2 dropped ``outputs`` / ``random_chunk_kwargs``;
            # v3 made ``noise_levels`` optional (None -> SI MAD);
            # v4 added ``threshold_unit``.
            4,
            None,
        ),
    )

    # Sorter names in ``_DEFAULT_CONTENTS`` that are NOT SpikeInterface
    # registered sorters and so must never be gated on
    # ``sis.installed_sorters()``. ``clusterless_thresholder`` is a
    # Spyglass-internal peak detector built on ``detect_peaks``.
    _NON_SI_SORTERS: frozenset[str] = frozenset({"clusterless_thresholder"})

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        The default-content catalog mirrors the designs.md
        ``SorterParameters`` section and includes MS4, MS5, KS4, SC2,
        TDC2, and clusterless_thresholder. Rows whose SpikeInterface
        sorter is NOT in ``spikeinterface.sorters.installed_sorters()``
        are skipped (logged at INFO), mirroring v1's availability gate
        at ``v1/sorting.py:184-189`` -- otherwise a user who inserts an
        uninstalled sorter's default row and then populates ``Sorting``
        hits an unhelpful "sorter not registered" error from SI. MS4 and
        KS4 are the common uninstalled cases (their Python wrappers exist
        even when the runtime/binary is absent, so ``available_sorters``
        is too lax -- ``installed_sorters`` is the right gate).

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
          ``insert_default`` uses (see ``v1/sorting.py:184-189`` and the
          install-gate rationale at :meth:`insert_default`). v1 auto-
          inserted for every ``available_sorters()`` entry, but
          ``get_default_sorter_params`` succeeds for wrapper-only sorters
          whose binary is absent (e.g. ``kilosort2_5``, ``ironclust``), so
          enumerating ``available_sorters()`` alone would ship rows that
          fail at ``Sorting.populate`` time with an unhelpful "sorter not
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
            rows.append((sorter, "default", validated, 1, None))
        if skipped_not_installed:
            logger.info(
                "insert_default_legacy_si_sorters: skipping "
                f"{sorted(skipped_not_installed)} -- available in "
                "SpikeInterface but not in installed_sorters() on this "
                "platform (a 'default' row would fail at populate time)."
            )
        cls.insert(rows, skip_duplicates=True)


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` or ``ConcatenatedRecordingSource`` exists for
    each selection row. The runtime helper today rejects the concat
    path with a clear "not implemented yet" error; the schema is final
    so the validator can be relaxed without a migration once the concat
    materializer lands.

    Whether an artifact pass was applied is recorded by the presence or
    absence of an ``ArtifactSource`` part row (zero-or-one), NOT by a
    nullable FK on the master. A nullable FK conflates "no artifact
    pass" with "match anything" in a restriction and forces every reader
    to special-case ``None``; the part row makes "no artifact pass" a
    plain "no ``ArtifactSource`` row" that is queryable, joinable, and
    impossible to alias. ``ArtifactSource`` is independent of the
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
        definition = """
        -> master
        ---
        -> Recording
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> ConcatenatedRecording
        """

    class ArtifactSource(SpyglassMixinPart):
        """Optional artifact pass for a sorting selection (zero-or-one).

        Present iff an artifact detection was configured for the sort;
        absent means "no artifact pass." Deliberately separate from the
        recording-source parts so ``resolve_source``'s "exactly one
        recording source" invariant is unaffected -- read it through
        :meth:`SortingSelection.resolve_artifact`.
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
        ``artifact_id`` is optional: when supplied (non-None), an
        ``ArtifactSource`` part row is inserted recording the artifact
        pass; when omitted/None, no ``ArtifactSource`` row is created.
        The find-existing path keys on the presence/absence and identity
        of that part row, so an artifact-backed and an artifact-free
        selection for the same ``(recording_id, sorter,
        sorter_params_name)`` are distinct, idempotent rows.

        Raises
        ------
        ValueError
            If zero or both source keys are supplied.
        NotImplementedError
            If ``concat_recording_id`` is supplied -- concat-source
            sorting is not yet implemented.
        DuplicateSelectionError
            If more than one master+source row matches.
        """
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
        )

        has_recording = "recording_id" in key
        has_concat = "concat_recording_id" in key
        if has_recording == has_concat:
            raise ValueError(
                "SortingSelection.insert_selection requires exactly one "
                "source key. Provide either recording_id (single-session) "
                "or concat_recording_id (concat). Got: "
                f"recording_id={'set' if has_recording else 'unset'}, "
                f"concat_recording_id={'set' if has_concat else 'unset'}."
            )
        if has_concat:
            raise NotImplementedError(
                "SortingSelection.insert_selection: concatenated "
                "recording sorting is not implemented yet. Use a single "
                "recording_id source for now."
            )

        for required in ("sorter", "sorter_params_name"):
            if required not in key:
                raise ValueError(
                    f"SortingSelection.insert_selection requires "
                    f"{required!r} in key."
                )

        master_restriction = {
            "sorter": key["sorter"],
            "sorter_params_name": key["sorter_params_name"],
        }
        source_restriction = {"recording_id": key["recording_id"]}
        artifact_id = key.get("artifact_id")
        # ``resolve_artifact`` reads the uuid column back as a ``uuid.UUID``;
        # normalize a caller-supplied ``artifact_id`` (which may be a str) so
        # the find-existing comparison below is UUID-vs-UUID. A str would
        # otherwise never equal the stored UUID, so an idempotent re-insert
        # would miss its match and create a duplicate sort.
        if artifact_id is not None:
            artifact_id = uuid.UUID(str(artifact_id))

        # Find existing: a master with the same sorter + recording source
        # AND the same artifact-source state (present-with-this-artifact_id
        # vs absent). "No artifact pass" is "no ArtifactSource row", so the
        # two cases never alias onto each other.
        candidates = (
            (cls * cls.RecordingSource)
            & master_restriction
            & source_restriction
        ).fetch("KEY", as_dict=True)
        matches = []
        for cand in candidates:
            cand_master = {
                k: v for k, v in cand.items() if k in cls.primary_key
            }
            existing_artifact = cls.resolve_artifact(cand_master)
            if existing_artifact == artifact_id:
                matches.append(cand_master)
        unique = {tuple(sorted(d.items())) for d in matches}
        if len(unique) == 1:
            return dict(next(iter(unique)))
        if len(unique) > 1:
            raise DuplicateSelectionError(
                f"SortingSelection has {len(unique)} master rows for "
                f"{master_restriction | source_restriction} with "
                f"artifact_id={artifact_id}. v2 inserts via this helper "
                "should not produce duplicates."
            )

        # Translate the would-be DataJoint FK IntegrityError into a
        # clear "missing default row" message before the inserts attempt.
        from spyglass.spikesorting.v2.utils import _ensure_lookup_row_exists

        _ensure_lookup_row_exists(
            SorterParameters,
            {
                "sorter": key["sorter"],
                "sorter_params_name": key["sorter_params_name"],
            },
            helper_name="SortingSelection.insert_selection",
            insert_default_path="SorterParameters.insert_default()",
        )
        cls._validate_artifact_source_for_recording(
            recording_id=key["recording_id"],
            artifact_id=artifact_id,
        )

        new_master_key = {
            **master_restriction,
            "sorting_id": uuid.uuid4(),
        }
        new_part_key = {
            "sorting_id": new_master_key["sorting_id"],
            **source_restriction,
        }
        with transaction_or_noop(cls.connection):
            cls.insert1(new_master_key)
            cls.RecordingSource.insert1(new_part_key)
            if artifact_id is not None:
                cls.ArtifactSource.insert1(
                    {
                        "sorting_id": new_master_key["sorting_id"],
                        "artifact_id": artifact_id,
                    }
                )
        return {k: new_master_key[k] for k in cls.primary_key}

    @classmethod
    def _validate_artifact_source_for_recording(
        cls,
        *,
        recording_id,
        artifact_id,
    ) -> None:
        """Ensure an artifact pass is valid for a sorting recording.

        Single-recording artifacts may only be linked to that exact
        ``recording_id``. Shared-group artifacts may be linked to any member
        recording in the group. This keeps artifact masks from one recording
        in a session from being silently applied to a different recording.
        """
        if artifact_id is None:
            return

        from spyglass.spikesorting.v2.artifact import (
            ArtifactSelection,
            SharedArtifactGroup,
        )

        artifact_key = {"artifact_id": artifact_id}
        if not (ArtifactDetection & artifact_key):
            raise ValueError(
                "SortingSelection.insert_selection: artifact_id "
                f"{artifact_id!r} is not in ArtifactDetection. Populate "
                "ArtifactDetection before linking an artifact pass to a sort."
            )

        source = ArtifactSelection.resolve_source(artifact_key)
        target_recording_id = str(recording_id)
        if source.kind == "recording":
            artifact_recording_id = str(source.key["recording_id"])
            if artifact_recording_id != target_recording_id:
                raise ValueError(
                    "SortingSelection.insert_selection: artifact_id "
                    f"{artifact_id!r} belongs to recording_id="
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
                    "SortingSelection.insert_selection: artifact_id "
                    f"{artifact_id!r} belongs to shared artifact group "
                    f"{group_name!r}, which does not include requested "
                    f"recording_id={target_recording_id!r}."
                )
            return

        raise ValueError(
            "SortingSelection.insert_selection: artifact_id "
            f"{artifact_id!r} has unsupported source kind {source.kind!r}."
        )

    @classmethod
    def prune_orphaned_selections(cls, dry_run: bool = True) -> list[dict]:
        """Find or delete master rows that have no source-part row.

        See ``ArtifactSelection.prune_orphaned_selections`` for the full
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
    def resolve_artifact(cls, key: dict):
        """Return the ``artifact_id`` for a selection, or ``None``.

        Reads the optional ``ArtifactSource`` part row. Returns the
        ``artifact_id`` when an artifact pass was configured, else
        ``None`` (no ``ArtifactSource`` row = no artifact pass). This is
        the single accessor every reader uses instead of a nullable-FK
        column lookup.

        Raises
        ------
        SchemaBypassError
            If more than one ``ArtifactSource`` row exists for ``key``
            (the part is zero-or-one by construction).
        """
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_key = {k: v for k, v in key.items() if k in cls.primary_key}
        rows = (cls.ArtifactSource & master_key).fetch("artifact_id")
        if len(rows) > 1:
            raise SchemaBypassError(
                f"SortingSelection {master_key} has {len(rows)} "
                "ArtifactSource rows; expected zero or one."
            )
        return rows[0] if len(rows) == 1 else None


@schema
class Sorting(SpyglassMixin, dj.Computed):
    """Sorted units NWB + SortingAnalyzer folder.

    ``make()`` applies post-motion preprocessing, runs the sorter,
    removes excess spikes, builds a
    ``SortingAnalyzer(format="binary_folder", sparse=True)``, computes
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
    analyzer_folder: varchar(255)
    n_units: int
    time_of_sort: datetime    # populate wall-clock; native DataJoint datetime (v1 stored a Unix-epoch int at v1/sorting.py:239 -- a DataJoint-type workaround no longer needed)
    """

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        Brain region is reached through ``Sorting.Unit * Electrode *
        BrainRegion`` (NON-NULL on Spyglass's ``Electrode``); see
        shared-contracts ``Unit-Level Brain Region Tracing``. For concat
        sorts the Electrode FK is anchored to the FIRST member's row.
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
        """
        from spyglass.spikesorting.v2.recording import RecordingSelection

        source = SortingSelection.resolve_source(key)
        if source.kind != "recording":
            raise NotImplementedError(
                "Sorting.make: concatenated_recording source is not yet "
                "implemented."
            )

        sel_row = (SortingSelection & key).fetch1()
        # The artifact pass lives on the zero-or-one ``ArtifactSource``
        # part, not a nullable ``artifact_id`` FK on the master, so
        # ``sel_row`` does not carry an ``artifact_id`` key. Resolve it
        # once here and stash it on ``sel_row`` so the
        # downstream readers (obs_intervals derivation below,
        # make_compute's artifact-mask gate, _rebuild_analyzer_folder)
        # see the artifact id without re-querying. Without this the
        # ``sel_row.get("artifact_id")`` reads would always be None and
        # every artifact-backed sort would silently skip artifact masking.
        sel_row["artifact_id"] = SortingSelection.resolve_artifact(key)
        sorter_row = (
            SorterParameters
            & {
                "sorter": sel_row["sorter"],
                "sorter_params_name": sel_row["sorter_params_name"],
            }
        ).fetch1()
        nwb_file_name = (
            RecordingSelection & {"recording_id": source.key["recording_id"]}
        ).fetch1("nwb_file_name")

        # Pre-fetch the observation-interval window so
        # ``_write_units_nwb`` can write ``obs_intervals=`` on every
        # ``add_unit`` call. Downstream firing-rate computations
        # need the artifact-removed valid_times to know which
        # segments of the recording the sort actually observed --
        # without this column the units NWB looks like the unit
        # was observed across the full session even where the
        # artifact mask blanked the signal. Matches v1 at
        # ``v1/sorting.py:597``. When ``artifact_id`` is unset the
        # caller (make_compute) falls back to the recording's
        # full timestamp envelope.
        if sel_row.get("artifact_id") is not None:
            from spyglass.spikesorting.v2.utils import (
                artifact_interval_list_name,
            )

            obs_intervals = (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": artifact_interval_list_name(
                        sel_row["artifact_id"]
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
        )

    def make_compute(
        self,
        key,
        source,
        sel_row,
        sorter_row,
        nwb_file_name,
        obs_intervals,
    ):
        """Sort, build analyzer, stage Units NWB outside any DB transaction.

        The long-running steps run here:

        - load the cached preprocessed recording,
        - apply the artifact mask if ``artifact_id`` is set,
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
        """
        import shutil as _shutil

        recording_id = source.key["recording_id"]
        recording = Recording().get_recording({"recording_id": recording_id})

        if sel_row.get("artifact_id") is not None and obs_intervals is not None:
            recording = self._apply_artifact_mask(
                recording=recording,
                valid_times=obs_intervals,
                artifact_id=sel_row.get("artifact_id"),
                recording_id=recording_id,
            )

        sorter = sorter_row["sorter"]
        sorter_params = dict(sorter_row["params"])
        # ``schema_version`` is Pydantic bookkeeping; the SI sorter
        # wrapper does not accept it.
        sorter_params.pop("schema_version", None)

        # Resolve job_kwargs ONCE per compute stage
        # (shared-contracts.md "Job-Kwargs Resolution" invariant).
        # The resolved dict flows into BOTH ``sis.run_sorter`` AND
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
        # analyzer uses the same resolved value as the sorter.
        analyzer_folder = self._build_analyzer(
            sorting=sorting_obj,
            recording=recording,
            key=key,
            sorter_row=sorter_row,
            job_kwargs=job_kwargs,
        )
        analysis_file_name = None
        try:
            analysis_file_name, units_object_id = self._write_units_nwb(
                sorting=sorting_obj,
                recording=recording,
                nwb_file_name=nwb_file_name,
                obs_intervals=obs_intervals,
            )
        except Exception:
            # Mode A cleanup: analyzer folder was created but the
            # units NWB write failed. Remove the analyzer folder so
            # a half-built scratch dir does not leak.
            try:
                if analyzer_folder.exists():
                    _shutil.rmtree(analyzer_folder, ignore_errors=False)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting.make_compute: failed to remove analyzer "
                    f"folder {analyzer_folder!r}: {cleanup_exc!r}"
                )
            raise

        return SortingComputed(
            sorting_obj=sorting_obj,
            analysis_file_name=analysis_file_name,
            units_object_id=units_object_id,
            analyzer_folder=analyzer_folder,
            recording_id=recording_id,
            nwb_file_name=nwb_file_name,
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
        """
        import datetime as _dt
        import pathlib as _pathlib
        import shutil as _shutil

        try:
            # no-op when framework transaction is active; kept defensively
            # so an out-of-populate caller still gets atomic registration.
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "object_id": units_object_id,
                        "analyzer_folder": str(analyzer_folder),
                        "n_units": len(sorting_obj.unit_ids),
                        "time_of_sort": _dt.datetime.now(),
                    }
                )
                self._populate_unit_part(
                    sorting=sorting_obj,
                    recording_id=recording_id,
                    nwb_file_name=nwb_file_name,
                    key=key,
                    analyzer_folder=analyzer_folder,
                )
        except Exception:
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                _pathlib.Path(abs_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting.make_insert: failed to clean up staged units "
                    f"NWB {analysis_file_name!r}: {cleanup_exc!r}"
                )
            # Mode B cleanup: analyzer folder also needs removal so a
            # rolled-back populate does not leave a 5-50 GB orphan.
            try:
                if analyzer_folder.exists():
                    _shutil.rmtree(analyzer_folder, ignore_errors=False)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting.make_insert: failed to remove analyzer "
                    f"folder {analyzer_folder!r}: {cleanup_exc!r}"
                )
            raise

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
        timestamps, exactly as v1 does
        (``v1/sorting.py:spike_times_to_valid_samples`` +
        ``v1/curation.py:get_sorting``). An affine
        ``round((t - t_start) * fs)`` inverse (SI's ``NwbSortingExtractor``)
        is correct only on a uniform grid and shifts every frame after a
        gap by the accumulated gap (and can push frames past the
        gap-excluded sample count), so it is NOT used here.

        Returns a ``NumpySorting`` (segment frame indices, ``t_start=0``)
        matching v1's ``NumpySorting.from_unit_dict`` shape, so
        ``get_unit_spike_train(uid)`` yields the original recording
        frames and a downstream ``extract_waveforms`` / analyzer build
        aligns to the right samples.

        ``as_dataframe=True`` returns a pandas DataFrame whose
        **index is the unit_id** and which carries a ``spike_times``
        column (the stored ABSOLUTE seconds, read straight from the
        units NWB) -- mirrors v1's ``nwb.units.to_dataframe()`` shape at
        ``v1/curation.py:197-209``. The ``CurationV2.get_sorting``
        accessor uses the same flag + index and adds a
        ``curation_label`` column joined from ``CurationV2.UnitLabel``.

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
                return self._empty_spike_times_dataframe()
            return si.NumpySorting.from_unit_dict({}, sampling_frequency=fs)

        abs_times = self._read_units_abs_spike_times(abs_path)
        if as_dataframe:
            return self._abs_spike_times_dataframe(abs_times)
        return self._numpysorting_from_abs_times(abs_times, rec_row, fs)

    @staticmethod
    def _read_units_abs_spike_times(abs_path) -> dict:
        """Return ``{unit_id(int): abs_spike_times(np.ndarray seconds)}``.

        Reads the stored absolute spike times directly from a v2 units
        NWB (``nwbf.units.to_dataframe()``), so callers get the persisted
        wall-clock values exactly -- no affine round-trip. Returns ``{}``
        for an empty/absent Units table.
        """
        import numpy as _np
        import pynwb

        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            if nwbf.units is None or len(nwbf.units) == 0:
                return {}
            units_df = nwbf.units.to_dataframe()
        return {
            int(uid): _np.asarray(st, dtype=float)
            for uid, st in units_df["spike_times"].items()
        }

    @classmethod
    def _numpysorting_from_abs_times(cls, abs_times, recording_row, fs):
        """Build a ``NumpySorting`` from absolute spike times.

        Maps each unit's absolute spike times to recording frame indices
        with ``np.searchsorted`` against the recording's (possibly
        gap-preserving) timestamps -- the v1-parity readback that an
        affine inverse breaks across wall-clock gaps.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2.utils import _spike_times_to_frames

        recording_times = cls._recording_timestamps(recording_row)
        n_samples = int(recording_times.size)
        units_dict = {
            uid: _spike_times_to_frames(recording_times, st, n_samples, uid)
            for uid, st in abs_times.items()
        }
        return si.NumpySorting.from_unit_dict(
            [units_dict], sampling_frequency=fs
        )

    @staticmethod
    def _abs_spike_times_dataframe(abs_times):
        """DataFrame (index=unit_id) of absolute spike-time arrays."""
        import pandas as pd

        unit_ids = list(abs_times)
        return pd.DataFrame(
            {"spike_times": [abs_times[u] for u in unit_ids]},
            index=pd.Index(unit_ids, name="unit_id"),
        )

    @staticmethod
    def _empty_spike_times_dataframe():
        """Empty (index=unit_id) spike-times DataFrame for zero-unit sorts."""
        import pandas as pd

        return pd.DataFrame(
            {"spike_times": []},
            index=pd.Index([], name="unit_id", dtype=int),
        )

    @staticmethod
    def _recording_timestamps(recording_row):
        """Return the full timestamp vector of the upstream Recording.

        Reads the persisted ``ElectricalSeries`` timestamps -- which for
        disjoint sort intervals are gap-preserving (non-uniform). The SI
        readback in ``get_sorting`` maps absolute spike times back to
        frames via ``np.searchsorted`` against this vector; the affine
        ``t_start + i/fs`` assumption is wrong across wall-clock gaps.
        Reads only the timestamps dataset (not the traces), so it is far
        lighter than loading the full SI recording.
        """
        import numpy as _np
        import pynwb

        abs_path = AnalysisNwbfile.get_abs_path(
            recording_row["analysis_file_name"]
        )
        series_name = recording_row["electrical_series_path"].rsplit("/", 1)[-1]
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            series = nwbf.acquisition[series_name]
            return _np.asarray(series.timestamps[:], dtype=_np.float64)

    def get_analyzer(self, key: dict) -> "si.SortingAnalyzer":
        """Return the SortingAnalyzer; rebuild on missing folder.

        Recompute is in-place; the DataJoint row is not deleted on a
        missing analyzer folder.

        Raises ``ZeroUnitAnalyzerError`` for a zero-unit sort: SI cannot
        build a ``SortingAnalyzer`` over zero units, so no folder was
        written by ``_build_analyzer`` (it returns the would-be path).
        Loading it would surface a confusing SI/file error; this raises
        a clear signal instead. Use ``get_sorting()`` for the (empty)
        unit list.

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

        Returns
        -------
        si.SortingAnalyzer
            The loaded ``SortingAnalyzer`` for the sort, rebuilt in
            place if its folder was missing.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2.exceptions import (
            ZeroUnitAnalyzerError,
        )
        from spyglass.spikesorting.v2.utils import _analyzer_path

        if int((self & key).fetch1("n_units")) == 0:
            raise ZeroUnitAnalyzerError(
                "Sorting.get_analyzer: sorting_id="
                f"{key['sorting_id']!r} has zero units; no "
                "SortingAnalyzer exists (SI cannot build one over zero "
                "units). Use get_sorting() if you only need the empty "
                "unit list, or re-sort with a lower detect_threshold."
            )

        folder = _analyzer_path({"sorting_id": key["sorting_id"]})
        if not folder.exists():
            self._rebuild_analyzer_folder(key)
        return si.load_sorting_analyzer(folder)

    def _rebuild_analyzer_folder(self, key) -> None:
        """Rebuild the analyzer folder for an existing Sorting row.

        Reloads the canonical sorting from the units NWB so the
        rebuilt analyzer is bit-equivalent to the one Sorting.make
        wrote -- not a fresh, possibly nondeterministic, sort.
        """
        import shutil as _shutil

        from spyglass.spikesorting.v2.utils import _analyzer_path

        sel_row = (SortingSelection & key).fetch1()
        # Resolve the artifact from the ArtifactSource part (the master
        # has no artifact_id FK); without this the artifact-mask gate
        # below would never fire and a rebuilt analyzer for an
        # artifact-backed sort would omit the mask, diverging from what
        # Sorting.make wrote.
        sel_row["artifact_id"] = SortingSelection.resolve_artifact(key)
        source = SortingSelection.resolve_source(key)
        if source.kind != "recording":
            raise NotImplementedError(
                "Sorting._rebuild_analyzer_folder: concat source not yet "
                "implemented."
            )
        recording = Recording().get_recording(
            {"recording_id": source.key["recording_id"]}
        )
        if sel_row.get("artifact_id") is not None:
            from spyglass.common.common_interval import IntervalList
            from spyglass.spikesorting.v2.recording import RecordingSelection
            from spyglass.spikesorting.v2.utils import (
                artifact_interval_list_name,
            )

            nwb_file_name = (
                RecordingSelection
                & {"recording_id": source.key["recording_id"]}
            ).fetch1("nwb_file_name")
            valid_times = (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": artifact_interval_list_name(
                        sel_row["artifact_id"]
                    ),
                }
            ).fetch1("valid_times")
            recording = self._apply_artifact_mask(
                recording=recording,
                valid_times=valid_times,
                artifact_id=sel_row["artifact_id"],
                recording_id=source.key["recording_id"],
            )
        sorting_obj = self.get_sorting(key)
        # ``_build_analyzer`` writes a folder to disk; a mid-rebuild
        # failure would otherwise leak a partial scratch folder.
        # Removing it before re-raising keeps the rebuild path's
        # invariant ("analyzer folder reflects the canonical sort")
        # true under failure.
        folder = _analyzer_path({"sorting_id": key["sorting_id"]})
        try:
            self._build_analyzer(
                sorting=sorting_obj, recording=recording, key=key
            )
        except Exception:
            try:
                if folder.exists():
                    _shutil.rmtree(folder, ignore_errors=False)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting._rebuild_analyzer_folder: failed to remove "
                    f"partial analyzer folder {folder!r}: {cleanup_exc!r}"
                )
            raise

    def delete(self, *args, safemode=None, **kwargs):
        """Cascade-delete + analyzer-folder cleanup on disk.

        The ``analyzer_folder`` path column on ``Sorting`` is not
        tracked by DataJoint, so a plain ``.delete()`` would leave
        the 5-50 GB scratch folder on disk per row. Mirrors
        ``ArtifactDetection.delete``'s IntervalList cleanup pattern:
        collect every folder path BEFORE the cascade delete (the row
        needed to compute the path is gone after), then call
        ``super().delete()``, then ``shutil.rmtree`` each collected
        path. ``ignore_errors=False`` so a permission error surfaces
        loudly rather than getting swallowed.

        Two cleanup paths already cover other points in the
        ``analyzer_folder`` lifecycle: ``_run_sorter`` cleans the
        sorter scratch ``TemporaryDirectory`` on successful sort,
        and the make_compute / make_insert except blocks clean the
        folder on populate failure. This override closes the third
        lifecycle event: row deletion.
        """
        import shutil as _shutil

        from spyglass.spikesorting.v2.utils import _analyzer_path

        keyed_folders = [
            (row, _analyzer_path({"sorting_id": row["sorting_id"]}))
            for row in self.fetch("KEY", as_dict=True)
        ]
        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)
        # Only remove a folder whose DB row was ACTUALLY deleted. A cancelled
        # confirmation prompt (user answers "no") or an empty restriction
        # leaves the rows in place and returns normally -- removing their
        # 5-50 GB analyzer scratch then would destroy data for a row the user
        # chose to keep.
        for row, folder in keyed_folders:
            if folder.exists() and not (Sorting & row):
                _shutil.rmtree(folder, ignore_errors=False)

    @classmethod
    def find_orphaned_analyzer_folders(cls, *, dry_run: bool = True) -> dict:
        """Audit 5-50 GB analyzer-folder disk leaks; never auto-delete DB rows.

        Each populated sort writes a 5-50 GB ``analyzer_folder`` of
        regeneratable scratch outside the DataJoint-tracked store. The
        ``Sorting.delete`` override cleans it up on row delete, but an external
        path that bypasses the override (raw SQL delete, scripted
        ``dj.Table.connection.query``) leaks the folder. This periodic audit
        mirrors ``prune_orphaned_selections`` and reports two leak classes:

        - **DB-side orphan**: a ``Sorting`` row whose ``analyzer_folder`` no
          longer exists on disk (the regeneratable scratch was removed out of
          band). Reported only -- deleting the *row* is a destructive DB
          operation the human decides on (per the Spyglass destructive-op
          contract); this method NEVER auto-deletes a row.
        - **Disk-side orphan**: an on-disk folder under the analyzer root that
          no ``Sorting`` row references (the row was deleted via a path that
          bypassed the ``delete`` override). Safe to delete after inspection.

        **Zero-unit carve-out.** Rows with ``n_units == 0`` are NOT DB-side
        orphans: ``_build_analyzer`` short-circuits before writing a folder and
        ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` before reading the
        path, so an absent folder is expected. The ``analyzer_folder`` column
        is NOT-NULL ``varchar(255)`` and still carries the would-be path (there
        is no None/sentinel value), so the carve-out is keyed on
        ``(Sorting & {"n_units": 0})``, NOT a string match on the column.

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
            ``{"db_side": [{"sorting_id", "analyzer_folder"}, ...],
            "disk_side": [folder_path_str, ...]}``.
        """
        import shutil as _shutil

        import datajoint as dj

        from spyglass.spikesorting.v2.utils import _analyzer_path

        # DB-side: units-bearing rows whose stored folder is gone on disk.
        # The n_units==0 carve-out excludes legitimately folder-less rows.
        db_side = [
            row
            for row in (cls & "n_units > 0").fetch(
                "sorting_id", "analyzer_folder", as_dict=True
            )
            if not Path(row["analyzer_folder"]).exists()
        ]

        # Disk-side: folders under the analyzer root that no row (any n_units)
        # references. The root is the shared parent of every _analyzer_path.
        referenced = {str(p) for p in cls.fetch("analyzer_folder")}
        analyzer_root = _analyzer_path({"sorting_id": "x"}).parent
        disk_side = []
        if analyzer_root.exists():
            for child in sorted(analyzer_root.iterdir()):
                if child.is_dir() and str(child) not in referenced:
                    disk_side.append(str(child))

        logger.info(
            "Sorting.find_orphaned_analyzer_folders: "
            f"{len(db_side)} DB-side orphan(s) (row present, folder missing), "
            f"{len(disk_side)} disk-side orphan(s) (folder present, no row)."
        )
        for row in db_side:
            logger.info(
                "  DB-side orphan: sorting_id=%s analyzer_folder=%s",
                row["sorting_id"],
                row["analyzer_folder"],
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
                _shutil.rmtree(folder, ignore_errors=False)
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
        recording, valid_times, *, artifact_id=None, recording_id=None
    ):
        """Zero out the complement of ``valid_times`` on the recording.

        ``valid_times`` is the artifact-removed (start, end) seconds
        array from the upstream ``IntervalList``; ``make_fetch``
        already fetched it as ``obs_intervals`` so ``make_compute``
        passes it through here instead of re-issuing the DB lookup
        (the tri-part contract forbids DB I/O inside compute).

        ``artifact_id`` / ``recording_id`` are used only to make the
        empty-``valid_times`` error message actionable.

        Raises
        ------
        EmptyArtifactValidTimesError
            If ``valid_times`` is empty -- masking would zero the whole
            recording, so the sort must fail loudly instead of running
            over all-zeros.
        ValueError
            If ``valid_times`` is not an ``(n, 2)`` array, has an
            interval with ``end < start``, or is not sorted-by-start and
            non-overlapping. The complement walker assumes monotonic,
            disjoint input; an unsorted/overlapping list would silently
            under-mask. (The fetched ``obs_intervals`` are monotonic in
            practice; this guards a hand-built curation override. Strict
            input is intentional -- silent sort/merge is deferred.)
        """
        import numpy as _np
        import spikeinterface.preprocessing as sip

        from spyglass.spikesorting.v2.exceptions import (
            EmptyArtifactValidTimesError,
        )

        valid_times = _np.asarray(valid_times, dtype=float)
        if valid_times.size == 0:
            raise EmptyArtifactValidTimesError(
                "Artifact-removed valid_times is empty for "
                f"artifact_id={artifact_id!r}, "
                f"recording_id={recording_id!r}: the artifact pass kept "
                "zero seconds of the recording. Masking would zero the "
                "entire recording and the sort would run over all-zeros. "
                "Re-run ArtifactDetection with looser thresholds or "
                "override the artifact selection."
            )
        if valid_times.ndim != 2 or valid_times.shape[1] != 2:
            raise ValueError(
                "_apply_artifact_mask: valid_times must be an (n, 2) array "
                f"of (start, end) seconds; got shape {valid_times.shape}."
            )
        starts = valid_times[:, 0]
        ends = valid_times[:, 1]
        if _np.any(ends < starts):
            raise ValueError(
                "_apply_artifact_mask: valid_times has an interval whose "
                "end precedes its start; each interval must be "
                "(start <= end)."
            )
        if valid_times.shape[0] > 1 and (
            _np.any(_np.diff(starts) < 0) or _np.any(starts[1:] < ends[:-1])
        ):
            raise ValueError(
                "_apply_artifact_mask: valid_times must be sorted by start "
                "time and non-overlapping (the complement walker assumes "
                f"monotonic, disjoint input); got {valid_times.tolist()!r}. "
                "Sort and merge the intervals before passing them."
            )

        timestamps = recording.get_times()
        # Walk the valid intervals left-to-right, collecting the
        # complement (artifact gaps) as a list of (start_frame,
        # end_frame) pairs. Each pair is materialized via
        # ``np.arange`` and concatenated -- ~100x faster than the
        # equivalent ``list.extend(range(start, end))`` for
        # multi-million-sample recordings (Python-int boxing for
        # every frame is the bottleneck of the loop form).
        frame_ranges: list[tuple[int, int]] = []
        cursor = timestamps[0]
        for vs, ve in valid_times:
            if vs > cursor:
                start = int(_np.searchsorted(timestamps, cursor))
                end = int(_np.searchsorted(timestamps, vs))
                if end > start:
                    frame_ranges.append((start, end))
            cursor = max(cursor, ve)
        if cursor < timestamps[-1]:
            start = int(_np.searchsorted(timestamps, cursor))
            end = len(timestamps)
            if end > start:
                frame_ranges.append((start, end))

        # Drop pure inter-chunk-gap ranges. For a DISJOINT recording the
        # gap-respecting valid_times leave a single boundary frame between
        # two chunks; the complement walk emits it as a width-1 range whose
        # successor is a wall-clock discontinuity. That frame is the last
        # real sample of the preceding chunk (valid) -- masking it would
        # zero a good sample per gap. A genuine 1-frame artifact instead
        # has ~1-sample spacing to its neighbor, so it is kept. (A 1-sample
        # artifact landing exactly on a chunk's final sample is the lone
        # uncovered edge; negligible at the chunk boundary.)
        sample_period = 1.0 / float(recording.get_sampling_frequency())
        frame_ranges = [
            (s, e)
            for (s, e) in frame_ranges
            if not (
                e - s == 1
                and e < len(timestamps)
                and (timestamps[e] - timestamps[s]) > 1.5 * sample_period
            )
        ]

        if not frame_ranges:
            return recording

        artifact_frames = _np.concatenate(
            [_np.arange(s, e, dtype=_np.int64) for s, e in frame_ranges]
        )
        # SI's ``list_triggers`` is documented as a list-of-arrays, one
        # per event channel. We pass a single numpy array wrapping ALL
        # artifact frames so the mask zeros every artifact sample
        # exactly once, independent of how SI interprets a bare Python
        # list of scalars in future minor versions.
        #
        # ``ms_before=None, ms_after=None`` is the documented "single
        # sample" mode and uses the ``pad is None`` code path in SI
        # 0.104's RemoveArtifactsRecordingSegment.get_traces (each
        # trigger zeros exactly its own frame). ``ms_before=0.0,
        # ms_after=0.0`` looks equivalent but triggers a boundary
        # bug in SI 0.104 where the first artifact frame in any
        # contiguous run is left unmasked (the ``trig - pad[0] <= 0``
        # branch assigns to an empty slice).
        return sip.remove_artifacts(
            recording=recording,
            list_triggers=[artifact_frames],
            ms_before=None,
            ms_after=None,
            mode="zeros",
        )

    # Sorters that ship as MATLAB containers in SpikeInterface. These
    # need ``singularity_image=True`` and a small kwarg-strip carve-out
    # so the v1-style default Lookup rows survive containerization.
    # The check is name-only; users who insert custom rows for other
    # MATLAB sorters can extend this set by subclassing.
    _MATLAB_SORTERS = ("kilosort2_5", "kilosort3", "ironclust")
    _MATLAB_SORTER_STRIP_KWARGS = (
        "tempdir",
        "mp_context",
        "max_threads_per_process",
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

        Not an SI registered sorter; uses
        ``spikeinterface.sortingcomponents.peak_detection.detect_peaks``
        directly and wraps the result in a ``NumpySorting``.

        ``noise_levels`` handling: when the params row supplies a
        non-``None`` value (e.g. ``default_clusterless`` ships
        ``noise_levels=[1.0]``), SI's ``locally_exclusive`` interprets
        ``detect_threshold`` in the recording's amplitude units. With
        ``threshold_unit="uv"`` this method scales the recording to
        microvolts (``scale_to_uV``, using the stored NWB gain) before
        ``detect_peaks``, so ``detect_threshold`` is a TRUE microvolt
        threshold -- honoring the label v1 used at ``v1/sorting.py:177``
        but never actually enforced (v1 thresholded in raw counts). For
        Frank-lab data gain==1 uV/count so this is a no-op; for
        non-unity-gain rigs it is the fix. A scalar (singleton list) is
        broadcast to length ``n_channels``
        because SI's ``locally_exclusive`` indexes ``noise_levels[chan]
        * detect_threshold`` per channel. When the params row omits
        ``noise_levels`` (default in v2, and what the smoke /
        synthetic-fixture rows do), SI computes per-channel MAD
        internally and ``detect_threshold`` is interpreted as a MAD
        multiplier -- which is what the v1 baseline-capture script
        relies on for the ``smoke_clusterless_5uv`` row to find any
        peaks on the MEArec fixture.
        """
        import numpy as _np
        import spikeinterface as si
        from spikeinterface.sortingcomponents.peak_detection import (
            detect_peaks,
        )

        params = dict(sorter_params)
        # v1-era kwarg rename: SI 0.99 ``local_radius_um`` became
        # ``radius_um`` in 0.101+.
        if "local_radius_um" in params:
            params["radius_um"] = params.pop("local_radius_um")
        # SI 0.104 ``detect_peaks`` rejects v1's stale routing hints
        # via the new ``(method, method_kwargs, job_kwargs)`` shape:
        # ``outputs`` was a Spyglass-only routing hint, and
        # ``random_chunk_kwargs`` was renamed to
        # ``random_slices_kwargs`` and is now managed internally.
        for stale in ("outputs", "random_chunk_kwargs"):
            params.pop(stale, None)

        # ``threshold_unit`` is a Spyglass-side knob, not a detect_peaks
        # kwarg: strip it and use it to resolve the noise_levels
        # precedence (explicit noise_levels win; otherwise "uv" -> [1.0]
        # so detect_threshold reads in the recording's native units --
        # raw ADC counts under v2's gain-free preprocessing, not true uV
        # unless gain-scaled; "mad" -> None so SI estimates per-channel
        # MAD).
        threshold_unit = params.pop("threshold_unit", "mad")
        nl_in = _clusterless_noise_levels(
            params.get("noise_levels"), threshold_unit
        )
        if nl_in is None:
            # Drop the key entirely so ``detect_peaks`` falls through to
            # SI's per-channel MAD estimation path.
            params.pop("noise_levels", None)
            # Pin SI's random-chunk sampling inside
            # ``get_noise_levels`` (the path detect_peaks takes when
            # noise_levels is absent) to a deterministic seed.
            # PR #3359 (merged 2024-10-25) changed SI's default from
            # ``seed=0`` to ``seed=None``, making the per-channel MAD
            # non-deterministic across runs on the same input --
            # which at a detect_threshold of 5 (a MAD multiplier, not
            # 5σ) can flip ~10-20 borderline peaks per shank. Per PR
            # #3359's stated principle
            # (*"seed must be explicit and no implicit"*) Spyglass IS
            # the explicit-seeder. Same fix + same user-override
            # mechanism as the ``sip.whiten`` pin in
            # ``_run_si_sorter`` -- set ``random_seed`` in the per-
            # row ``SorterParameters.job_kwargs`` blob to override.
            _random_seed = (job_kwargs or {}).get("random_seed", 0)
            params.setdefault("random_slices_kwargs", {"seed": _random_seed})
        else:
            n_channels = recording.get_num_channels()
            # Reject an explicit noise_levels of the wrong length BEFORE
            # broadcasting / indexing it per channel: a singleton is
            # broadcast, an n_channels-length array is used as-is, and
            # any other explicit length is a configuration error (it
            # would otherwise mis-index inside SI's ``locally_exclusive``).
            _assert_noise_levels_length(nl_in, n_channels)
            nl = _np.asarray(nl_in, dtype=_np.float64)
            if nl.size == 1:
                nl = _np.full(n_channels, float(nl[0]), dtype=_np.float64)
            params["noise_levels"] = nl

        method = params.pop("method", "locally_exclusive")
        # ``random_seed`` is a Spyglass-side knob (already threaded into
        # ``random_slices_kwargs`` above for the noise_levels=None path);
        # it is NOT a valid SI job kwarg, so leaving it in ``job_kwargs``
        # makes ``detect_peaks`` -> ``fix_job_kwargs`` raise
        # ``AssertionError: random_seed is not a valid job keyword
        # argument``. Strip it before the call.
        detect_job_kwargs = {
            k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"
        }
        if threshold_unit == "uv":
            # Honor the "uv" label: scale the detector's input from raw ADC
            # counts to microvolts using the recording's STORED gain/offset
            # (the NWB ElectricalSeries conversion/offset, reloaded onto the
            # recording's channel gains by se.read_nwb_recording). With
            # noise_levels=[1.0], detect_threshold is then a genuine
            # microvolt threshold. For Frank-lab data (gain==1 uV/count) this
            # is a no-op; for non-unity-gain rigs (e.g. Intan ~0.195) it
            # converts a previously raw-count threshold into true uV.
            # Diverges from v1, which always thresholded in raw counts.
            import spikeinterface.preprocessing as sip

            if recording.get_channel_gains() is None:
                raise ValueError(
                    "clusterless_thresholder threshold_unit='uv' requires the "
                    "recording to carry channel gains (the NWB ElectricalSeries "
                    "conversion); none are set. Use threshold_unit='mad' for a "
                    "gain-free relative threshold."
                )
            recording = sip.scale_to_uV(recording)
        detected = detect_peaks(
            recording,
            method=method,
            method_kwargs=params,
            job_kwargs=(detect_job_kwargs or None),
        )
        # SI 0.104 renamed ``from_times_labels`` to
        # ``from_samples_and_labels`` (sample indices); ``detect_peaks``
        # already returns sample indices.
        return si.NumpySorting.from_samples_and_labels(
            samples_list=detected["sample_index"],
            labels_list=_np.zeros(len(detected), dtype=_np.int32),
            sampling_frequency=recording.get_sampling_frequency(),
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

        Scratch is anchored under ``spyglass.settings.temp_dir`` via
        ``tempfile.TemporaryDirectory`` so the dir is cleaned on
        successful exit AND on raise (fixes the tempdir leak).
        ``os.chmod 0o777`` makes it world-writable so SI sorter
        subprocesses with a different uid (rootless container, slurm
        scenarios) can write into it.

        External float64 whitening matches v1 (``v1/sorting.py:428-430``):
        if the sorter asks for whitening, run it externally at float64
        and turn the sorter's internal whitening off so we do not
        whiten twice. Runs AFTER the upstream artifact mask was
        applied in ``Sorting.make_compute`` -- artifact-masked frames
        should not bias whitening's covariance estimate.

        MATLAB sorters (Kilosort 2.5 / 3, IronClust) get
        ``singularity_image=True`` and the strip-kwargs carve-out:
        ``tempdir`` / ``mp_context`` / ``max_threads_per_process`` do
        not survive containerization.
        """
        import os
        import tempfile

        import numpy as _np
        import spikeinterface as _si
        import spikeinterface.sorters as sis

        from spyglass.settings import temp_dir as _spyglass_temp_dir

        sorter_temp_dir = tempfile.TemporaryDirectory(
            prefix=f"sort_{sorting_id}_",
            dir=_spyglass_temp_dir,
        )
        patched_numpy_inf = False
        try:
            os.chmod(sorter_temp_dir.name, 0o777)

            # spikeextractors 0.9.11 (a transitive dep of SI's MS4
            # wrapper at ``spikeinterface/sorters/external/mountainsort4.py``)
            # references the removed ``numpy.Inf`` alias at
            # ``spikeextractors/extraction_tools.py:766``; that crashes
            # under numpy >= 2.0. Restore the alias only for the duration
            # of this call and delete it again in the ``finally`` -- a
            # persistent global mutation would make every later module
            # that probes ``hasattr(np, "Inf")`` (some scipy versions)
            # see a different numpy than import time. ``np.inf`` is what
            # ``np.Inf`` always aliased, so the patch is value-safe; only
            # its lifetime is scoped. TODO: drop once SI's MS4 wrapper /
            # spikeextractors stops referencing the removed alias.
            if sorter.lower() == "mountainsort4" and not hasattr(_np, "Inf"):
                _np.Inf = _np.inf
                patched_numpy_inf = True

            if sorter_params.get("whiten", False):
                import spikeinterface.preprocessing as sip

                # Pin SI's random-chunk-based covariance estimate
                # inside ``sip.whiten`` to a deterministic seed.
                # PR #3359 (merged 2024-10-25) changed SI's default
                # from ``seed=0`` to ``seed=None``, making the
                # whitening matrix non-deterministic across runs on
                # the same input. Per the PR author: *"seed must be
                # explicit and no implicit"* -- so Spyglass IS the
                # explicit-seeder. Empirically verified: 3 v2 MS4
                # runs with seed=0 produce identical (n_units,
                # median_fr); without the pin, 4 runs produced 4
                # distinct states.
                #
                # User override: set ``random_seed`` in the per-row
                # ``SorterParameters.job_kwargs`` blob to use a
                # different seed (for robustness studies / variance
                # characterization). Spyglass's default 0 matches v1
                # SI 0.99 behavior so re-runs of a parameter row are
                # reproducible by default.
                _random_seed = (job_kwargs or {}).get("random_seed", 0)
                recording = sip.whiten(
                    recording, dtype=_np.float64, seed=_random_seed
                )
                sorter_params = {**sorter_params, "whiten": False}

            # Resolved job_kwargs (n_jobs, chunk_duration, progress_bar,
            # etc.) install via ``si.set_global_job_kwargs`` and are
            # picked up by ``run_sorter`` through SI's global state.
            # They MUST NOT be splatted into ``run_sorter(**...)``: SI
            # 0.104's ``run_sorter`` signature has ``**sorter_params``
            # as the only catch-all, so any extra kwargs flow straight
            # into the sorter and trip strict per-sorter validators
            # (MS4, MS5, KS4 all raise ``Invalid parameters: [...]``
            # for pool_engine / n_jobs / chunk_duration / progress_bar
            # / mp_context / max_threads_per_worker).
            sj_kwargs = dict(job_kwargs or {})
            # ``random_seed`` is a Spyglass-side knob for SI's
            # random-chunk sampling (consumed by the ``sip.whiten``
            # call above and the clusterless ``random_slices_kwargs``
            # pin); strip before installing as a job kwarg because
            # SI's ``set_global_job_kwargs`` rejects unknown keys.
            sj_kwargs.pop("random_seed", None)
            previous_global = dict(_si.get_global_job_kwargs())
            if sj_kwargs:
                _si.set_global_job_kwargs(**sj_kwargs)
            run_kwargs = dict(
                sorter_name=sorter,
                recording=recording,
                folder=sorter_temp_dir.name,
                remove_existing_folder=True,
            )
            if sorter.lower() in Sorting._MATLAB_SORTERS:
                run_kwargs["singularity_image"] = True
                effective_params = {
                    k: v
                    for k, v in sorter_params.items()
                    if k not in Sorting._MATLAB_SORTER_STRIP_KWARGS
                }
            else:
                effective_params = sorter_params
            try:
                return sis.run_sorter(**run_kwargs, **effective_params)
            finally:
                if sj_kwargs:
                    # ``set_global_job_kwargs`` UPDATES (does not
                    # replace) the global, so a job kwarg the sort
                    # installed that was ABSENT from the prior global
                    # (chunk_size / total_memory / chunk_memory are not
                    # in SI's default global set) would leak into every
                    # later populate. Reset to the baseline first, then
                    # re-apply the captured prior global for an exact
                    # restore.
                    _si.reset_global_job_kwargs()
                    _si.set_global_job_kwargs(**previous_global)
        finally:
            # Undo the scoped ``np.Inf`` patch so the global numpy
            # module is left exactly as the rest of the process saw it.
            if patched_numpy_inf and hasattr(_np, "Inf"):
                del _np.Inf
            # ``TemporaryDirectory`` auto-cleans on garbage collection,
            # but the explicit ``.cleanup()`` in a ``finally`` makes
            # the cleanup point obvious and survives worker-process
            # exit predictably under the parallel-populate process
            # pool. Catch + log any cleanup error: if the sort itself
            # raised, a cleanup failure (e.g. a stale lock on a network
            # FS) must NOT replace the original sort exception -- that
            # would hide the real failure behind a misleading
            # PermissionError on the tempdir.
            try:
                sorter_temp_dir.cleanup()
            except Exception as cleanup_exc:
                logger.warning(
                    "Sorting._run_si_sorter: sorter_temp_dir cleanup "
                    f"failed for sorting_id={sorting_id}: {cleanup_exc!r}. "
                    "Original sort exception (if any) preserved."
                )

    @staticmethod
    def _remove_excess_spikes(sorting, recording):
        """Drop spikes whose sample index is outside the recording window."""
        import spikeinterface.curation as sic

        return sic.remove_excess_spikes(sorting, recording)

    @staticmethod
    def _build_analyzer(
        sorting,
        recording,
        key,
        *,
        sorter_row=None,
        job_kwargs=None,
    ):
        """Build the binary-folder SortingAnalyzer + base extensions.

        ``sorter_row`` is the already-fetched ``SorterParameters`` row
        from ``make_fetch``; passing it through avoids three redundant
        DB round-trips during ``make_compute`` (we cannot read inside
        ``make_compute`` per the tri-part contract). The rebuild path
        does not have the row pre-fetched, so it leaves
        ``sorter_row=None`` and pays the DB cost once -- that path is
        rare (missing analyzer folder) and not on the populate hot
        path, so the lookup is acceptable there.

        ``job_kwargs`` is the already-resolved dict from
        ``_resolved_job_kwargs`` -- when ``make_compute`` calls
        ``_run_sorter`` and ``_build_analyzer`` from the same
        invocation it resolves once and threads through; the rebuild
        path falls back to resolving locally (same DB-read tradeoff
        as ``sorter_row``).
        """
        import shutil as _shutil

        import spikeinterface as si

        from spyglass.spikesorting.v2.utils import (
            _analyzer_path,
            _resolved_job_kwargs,
        )

        folder = _analyzer_path({"sorting_id": key["sorting_id"]})

        # Zero-unit short-circuit BEFORE any I/O or DB fetch:
        # ``create_sorting_analyzer(sparse=True)`` -> ``estimate_sparsity``
        # -> ``random_spikes_selection`` crashes on ``np.concatenate([])``
        # for empty sortings. ``_populate_unit_part`` iterates an empty
        # ``sorting.unit_ids`` and writes zero Unit rows; the Sorting
        # master row commits with ``n_units=0``. Return the (not yet
        # created) folder path for the row; ``Sorting.get_analyzer``
        # raises ``ZeroUnitAnalyzerError`` for a zero-unit sort rather
        # than trying to load this never-built folder.
        if sorting.get_num_units() == 0:
            logger.warning(
                "Sorting._build_analyzer: sorting_id="
                f"{key.get('sorting_id')!r} has zero units; skipping "
                "analyzer build. Check ``detect_threshold`` / "
                "artifact masking if you expected non-zero output."
            )
            return folder

        folder.parent.mkdir(parents=True, exist_ok=True)

        if sorter_row is None:
            sorter_row = (
                SorterParameters
                & (
                    (SortingSelection & key).proj(
                        "sorter", "sorter_params_name"
                    )
                )
            ).fetch1()
        if job_kwargs is None:
            job_kwargs = _resolved_job_kwargs(sorter_row["job_kwargs"])

        try:
            analyzer = si.create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                sparse=True,
                format="binary_folder",
                folder=folder,
                return_in_uV=True,
                overwrite=True,
            )
            # ``random_seed`` is a Spyglass-side knob (consumed by the sorter
            # and the whitening pin in ``_run_si_sorter``), not a valid
            # ``SortingAnalyzer.compute`` keyword -- SI raises
            # "please remove {'random_seed'}". Strip it here, mirroring the
            # detect_peaks path. It stays in ``job_kwargs`` upstream so the
            # sorter/whitening still read the seed.
            analyzer_job_kwargs = {
                k: v for k, v in job_kwargs.items() if k != "random_seed"
            }
            analyzer.compute(
                ["random_spikes", "noise_levels", "templates", "waveforms"],
                extension_params={
                    "random_spikes": {
                        "max_spikes_per_unit": 500,
                        "method": "uniform",
                        # Pin the stochastic spike subsampling. When a
                        # unit has more than ``max_spikes_per_unit`` (500)
                        # spikes, ``random_spikes`` draws a uniform random
                        # subset; the SI 0.104 extension defaults
                        # ``seed=None`` (verified against
                        # ``ComputeRandomSpikes._set_params``), so an
                        # unseeded build selects a different subset each
                        # time -- and the persisted ``peak_amplitude_uv``
                        # / peak channel (computed from the subset's
                        # templates) drifts across rebuilds of the same
                        # sort. Defaults to 0 but honors the per-row
                        # ``job_kwargs={"random_seed": N}`` override, the
                        # same knob the whitening / noise_levels pins read
                        # in ``_run_si_sorter`` / ``_run_clusterless_
                        # thresholder`` (lines using
                        # ``(job_kwargs or {}).get("random_seed", 0)``) --
                        # so a user changing the seed gets a consistent
                        # seed across the sort AND the analyzer subsample.
                        "seed": (job_kwargs or {}).get("random_seed", 0),
                    },
                    "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
                },
                **analyzer_job_kwargs,
            )
        except Exception:
            try:
                if folder.exists():
                    _shutil.rmtree(folder, ignore_errors=False)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Sorting._build_analyzer: failed to remove partial "
                    f"analyzer folder {folder!r}: {cleanup_exc!r}"
                )
            raise
        return folder

    @staticmethod
    def _write_units_nwb(sorting, recording, nwb_file_name, obs_intervals=None):
        """Write a fresh AnalysisNwbfile containing only the v2 Units table.

        Spike times are stored in the recording's absolute timeline
        (``timestamps[sample_index]``) -- matching v1's convention --
        so downstream consumers can compare directly against the
        Recording's IntervalList valid_times. ``AnalysisNwbfile().create``
        already strips any parent ``/units`` from the analysis NWB so
        the v2 sort outputs are the only Units rows in the file
        (addresses #1437).

        Every unit row carries ``obs_intervals`` (the artifact-
        removed valid-time window the sort observed) and a
        ``curation_label`` placeholder list (``["uncurated"]``).
        Both columns mirror v1 at ``v1/sorting.py:583-598``;
        external readers that grep for either column on a
        pre-curation NWB now find them. ``obs_intervals`` defaults
        to the recording's full timestamp envelope when no artifact
        mask was applied (``obs_intervals=None``).
        """
        import numpy as _np
        import pynwb

        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name
        )
        analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

        timestamps = recording.get_times()
        if obs_intervals is None:
            # ``obs_intervals is None`` is the "no artifact pass" case:
            # the artifact pass is optional (an ArtifactSource part is
            # zero-or-one; no part / artifact_id=None means no masking),
            # so there is no artifact-removed IntervalList to read. The
            # recorded window(s) ARE the correct obs_intervals then -- the
            # sort observed every recorded sample. Split at wall-clock
            # discontinuities so a DISJOINT recording reports one interval
            # per recorded chunk rather than a single envelope spanning the
            # gaps (which would inflate the observation duration). For a
            # contiguous recording this collapses to a single
            # ``[t0, t_end]``, unchanged. (v1's FK was mandatory, so it
            # always had artifact-removed intervals here.)
            from spyglass.spikesorting.v2.utils import (
                _base_intervals_from_timestamps,
            )

            obs_intervals_arr = _np.asarray(
                _base_intervals_from_timestamps(
                    timestamps, recording.get_sampling_frequency()
                )
            )
        else:
            obs_intervals_arr = _np.asarray(obs_intervals)

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # ``curation_label`` is a scalar ``"uncurated"`` at sort
            # time, matching v1's pre-curation NWB shape at
            # ``v1/sorting.py:583-598``. External readers expecting
            # v1's shape do
            # ``nwb.units["curation_label"][i] == "uncurated"`` and
            # would silently fail an equality check against a list.
            # ``CurationV2.insert_curation`` rewrites this to the
            # indexed ragged-list shape at post-curation time, which
            # is the v1-curated shape. The pre-vs-post shape
            # discontinuity is inherited from v1 and is intentional.
            #
            # ``add_unit_column`` must be declared BEFORE any
            # ``add_unit`` call that passes the column as a kwarg;
            # pynwb rejects the kwarg as "extra keys" otherwise.
            # The scalar shape (no ``index=True``) matches v1.
            if len(sorting.unit_ids) > 0:
                nwbf.add_unit_column(
                    name="curation_label",
                    description=(
                        'Curation label scalar; ``"uncurated"`` at '
                        "sort time, refined to a per-unit label list "
                        "by CurationV2.insert_curation."
                    ),
                )
            for unit_id in sorting.unit_ids:
                spike_indices = sorting.get_unit_spike_train(unit_id=unit_id)
                # Map sample indices into the recording's wall-clock so
                # the stored spike times match Recording.get_times()
                # exactly. v1 uses this same convention.
                spike_times = timestamps[spike_indices]
                nwbf.add_unit(
                    spike_times=spike_times,
                    id=int(unit_id),
                    obs_intervals=obs_intervals_arr,
                    curation_label="uncurated",
                )
            # pynwb leaves ``nwbf.units = None`` if no add_unit() was
            # called, so a zero-unit sort would crash on .object_id.
            # Initialize an empty Units table explicitly (v1 has the
            # same guard at v1/sorting.py:578).
            if nwbf.units is None:
                nwbf.units = pynwb.misc.Units(
                    name="units",
                    description="Empty units table (sorter found zero units).",
                )
            units_object_id = nwbf.units.object_id
            io.write(nwbf)

        # The AnalysisNwbfile DB-row registration (.add) is deliberately
        # NOT done here -- ``Sorting.make`` registers it inside its
        # ``transaction_or_noop`` block so the row rolls back atomically
        # if any of the master / Unit-part inserts fail.
        return analysis_file_name, units_object_id

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

        import numpy as _np
        import spikeinterface as si
        from spikeinterface.core import template_tools

        from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError
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
        # most-negative one. v1 makes this configurable via the
        # ``peak_channel`` metric params; v2 threads it here at sort time.
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

        rows = []
        for unit_id in sorting.unit_ids:
            try:
                int_unit_id = int(unit_id)
            except (TypeError, ValueError) as exc:
                raise NonIntegerUnitIDError(
                    f"Sorting.make: sorter returned unit_id {unit_id!r} "
                    "that does not convert to int. v2's Sorting.Unit "
                    "stores int unit_ids; remap before insertion if "
                    "the sorter emits non-convertible IDs."
                ) from exc
            peak_chan = int(peak_channels[unit_id])
            if peak_chan not in electrode_by_id:
                raise RuntimeError(
                    f"Sorting.make: peak channel {peak_chan} for unit "
                    f"{int_unit_id} is not in sort group "
                    f"{sort_group_id} for {nwb_file_name!r}. Sort group "
                    "/ recording channel-id mismatch."
                )
            n_spikes = int(len(sorting.get_unit_spike_train(unit_id=unit_id)))
            rows.append(
                {
                    **key,
                    "unit_id": int_unit_id,
                    **{
                        k: electrode_by_id[peak_chan][k]
                        for k in (
                            "nwb_file_name",
                            "electrode_group_name",
                            "electrode_id",
                        )
                    },
                    "peak_amplitude_uv": float(peak_amplitudes[unit_id]),
                    "n_spikes": n_spikes,
                }
            )
        Sorting.Unit.insert(rows)
