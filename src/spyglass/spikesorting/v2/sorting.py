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
    _assert_schema_version_matches,
    _assert_v2_db_safe,
    _validate_params,
    find_orphaned_masters,
    transaction_or_noop,
    unit_brain_region_df,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
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
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    def insert1(self, row, **kwargs):
        row = dict(row)
        schema_cls = _get_sorter_schema(row["sorter"])
        row["params"] = _validate_params(schema_cls, row["params"])
        _assert_schema_version_matches(
            row, schema_cls, table_name="SorterParameters"
        )
        super().insert1(row, **kwargs)

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
                _get_sorter_schema("clusterless_thresholder"), {}
            ),
            # ClusterlessThresholderSchema bumped to schema_version=2
            # by dropping ``outputs`` and ``random_chunk_kwargs``;
            # this row tracks the same version. Other sorter rows are
            # unchanged at 1.
            2,
            None,
        ),
    )

    @classmethod
    def insert_default(cls):
        """Insert v2 default sorter rows if missing.

        The default-content list mirrors the designs.md
        ``SorterParameters`` section and includes MS4, MS5, KS4, SC2,
        TDC2, and clusterless_thresholder. MS4 + KS4 are not
        deterministic and ``clusterless_thresholder`` is a Spyglass
        peak-detection special case, not an SI registered sorter; see
        the per-sorter Pydantic schemas in
        ``spyglass.spikesorting.v2._params.sorter`` for the validated
        field surface.
        """
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Source part rows make the input shape explicit: exactly one of
    ``RecordingSource`` or ``ConcatenatedRecordingSource`` exists for
    each selection row. The runtime helper today rejects the concat
    path with a clear "not implemented yet" error; the schema is final
    so the validator can be relaxed without a migration once the concat
    materializer lands. ``artifact_id`` is a real DataJoint nullable FK
    to ``ArtifactDetection`` (not a loose UUID column) so DataJoint
    enforces referential integrity.
    """

    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    -> [nullable] ArtifactDetection
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

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Insert master + exactly one source part; return PK-only dict.

        Reads exactly one of ``recording_id`` (single-session) or
        ``concat_recording_id`` (concat) from ``key``; raises ValueError
        on zero or two sources. The concat path is rejected today with
        ``NotImplementedError`` because the concat materializer is gated.
        ``artifact_id`` is optional (the FK is nullable); concat sorts
        must leave it NULL.

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
        # artifact_id is optional; track whether the caller supplied it
        # so the find-existing-or-insert is precise.
        if "artifact_id" in key and key["artifact_id"] is not None:
            master_restriction["artifact_id"] = key["artifact_id"]
        source_part = cls.RecordingSource
        source_restriction = {"recording_id": key["recording_id"]}

        joined = (cls * source_part) & master_restriction & source_restriction
        existing = joined.fetch("KEY", as_dict=True)
        existing_master_keys = [
            {k: v for k, v in row.items() if k in cls.primary_key}
            for row in existing
        ]
        unique = {tuple(sorted(d.items())) for d in existing_master_keys}
        if len(unique) == 1:
            return dict(next(iter(unique)))
        if len(unique) > 1:
            raise DuplicateSelectionError(
                f"SortingSelection has {len(unique)} master rows for "
                f"{master_restriction | source_restriction}. v2 inserts "
                "via this helper should not produce duplicates."
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
            source_part.insert1(new_part_key)
        return {k: new_master_key[k] for k in cls.primary_key}

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
    time_of_sort: datetime
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
        if sel_row.get("artifact_id"):
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

        from spyglass.spikesorting.v2.recording import Recording

        recording_id = source.key["recording_id"]
        recording = Recording().get_recording({"recording_id": recording_id})

        if sel_row.get("artifact_id") and obs_intervals is not None:
            recording = self._apply_artifact_mask(
                recording=recording,
                valid_times=obs_intervals,
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

    def get_sorting(self, key, as_dataframe: bool = False):
        """Return the SpikeInterface BaseSorting backed by the units NWB.

        The v2 Units NWB does NOT carry an ``ElectricalSeries`` (the
        canonical preprocessed recording lives in the Recording's
        AnalysisNwbfile), so both ``sampling_frequency`` and ``t_start``
        must be supplied explicitly to ``NwbSortingExtractor`` --
        otherwise its auto-detection raises looking for a non-existent
        ElectricalSeries in the units NWB.

        ``t_start`` is the recording's first wall-clock timestamp,
        derived from the upstream Recording's AnalysisNwbfile via SI's
        ``read_nwb_recording``. This matches the absolute spike-times
        stored by ``_write_units_nwb`` (spike_times = timestamps[
        sample_index]) so ``sorting.get_unit_spike_train(uid)`` returns
        the original sample indices and ``return_times=True`` returns
        the recording-timeline times.

        ``as_dataframe=True`` returns a pandas DataFrame with
        ``unit_id`` + ``spike_times`` (in seconds) columns -- a
        pre-curation peek convenience mirroring v1's
        ``(SpikeSorting & key).fetch_nwb()`` notebook pattern. The
        ``CurationV2.get_sorting`` accessor uses the same
        ``as_dataframe`` flag with the same core columns and adds
        a ``curation_label`` column joined from
        ``CurationV2.UnitLabel``.
        """
        from spikeinterface.extractors import NwbSortingExtractor

        from spyglass.spikesorting.v2.recording import Recording

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        source = SortingSelection.resolve_source(key)
        recording_id = source.key["recording_id"]
        rec_row = (Recording & {"recording_id": recording_id}).fetch1()
        fs = float(rec_row["sampling_frequency"])
        t_start = self._recording_t_start(rec_row)
        si_sorting = NwbSortingExtractor(
            file_path=abs_path, sampling_frequency=fs, t_start=t_start
        )
        if not as_dataframe:
            return si_sorting

        import pandas as pd

        # v2's units NWB writer stores unit_ids as integers via
        # ``add_unit(id=int(kept_uid), ...)`` -- the cast back to int
        # here matches that contract. If SI ever returns string IDs
        # (e.g., an externally-ingested NWB with string-typed unit
        # IDs), this cast would raise; pre-curation NWBs written by
        # v2 itself always pass.
        unit_ids = [int(uid) for uid in si_sorting.unit_ids]
        return pd.DataFrame(
            {
                "unit_id": unit_ids,
                "spike_times": [
                    si_sorting.get_unit_spike_train(
                        unit_id=uid, return_times=True
                    )
                    for uid in si_sorting.unit_ids
                ],
            }
        )

    @staticmethod
    def _recording_t_start(recording_row) -> float:
        """Return the first timestamp of the upstream Recording.

        Opens the Recording's ``AnalysisNwbfile`` ``ElectricalSeries``
        and reads only the first timestamp value. Cheaper than loading
        the full SI recording -- HDF5 lets us index timestamps[0]
        without materializing the dataset.
        """
        import pynwb

        abs_path = AnalysisNwbfile.get_abs_path(
            recording_row["analysis_file_name"]
        )
        # electrical_series_path is "acquisition/ProcessedElectricalSeries"
        series_name = recording_row["electrical_series_path"].rsplit(
            "/", 1
        )[-1]
        with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
            nwbf = io.read()
            series = nwbf.acquisition[series_name]
            return float(series.timestamps[0])

    def get_analyzer(self, key):
        """Return the SortingAnalyzer; rebuild on missing folder.

        Recompute is in-place; the DataJoint row is not deleted on a
        missing analyzer folder.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2.utils import _analyzer_path

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
        from spyglass.spikesorting.v2.recording import Recording

        import shutil as _shutil

        from spyglass.spikesorting.v2.utils import _analyzer_path

        sel_row = (SortingSelection & key).fetch1()
        source = SortingSelection.resolve_source(key)
        if source.kind != "recording":
            raise NotImplementedError(
                "Sorting._rebuild_analyzer_folder: concat source not yet "
                "implemented."
            )
        recording = Recording().get_recording(
            {"recording_id": source.key["recording_id"]}
        )
        if sel_row.get("artifact_id"):
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

        folders_to_remove = [
            _analyzer_path({"sorting_id": row["sorting_id"]})
            for row in self.fetch("KEY", as_dict=True)
        ]
        if safemode is None:
            super().delete(*args, **kwargs)
        else:
            super().delete(*args, safemode=safemode, **kwargs)
        for folder in folders_to_remove:
            if folder.exists():
                _shutil.rmtree(folder, ignore_errors=False)

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ):
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts raise ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; the anchor-member output is
        labeled ``region_resolution='anchor_member'``.
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
                    "regions."
                )
            resolution = "anchor_member"
        else:
            resolution = "single_session"
        return unit_brain_region_df(self.Unit & key, resolution)

    # ---- Implementation helpers -----------------------------------------

    @staticmethod
    def _apply_artifact_mask(recording, valid_times):
        """Zero out the complement of ``valid_times`` on the recording.

        ``valid_times`` is the artifact-removed (start, end) seconds
        array from the upstream ``IntervalList``; ``make_fetch``
        already fetched it as ``obs_intervals`` so ``make_compute``
        passes it through here instead of re-issuing the DB lookup
        (the tri-part contract forbids DB I/O inside compute).
        """
        import numpy as _np
        import spikeinterface.preprocessing as sip

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

        For SI sorters the per-sort scratch directory is anchored to
        ``spyglass.settings.temp_dir`` via
        ``tempfile.TemporaryDirectory`` so the dir is cleaned on
        successful exit and on raise. The directory is also
        ``chmod 0o777``'d so SI sorter subprocesses with a different
        uid (rootless container, slurm scenarios) can write into it.

        MATLAB-based sorters (Kilosort 2.5 / 3, IronClust) get the
        ``singularity_image=True`` flag from SI's container API plus
        a small kwarg-strip carve-out so v1-style rows survive the
        containerized call.
        """
        import os
        import tempfile

        import numpy as _np
        import spikeinterface as si
        import spikeinterface.sorters as sis

        if sorter == "clusterless_thresholder":
            from spikeinterface.sortingcomponents.peak_detection import (
                detect_peaks,
            )

            params = dict(sorter_params)
            # v1-era kwarg rename: the SI 0.99 `local_radius_um` became
            # `radius_um` in 0.101+.
            if "local_radius_um" in params:
                params["radius_um"] = params.pop("local_radius_um")
            # ``noise_levels=[1.0]`` is DELIBERATELY forwarded to
            # ``detect_peaks`` (via SI 0.104's ``**old_kwargs`` ->
            # ``method_kwargs`` routing). With ``noise_levels``
            # absent, SI computes per-channel MAD and treats
            # ``detect_threshold`` as a MAD multiplier; with
            # ``noise_levels=[1.0]`` SI broadcasts that single value
            # to every channel so ``detect_threshold`` stays in
            # microvolts. Matches v1's deliberate choice at
            # ``v1/sorting.py:177,402-404``. A user's
            # ``detect_threshold=100.0`` was 100 µV in v1; without
            # this passthrough it would become ~100xMAD on noisy
            # channels, silently shifting detection ~5x.
            #
            # The remaining strip handles fields SI 0.104's
            # ``detect_peaks`` actively rejects: ``outputs`` (a
            # Spyglass routing hint that never had an SI meaning) and
            # ``random_chunk_kwargs`` (renamed to
            # ``random_slices_kwargs`` and now managed internally).
            for stale in ("outputs", "random_chunk_kwargs"):
                params.pop(stale, None)

            # Route resolved job_kwargs into detect_peaks via the
            # explicit kwarg. SI 0.104's signature is
            # ``detect_peaks(recording, method, method_kwargs,
            # ...)`` and rejects the legacy ``**old_kwargs`` mode
            # whenever ``job_kwargs`` is non-None; route per-method
            # kwargs (detect_threshold, peak_sign, noise_levels,
            # radius_um, etc.) through the explicit ``method_kwargs``
            # dict instead.
            #
            # SI 0.104's locally_exclusive computes
            # ``self.noise_levels * detect_threshold`` and indexes
            # the resulting ``abs_thresholds`` per channel. The v1-
            # stored shape ``[1.0]`` (single value) is meant to
            # represent "one microvolt per channel everywhere";
            # SI 0.99 broadcast it implicitly, SI 0.104 wants a
            # length-``n_channels`` array. Broadcast manually so a
            # 32-channel recording does not divide-by-zero on
            # missing per-channel entries inside the numba kernel.
            if "noise_levels" in params:
                nl = _np.asarray(params["noise_levels"], dtype=_np.float64)
                if nl.size == 1:
                    nl = _np.full(
                        recording.get_num_channels(),
                        float(nl[0]),
                        dtype=_np.float64,
                    )
                params["noise_levels"] = nl
            method = params.pop("method", "locally_exclusive")
            detected = detect_peaks(
                recording,
                method=method,
                method_kwargs=params,
                job_kwargs=(job_kwargs or None),
            )
            # SI 0.104 renamed ``from_times_labels`` to
            # ``from_samples_and_labels`` (sample indices) /
            # ``from_times_and_labels`` (absolute seconds). We pass
            # sample indices from ``detect_peaks``.
            return si.NumpySorting.from_samples_and_labels(
                samples_list=detected["sample_index"],
                labels_list=_np.zeros(len(detected), dtype=_np.int32),
                sampling_frequency=recording.get_sampling_frequency(),
            )

        # SI sorter path. Anchor scratch under Spyglass's temp_dir so
        # a disk-full surface fails in a known location and the dir
        # is auto-removed on context exit (fix for the tempdir leak).
        # ``os.chmod`` makes the dir world-writable so a sorter
        # subprocess running under a different uid can write into it.
        from spyglass.settings import temp_dir as _spyglass_temp_dir

        sorter_temp_dir = tempfile.TemporaryDirectory(
            prefix=f"sort_{sorting_id}_",
            dir=_spyglass_temp_dir,
        )
        try:
            os.chmod(sorter_temp_dir.name, 0o777)

            # Restore v1's external float64 whitening. The upstream
            # ``Recording._apply_pre_motion_preprocessing`` runs
            # bandpass + reference at float64; pre-whitening here
            # keeps the whitening step at the same precision. MS4's
            # internal whitening operates at float32 (see
            # Mountainsort4Sorter wrapper), which v1 deliberately
            # bypassed for precision parity. Mirrors
            # ``v1/sorting.py:428-430``: if the sorter asks for
            # whitening, run it externally at float64 and turn the
            # sorter's internal whitening off so we do not whiten
            # twice. Runs AFTER the clusterless branch returns and
            # AFTER the upstream artifact mask was applied in
            # ``Sorting.make_compute`` -- artifact-masked frames
            # should not bias whitening's covariance estimate.
            if sorter_params.get("whiten", False):
                import spikeinterface.preprocessing as sip

                recording = sip.whiten(recording, dtype=_np.float64)
                sorter_params = {**sorter_params, "whiten": False}

            # MATLAB sorter carve-out. The ``singularity_image=True``
            # flag triggers SI's container runner; some v1-style
            # kwargs do not survive the container boundary
            # (``tempdir`` clashes with the container's own workspace,
            # ``mp_context`` / ``max_threads_per_process`` are noisy
            # in containerized runs) so they are stripped only for
            # these sorters.
            # Resolved job_kwargs (n_jobs, chunk_duration, etc.) flow
            # into ``sis.run_sorter`` via the ``**sorter_params``
            # catch-all -- SI splits them into per-sorter and SI
            # control args internally. Merging via dict update lets a
            # per-row ``job_kwargs`` override anything the row's
            # sorter_params happened to set.
            sj_kwargs = job_kwargs or {}
            if sorter.lower() in Sorting._MATLAB_SORTERS:
                clean_params = {
                    k: v
                    for k, v in sorter_params.items()
                    if k not in Sorting._MATLAB_SORTER_STRIP_KWARGS
                }
                return sis.run_sorter(
                    sorter_name=sorter,
                    recording=recording,
                    folder=sorter_temp_dir.name,
                    remove_existing_folder=True,
                    singularity_image=True,
                    **clean_params,
                    **sj_kwargs,
                )

            return sis.run_sorter(
                sorter_name=sorter,
                recording=recording,
                folder=sorter_temp_dir.name,
                remove_existing_folder=True,
                **sorter_params,
                **sj_kwargs,
            )
        finally:
            # ``TemporaryDirectory`` auto-cleans on garbage collection,
            # but the explicit ``.cleanup()`` in a ``finally`` makes
            # the cleanup point obvious and survives the worker
            # process exit predictably under the parallel-populate
            # process pool.
            sorter_temp_dir.cleanup()

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
        import spikeinterface as si

        from spyglass.spikesorting.v2.utils import (
            _analyzer_path,
            _resolved_job_kwargs,
        )

        folder = _analyzer_path({"sorting_id": key["sorting_id"]})
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

        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=True,
            format="binary_folder",
            folder=folder,
            return_in_uV=True,
            overwrite=True,
        )
        analyzer.compute(
            ["random_spikes", "noise_levels", "templates", "waveforms"],
            extension_params={
                "random_spikes": {
                    "max_spikes_per_unit": 500,
                    "method": "uniform",
                },
                "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
            },
            **job_kwargs,
        )
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
            # Fall back to the full recording window when no artifact
            # IntervalList was supplied; matches the no-mask sort
            # semantics (the sort observed every sample).
            obs_intervals_arr = _np.asarray(
                [[float(timestamps[0]), float(timestamps[-1])]]
            )
        else:
            obs_intervals_arr = _np.asarray(obs_intervals)

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # Declare the ``curation_label`` indexed list column
            # BEFORE adding any unit so v1-style readers that grep
            # for the column on a pre-curation NWB find it.
            # Conditional on at least one unit being written;
            # ``pynwb`` cannot infer dtype on an empty
            # ``add_unit_column`` followed by ``io.write``.
            if len(sorting.unit_ids) > 0:
                nwbf.add_unit_column(
                    name="curation_label",
                    description=(
                        "Curation label list (placeholder "
                        "``[\"uncurated\"]`` at sort time; refined "
                        "by CurationV2.insert_curation)."
                    ),
                    index=True,
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
                    curation_label=["uncurated"],
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
        """
        import numpy as _np
        import spikeinterface as si
        from spikeinterface.core import template_tools

        from spyglass.spikesorting.v2.exceptions import NonIntegerUnitIDError
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
            SortGroupV2,
        )

        analyzer = si.load_sorting_analyzer(analyzer_folder)
        peak_channels = template_tools.get_template_extremum_channel(
            analyzer, outputs="id"
        )
        peak_amplitudes = template_tools.get_template_extremum_amplitude(
            analyzer
        )

        sort_group_id = int(
            (
                RecordingSelection & {"recording_id": recording_id}
            ).fetch1("sort_group_id")
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
            n_spikes = int(
                len(sorting.get_unit_spike_train(unit_id=unit_id))
            )
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
                    "peak_amplitude_uv": float(
                        _np.abs(peak_amplitudes[unit_id])
                    ),
                    "n_spikes": n_spikes,
                }
            )
        Sorting.Unit.insert(rows)
