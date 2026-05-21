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

import datajoint as dj

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
    _assert_v2_db_safe,
    _validate_params,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

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
            1,
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
            If ``concat_recording_id`` is supplied (Phase 3 work).
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

        new_master_key = {
            **master_restriction,
            "sorting_id": uuid.uuid4(),
        }
        new_part_key = {
            "sorting_id": new_master_key["sorting_id"],
            **source_restriction,
        }
        if cls.connection.in_transaction:
            cls.insert1(new_master_key)
            source_part.insert1(new_part_key)
        else:
            with cls.connection.transaction:
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
        all_masters = cls.fetch("KEY", as_dict=True)
        orphans: list[dict] = []
        for master in all_masters:
            rec_count = len(cls.RecordingSource & master)
            concat_count = len(cls.ConcatenatedRecordingSource & master)
            if rec_count + concat_count == 0:
                orphans.append(master)
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

    def make(self, key):
        """Sort, build analyzer, write Units NWB, populate Unit part.

        Re-checks ``SortingSelection.resolve_source(key)`` at entry per
        the shared-contracts Source Part Pattern. Single-recording path
        only today; concat raises ``NotImplementedError``.
        """
        import datetime as _dt

        from spyglass.spikesorting.v2.recording import (
            Recording,
            RecordingSelection,
        )

        source = SortingSelection.resolve_source(key)
        if source.kind != "recording":
            raise NotImplementedError(
                "Sorting.make: concatenated_recording source is not yet "
                "implemented (Phase 3 work)."
            )

        sel_row = (SortingSelection & key).fetch1()
        recording_id = source.key["recording_id"]
        recording = Recording().get_recording({"recording_id": recording_id})

        if sel_row.get("artifact_id"):
            recording = self._apply_artifact_mask(
                recording=recording,
                artifact_id=sel_row["artifact_id"],
                recording_id=recording_id,
            )

        sorter_row = (
            SorterParameters
            & {
                "sorter": sel_row["sorter"],
                "sorter_params_name": sel_row["sorter_params_name"],
            }
        ).fetch1()
        sorter = sorter_row["sorter"]
        sorter_params = dict(sorter_row["params"])
        # ``schema_version`` is Pydantic bookkeeping; the SI sorter
        # wrapper does not accept it.
        sorter_params.pop("schema_version", None)

        sorting_obj = self._run_sorter(
            sorter=sorter,
            sorter_params=sorter_params,
            recording=recording,
            sorting_id=key["sorting_id"],
        )
        sorting_obj = self._remove_excess_spikes(sorting_obj, recording)

        analyzer_folder = self._build_analyzer(
            sorting=sorting_obj, recording=recording, key=key
        )

        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        analysis_file_name, units_object_id = self._write_units_nwb(
            sorting=sorting_obj,
            recording=recording,
            nwb_file_name=nwb_file_name,
        )

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

    # ---- Accessors -------------------------------------------------------

    def get_sorting(self, key):
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
        return NwbSortingExtractor(
            file_path=abs_path, sampling_frequency=fs, t_start=t_start
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
            recording = self._apply_artifact_mask(
                recording=recording,
                artifact_id=sel_row["artifact_id"],
                recording_id=source.key["recording_id"],
            )
        sorting_obj = self.get_sorting(key)
        self._build_analyzer(
            sorting=sorting_obj, recording=recording, key=key
        )

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ):
        """Per-unit brain regions via Sorting.Unit * Electrode * BrainRegion.

        Single-session sorts return ``region_resolution='single_session'``.
        Concat sorts (when implemented) raise
        ``ConcatBrainRegionAmbiguousError`` unless
        ``allow_anchor_member=True``; the anchor-member output is
        labeled ``region_resolution='anchor_member'``.
        """
        import pandas as pd

        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_region import BrainRegion

        source = SortingSelection.resolve_source(key)
        if source.kind == "concatenated_recording":
            from spyglass.spikesorting.v2.exceptions import (
                ConcatBrainRegionAmbiguousError,
            )

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

        joined = (
            (self.Unit & key) * _Electrode * BrainRegion
        ).fetch(
            "unit_id",
            "electrode_id",
            "region_name",
            "subregion_name",
            "subsubregion_name",
            as_dict=True,
        )
        df = pd.DataFrame(joined)
        df["region_resolution"] = resolution
        return df

    # ---- Implementation helpers -----------------------------------------

    @staticmethod
    def _apply_artifact_mask(recording, artifact_id, recording_id):
        """Zero out artifact intervals on the recording.

        Looks up the IntervalList row written by ArtifactDetection.make
        and uses ``sip.remove_artifacts(mode='zeros')`` to mask the
        complement of the artifact-removed valid times.
        """
        import numpy as _np
        import spikeinterface.preprocessing as sip

        from spyglass.common.common_interval import IntervalList
        from spyglass.spikesorting.v2.recording import RecordingSelection

        nwb_file_name = (
            RecordingSelection & {"recording_id": recording_id}
        ).fetch1("nwb_file_name")
        interval_list_name = f"artifact_{artifact_id}"
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")

        timestamps = recording.get_times()
        artifact_frames = []
        cursor = timestamps[0]
        for vs, ve in valid_times:
            if vs > cursor:
                start = int(_np.searchsorted(timestamps, cursor))
                end = int(_np.searchsorted(timestamps, vs))
                artifact_frames.extend(range(start, end))
            cursor = max(cursor, ve)
        if cursor < timestamps[-1]:
            start = int(_np.searchsorted(timestamps, cursor))
            artifact_frames.extend(range(start, len(timestamps)))

        if not artifact_frames:
            return recording
        # SI's ``list_triggers`` is documented as a list-of-arrays, one
        # per event channel. We pass a single numpy array wrapping ALL
        # artifact frames so the mask zeros every artifact sample
        # exactly once, independent of how SI interprets a bare Python
        # list of scalars in future minor versions.
        return sip.remove_artifacts(
            recording=recording,
            list_triggers=[_np.asarray(artifact_frames, dtype=_np.int64)],
            ms_before=0.0,
            ms_after=0.0,
            mode="zeros",
        )

    @staticmethod
    def _run_sorter(sorter, sorter_params, recording, sorting_id):
        """Dispatch sort execution; clusterless_thresholder vs SI sorters."""
        import tempfile

        import numpy as _np
        import spikeinterface as si
        import spikeinterface.sorters as sis

        if sorter == "clusterless_thresholder":
            from spikeinterface.sortingcomponents.peak_detection import (
                detect_peaks,
            )

            params = dict(sorter_params)
            if "local_radius_um" in params:
                params["radius_um"] = params.pop("local_radius_um")
            params.pop("outputs", None)

            detected = detect_peaks(recording, **params)
            return si.NumpySorting.from_times_labels(
                times_list=detected["sample_index"],
                labels_list=_np.zeros(len(detected), dtype=_np.int32),
                sampling_frequency=recording.get_sampling_frequency(),
            )

        tmpdir = tempfile.mkdtemp(prefix=f"sort_{sorting_id}_")
        return sis.run_sorter(
            sorter_name=sorter,
            recording=recording,
            folder=tmpdir,
            remove_existing_folder=True,
            **sorter_params,
        )

    @staticmethod
    def _remove_excess_spikes(sorting, recording):
        """Drop spikes whose sample index is outside the recording window."""
        import spikeinterface.curation as sic

        return sic.remove_excess_spikes(sorting, recording)

    @staticmethod
    def _build_analyzer(sorting, recording, key):
        """Build the binary-folder SortingAnalyzer + base extensions."""
        import spikeinterface as si

        from spyglass.spikesorting.v2.utils import (
            _analyzer_path,
            _resolved_job_kwargs,
        )

        folder = _analyzer_path({"sorting_id": key["sorting_id"]})
        folder.parent.mkdir(parents=True, exist_ok=True)

        sorter_row = (
            SorterParameters
            & {
                "sorter": (SortingSelection & key).fetch1("sorter"),
                "sorter_params_name": (
                    SortingSelection & key
                ).fetch1("sorter_params_name"),
            }
        ).fetch1()
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
    def _write_units_nwb(sorting, recording, nwb_file_name):
        """Write a fresh AnalysisNwbfile containing only the v2 Units table.

        Spike times are stored in the recording's absolute timeline
        (``timestamps[sample_index]``) -- matching v1's convention --
        so downstream consumers can compare directly against the
        Recording's IntervalList valid_times. ``AnalysisNwbfile().create``
        already strips any parent ``/units`` from the analysis NWB so
        the v2 sort outputs are the only Units rows in the file
        (addresses #1437).
        """
        import pynwb

        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name
        )
        analysis_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

        timestamps = recording.get_times()
        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            for unit_id in sorting.unit_ids:
                spike_indices = sorting.get_unit_spike_train(unit_id=unit_id)
                # Map sample indices into the recording's wall-clock so
                # the stored spike times match Recording.get_times()
                # exactly. v1 uses this same convention.
                spike_times = timestamps[spike_indices]
                nwbf.add_unit(spike_times=spike_times, id=int(unit_id))
            units_object_id = nwbf.units.object_id
            io.write(nwbf)

        AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
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
