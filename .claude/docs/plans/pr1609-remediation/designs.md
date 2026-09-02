# Designs — per-component code and rationale

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Each section is referenced by anchor from a phase file. Line references are to branch `spikesorting-v2` at `ad2b9626`; see [overview.md](overview.md#current-codebase-integration-points) for the verified list.

## si-compat-shim

**Problem.** Ten v0/v1 call sites use SpikeInterface names that exist in exactly one of the two pinned versions: 0.99.1 has `si.load_extractor` and `NumpySorting.from_times_labels`; 0.104.3 has `si.load` and `NumpySorting.from_samples_and_labels`. Under the new package pin the read paths crash with `AttributeError` instead of the guarded `RuntimeError`.

**Design.** One stdlib-plus-SpikeInterface module, imported lazily where used, no version parsing (attribute presence is the contract):

```python
# src/spyglass/spikesorting/_si_compat.py
"""SpikeInterface API names that differ between the 0.99 and 0.101+ lines.

The package pins SpikeInterface 0.104 but existing v0/v1 rows were written
under 0.99. Read paths (loading a saved recording or sorting) must work under
both, so they go through these wrappers instead of calling a version-specific
name. Attribute presence is the dispatch: no version parsing.
"""

from __future__ import annotations


def load_extractor(source):
    """Load a saved SpikeInterface recording or sorting.

    Parameters
    ----------
    source : str | pathlib.Path | dict
        Folder, ``.json`` / ``.pkl`` path, or the dict form that
        ``BaseExtractor.to_dict`` produces (v0 artifact detection passes a dict).

    Returns
    -------
    si.BaseRecording | si.BaseSorting
    """
    import spikeinterface as si

    loader = getattr(si, "load_extractor", None) or si.load
    return loader(source)


def numpy_sorting_from_samples_and_labels(
    samples, labels, sampling_frequency, unit_ids=None
):
    """Build a ``NumpySorting`` from sample indices and unit labels.

    ``from_times_labels`` (0.99) and ``from_samples_and_labels`` (0.101+) share
    positional semantics ``(samples, labels, sampling_frequency, unit_ids)``.
    """
    from spikeinterface.core import NumpySorting

    ctor = getattr(NumpySorting, "from_samples_and_labels", None) or (
        NumpySorting.from_times_labels
    )
    return ctor(samples, labels, sampling_frequency, unit_ids=unit_ids)
```

**Call-site rewrite.** Replace `si.load_extractor(x)` with `load_extractor(x)` (import from `spyglass.spikesorting._si_compat`) at the ten sites listed in the overview. Replace the two `si.NumpySorting.from_times_labels(times_list=..., labels_list=..., sampling_frequency=...)` calls (`v0/spikesorting_sorting.py:274`, `v1/sorting.py:426`) with the positional shim call. Keep every existing `_require_legacy_si_environment(...)` guard on populate paths; the shim only makes READ paths and the guarded paths' first statements not crash before the guard is reached.

**Verification that cannot be automated in one env.** Whether `si.load` (0.104) opens folders written by 0.99 `recording.save()` is an SI compatibility question. Before claiming read paths work, run in `spyglass_spikesorting_v2` against one real v0 `SpikeSortingRecording` folder and one v0 sorting folder from the lab's data (or a folder produced by the legacy env on the smoke fixture):

```python
from spyglass.spikesorting._si_compat import load_extractor
rec = load_extractor("/path/to/v0/recording_folder"); print(rec)
srt = load_extractor("/path/to/v0/sorting_folder"); print(srt)
```

If 0.104 refuses a 0.99 folder, wrap the failure: catch the SI exception inside `load_extractor` and re-raise `RuntimeError(_legacy_runtime_message(f"loading {source}"))` from `_legacy_runtime.py:33-44`, and say so in the CHANGELOG. Do not silently fall back.

## es-selection

**Problem.** `get_raw_eseries_path` (`nwb_helper_fn.py:342-389`) picks the first acquisition `ElectricalSeries` in h5py name order; `Raw` ingestion (`common_ephys.py:299-304` + `ingestion.py:178-196`) picks by a sanitized name set. On a file with `analog-series` + `e-series` the two disagree and the sort runs on the wrong signal with no error (master's SI 0.99 raised on such files).

**Design.** Make the name set a single constant in `nwb_helper_fn.py` (lowest layer; `common_ephys` imports from it), select only names in that set, and refuse zero or multiple matches:

```python
# src/spyglass/utils/nwb_helper_fn.py (module level, near the top)
RAW_ELECTRICAL_SERIES_NAMES = (
    "e-series",
    "electricalseries",
    "ephys",
    "electrophysiology",
)


def sanitize_nwb_object_name(name):
    """Case- and space-insensitive form used to match NWB object names."""
    return name.lower().replace(" ", "") if name else None
```

`common_ephys.py:299-304` becomes `_source_nwb_object_name = list(RAW_ELECTRICAL_SERIES_NAMES)`; `ingestion.py:200-203` `sanitize_nwb_object_name` delegates to the helper (or is replaced by an import) so there is one implementation.

Selection body replacing `nwb_helper_fn.py:380-389`:

```python
    wanted = {sanitize_nwb_object_name(n) for n in RAW_ELECTRICAL_SERIES_NAMES}
    matches = [n for n in names if sanitize_nwb_object_name(n) in wanted]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one raw acquisition ElectricalSeries named one of "
            f"{list(RAW_ELECTRICAL_SERIES_NAMES)} in {nwb_file_path}; found "
            f"{matches or 'none'} among acquisition ElectricalSeries {names}. "
            "Pass electrical_series_path='acquisition/<name>' explicitly to "
            "select one."
        )
    return f"acquisition/{matches[0]}"
```

Rationale for refusing multiple matches instead of taking the first: `Raw` iterates `nwb_file.objects` (pynwb insertion order) while this helper iterates h5py (alphabetical); "first" is not the same object in both. Loud beats a coin flip. The `electrical_series_path` override already exists at `spikesorting/utils.py:297`.

## multi-source-raise

Replace `_warn_multi_source` (`dj_merge_tables.py:793-808`) with:

```python
    def _raise_multi_source(self, sources) -> None:
        """Refuse a restriction that resolves to more than one source part.

        Each source resolves to a different parent class, so fetching across
        them returns files/ids from unrelated tables. Callers that aggregate
        across sources on purpose pass ``multi_source=True``.
        """
        raise ValueError(
            f"Merge.fetch_nwb: restriction spans {len(sources)} sources "
            f"({sorted(sources)}). Restrict to a single source, or pass "
            "multi_source=True to fetch across all of them."
        )
```

Both call sites (`:692-693`, `:765-766`) call the new name. Delete the `disable_warning` parameter from the signature at `:591` and its docstring entry. `SpikeSortingOutput.get_spike_times` (`spikesorting_merge.py:497`) aggregates across a group's merge ids by design: pass `multi_source=True` there. `SortedSpikesGroup.fetch_spike_data` (`group.py:201-202`) already opts in.

## export-file-retention

`common_usage.py:544-547`: keep the `Table` part deletion; for `File`, skip the deletion when any child table still references those rows, because `delete_quick` cannot cascade and the DANDI rows are upload provenance that must survive a re-export:

```python
            for export_id in overlap:
                id_dict = {"export_id": export_id}
                (self.Table & id_dict).delete_quick()
                files = self.File & id_dict
                referenced = [
                    child.full_table_name
                    for child in self.File().children(as_objects=True)
                    if child & files
                ]
                if referenced:
                    logger.info(
                        f"Keeping Export.File rows for superseded export_id "
                        f"{export_id}: still referenced by {referenced}"
                    )
                    continue
                files.delete_quick()
```

Alternative rejected: `(self.File & id_dict).super_delete(safemode=False)` cascades and deletes `DandiPath` rows, destroying the record of what was uploaded.

## unitannotation-migration

Two classmethods on `UnitAnnotation` (`unit_annotation.py`, after `add_annotation`). Both use the existing helpers `_get_spike_obj_name` and `_get_nwb_unit_ids` (`group.py:402-413`).

```python
    @classmethod
    def audit_positional_unit_ids(cls):
        """List merge ids whose NWB unit ids are not the dense range ``0..n-1``.

        For such merge ids a ``unit_id`` written under the older positional
        contract (index into the spike-times list) and one written under the
        current contract (NWB units-table id) can differ. Read-only and
        idempotent. Returns a DataFrame with columns ``spikesorting_merge_id``,
        ``n_units``, ``true_unit_ids``, ``stored_unit_ids``.
        """
        import pandas as pd

        rows = []
        for merge_id in set(cls.fetch("spikesorting_merge_id")):
            nwb_file = (
                SpikeSortingOutput & {"merge_id": merge_id}
            ).fetch_nwb()[0]
            name = _get_spike_obj_name(nwb_file, allow_empty=True)
            true_ids = _get_nwb_unit_ids(nwb_file, name) if name else []
            if true_ids == list(range(len(true_ids))):
                continue
            stored = sorted(
                int(u)
                for u in (cls & {"spikesorting_merge_id": merge_id}).fetch(
                    "unit_id"
                )
            )
            rows.append(
                dict(
                    spikesorting_merge_id=merge_id,
                    n_units=len(true_ids),
                    true_unit_ids=true_ids,
                    stored_unit_ids=stored,
                )
            )
        return pd.DataFrame(rows)

    @classmethod
    def migrate_positional_unit_ids(cls, *, dry_run: bool = True) -> dict:
        """Remap ``unit_id`` from positional index to NWB unit id, once.

        Run exactly once, immediately after upgrading to the release that
        changed ``UnitAnnotation.unit_id`` to the NWB units-table id, and before
        any new annotations are written. NOT idempotent: a second run would
        remap already-correct ids. Aborts without writing if any stored id is
        not a valid positional index (``>= n_units``), since such a row cannot
        have been written positionally.

        Returns ``{merge_id: {old_unit_id: new_unit_id}}`` (only changed ids).
        """
        audit = cls.audit_positional_unit_ids()
        plan = {}
        for row in audit.itertuples(index=False):
            true_ids = row.true_unit_ids
            invalid = [u for u in row.stored_unit_ids if u >= len(true_ids)]
            if invalid:
                raise ValueError(
                    f"UnitAnnotation rows for {row.spikesorting_merge_id} have "
                    f"unit_id(s) {invalid} >= n_units={len(true_ids)}; they "
                    "cannot be positional indices. Resolve by hand before "
                    "migrating."
                )
            mapping = {u: true_ids[u] for u in row.stored_unit_ids if true_ids[u] != u}
            if mapping:
                plan[row.spikesorting_merge_id] = mapping
        if dry_run or not plan:
            return plan
        with cls.connection.transaction:
            for merge_id, mapping in plan.items():
                restr = {"spikesorting_merge_id": merge_id}
                masters = (cls & restr).fetch(as_dict=True)
                parts = (cls.Annotation & restr).fetch(as_dict=True)
                (cls.Annotation & restr).delete_quick()
                (cls & restr).delete_quick()
                remap = lambda r: {**r, "unit_id": mapping.get(r["unit_id"], r["unit_id"])}
                cls.insert([remap(r) for r in masters])
                cls.Annotation.insert([remap(r) for r in parts])
        return plan
```

Delete-then-insert is required because `unit_id` is in the primary key (DataJoint cannot update a PK). Doing it per merge id inside one transaction avoids transient PK collisions when an old id equals another row's new id.

## concat-merge-gate

In `CurationV2.insert_curation` (`curation.py`), before the transaction block that ends at `:1384-1392`, resolve the source once and gate the merge registration:

```python
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        register_merge = source.kind == "recording"
```

Inside the transaction, replace the unconditional `SpikeSortingOutput._merge_insert(...)` with:

```python
            if register_merge:
                SpikeSortingOutput._merge_insert(
                    [{"sorting_id": sorting_id, "curation_id": curation_id}],
                    part_name="CurationV2",
                    skip_duplicates=True,
                )
            else:
                logger.warning(CONCAT_MERGE_GATE_MESSAGE.format(sorting_id=sorting_id))
```

with, at module level in `curation.py`:

```python
CONCAT_MERGE_GATE_MESSAGE = (
    "CurationV2 for concat-backed sorting {sorting_id} was NOT registered in "
    "SpikeSortingOutput: its spike times are on the concatenated recording's "
    "synthetic 0-based timeline (member wall-clock gaps are dropped), so a "
    "downstream consumer keyed by nwb_file_name would misalign them. Use "
    "ConcatenatedRecording.split_sorting_by_session for per-member frames; "
    "per-member decodable rows are a planned addition."
)
```

Also add a read-only audit for trial DBs that already registered concat rows:

```python
    @classmethod
    def audit_concat_merge_rows(cls) -> list[dict]:
        """Return SpikeSortingOutput.CurationV2 rows whose sorting is concat-backed."""
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        concat_sortings = SortingSelection.ConcatenatedRecordingSource.fetch("sorting_id")
        return (
            SpikeSortingOutput.CurationV2 & [{"sorting_id": s} for s in concat_sortings]
        ).fetch("KEY") if len(concat_sortings) else []
```

Run-summary plumbing: `_pipeline_run.py:804-807` reads `merge_id` with `fetch1`; for a concat run that raises. Change to `fetch("merge_id")` and set `merge_id = ids[0] if len(ids) else None`; append a warning string to `run_summary["warnings"]` when `None`. Type `root_merge_id: "UUID | None"` at `_pipeline_types.py:121`; docstring `_pipeline_run.py:330-351` states concat runs return `None` merge ids.

Phase 2 removes the gate's warning branch for concat sorts by populating member rows instead (see [concat-member-curation](#concat-member-curation)); the gate itself stays (the concat `CurationV2` row is still never registered directly).

## unitmatch-baseline

`_unitmatch_backend.py:237-249`. The bundle is symmetric (`ms_before == ms_after`, comment at `:144`), so the peak sits at `spike_width // 2`. Derive the baseline window from the waveform shape instead of the literal 15:

```python
def _zero_center(waveform: np.ndarray) -> np.ndarray:
    """Subtract each template's pre-spike baseline (SI templates carry a DC offset).

    The bundle is written with ``ms_before == ms_after`` so the peak sits at
    ``spike_width // 2``; the first quarter of the window is guaranteed
    pre-spike for any user ``ms_before``. ``waveform`` is
    ``(n_units, spike_width, n_channels, 2)``.
    """
    n_baseline = max(1, waveform.shape[1] // 4)
    baseline = waveform[:, :n_baseline, :, :].mean(axis=1)
    return waveform - baseline[:, np.newaxis, :, :]
```

Default `ms_before=1.5` at 30 kHz gives `spike_width=90`, `n_baseline=22` (previously 15): a longer, still pre-spike baseline. At `ms_before=0.3` (`spike_width=18`) it gives 4 samples instead of overlapping the peak.

## file-tracking-lazy-v2

`common_file_tracking.py:125-133`: do not import `spyglass.spikesorting.v2.recompute` (its import declares schemas) unless the schema already exists:

```python
        import datajoint as dj

        # Literal on purpose: importing the v2 module to read its schema name
        # would itself declare the schema on a v1-only database.
        if "spikesorting_v2_recompute" not in dj.list_schemas():
            return deleted
        try:
            from spyglass.spikesorting.v2.recompute import (
                RecordingArtifactRecompute as V2RecordingArtifactRecompute,
            )
            ...
```

A test in the v2 suite asserts `recompute.schema.database == "spikesorting_v2_recompute"` so the literal cannot drift.

## concat-member-curation

**Goal.** One decodable `SpikeSortingOutput` row per member session of a curated concat sort, with wall-clock spike times and the concat unit ids preserved.

**Why a new table rather than fixing timestamps on the concat row.** `SortedSpikesGroup` is `-> Session` keyed (`group.py:63-74`), decoding restricts by the session's `IntervalList`, and every `AnalysisNwbfile` belongs to one parent NWB. A single row cannot span sessions.

**Table** (new module `src/spyglass/spikesorting/v2/concat_member_curation.py`, new schema `spikesorting_v2_concat_curation`; additive):

```python
schema = dj.schema("spikesorting_v2_concat_curation")


@schema
class ConcatMemberCuration(SpyglassMixin, dj.Computed):
    """Per-member, wall-clock-aligned view of a curated concat sort.

    One row per (concat CurationV2, member_index). Spike frames are split back
    into the member's local sample frame with ``MemberBoundary`` and mapped to
    that member's Recording timestamps, so downstream consumers keyed by
    ``nwb_file_name`` (SortedSpikesGroup, decoding) see the same units and ids
    on every member session.
    """

    definition = """
    -> CurationV2
    member_index: int
    ---
    -> Session                      # the member session (from MemberSnapshot)
    -> AnalysisNwbfile
    object_id: varchar(72)
    n_units: int
    """

    @property
    def key_source(self):
        # Concat-backed curations only, expanded to their frozen members.
        return (
            CurationV2
            * SortingSelection.ConcatenatedRecordingSource
            * ConcatenatedRecordingSelection.MemberSnapshot
        ).proj()
```

`SortingSelection.ConcatenatedRecordingSource` carries `concat_recording_id` (`sorting.py:748-754`); `MemberSnapshot` is keyed by `(concat_recording_id, member_index)` and carries `nwb_file_name`, `sort_group_id`, `interval_list_name`, `team_name`, `recording_id` as plain columns (`session_group.py:482-496`). Confirm the exact column names by reading the `MemberSnapshot.definition` before writing `key_source`.

**make** (tri-part is not required; the compute is a read-split-write over one NWB and runs in seconds):

1. `fetch`: the concat `CurationV2` row (`analysis_file_name`, `object_id`), the `ConcatenatedRecording` row (`n_samples`, `sampling_frequency`), the ordered `MemberBoundary` `end_sample` list, the member's `MemberSnapshot` row and its `Recording` row (`analysis_file_name`, `electrical_series_path`), labels from `CurationV2.UnitLabel & key`, and the `MergeGroup` rows for provenance.
2. `compute`:
   - `abs_times, sample_indices = read_units_abs_times_and_sample_indices(curated_abs_path)` (`_units_nwb.py:100`) → concat-frame sample indices per unit.
   - `per_member = split_unit_spike_trains(sample_indices, boundaries, total_n_samples=n_samples)` (`_concat_recording.py:297`) → pick `per_member[member_index]` (local frames, unit ids preserved, empty arrays for absent units).
   - `ts = recording_timestamps(member_recording_row)` (`_units_nwb.py:531`); `abs_times_by_uid = {uid: ts[frames] for uid, frames in local.items()}` (frames are already `< len(ts)` by the split's conservation check; assert it).
   - `obs_intervals_by_uid` from the member Recording's valid intervals via `_base_intervals_from_recording` (`_units_nwb.py:519`) or the member `IntervalList` valid times; one interval set shared by all units.
   - Write with `_write_curated_units_nwb_body(analysis_file_name=<new staged AnalysisNwbfile for member nwb_file_name>, nwb_file_name=<member>, kept_unit_to_contributors=<from MergeGroup rows>, apply_merge=<concat row's merges_applied>, labels=<UnitLabel dict>, abs_times_by_uid=..., sample_indices_by_uid=local, obs_intervals_by_uid=..., curation_header=<concat curation provenance + member_index>, merge_group_rows=...)` (`_units_nwb.py:909`).
3. `insert`: `AnalysisNwbfile().add(...)`, `self.insert1(...)`, then `SpikeSortingOutput._merge_insert([key], part_name="ConcatMemberCuration", skip_duplicates=True)`, in one transaction (mirror `curation.py:1360-1392`).

**Merge part** in `spikesorting_merge.py`: probe `ConcatMemberCuration` the same way `_probe_v2_curation` probes `CurationV2` (`:38-62`), add to `source_class_dict` (`:104-110`), and declare:

```python
    if ConcatMemberCuration is not None:

        class ConcatMemberCuration(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> ConcatMemberCuration
            """
```

**Source contract.** The merge dispatch needs `get_recording`, `get_sorting`, `get_sort_group_info`, and `fetch_nwb` on the source class (`spikesorting_merge.py:353-372`, `:456-512`). Implement on `ConcatMemberCuration`: `get_recording` → `Recording().get_recording(member recording key)`; `get_sorting` → `numpysorting_from_abs_times(abs_times, member_recording_row, fs)` (`_units_nwb.py:214`) over this row's NWB; `get_sort_group_info` → delegate to `CurationV2.get_sort_group_info` restricted to the member's `sort_group_id` (regions are unambiguous per member, so no `ConcatBrainRegionAmbiguousError`); `fetch_nwb` from the mixin with `_nwb_table = AnalysisNwbfile`.

**Orchestration.** `run_v2_pipeline` concat path: after the curation stage, `ConcatMemberCuration.populate(curation_key)` and report `member_merge_ids: dict[str, UUID]` (`nwb_file_name → merge_id`) in the run summary alongside the `None` root/analysis merge ids; `describe_run` prints them. `run_v2_unit_match` continues to consume the concat `CurationV2` (not member rows).
