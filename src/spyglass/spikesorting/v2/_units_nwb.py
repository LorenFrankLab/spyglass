"""Units-NWB read/write IO for v2 sorts.

These functions are the filesystem-IO core behind ``Sorting`` (and reused
by ``CurationV2``): reading a units NWB's stored ABSOLUTE spike times,
mapping those times back to recording frames, reading the recording's
persisted timestamp vector, and writing the pre-curation sorting-units
NWB. They take already-resolved paths / SpikeInterface objects / fetched
row dicts and do pynwb IO; they hold no DataJoint table knowledge of
their own.

Why this lives in its own module rather than in ``sorting.py``:
``sorting.py`` is a DataJoint *schema* module -- importing it activates
``dj.schema(...)`` and the source-part / merge dependencies. The units-NWB
IO needs none of that, so ``Sorting`` becomes a thin orchestrator (fetch
row -> resolve path -> call these -> insert) and ``CurationV2`` can reach
the SHARED readback helpers here instead of reaching into ``Sorting``'s
private methods. Same "thin DataJoint shell over pure/IO services"
direction as ``_artifact_compute`` / ``_selection_identity`` /
``_analyzer_cache`` / ``_curation_transforms``.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection at import: like ``_analyzer_cache``, the only DataJoint
dependency (``AnalysisNwbfile`` for path resolution / file creation) is
imported lazily at call time. The IO itself is pynwb against the
filesystem, not the database.
"""

from __future__ import annotations


def read_units_abs_spike_times(abs_path) -> dict:
    """Return ``{unit_id(int): abs_spike_times(np.ndarray seconds)}``.

    Reads the stored absolute spike times directly from a v2 units
    NWB (``nwbf.units.to_dataframe()``), so callers get the persisted
    wall-clock values exactly -- no affine round-trip. Returns ``{}``
    for an empty/absent Units table.
    """
    import numpy as _np
    import pynwb

    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        if nwbf.units is None or len(nwbf.units) == 0:
            return {}
        units_df = nwbf.units.to_dataframe()
    return {
        int(uid): _np.asarray(st, dtype=float)
        for uid, st in units_df["spike_times"].items()
    }


def numpysorting_from_abs_times(abs_times, recording_row, fs):
    """Build a ``NumpySorting`` from absolute spike times.

    Maps each unit's absolute spike times to recording frame indices
    with ``np.searchsorted`` against the recording's (possibly
    gap-preserving) timestamps -- the v1-parity readback that an
    affine inverse breaks across wall-clock gaps.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2.utils import _spike_times_to_frames

    recording_times = recording_timestamps(recording_row)
    n_samples = int(recording_times.size)
    units_dict = {
        uid: _spike_times_to_frames(recording_times, st, n_samples, uid)
        for uid, st in abs_times.items()
    }
    return si.NumpySorting.from_unit_dict([units_dict], sampling_frequency=fs)


def abs_spike_times_dataframe(abs_times):
    """DataFrame (index=unit_id) of absolute spike-time arrays."""
    import pandas as pd

    unit_ids = list(abs_times)
    return pd.DataFrame(
        {"spike_times": [abs_times[u] for u in unit_ids]},
        index=pd.Index(unit_ids, name="unit_id"),
    )


def empty_spike_times_dataframe():
    """Empty (index=unit_id) spike-times DataFrame for zero-unit sorts."""
    import pandas as pd

    return pd.DataFrame(
        {"spike_times": []},
        index=pd.Index([], name="unit_id", dtype=int),
    )


def recording_timestamps(recording_row):
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

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    abs_path = AnalysisNwbfile.get_abs_path(recording_row["analysis_file_name"])
    series_name = recording_row["electrical_series_path"].rsplit("/", 1)[-1]
    with pynwb.NWBHDF5IO(path=abs_path, mode="r", load_namespaces=True) as io:
        nwbf = io.read()
        series = nwbf.acquisition[series_name]
        return _np.asarray(series.timestamps[:], dtype=_np.float64)


def write_sorting_units_nwb(
    sorting, recording, nwb_file_name, obs_intervals=None
):
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

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    analysis_file_name = AnalysisNwbfile().create(nwb_file_name=nwb_file_name)
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
