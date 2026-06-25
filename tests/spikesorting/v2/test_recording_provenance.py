"""Recording / Units NWB provenance.

Covers the ``ElectricalSeries.filtering`` provenance string reflecting the
preprocessing steps that actually ran, and the per-unit ``obs_intervals``
fallback to the recorded windows when no artifact pass was applied (vs the
artifact IntervalList's valid_times when one was).
"""

from __future__ import annotations

import numpy as np
import pytest


def _fresh_unit_producing_selection(populated_sorting):
    """Build a fresh MS5 ``SortingSelection`` on the fixture's
    recording+artifact (NOT yet populated); return its ``{"sorting_id"}``.

    The fallback test needs a sort that actually yields units so the Units NWB
    carries real ``obs_intervals``; the clusterless ``default`` row finds zero
    peaks on the MEArec smoke fixture. MS5 produces units.

    The selection is ARTIFACT-FREE (no ``artifact_detection_id``) so it is a DISTINCT
    row from the package fixture's artifact-backed MS5 sort -- otherwise
    ``insert_selection`` (find-existing-or-insert) would return the shared
    fixture's sort and a destructive test would delete shared state. The
    caller owns populate + teardown.
    """
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    SorterParameters.insert_default()
    return SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    )


@pytest.mark.usefixtures("dj_conn")
def test_filtering_description_reflects_actual_steps():
    """The persisted ``ElectricalSeries.filtering`` provenance is built from
    the preprocessing steps that ACTUALLY ran, not the old hardcoded
    "Bandpass filter + common reference" that misdescribed the no_filter /
    reference_mode='none' artifact.
    """
    from spyglass.spikesorting.v2._params.preprocessing import (
        BandpassFilterParams,
    )
    from spyglass.spikesorting.v2.recording import Recording

    bp = BandpassFilterParams(freq_min=300.0, freq_max=6000.0)
    no_ps = {"phase_shift": False}

    # No preprocessing at all -> must not claim a filter or a reference step.
    none = Recording._filtering_description(None, "none", no_ps)
    assert none == "none (raw, no preprocessing)"
    assert "bandpass" not in none.lower()
    assert "common reference" not in none.lower()

    # Bandpass only (reference_mode='none') -> no reference claim.
    bp_only = Recording._filtering_description(bp, "none", no_ps)
    assert "bandpass filter 300-6000 Hz" in bp_only
    assert "common reference" not in bp_only

    # Bandpass + common reference -> both steps named, in the RUNTIME APPLY
    # order (bandpass first, then reference -- the order is non-commutative on
    # the global-median branch, so the provenance must track the apply order).
    both = Recording._filtering_description(bp, "global_median", no_ps)
    assert "bandpass filter 300-6000 Hz" in both
    assert "common reference (global_median)" in both
    assert both.index("bandpass filter") < both.index(
        "common reference"
    ), "provenance must list bandpass before reference (the apply order)"

    # Phase-shift is named ONLY when the applied-step report says it ran, and
    # listed first; a requested-but-skipped phase-shift (report False) is not
    # claimed -- the provenance tracks what RAN, not what was requested.
    with_ps = Recording._filtering_description(
        bp, "global_median", {"phase_shift": True}
    )
    assert with_ps.startswith("phase-shift (ADC); bandpass filter 300-6000 Hz")
    assert "phase-shift" not in both


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_obs_intervals_recorded_windows_fallback(populated_sorting):
    """``obs_intervals=None`` (no artifact pass) writes the recorded
    window(s); an artifact-backed sort writes the artifact IntervalList's
    valid_times.

    The Units NWB carries ``obs_intervals`` per unit for downstream
    firing-rate windows. When no artifact mask was applied the fallback is
    the recording's recorded window(s) -- split at wall-clock
    discontinuities, so a contiguous recording yields a single interval
    (asserted here) and a DISJOINT recording yields one per chunk (asserted
    in ``test_obs_intervals_no_artifact_respects_disjoint_gap``). When an
    artifact pass exists the obs_intervals come from its IntervalList (the
    make_fetch path), not the fallback. We populate an artifact-FREE MS5
    sort for the fallback and reuse the artifact-backed fixture for the
    IntervalList path.
    """
    import pynwb

    from spyglass.common import IntervalList
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
    )

    def _first_unit_obs_intervals(sort_pk):
        analysis_file_name = (Sorting & sort_pk).fetch1("analysis_file_name")
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        with pynwb.NWBHDF5IO(
            path=abs_path, mode="r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            return np.asarray(nwbf.units["obs_intervals"][0])

    # --- artifact-FREE sort: full-envelope fallback -----------------------
    free_pk = _fresh_unit_producing_selection(populated_sorting)
    try:
        Sorting.populate(free_pk, reserve_jobs=False)
        recording_id = SortingSelection.resolve_source(free_pk).key[
            "recording_id"
        ]
        rec = Recording().get_recording({"recording_id": recording_id})
        times = rec.get_times()
        obs = _first_unit_obs_intervals(free_pk)
        assert obs.shape == (1, 2), (
            "no-artifact sort over a CONTIGUOUS recording must observe one "
            "recorded interval (base intervals collapse to the envelope)"
        )
        assert abs(obs[0][0] - float(times[0])) < 1e-6
        assert abs(obs[0][1] - float(times[-1])) < 1e-6
    finally:
        (Sorting & free_pk).delete(safemode=False)
        (SortingSelection & free_pk).delete(safemode=False)

    # --- artifact-backed fixture: obs_intervals from the IntervalList -----
    artifact_detection_id = SortingSelection.resolve_artifact_detection(
        populated_sorting
    )
    assert (
        artifact_detection_id is not None
    ), "fixture sort should be artifact-backed"
    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    nwb_file_name = (
        RecordingSelection & {"recording_id": recording_id}
    ).fetch1("nwb_file_name")
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": artifact_detection_interval_list_name(
                artifact_detection_id
            ),
        }
    ).fetch1("valid_times")
    obs = _first_unit_obs_intervals(populated_sorting)
    np.testing.assert_allclose(obs, np.asarray(valid_times), rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# channel_name resolution on a real-NWB-shape fixture.
#
# ``Recording._spikeinterface_channel_ids`` resolves channel ids from the raw
# NWB ``channel_name`` column when present and falls back to the integer
# ``electrode_id`` when absent. This exercises both branches on a fixture built
# to match production Frank-lab NWB shape.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "channel_names", [None, ["ch_a", "ch_b", "ch_c", "ch_d"]]
)
def test_channel_name_resolution_path_real_nwb(
    dj_conn, tmp_path, monkeypatch, channel_names
):
    """``_spikeinterface_channel_ids`` resolves channel ids from the raw
    NWB ``channel_name`` column when present, and falls back to integer
    ``electrode_id`` when absent.

    The MEArec fixtures omit ``channel_name`` so only the integer-fallback
    branch was exercised; production Frank-lab NWBs carry the column. This
    test builds a 4-contact NWB via the fixture builder's ``channel_names``
    parameter (injecting the column) and via the default (no column), then
    asserts the resolved SpikeInterface channel ids match each branch's
    expected mapping.
    """
    from datetime import datetime, timezone

    import pynwb

    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        tetrode_probe_layout,
    )
    from spyglass.spikesorting.v2.recording import Recording

    nwbfile = pynwb.NWBFile(
        session_description="channel_name resolution fixture",
        identifier="a19-chan-name",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    _add_probe_and_electrodes(
        nwbfile,
        tetrode_probe_layout(),
        targeted_location="hpc",
        channel_names=channel_names,
    )
    out = tmp_path / "a19_channel_name_fixture.nwb"
    with pynwb.NWBHDF5IO(str(out), mode="w") as io:
        io.write(nwbfile)

    # _spikeinterface_channel_ids resolves the raw path via Nwbfile; redirect
    # it to our standalone fixture (no ingestion needed for the lookup).
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(out))
    )

    spyglass_ids = [0, 1, 2, 3]
    resolved = Recording._spikeinterface_channel_ids(
        "a19_channel_name_fixture.nwb", spyglass_ids
    )

    if channel_names is None:
        assert resolved == [0, 1, 2, 3], (
            "integer-fallback branch must return int electrode_ids; got "
            f"{resolved!r}"
        )
        assert all(isinstance(c, int) for c in resolved)
    else:
        assert resolved == channel_names, (
            "channel_name branch must resolve to the injected string names "
            f"in electrode order; got {resolved!r}"
        )


# --------------------------------------------------------------------------- #
# channel_name lookup must map electrode_id -> electrodes-table ROW index.
#
# The raw NWB ``channel_name`` column is positional: row k holds channel k's
# name. Spyglass electrode ids are NOT guaranteed to equal row positions
# (production NWBs can be non-contiguous, non-zero-based, or reordered).
# Indexing ``channel_name`` by the electrode id treats the id as a row index,
# so v2 could slice raw channel B and rename it as electrode A -- the sorter
# then sees a different channel's signal than the row implies. The mapping must
# go through the electrodes-table id->row resolution the write path already
# uses (``get_electrode_indices``).
# --------------------------------------------------------------------------- #


def _write_electrode_table_nwb(path, electrode_ids, *, channel_names=None):
    """Write a minimal NWB whose electrodes table has ``electrode_ids`` (in row
    order) and an optional positional ``channel_name`` column.

    ``electrode_ids[k]`` becomes the id of electrodes-table row ``k``;
    ``channel_names[k]`` (when given) is that row's channel name. Pass
    non-contiguous / shuffled ids so id-as-row-index and a correct id->row
    mapping give different answers.
    """
    from datetime import datetime, timezone

    import pynwb

    nwbfile = pynwb.NWBFile(
        session_description="electrode id->row mapping fixture",
        identifier="electrode-id-mapping",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe0")
    electrode_group = nwbfile.create_electrode_group(
        name="0", description="test group", location="hpc", device=device
    )
    for eid in electrode_ids:
        nwbfile.add_electrode(id=int(eid), location="hpc", group=electrode_group)
    if channel_names is not None:
        nwbfile.add_electrode_column(
            name="channel_name",
            description="Per-channel string name (Frank-lab production NWBs)",
            data=list(channel_names),
        )
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_channel_name_maps_by_electrode_id_not_row(
    dj_conn, tmp_path, monkeypatch
):
    """``spikeinterface_channel_ids`` resolves each electrode id to its
    electrodes-table ROW, then reads ``channel_name`` -- it must NOT treat the
    electrode id as a row index.

    The electrodes table has non-contiguous ids ``[10, 11, 12, 13]`` in row
    order, so the buggy ``channel_names[int(electrode_id)]`` would index row
    12 / row 10 (out of range) instead of the rows holding electrode ids 12
    and 10. Requesting ids ``[12, 10]`` must return their own channel names.
    """
    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2.recording import Recording

    ids_in_row_order = [10, 11, 12, 13]
    names_in_row_order = ["c10", "c11", "c12", "c13"]
    path = tmp_path / "noncontiguous_ids.nwb"
    _write_electrode_table_nwb(
        path, ids_in_row_order, channel_names=names_in_row_order
    )
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(path))
    )

    resolved = Recording._spikeinterface_channel_ids(
        "noncontiguous_ids.nwb", [12, 10]
    )
    assert resolved == ["c12", "c10"], (
        "channel_name lookup must map electrode id -> table row (id 12 -> row "
        "2 -> 'c12', id 10 -> row 0 -> 'c10'), not index the column by the id "
        f"as a row position; got {resolved!r}"
    )


def test_missing_electrode_id_raises(dj_conn, tmp_path, monkeypatch):
    """A requested electrode id absent from the electrodes table raises,
    rather than silently mis-indexing the ``channel_name`` column.

    With non-contiguous ids ``[10, 11, 12, 13]`` the buggy
    ``channel_names[int(electrode_id)]`` would silently return rows 1 and 2
    for the absent ids 1 and 2 (a wrong-channel mapping with no error). The
    id->row mapping has no entry for those ids and must raise.
    """
    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2.recording import Recording

    path = tmp_path / "missing_id.nwb"
    _write_electrode_table_nwb(
        path, [10, 11, 12, 13], channel_names=["c10", "c11", "c12", "c13"]
    )
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(path))
    )

    with pytest.raises(ValueError, match="electrodes table"):
        Recording._spikeinterface_channel_ids("missing_id.nwb", [1, 2])


@pytest.mark.slow
def test_restrict_recording_carries_correct_traces(
    dj_conn, tmp_path, monkeypatch
):
    """After ``restrict_recording`` renames sliced SI channels back to electrode
    ids, each renamed electrode id carries ITS OWN raw trace.

    Builds an electrodes table whose ids are a shuffled in-range permutation
    ``[2, 3, 0, 1]`` (so id-as-row-index silently maps to the wrong channel)
    with distinguishable per-channel constant traces. Requesting electrode ids
    ``[0, 1]`` must slice the SI channels for rows holding ids 0 and 1 (rows 2
    and 3), so the renamed recording's id 0 carries row-2's trace and id 1
    carries row-3's. The buggy id-as-row lookup would instead carry rows 0/1.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.common import Nwbfile
    from spyglass.spikesorting.v2._recording_restriction import (
        restrict_recording,
    )

    # Row order: id 2 -> row 0 (channel s0), id 3 -> row 1 (s1),
    #            id 0 -> row 2 (s2), id 1 -> row 3 (s3).
    ids_in_row_order = [2, 3, 0, 1]
    si_channel_names = ["s0", "s1", "s2", "s3"]
    path = tmp_path / "shuffled_traces.nwb"
    _write_electrode_table_nwb(
        path, ids_in_row_order, channel_names=si_channel_names
    )
    monkeypatch.setattr(
        Nwbfile, "get_abs_path", staticmethod(lambda *a, **k: str(path))
    )

    # Distinguishable constant traces: SI channel ``s{k}`` carries value
    # k * 1000, so a mis-mapped channel is unambiguous on readback.
    fs, n_samples, n_channels = 1000.0, 1000, 4
    traces = np.empty((n_samples, n_channels), dtype="float32")
    for k in range(n_channels):
        traces[:, k] = float(k * 1000)
    recording = NumpyRecording(
        traces_list=[traces],
        sampling_frequency=fs,
        channel_ids=si_channel_names,
    )

    valid_times = np.array([[0.0, 1.0]])
    sliced, _override, _n = restrict_recording(
        recording=recording,
        nwb_file_name="shuffled_traces.nwb",
        interval_list_name="raw data valid times",
        sort_group_channel_ids=[0, 1],
        reference_mode="none",
        reference_electrode_id=None,
        sort_valid_times=valid_times,
        raw_valid_times=valid_times,
        min_segment_length=0.0,
        bad_channel_handling="remove",
        bad_channel_ids=(),
    )

    assert list(sliced.get_channel_ids()) == [0, 1]
    # id 0 -> row 2 -> SI channel s2 -> trace value 2000.
    assert float(sliced.get_traces(channel_ids=[0])[0, 0]) == 2000.0, (
        "renamed electrode id 0 must carry its own raw trace (row 2 / s2 = "
        "2000), not the id-as-row-index channel"
    )
    # id 1 -> row 3 -> SI channel s3 -> trace value 3000.
    assert float(sliced.get_traces(channel_ids=[1])[0, 0]) == 3000.0, (
        "renamed electrode id 1 must carry its own raw trace (row 3 / s3 = "
        "3000), not the id-as-row-index channel"
    )
