"""Recording / Units NWB provenance.

Covers the ``ElectricalSeries.filtering`` string, source provenance,
``obs_intervals`` and ``channel_name`` resolution.
"""

from __future__ import annotations

import numpy as np
import pytest


def _fresh_unit_producing_selection(populated_sorting):
    """Build a fresh MS5 ``SortingSelection`` on the fixture's
    recording+artifact (NOT yet populated); return its ``{"sorting_id"}``.

    MS5 yields units on the smoke fixture. Artifact-free, so it is distinct
    from the shared artifact-backed sort. The caller owns populate and
    teardown.
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


def test_filtering_description_reflects_actual_steps():
    """``ElectricalSeries.filtering`` names only the steps that ran."""
    from spyglass.spikesorting.v2._params.preprocessing import (
        BandpassFilterParams,
    )
    from spyglass.spikesorting.v2._recording.preprocessing import (
        filtering_description,
    )

    bp = BandpassFilterParams(freq_min=300.0, freq_max=6000.0)
    no_ps = {"phase_shift": False}

    # No preprocessing at all -> must not claim a filter or a reference step.
    none = filtering_description(None, "none", no_ps)
    assert none == "none (raw, no preprocessing)"
    assert "bandpass" not in none.lower()
    assert "common reference" not in none.lower()

    # Bandpass only (reference_mode='none') -> no reference claim.
    bp_only = filtering_description(bp, "none", no_ps)
    assert "bandpass filter 300-6000 Hz" in bp_only
    assert "common reference" not in bp_only

    # Apply order: bandpass, then reference (non-commutative for median).
    both = filtering_description(bp, "global_median", no_ps)
    assert "bandpass filter 300-6000 Hz" in both
    assert "common reference (global_median)" in both
    assert both.index("bandpass filter") < both.index(
        "common reference"
    ), "provenance must list bandpass before reference (the apply order)"

    # Phase-shift is named first, and only if the step report says it ran.
    with_ps = filtering_description(bp, "global_median", {"phase_shift": True})
    assert with_ps.startswith("phase-shift (ADC); bandpass filter 300-6000 Hz")
    assert "phase-shift" not in both


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_recording_nwb_carries_source_provenance(populated_sorting):
    """The Recording artifact NWB embeds the full source-provenance set.

    A shared NWB must be interpretable without the database.
    """
    import spikeinterface as si

    from spyglass.common import Raw
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._storage.provenance import (
        RECORDING_PROVENANCE,
        read_provenance_values,
    )
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    sel = (RecordingSelection & {"recording_id": recording_id}).fetch1()
    nwb_file_name = sel["nwb_file_name"]
    sort_group_id = int(sel["sort_group_id"])

    reference_mode = str(
        (
            SortGroupV2
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch1("reference_mode")
    )
    raw_object_id = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
        "raw_object_id"
    )
    pp_row = (
        PreprocessingParameters
        & {"preprocessing_params_name": sel["preprocessing_params_name"]}
    ).fetch1()
    bad_channel_handling = PreprocessingParamsSchema.model_validate(
        pp_row["params"]
    ).bad_channel_handling

    analysis_file_name = (Recording & {"recording_id": recording_id}).fetch1(
        "analysis_file_name"
    )
    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

    prov = read_provenance_values(abs_path, RECORDING_PROVENANCE)

    # JSON has no UUID type.
    assert prov["recording_id"] == str(recording_id)
    assert prov["raw_object_id"] == raw_object_id
    assert prov["preprocessing_params_name"] == sel["preprocessing_params_name"]
    assert prov["sort_group_id"] == sort_group_id
    assert prov["reference_mode"] == reference_mode
    assert prov["bad_channel_handling"] == str(bad_channel_handling)
    assert prov["spikeinterface_version"] == si.__version__


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_obs_intervals_recorded_windows_fallback(populated_sorting):
    """``obs_intervals=None`` (no artifact pass) writes the recorded
    window(s); an artifact-backed sort writes the artifact IntervalList's
    valid_times.

    A contiguous recording yields one interval; the disjoint case is in
    ``test_obs_intervals_no_artifact_respects_disjoint_gap``.
    """
    import pynwb

    from spyglass.common import IntervalList
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._artifacts.naming import (
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
# channel_name resolution.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "channel_names", [None, ["ch_a", "ch_b", "ch_c", "ch_d"]]
)
def test_channel_name_resolution_path_real_nwb(
    dj_conn, tmp_path, channel_names
):
    """``spikeinterface_channel_ids`` resolves channel ids from the raw
    NWB ``channel_name`` column when present, and falls back to integer
    ``electrode_id`` when absent.

    MEArec fixtures omit the column; Frank-lab NWBs carry it.
    """
    from datetime import datetime, timezone

    import pynwb

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        tetrode_probe_layout,
    )
    from spyglass.spikesorting.v2._recording.geometry import (
        spikeinterface_channel_ids,
    )

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

    # Reads the file directly; no ingestion needed.
    spyglass_ids = [0, 1, 2, 3]
    resolved = spikeinterface_channel_ids(str(out), spyglass_ids)

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
# ``channel_name`` is positional; electrode ids need not equal row positions.
# --------------------------------------------------------------------------- #


def _write_electrode_table_nwb(path, electrode_ids, *, channel_names=None):
    """Write a minimal NWB whose electrodes table has ``electrode_ids`` (in row
    order) and an optional positional ``channel_name`` column.

    ``electrode_ids[k]`` is row ``k``'s id and ``channel_names[k]`` its name.
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
        nwbfile.add_electrode(
            id=int(eid), location="hpc", group=electrode_group
        )
    if channel_names is not None:
        nwbfile.add_electrode_column(
            name="channel_name",
            description="Per-channel string name (Frank-lab production NWBs)",
            data=list(channel_names),
        )
    with pynwb.NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)


def test_channel_name_maps_by_electrode_id_not_row(dj_conn, tmp_path):
    """``spikeinterface_channel_ids`` maps each electrode id to its row before
    reading ``channel_name``; ids ``[10, 11, 12, 13]`` are non-contiguous.
    """
    from spyglass.spikesorting.v2._recording.geometry import (
        spikeinterface_channel_ids,
    )

    ids_in_row_order = [10, 11, 12, 13]
    names_in_row_order = ["c10", "c11", "c12", "c13"]
    path = tmp_path / "noncontiguous_ids.nwb"
    _write_electrode_table_nwb(
        path, ids_in_row_order, channel_names=names_in_row_order
    )

    resolved = spikeinterface_channel_ids(str(path), [12, 10])
    assert resolved == ["c12", "c10"], (
        "channel_name lookup must map electrode id -> table row (id 12 -> row "
        "2 -> 'c12', id 10 -> row 0 -> 'c10'), not index the column by the id "
        f"as a row position; got {resolved!r}"
    )


def test_missing_electrode_id_raises(dj_conn, tmp_path):
    """An electrode id absent from the electrodes table raises."""
    from spyglass.spikesorting.v2._recording.geometry import (
        spikeinterface_channel_ids,
    )

    path = tmp_path / "missing_id.nwb"
    _write_electrode_table_nwb(
        path, [10, 11, 12, 13], channel_names=["c10", "c11", "c12", "c13"]
    )

    with pytest.raises(ValueError, match="'missing_id.nwb' electrodes table"):
        spikeinterface_channel_ids(str(path), [1, 2])


@pytest.mark.slow
def test_channel_selection_carries_correct_traces(dj_conn, tmp_path):
    """After ``select_sort_group_channels`` renames sliced SI channels back to
    electrode ids, each renamed electrode id carries ITS OWN raw trace.

    Row ids are the in-range permutation ``[2, 3, 0, 1]``.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._recording.restriction import (
        select_sort_group_channels,
    )

    # Row order: id 2 -> row 0 (channel s0), id 3 -> row 1 (s1),
    #            id 0 -> row 2 (s2), id 1 -> row 3 (s3).
    ids_in_row_order = [2, 3, 0, 1]
    si_channel_names = ["s0", "s1", "s2", "s3"]
    path = tmp_path / "shuffled_traces.nwb"
    _write_electrode_table_nwb(
        path, ids_in_row_order, channel_names=si_channel_names
    )

    # SI channel ``s{k}`` carries the constant k * 1000.
    fs, n_samples, n_channels = 1000.0, 1000, 4
    traces = np.empty((n_samples, n_channels), dtype="float32")
    for k in range(n_channels):
        traces[:, k] = float(k * 1000)
    recording = NumpyRecording(
        traces_list=[traces],
        sampling_frequency=fs,
        channel_ids=si_channel_names,
    )

    sliced = select_sort_group_channels(
        recording,
        nwb_file_abs_path=str(path),
        sort_group_channel_ids=[0, 1],
        reference_mode="none",
        reference_electrode_id=None,
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
