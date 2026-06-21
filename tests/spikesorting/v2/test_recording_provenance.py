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
