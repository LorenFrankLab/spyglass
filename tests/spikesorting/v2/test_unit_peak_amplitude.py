"""``Sorting.Unit.peak_amplitude_uv`` is the template extremum on the
attributed electrode.

Guards two coupled failure modes of sort-time unit attribution:

* Value: ``get_template_extremum_amplitude`` defaults to ``mode="at_index"``
  (the value at the alignment sample), which under-reports the true peak.
* Channel: ``at_index`` re-picks its own best channel, which could differ from
  the electrode FK (``get_template_extremum_channel``, ``mode="extremum"``).

v2 passes ``mode="extremum"`` (and the configured ``peak_sign``) to the
amplitude call, so the stored amplitude is the template PEAK on the SAME
channel as the attributed electrode. This test recomputes the extremum
directly from the analyzer's template array (an independent code path, not
``template_tools``) and asserts equality + channel consistency.

The MountainSort5 integration fixture is trough-aligned, so it checks the
persisted invariant. The synthetic row-producer regression deliberately
places the strongest peak after alignment on a different electrode, with
distinct gains and positive/negative polarity, to distinguish the modes.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_peak_amplitude_is_extremum_on_attributed_electrode(populated_sorting):
    import spikeinterface as si

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from spyglass.spikesorting.v2.utils import resolve_peak_sign

    # The analyzer folder is not a column; load via the accessor, which
    # resolves the path from sorting_id and rebuilds on miss.
    analyzer = Sorting().get_analyzer(populated_sorting)
    templates = analyzer.get_extension("templates").get_data()
    analyzer_unit_ids = [int(u) for u in analyzer.unit_ids]
    chan_ids = [int(c) for c in analyzer.channel_ids]

    params = (
        SortingSelection * SorterParameters
        & {"sorting_id": populated_sorting["sorting_id"]}
    ).fetch1("params")
    peak_sign = resolve_peak_sign(params)

    unit_rows = (Sorting.Unit & populated_sorting).fetch(
        "unit_id", "electrode_id", "peak_amplitude_uv", as_dict=True
    )
    assert unit_rows, "fixture must produce >=1 sorted unit"

    for row in unit_rows:
        uid = int(row["unit_id"])
        template = templates[analyzer_unit_ids.index(uid)]  # (n_time, n_chan)

        # Independent recompute of the per-channel extremum for this
        # peak_sign, then the extremum channel + its magnitude.
        if peak_sign == "neg":
            chan_peak = template.min(axis=0)
            best_idx = int(np.argmin(chan_peak))
            expected_amp = abs(float(chan_peak[best_idx]))
        elif peak_sign == "pos":
            chan_peak = template.max(axis=0)
            best_idx = int(np.argmax(chan_peak))
            expected_amp = float(chan_peak[best_idx])
        else:  # both
            chan_peak = np.abs(template).max(axis=0)
            best_idx = int(np.argmax(chan_peak))
            expected_amp = float(chan_peak[best_idx])

        # Channel: amplitude is on the SAME channel as the attributed electrode.
        assert chan_ids[best_idx] == int(row["electrode_id"]), (
            f"unit {uid}: peak amplitude channel {chan_ids[best_idx]} != "
            f"attributed electrode {int(row['electrode_id'])}"
        )
        # Value: stored amplitude is the extremum (not the at-index value).
        assert np.isclose(
            abs(float(row["peak_amplitude_uv"])),
            expected_amp,
            rtol=1e-2,
            atol=1e-2,
        ), (
            f"unit {uid}: stored peak_amplitude_uv "
            f"{row['peak_amplitude_uv']} != template extremum {expected_amp}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "sorter_params, sign",
    [
        ({"peak_sign": "neg"}, -1),
        ({"peak_sign": "pos"}, 1),
        ({"detect_sign": -1}, -1),
        ({"detect_sign": 1}, 1),
        ({"peak_sign": "both"}, -1),
    ],
)
def test_row_producer_uses_off_alignment_extremum_and_configured_sign(
    tmp_path, sorter_params, sign
):
    """The actual row producer connects polarity, peak time, gains and IDs."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from spyglass.spikesorting.v2._sorting_units import (
        build_unit_rows_from_analyzer,
    )

    frames = np.array([500, 1500, 2500])
    traces = np.zeros((10_000, 2), dtype=np.float32)
    # At the detection frame electrode 24 leads (20 counts * 3 = 60 uV).
    # The true extremum is one sample later on electrode 12 (50 * 2 = 100 uV).
    traces[frames, 1] = sign * 20
    traces[frames + 1, 0] = sign * 50
    recording = si.NumpyRecording(
        traces, sampling_frequency=1000.0, channel_ids=[12, 24]
    )
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    recording.set_channel_gains([2.0, 3.0])
    recording.set_channel_offsets(0.0)
    recording = recording.save(
        folder=tmp_path / "recording", n_jobs=1, progress_bar=False
    )
    sorting = si.NumpySorting.from_unit_dict(
        {17: frames}, sampling_frequency=1000.0
    )
    folder = tmp_path / "off_alignment.analyzer"
    build_analyzer(
        sorting,
        recording,
        {"sorting_id": "off_alignment"},
        sorter_row={"job_kwargs": {}},
        job_kwargs={"n_jobs": 1, "progress_bar": False, "random_seed": 0},
        analyzer_folder=folder,
        waveform_params={
            "ms_before": 2.0,
            "ms_after": 3.0,
            "max_spikes_per_unit": 10,
            "whiten": False,
            "purpose": "display",
            "sparsity": {"method": "dense"},
        },
    )
    electrode_by_id = {
        electrode_id: {
            "nwb_file_name": "synthetic.nwb",
            "electrode_group_name": "probe",
            "electrode_id": electrode_id,
        }
        for electrode_id in (12, 24)
    }
    rows = build_unit_rows_from_analyzer(
        sorting=sorting,
        analyzer_folder=folder,
        sorter_row={"params": sorter_params},
        electrode_by_id=electrode_by_id,
        sort_group_id=0,
        nwb_file_name="synthetic.nwb",
        key={"sorting_id": "off_alignment"},
    )
    assert rows == [
        {
            "sorting_id": "off_alignment",
            "unit_id": 17,
            "nwb_file_name": "synthetic.nwb",
            "electrode_group_name": "probe",
            "electrode_id": 12,
            "peak_amplitude_uv": 100.0,
            "n_spikes": 3,
        }
    ]
