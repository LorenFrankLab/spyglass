"""``Sorting.Unit.peak_amplitude_uv`` is the template extremum on the
attributed electrode.

Regression guard for the two coupled defects in sort-time unit attribution:

* F1: ``get_template_extremum_amplitude`` defaulted to ``mode="at_index"``
  (the value at the alignment sample), under-reporting the true peak.
* F2: ``at_index`` re-picks its own best channel, which could differ from
  the electrode FK (``get_template_extremum_channel``, ``mode="extremum"``).

The fix passes ``mode="extremum"`` (and the configured ``peak_sign``) to the
amplitude call, so the stored amplitude is the template PEAK on the SAME
channel as the attributed electrode. This test recomputes the extremum
directly from the analyzer's template array (an independent code path, not
``template_tools``) and asserts equality + channel consistency.

Scope note: this runs on the MountainSort5 ``populated_sorting`` fixture,
whose templates are aligned to the trough -- so ``at_index`` coincides with
``extremum`` and the fix is a no-op here (verified: the test passes with and
without ``mode="extremum"``). It is therefore an *invariant* guard
(``peak_amplitude_uv`` is the extremum on the attributed electrode), not a
behavioral repro. The fix's behavioral effect is on sorters whose detection
alignment differs from the template peak; a discriminating clusterless repro
is not yet wired up.
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

    # The analyzer folder is no longer a column; load via the accessor,
    # which resolves the path from sorting_id and rebuilds on miss.
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

        # F2: amplitude is on the SAME channel as the attributed electrode.
        assert chan_ids[best_idx] == int(row["electrode_id"]), (
            f"unit {uid}: peak amplitude channel {chan_ids[best_idx]} != "
            f"attributed electrode {int(row['electrode_id'])}"
        )
        # F1: stored amplitude is the extremum (not the at-index value).
        assert np.isclose(
            abs(float(row["peak_amplitude_uv"])),
            expected_amp,
            rtol=1e-2,
            atol=1e-2,
        ), (
            f"unit {uid}: stored peak_amplitude_uv "
            f"{row['peak_amplitude_uv']} != template extremum {expected_amp}"
        )
