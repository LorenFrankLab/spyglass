"""Match separately sorted, disjoint epochs on one original NWB clock."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pipeline


def test_disjoint_epochs_match_without_inflating_detected_sessions(
    daily_concat_match_inputs, monkeypatch
):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import OverlappingMatchInputError
    from spyglass.spikesorting.v2.matcher_protocol import get_input_preparer
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for
    from tests.spikesorting.v2._sorter_stub import plant_sorter
    from tests.spikesorting.v2._unitmatch_helpers import (
        install_fixture_pairer,
        restore_matcher_registry,
    )
    from tests.spikesorting.v2.conftest import _plant_spread_unit

    fx = daily_concat_match_inputs
    early = fx["curations"]["single_a_first"]
    plant_sorter(monkeypatch, _plant_spread_unit)
    late_sort = SortingSelection.insert_selection(
        {
            **fx["recording_keys"]["a_second"],
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    )
    Sorting.populate(late_sort, reserve_jobs=False)
    late = CurationV2.insert_curation(sorting_key=late_sort)
    params_name = "disjoint_epoch_pairer_params"
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="disjoint_epoch_pairer",
        matcher_params_name=params_name,
        pairs=[[0, 0]],
    )

    def extract(directory, recording, sorting, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(
        get_input_preparer("disjoint_epoch_pairer"), "extract", extract
    )
    pk = None
    try:
        # Reverse caller order: original acquisition position must win.
        pk = UnitMatchSelection.insert_inputs([late, early], params_name)
        assert (
            UnitMatchSelection.insert_inputs([early, late], params_name) == pk
        )
        recordings = (UnitMatchSelection.InputRecording & pk).fetch(
            as_dict=True, order_by="input_index"
        )
        assert len(recordings) == 2
        assert {r["nwb_file_name"] for r in recordings} == {
            fx["nwb_file_names"]["a"]
        }
        assert (
            recordings[0]["source_coverage"]["sample_ranges"][-1][1]
            <= recordings[1]["source_coverage"]["sample_ranges"][0][0]
        )
        with pytest.raises(OverlappingMatchInputError, match="raw samples"):
            UnitMatchSelection.insert_inputs(
                [fx["curations"]["single_a"], early], params_name
            )
        UnitMatch.populate(pk, reserve_jobs=False)
        TrackedUnit.populate(pk, reserve_jobs=False)
        tracked = (TrackedUnit & pk).fetch1()
        assert (
            tracked["n_matching_inputs"],
            tracked["n_sessions_detected"],
        ) == (2, 1)
        assert len(UnitMatch.Pair & pk) == 1
        db_inputs, db_recordings = UnitMatch().get_input_provenance(pk)
        nwb_inputs, nwb_recordings = UnitMatch().get_input_provenance(
            pk, from_nwb=True
        )
        assert db_inputs.equals(nwb_inputs)
        assert db_recordings[list(nwb_recordings.columns)].equals(
            nwb_recordings
        )
        assert all(
            snapshot["included_unit_ids"] == [0]
            and snapshot["policy_name"] == "v2_unflagged_units"
            and snapshot["selection_hash"]
            for snapshot in nwb_inputs["selection_snapshot"]
        )
        assert nwb_recordings["source_coverage"].tolist() == [
            row["source_coverage"] for row in recordings
        ]
        spikes = TrackedUnit().get_member_spike_times(pk)
        assert len(spikes) == 2
        assert len(set(spikes["interval_list_name"])) == 2
        for _, row in spikes.iterrows():
            expected = CurationV2.get_sorting(
                {
                    "sorting_id": row["sorting_id"],
                    "curation_id": row["curation_id"],
                },
                as_dataframe=True,
            ).loc[row["unit_id"], "spike_times"]
            np.testing.assert_array_equal(row["spike_times"], expected)
    finally:
        if pk is not None:
            (UnitMatchSelection & pk).super_delete(warn=False)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        restore_matcher_registry(saved)
        clear_curations_for(late_sort)
        (SortingSelection & late_sort).super_delete(warn=False)
