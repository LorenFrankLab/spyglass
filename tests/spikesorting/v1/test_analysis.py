import pytest


@pytest.mark.skip(reason="JAX issues")
def test_analysis_units(pop_annotations):
    selected_spike_times, selected_unit_ids = pop_annotations.fetch_unit_spikes(
        return_unit_ids=True
    )

    assert selected_spike_times[0].shape[0] > 0, "Found no spike times"

    units = [d["unit_id"] for d in selected_unit_ids]
    assert 0 in units, "Unexpected unit ids"
