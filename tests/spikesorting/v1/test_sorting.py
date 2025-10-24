import pytest


@pytest.mark.slow
def test_sorting(spike_v1, pop_sort):
    """Integration test for SpikeSorting with real spikeinterface operations.

    This test validates the full spike sorting pipeline including:
    - Real spikeinterface spike sorting (~90s)
    - Real file I/O operations (~5s)
    - Full database integration

    Runtime: ~95s
    For faster feedback, use test_sorting_unit.py with mocked operations (~2-3s).
    """
    n_sorts = len(spike_v1.SpikeSorting & pop_sort)
    assert n_sorts >= 1, "SpikeSorting population failed"
