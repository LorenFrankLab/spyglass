"""``build_analyzer`` projects a 3D probe to 2D before building the analyzer.

Spyglass stores electrode geometry in 3D (z is typically 0), but the
SortingAnalyzer ``unit_locations`` extension and the spikeinterface-gui probe
view assume 2D contact positions and raise
``could not broadcast input array from shape (3,) into shape (2,)`` on a 3D
probe. ``build_analyzer`` projects the probe to 2D so both work; this is the
regression guard. It is a fast, DB-free unit test: it drives ``build_analyzer``
directly with a synthetic recording carrying a 3D probe, passing ``sorter_row``
/ ``job_kwargs`` / ``analyzer_folder`` so no database read happens.
"""

from __future__ import annotations

import pytest
import spikeinterface.full as si

from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer


@pytest.fixture
def recording_3d_and_sorting():
    """A synthetic recording (forced to a 3D probe) and its sorting."""
    recording, sorting = si.generate_ground_truth_recording(
        durations=[10.0], num_channels=8, num_units=4, seed=0
    )
    recording = recording.set_probe(recording.get_probe().to_3d(axes="xy"))
    assert recording.get_probe().ndim == 3
    return recording, sorting


class TestBuildAnalyzerProbeProjection:
    def test_analyzer_probe_is_2d(self, recording_3d_and_sorting, tmp_path):
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
        )

        analyzer = si.load_sorting_analyzer(folder)
        assert analyzer.get_probe().ndim == 2

    def test_unit_locations_computes_after_projection(
        self, recording_3d_and_sorting, tmp_path
    ):
        # Without the 2D projection this raises ValueError:
        # "could not broadcast input array from shape (3,) into shape (2,)".
        recording, sorting = recording_3d_and_sorting
        folder = tmp_path / "sort.analyzer"

        build_analyzer(
            sorting,
            recording,
            key={"sorting_id": "test-3d-probe"},
            sorter_row={"job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=folder,
        )

        analyzer = si.load_sorting_analyzer(folder)
        analyzer.compute("unit_locations")
        unit_locations = analyzer.get_extension("unit_locations").get_data()
        assert unit_locations.shape[0] == sorting.get_num_units()
