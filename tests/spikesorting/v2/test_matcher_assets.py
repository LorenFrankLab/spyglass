"""Required matching artifacts are checked without healing or waveform reads."""

import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from spyglass.spikesorting.v2._matching import assets

pytestmark = pytest.mark.unit


class ReadOnlyFiles:
    def __init__(self, names):
        self.names = names

    def __and__(self, key):
        return SimpleNamespace(
            fetch1=lambda field: self.names[key["sorting_id"]]
        )


@pytest.fixture
def artifact_inputs(monkeypatch, tmp_path):
    paths = {}
    for name in ("first", "second"):
        path = paths[f"{name}_units"] = tmp_path / f"{name}_units.nwb"
        with h5py.File(path, "w") as handle:
            handle.attrs["neurodata_type"] = "NWBFile"
            units = handle.create_group("units")
            units.create_dataset("id", data=[0, 1])
            for column in (
                "spike_times",
                "spike_sample_index",
                "obs_intervals",
            ):
                units.create_dataset(column, data=[])
                units.create_dataset(
                    f"{column}_index", data=np.array([0, 0], dtype=np.uint64)
                )
        path = paths[f"{name}_sorting"] = tmp_path / f"{name}_sorting.nwb"
        path.touch()
        path = paths[f"{name}_traces"] = tmp_path / f"{name}_traces.nwb"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "acquisition/traces/data", shape=(10, 2), dtype="f4"
            )

    monkeypatch.setattr(
        assets, "stored_analysis_path", lambda name: str(paths[name])
    )
    curations = ReadOnlyFiles(
        {name: f"{name}_units" for name in ("first", "second")}
    )
    sortings = ReadOnlyFiles(
        {name: f"{name}_sorting" for name in ("first", "second")}
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.curation",
        SimpleNamespace(CurationV2=curations),
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.sorting",
        SimpleNamespace(
            Sorting=sortings,
            SortingSelection=SimpleNamespace(
                resolve_effective_source=lambda key: SimpleNamespace(
                    traces=SimpleNamespace(
                        row={
                            "analysis_file_name": f"{key['sorting_id']}_traces",
                            "electrical_series_path": "acquisition/traces",
                        }
                    )
                )
            ),
        ),
    )

    def statistics(path, **kwargs):
        assert path.endswith("_sorting.nwb")
        return [(0, 10)]

    monkeypatch.setattr(
        "spyglass.spikesorting.v2._storage.units_nwb.read_sorting_statistics_spans",
        statistics,
    )
    inputs = [
        {
            "sorting_id": name,
            "curation_id": 1,
            "selection_snapshot": {"labels_by_unit": {"0": [], "1": []}},
            "recordings": [{"end_sample": 10}],
        }
        for name in ("first", "second")
    ]
    return SimpleNamespace(inputs=inputs, paths=paths)


@pytest.mark.parametrize("count", [1, 2])
def test_available_assets_are_read_only_and_do_not_read_spikes_or_traces(
    artifact_inputs, monkeypatch, count
):
    original = h5py.Dataset.__getitem__

    def metadata_only(dataset, key):
        assert dataset.name.endswith(
            "/id"
        ), "Preflight read signal or spike payload."
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", metadata_only)
    assets.validate_match_input_assets(artifact_inputs.inputs[:count])


@pytest.mark.parametrize(
    "missing", ["first_units", "second_units", "first_sorting", "first_traces"]
)
def test_missing_required_asset_blocks_before_execution(
    artifact_inputs, missing
):
    artifact_inputs.paths[missing].unlink()
    with pytest.raises(FileNotFoundError, match="before planning"):
        assets.validate_match_input_assets(artifact_inputs.inputs)
    assert not artifact_inputs.paths[missing].exists()


def test_single_input_needs_curated_units_but_no_inference_assets(
    artifact_inputs,
):
    artifact_inputs.paths["first_sorting"].unlink()
    artifact_inputs.paths["first_traces"].unlink()
    assets.validate_match_input_assets(artifact_inputs.inputs[:1])
    artifact_inputs.paths["first_units"].unlink()
    with pytest.raises(FileNotFoundError, match="curated Units"):
        assets.validate_match_input_assets(artifact_inputs.inputs[:1])


def test_curated_artifact_must_contain_the_exact_unit_namespace(
    artifact_inputs,
):
    with h5py.File(artifact_inputs.paths["first_units"], "a") as handle:
        handle["units/id"][:] = np.array([0, 7])
    with pytest.raises(ValueError, match="Units namespace differs"):
        assets.validate_match_input_assets(artifact_inputs.inputs)


def test_required_units_columns_are_checked_without_materialization(
    artifact_inputs,
):
    with h5py.File(artifact_inputs.paths["first_units"], "a") as handle:
        del handle["units/spike_sample_index"]
    with pytest.raises(ValueError, match="required Units columns"):
        assets.validate_match_input_assets(artifact_inputs.inputs)


@pytest.mark.parametrize(
    "damage", ["missing", "wrong_rows", "wrong_type", "noninteger"]
)
def test_ragged_units_indexes_require_one_integer_offset_per_unit(
    artifact_inputs, damage
):
    with h5py.File(artifact_inputs.paths["first_units"], "a") as handle:
        del handle["units/spike_sample_index_index"]
        if damage == "wrong_rows":
            handle.create_dataset("units/spike_sample_index_index", data=[0])
        elif damage == "wrong_type":
            handle.create_group("units/spike_sample_index_index")
        elif damage == "noninteger":
            handle.create_dataset(
                "units/spike_sample_index_index", data=[0.0, 0.0]
            )
    with pytest.raises(
        ValueError,
        match="required Units columns|invalid Units datasets|integer offset",
    ):
        assets.validate_match_input_assets(artifact_inputs.inputs)


@pytest.mark.parametrize("column", ["id", "spike_times", "obs_intervals"])
def test_required_units_columns_must_be_datasets(artifact_inputs, column):
    with h5py.File(artifact_inputs.paths["first_units"], "a") as handle:
        del handle[f"units/{column}"]
        handle.create_group(f"units/{column}")
    with pytest.raises(ValueError, match="invalid Units datasets"):
        assets.validate_match_input_assets(artifact_inputs.inputs)


@pytest.mark.parametrize("damage", ["missing_series", "wrong_frames"])
def test_effective_trace_layout_is_checked_without_a_geometry_hook(
    artifact_inputs, damage
):
    with h5py.File(artifact_inputs.paths["first_traces"], "a") as handle:
        del handle["acquisition/traces/data"]
        if damage == "wrong_frames":
            handle.create_dataset(
                "acquisition/traces/data", shape=(9, 2), dtype="f4"
            )
    with pytest.raises(
        ValueError, match="missing ElectricalSeries|frame count differs"
    ):
        assets.validate_match_input_assets(artifact_inputs.inputs)


@pytest.mark.parametrize("damage", ["group", "one_dimension", "no_channels"])
def test_effective_trace_data_requires_frames_and_nonempty_channels(
    artifact_inputs, damage
):
    with h5py.File(artifact_inputs.paths["first_traces"], "a") as handle:
        del handle["acquisition/traces/data"]
        if damage == "group":
            handle.create_group("acquisition/traces/data")
        else:
            shape = (10,) if damage == "one_dimension" else (10, 0)
            handle.create_dataset(
                "acquisition/traces/data", shape=shape, dtype="f4"
            )
    with pytest.raises(ValueError, match="two-dimensional dataset"):
        assets.validate_match_input_assets(artifact_inputs.inputs)


def test_missing_persisted_statistics_is_an_actionable_preflight_error(
    artifact_inputs, monkeypatch
):
    def absent(*args, **kwargs):
        raise RuntimeError(
            "Sorting has no persisted statistics spans; repopulate it."
        )

    monkeypatch.setattr(
        "spyglass.spikesorting.v2._storage.units_nwb.read_sorting_statistics_spans",
        absent,
    )
    with pytest.raises(ValueError, match="no persisted statistics spans"):
        assets.validate_match_input_assets(artifact_inputs.inputs)


def test_importable_package_name_does_not_prove_matching_runtime_ready(
    monkeypatch,
):
    from spyglass.spikesorting.v2._matching import unitmatch_backend

    def unavailable():
        raise ImportError(
            "UnitMatchPy is installed but its native dependency is unavailable."
        )

    monkeypatch.setattr(unitmatch_backend, "_require_unitmatch", unavailable)
    with pytest.raises(ImportError, match="native dependency is unavailable"):
        assets.validate_match_backend_availability(
            "unitmatch", needs_pair_inference=True
        )
    # A zero-pair input needs only producer metadata; custom runtimes own their
    # checks rather than importing the default backend's dependencies.
    assets.validate_match_backend_availability(
        "unitmatch", needs_pair_inference=False
    )
    assets.validate_match_backend_availability(
        "custom", needs_pair_inference=True
    )
