"""Read-only checks for the artifacts a matching worker will consume."""

from __future__ import annotations

from pathlib import Path

from spyglass.spikesorting.v2._matching.coverage import stored_analysis_path


def validate_match_backend_availability(name, *, needs_pair_inference):
    """Check the default runtime imports, including transitive dependencies."""
    if name == "unitmatch" and needs_pair_inference:
        from spyglass.spikesorting.v2._matching.unitmatch_backend import (
            _require_unitmatch,
        )

        _require_unitmatch()


def _required_path(name, *, label, recovery):
    try:
        path = stored_analysis_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label}: {exc} {recovery}") from exc
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{label} is missing its local NWB artifact {name!r}. {recovery}"
        )
    return path


def _validate_units_layout(path, expected_units, *, label):
    """Check the namespace and required columns without reading spike trains."""
    import h5py

    from spyglass.spikesorting._numerical import integer_vector
    from spyglass.spikesorting.v2._storage.units_nwb import (
        SPIKE_SAMPLE_INDEX_COLUMN,
    )

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("neurodata_type") != "NWBFile":
            raise ValueError(f"{label} artifact is not an NWBFile.")
        units = handle.get("units")
        ragged_columns = {
            "spike_times",
            SPIKE_SAMPLE_INDEX_COLUMN,
            "obs_intervals",
        }
        required = (
            {"id"}
            | ragged_columns
            | {f"{column}_index" for column in ragged_columns}
        )
        missing = (
            required
            if not isinstance(units, h5py.Group)
            else required.difference(units)
        )
        if missing:
            raise ValueError(
                f"{label} artifact is missing required Units columns {sorted(missing)}. "
                "Restore the original artifact before matching."
            )
        invalid = sorted(
            name
            for name in required
            if not isinstance(units[name], h5py.Dataset)
        )
        if invalid:
            raise ValueError(
                f"{label} artifact has invalid Units datasets {invalid}. "
                "Restore the original artifact before matching."
            )
        identifiers = integer_vector(units["id"][:], name="unit_id")
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(
            expected_units
        ):
            raise ValueError(
                f"{label} artifact's Units namespace differs from its frozen curation. "
                "Restore the original artifact before matching."
            )
        for column in sorted(ragged_columns):
            index = units[f"{column}_index"]
            if (
                index.shape != (len(identifiers),)
                or index.dtype.kind not in "iu"
            ):
                raise ValueError(
                    f"{label} artifact's Units index {index.name!r} must have one "
                    "integer offset per unit. Restore the original artifact before matching."
                )


def _validate_trace_layout(path, series_path, expected_frames, *, label):
    """Read dataset shape metadata, without reading signal or timestamp arrays."""
    import h5py

    with h5py.File(path, "r") as handle:
        series = handle.get(series_path)
        if not isinstance(series, h5py.Group) or "data" not in series:
            raise ValueError(
                f"{label} artifact is missing ElectricalSeries {series_path!r}. "
                "Rebuild its recording explicitly before planning."
            )
        data = series["data"]
        if (
            not isinstance(data, h5py.Dataset)
            or data.ndim != 2
            or data.shape[1] < 1
        ):
            raise ValueError(
                f"{label} artifact's ElectricalSeries data must be a two-dimensional "
                "dataset with at least one channel. Rebuild its recording explicitly before planning."
            )
        if data.shape[0] != expected_frames:
            raise ValueError(
                f"{label} artifact's frame count differs from its frozen source. "
                "Rebuild its recording explicitly before planning."
            )


def validate_match_input_assets(inputs):
    """Check every required local artifact without resolving or rebuilding it.

    Every run reads curated Units, including a single-input zero-pair run.
    Multi-input runs also need the sort's statistics spans and effective trace
    artifact, even when the backend has no channel-geometry validation hook.
    Only Units identifiers, provenance, and dataset metadata are read here.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._storage.units_nwb import (
        read_sorting_statistics_spans,
    )

    inputs = list(inputs)
    for item in inputs:
        key = {
            "sorting_id": item["sorting_id"],
            "curation_id": item["curation_id"],
        }
        label = f"Curation {item['sorting_id']}/{item['curation_id']}"
        units_path = _required_path(
            (CurationV2 & key).fetch1("analysis_file_name"),
            label=label,
            recovery="Restore its curated Units artifact before planning; no rebuild is attempted.",
        )
        _validate_units_layout(
            units_path,
            map(int, item["selection_snapshot"]["labels_by_unit"]),
            label=label,
        )
        if len(inputs) == 1:
            continue
        sorting_key = {"sorting_id": item["sorting_id"]}
        sorting_path = _required_path(
            (Sorting & sorting_key).fetch1("analysis_file_name"),
            label=f"Sorting {item['sorting_id']} statistics spans",
            recovery="Restore its sorting artifact before planning; no rebuild is attempted.",
        )
        try:
            read_sorting_statistics_spans(
                sorting_path, sorting_id=item["sorting_id"]
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        traces = SortingSelection.resolve_effective_source(sorting_key).traces
        trace_path = _required_path(
            traces.row["analysis_file_name"],
            label=f"Sorting {item['sorting_id']} effective traces",
            recovery="Rebuild its recording explicitly before planning.",
        )
        _validate_trace_layout(
            trace_path,
            traces.row["electrical_series_path"],
            max(row["end_sample"] for row in item["recordings"]),
            label=f"Sorting {item['sorting_id']} effective traces",
        )
