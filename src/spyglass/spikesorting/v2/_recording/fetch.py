"""DB inputs for a ``Recording`` populate (``Recording.make_fetch``).

DataJoint calls ``make_fetch`` twice and compares the results, so every value
is deterministic. DataJoint tables are imported inside the functions.
"""

from __future__ import annotations

from typing import NamedTuple

from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording.geometry import (
    fetch_interior_bad_channel_ids,
    fetch_sort_group_probe_info,
)
from spyglass.spikesorting.v2._core.selection_identity import (
    recording_input_hash,
)
from spyglass.spikesorting.v2._core.reference_resolution import (
    _validate_reference_fields,
)

from spyglass.spikesorting.v2._recording.types import RecordingFetched


class ResolvedRecordingInputs(NamedTuple):
    """Live sort-group construction inputs and their content hash.

    Attributes
    ----------
    channel_ids : list
        Sort-group electrode ids, sorted as ints.
    reference_mode : str
        ``SortGroupV2.reference_mode``.
    reference_electrode_id : int or None
        ``SortGroupV2.reference_electrode_id``.
    preprocessing_params : PreprocessingParamsSchema
        The validated ``PreprocessingParameters.params`` blob.
    bad_channel_ids : tuple
        Interior curated-bad electrode ids the ``interpolate`` path fills;
        empty for ``remove``.
    input_hash : str
        :func:`recording_input_hash` of the inputs above.
    """

    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    preprocessing_params: PreprocessingParamsSchema
    bad_channel_ids: tuple
    input_hash: str


def resolve_recording_inputs(
    nwb_file_name: str,
    sort_group_id: int,
    preprocessing_params_name: str,
) -> ResolvedRecordingInputs:
    """Read a recording's live construction inputs and hash them.

    The single read path for selection and populate, so the two cannot
    disagree.

    Parameters
    ----------
    nwb_file_name : str
        The raw NWB file.
    sort_group_id : int
        The sort group whose membership + reference are read.
    preprocessing_params_name : str
        Selects the ``PreprocessingParameters`` row whose
        ``bad_channel_handling`` decides whether the interpolate bad-channel set
        enters the content.

    Returns
    -------
    ResolvedRecordingInputs
        The resolved inputs and their ``recording_input_hash``.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        SortGroupV2,
    )

    sort_group_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": int(sort_group_id),
    }
    channel_ids = sorted(
        (SortGroupV2.SortGroupElectrode & sort_group_key).fetch("electrode_id"),
        key=int,
    )
    reference_mode, reference_electrode_id = (
        SortGroupV2 & sort_group_key
    ).fetch1("reference_mode", "reference_electrode_id")
    reference_mode = str(reference_mode)
    reference_electrode_id = (
        None if reference_electrode_id is None else int(reference_electrode_id)
    )
    # DeepHash-stable: its __dict__ is primitives.
    preprocessing_params = PreprocessingParamsSchema.model_validate(
        (
            PreprocessingParameters
            & {"preprocessing_params_name": preprocessing_params_name}
        ).fetch1("params")
    )
    bad_channel_ids: tuple = ()
    if preprocessing_params.bad_channel_handling == "interpolate":
        bad_channel_ids = fetch_interior_bad_channel_ids(
            nwb_file_name, channel_ids
        )
    return ResolvedRecordingInputs(
        channel_ids=channel_ids,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        preprocessing_params=preprocessing_params,
        bad_channel_ids=bad_channel_ids,
        input_hash=recording_input_hash(
            electrode_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            interpolated_bad_channel_ids=bad_channel_ids,
        ),
    )


def fetch_recording_inputs(key: dict) -> RecordingFetched:
    """Read every DB input ``Recording.make_compute`` needs.

    Parameters
    ----------
    key : dict
        Restriction selecting a single ``Recording`` row.

    Returns
    -------
    RecordingFetched
        DB-side inputs unpacked positionally into ``make_compute``.

    Raises
    ------
    ValueError
        If the sort group has no electrodes.
    RecordingInputDriftError
        If the sort group's inputs no longer hash to the selection's
        ``recording_input_hash``.
    """
    from spyglass.common.common_ephys import Raw
    from spyglass.common.common_interval import IntervalList
    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2.recording import RecordingSelection

    sel = (RecordingSelection & key).fetch1()
    nwb_file_name = sel["nwb_file_name"]
    sort_group_id = int(sel["sort_group_id"])
    interval_list_name = sel["interval_list_name"]

    resolved = resolve_recording_inputs(
        nwb_file_name, sort_group_id, sel["preprocessing_params_name"]
    )
    channel_ids = resolved.channel_ids
    if len(channel_ids) == 0:
        raise ValueError(
            f"Recording.make: sort group {sort_group_id} for "
            f"{nwb_file_name!r} has zero electrodes."
        )
    # A bypassing write (e.g. update1) can skip SortGroupV2.insert's check.
    _validate_reference_fields(
        {
            "reference_mode": resolved.reference_mode,
            "reference_electrode_id": resolved.reference_electrode_id,
        }
    )
    sort_valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
    ).fetch1("valid_times")
    raw_valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    # Compute reads the raw ElectricalSeries by this id, not by scanning
    # acquisition, so a multi-series NWB cannot feed a different source.
    raw_object_id = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
        "raw_object_id"
    )
    # Decides whether the legacy ``tetrode_12.5`` geometry repair applies.
    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    stored_input_hash = sel["recording_input_hash"]
    live_input_hash = resolved.input_hash
    if live_input_hash != stored_input_hash:
        from spyglass.spikesorting.v2.exceptions import (
            RecordingInputDriftError,
        )

        raise RecordingInputDriftError(
            f"Recording {sel['recording_id']} was selected with input "
            f"fingerprint {stored_input_hash} but the live sort-group "
            f"inputs now hash to {live_input_hash}: the electrode "
            "membership, reference, or interpolate bad-channel set "
            "changed after insert_selection. Re-run insert_selection "
            "for the current inputs (it mints a new recording_id for "
            "the changed set), or restore the sort group's membership / "
            "bad-channel flags."
        )
    return RecordingFetched(
        sel=sel,
        channel_ids=channel_ids,
        reference_mode=resolved.reference_mode,
        reference_electrode_id=resolved.reference_electrode_id,
        sort_valid_times=sort_valid_times,
        raw_valid_times=raw_valid_times,
        preprocessing_params=resolved.preprocessing_params,
        probe_types=probe_types,
        electrode_group_names=electrode_group_names,
        bad_channel_ids=resolved.bad_channel_ids,
        raw_object_id=raw_object_id,
        raw_path=Nwbfile().get_abs_path(nwb_file_name),
    )
