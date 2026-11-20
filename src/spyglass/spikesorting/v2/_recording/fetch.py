"""DB inputs for a ``Recording`` populate (``Recording.make_fetch``).

DataJoint calls ``make_fetch`` twice and compares the results, so every value
is deterministic. ``recording`` imports this module, so its tables are
imported inside the functions.
"""

from __future__ import annotations

from typing import NamedTuple

from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording.geometry import (
    fetch_interior_bad_channel_ids,
    fetch_sort_group_contact_positions,
    fetch_sort_group_probe_info,
    retained_channel_ids,
)
from spyglass.spikesorting.v2._core.selection_identity import (
    recording_input_hash,
)
from spyglass.spikesorting.v2._core.reference_resolution import (
    _validate_reference_fields,
)

from spyglass.spikesorting.v2._recording.restriction import (
    intended_save_intervals,
)
from spyglass.spikesorting.v2._recording.types import RecordingFetched
from spyglass.common.common_ephys import Raw
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.spikesorting.v2.exceptions import RecordingInputDriftError


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
        The validated ``PreprocessingParameters.preprocessing_params`` blob.
    bad_channel_ids : tuple
        Interior curated-bad electrode ids the ``interpolate`` path fills;
        empty for ``remove``.
    sort_valid_times : numpy.ndarray
        The sort interval's ``valid_times``, shape ``(n_intervals, 2)`` in
        seconds.
    raw_valid_times : numpy.ndarray
        The session's ``raw data valid times``, shape ``(n_intervals, 2)`` in
        seconds.
    input_hash : str
        :func:`recording_input_hash` of the inputs above, including the saved
        intervals they cut.
    """

    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    preprocessing_params: PreprocessingParamsSchema
    bad_channel_ids: tuple
    sort_valid_times: object
    raw_valid_times: object
    input_hash: str


class SortGroupChannels(NamedTuple):
    """A sort group's channels as the recording stage reads them.

    Attributes
    ----------
    channel_ids : list[int]
        Sort-group electrode ids, sorted.
    reference_mode : str
        ``SortGroupV2.reference_mode``.
    reference_electrode_id : int or None
        ``SortGroupV2.reference_electrode_id``.
    preprocessing_params : PreprocessingParamsSchema
        The validated ``PreprocessingParameters.preprocessing_params`` blob.
    bad_channel_ids : tuple
        Interior curated-bad electrode ids the ``interpolate`` path fills;
        empty for ``remove``.
    """

    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    preprocessing_params: PreprocessingParamsSchema
    bad_channel_ids: tuple


def resolve_sort_group_channels(
    nwb_file_name: str, sort_group_id: int, preprocessing_params_name: str
) -> SortGroupChannels:
    """Read a sort group's members, reference and interpolated bad channels.

    The one read of the channel set, shared by recording inputs and the
    preflight geometry checks.

    Raises
    ------
    ValueError
        On the ``interpolate`` path, if a needed position or probe link is
        missing (:func:`fetch_interior_bad_channel_ids`).
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
        ).fetch1("preprocessing_params")
    )
    bad_channel_ids: tuple = ()
    if preprocessing_params.bad_channel_handling == "interpolate":
        bad_channel_ids = fetch_interior_bad_channel_ids(
            nwb_file_name, channel_ids
        )
    return SortGroupChannels(
        channel_ids=channel_ids,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        preprocessing_params=preprocessing_params,
        bad_channel_ids=bad_channel_ids,
    )


class SortGroupContacts(NamedTuple):
    """The registered contacts a sort group's recording keeps.

    Attributes
    ----------
    n_members : int
        Number of sort-group members (the tetrode repair needs four).
    channel_ids : list[int]
        The retained electrode ids (:func:`retained_channel_ids`), sorted.
    positions : numpy.ndarray
        ``(n_channels, 3)`` ``Probe.Electrode`` positions, NaN where missing.
    probe_types, electrode_group_names, probe_shanks : tuple
        Per-channel metadata, aligned with ``channel_ids``.
    """

    n_members: int
    channel_ids: list
    positions: object
    probe_types: tuple
    electrode_group_names: tuple
    probe_shanks: tuple


def fetch_sort_group_contacts(
    nwb_file_name: str, sort_group_id: int, preprocessing_params_name: str
) -> "SortGroupContacts | None":
    """Fetch the registered geometry of the contacts a recording keeps.

    Preflight and plotting read ``Probe.Electrode``; the recording stage
    reads the session's electrodes table, so a session that disagrees with
    its probe can still fail at ``Recording.make``.

    Returns
    -------
    SortGroupContacts or None
        ``None`` for a sort group without members.

    Raises
    ------
    ValueError
        As :func:`resolve_sort_group_channels`.
    """
    from spyglass.common.common_ephys import Electrode

    channels = resolve_sort_group_channels(
        nwb_file_name, sort_group_id, preprocessing_params_name
    )
    if not channels.channel_ids:
        return None
    channel_ids = retained_channel_ids(
        channels.channel_ids,
        reference_mode=channels.reference_mode,
        reference_electrode_id=channels.reference_electrode_id,
        bad_channel_handling=channels.preprocessing_params.bad_channel_handling,
        bad_channel_ids=channels.bad_channel_ids,
    )
    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    probe_shanks = tuple(
        (
            Electrode
            & {"nwb_file_name": nwb_file_name}
            & [{"electrode_id": c} for c in channel_ids]
        ).fetch("probe_shank", order_by="electrode_id")
    )
    return SortGroupContacts(
        n_members=len(channels.channel_ids),
        channel_ids=channel_ids,
        positions=fetch_sort_group_contact_positions(
            nwb_file_name, channel_ids
        ),
        probe_types=probe_types,
        electrode_group_names=electrode_group_names,
        probe_shanks=probe_shanks,
    )


def resolve_recording_inputs(
    nwb_file_name: str,
    sort_group_id: int,
    preprocessing_params_name: str,
    interval_list_name: str,
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
        enters the content, and whose ``min_segment_length`` shapes the cut.
    interval_list_name : str
        The sort interval, intersected with the raw valid times.

    Returns
    -------
    ResolvedRecordingInputs
        The resolved inputs and their ``recording_input_hash``.

    Raises
    ------
    ValueError
        If the sort interval or the ``raw data valid times`` IntervalList is
        missing for ``nwb_file_name``.
    """
    channels = resolve_sort_group_channels(
        nwb_file_name, sort_group_id, preprocessing_params_name
    )
    channel_ids = channels.channel_ids
    reference_mode = channels.reference_mode
    reference_electrode_id = channels.reference_electrode_id
    preprocessing_params = channels.preprocessing_params
    bad_channel_ids = channels.bad_channel_ids
    sort_valid_times = _fetch_valid_times(nwb_file_name, interval_list_name)
    raw_valid_times = _fetch_valid_times(nwb_file_name, "raw data valid times")
    saved_intervals = intended_save_intervals(
        sort_valid_times,
        raw_valid_times,
        preprocessing_params.min_segment_length,
    )
    return ResolvedRecordingInputs(
        channel_ids=channel_ids,
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        preprocessing_params=preprocessing_params,
        bad_channel_ids=bad_channel_ids,
        sort_valid_times=sort_valid_times,
        raw_valid_times=raw_valid_times,
        input_hash=recording_input_hash(
            electrode_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            interpolated_bad_channel_ids=bad_channel_ids,
            saved_intervals=saved_intervals,
        ),
    )


def _fetch_valid_times(nwb_file_name: str, interval_list_name: str):
    """Return one IntervalList's ``valid_times``, shape ``(n_intervals, 2)``.

    Raises
    ------
    ValueError
        If the IntervalList row does not exist.
    """
    rel = IntervalList & {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }
    if not rel:
        raise ValueError(
            f"IntervalList {interval_list_name!r} does not exist for "
            f"{nwb_file_name!r}; a recording needs it to resolve its time "
            "cut. Insert the interval list (or check its name) first."
        )
    return rel.fetch1("valid_times")


def assert_recording_inputs_current(
    sel: dict, resolved: ResolvedRecordingInputs | None = None
) -> ResolvedRecordingInputs:
    """Raise if a recording's live inputs no longer hash to its selection.

    The one live-versus-stored comparison: ``Recording.make_fetch``, the
    recompute delete gate and ``audit_recording_input_hashes`` all use it.
    The hash covers the electrode membership, reference, interpolated bad
    channels, and the sort interval / raw valid times. Re-running
    ``insert_selection`` mints a new ``recording_id`` for changed inputs.

    Parameters
    ----------
    sel : dict
        A ``RecordingSelection`` row.
    resolved : ResolvedRecordingInputs, optional
        The live inputs if already read; resolved here otherwise.

    Returns
    -------
    ResolvedRecordingInputs
        The live inputs.

    Raises
    ------
    RecordingInputDriftError
        If the live inputs hash differently from ``recording_input_hash``.
    ValueError
        If the sort interval or ``raw data valid times`` IntervalList is gone.
    """
    if resolved is None:
        resolved = resolve_recording_inputs(
            sel["nwb_file_name"],
            int(sel["sort_group_id"]),
            sel["preprocessing_params_name"],
            sel["interval_list_name"],
        )
    stored_input_hash = sel["recording_input_hash"]
    live_input_hash = resolved.input_hash
    if live_input_hash != stored_input_hash:
        raise RecordingInputDriftError(
            f"Recording {sel['recording_id']} was selected with input "
            f"fingerprint {stored_input_hash}, but its live sort group or "
            f"intervals now hash to {live_input_hash}. Re-run "
            "insert_selection for the current inputs, or restore the sort "
            "group or IntervalList."
        )
    return resolved


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
    from spyglass.spikesorting.v2.recording import RecordingSelection

    sel = (RecordingSelection & key).fetch1()
    nwb_file_name = sel["nwb_file_name"]
    sort_group_id = int(sel["sort_group_id"])
    interval_list_name = sel["interval_list_name"]

    resolved = resolve_recording_inputs(
        nwb_file_name,
        sort_group_id,
        sel["preprocessing_params_name"],
        interval_list_name,
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
    # Compute reads the raw ElectricalSeries by this id, not by scanning
    # acquisition, so a multi-series NWB cannot feed a different source.
    raw_object_id = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
        "raw_object_id"
    )
    # Decides whether the legacy ``tetrode_12.5`` geometry repair applies.
    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    assert_recording_inputs_current(sel, resolved)
    return RecordingFetched(
        sel=sel,
        channel_ids=channel_ids,
        reference_mode=resolved.reference_mode,
        reference_electrode_id=resolved.reference_electrode_id,
        sort_valid_times=resolved.sort_valid_times,
        raw_valid_times=resolved.raw_valid_times,
        preprocessing_params=resolved.preprocessing_params,
        probe_types=probe_types,
        electrode_group_names=electrode_group_names,
        bad_channel_ids=resolved.bad_channel_ids,
        raw_object_id=raw_object_id,
        raw_path=Nwbfile().get_abs_path(nwb_file_name),
    )
