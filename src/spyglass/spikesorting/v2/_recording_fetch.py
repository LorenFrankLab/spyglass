"""DB inputs for a ``Recording`` populate.

:func:`fetch_recording_inputs` is the body of ``Recording.make_fetch``. It
reads the selection row, the sort group's electrodes and reference, the sort
and raw-data intervals, the raw source's object id, the validated
preprocessing parameters, the per-channel probe metadata and, for the
``interpolate`` path, the interior bad channels. It then checks that these
inputs still hash to the selection's ``recording_input_hash``. DataJoint calls
``make_fetch`` twice per populate and compares the two results, so every value
is returned in a deterministic form.

Imports without the DB layer: the DataJoint tables are imported inside the
function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording_geometry import (
    fetch_interior_bad_channel_ids,
    fetch_sort_group_probe_info,
)
from spyglass.spikesorting.v2._selection_identity import (
    recording_input_hash,
)
from spyglass.spikesorting.v2.utils import _validate_reference_fields

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.recording import RecordingFetched


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
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingFetched,
        RecordingSelection,
        SortGroupV2,
    )

    sel = (RecordingSelection & key).fetch1()
    nwb_file_name = sel["nwb_file_name"]
    sort_group_id = int(sel["sort_group_id"])
    interval_list_name = sel["interval_list_name"]

    channel_ids = sorted(
        (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch("electrode_id"),
        key=int,
    )
    if len(channel_ids) == 0:
        raise ValueError(
            f"Recording.make: sort group {sort_group_id} for "
            f"{nwb_file_name!r} has zero electrodes."
        )
    reference_mode, reference_electrode_id = (
        SortGroupV2
        & {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
        }
    ).fetch1("reference_mode", "reference_electrode_id")
    reference_mode = str(reference_mode)
    reference_electrode_id = (
        None if reference_electrode_id is None else int(reference_electrode_id)
    )
    # Re-validate the reference fields on the READ side. The
    # mode-vs-id biconditional is enforced in SortGroupV2.insert1/
    # insert, but a row written through a path that bypasses those
    # overrides (e.g. raw update1) could carry a stray
    # reference_electrode_id under a non-"specific" mode -- which the
    # downstream dispatch silently ignores. Validate here so such a
    # row fails loudly at populate instead of silently dropping the
    # operator's intent.
    _validate_reference_fields(
        {
            "reference_mode": reference_mode,
            "reference_electrode_id": reference_electrode_id,
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
    # The exact NWB object the common ``Raw`` row was ingested from. The
    # compute step resolves the raw ElectricalSeries by this id (not by
    # scanning acquisition) so a multi-ElectricalSeries / repacked NWB
    # cannot silently feed a different source than the selection lineage.
    raw_object_id = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
        "raw_object_id"
    )
    # Pre-fetch + validate the preprocessing params here so the
    # compute stage does not need to re-read them. The validated
    # model is DeepHash-stable across the two ``make_fetch``
    # calls DataJoint makes (Pydantic ``__dict__`` is primitives).
    preprocessing_row = (
        PreprocessingParameters
        & {"preprocessing_params_name": sel["preprocessing_params_name"]}
    ).fetch1()
    preprocessing_params = PreprocessingParamsSchema.model_validate(
        preprocessing_row["params"]
    )
    # Pull the job_kwargs blob so ``make_compute`` can call
    # ``_resolved_job_kwargs(preprocessing_job_kwargs)`` -- every v2 compute
    # stage resolves job_kwargs once so the override channels are
    # honored consistently.
    # Currently the Recording write path streams via HDMF's
    # chunked iterator and does not consume SI-style job_kwargs;
    # the resolver call is kept so the override channels
    # (``dj.config['custom']['spikesorting_v2_job_kwargs']`` and
    # the per-row blob) are testable here and so future
    # consumers can pick up the resolved dict without retrofitting
    # the fetch.
    preprocessing_job_kwargs = preprocessing_row.get("job_kwargs")
    # Per-channel ``probe_type`` + ``electrode_group_name`` for the
    # sort group, used by ``make_compute`` to decide whether the
    # legacy ``tetrode_12.5`` probe-geometry patch applies. Fetched
    # here so ``make_compute`` stays DB-I/O free per the tri-part
    # contract.
    probe_types, electrode_group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    # The sort group's interior curated-bad channels to re-include and fill
    # on the ``interpolate`` path -- empty for ``remove`` (which re-includes
    # nothing, so the slice is exactly the members). Sorted
    # tuple so the fetched value is DeepHash-stable like the others.
    bad_channel_ids: tuple = ()
    if preprocessing_params.bad_channel_handling == "interpolate":
        bad_channel_ids = fetch_interior_bad_channel_ids(
            nwb_file_name, channel_ids
        )
    # Verify the recording's resolved construction inputs still match the
    # content-addressed recording_input_hash the recording_id was minted
    # from. A mismatch means the sort group's membership/reference or the
    # interpolate bad-channel set changed after insert_selection, so building
    # here would produce content that no longer matches recording_id. Rows
    # with a NULL hash (an allow_direct_insert bypass, or a row predating the
    # column) never recorded a fingerprint, so they opt out -- every
    # insert_selection row carries one.
    stored_input_hash = sel.get("recording_input_hash")
    if stored_input_hash is not None:
        live_input_hash = recording_input_hash(
            electrode_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            interpolated_bad_channel_ids=bad_channel_ids,
        )
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
        reference_mode=reference_mode,
        reference_electrode_id=reference_electrode_id,
        sort_valid_times=sort_valid_times,
        raw_valid_times=raw_valid_times,
        preprocessing_params=preprocessing_params,
        preprocessing_job_kwargs=preprocessing_job_kwargs,
        probe_types=probe_types,
        electrode_group_names=electrode_group_names,
        bad_channel_ids=bad_channel_ids,
        raw_object_id=raw_object_id,
        raw_path=Nwbfile().get_abs_path(nwb_file_name),
    )
