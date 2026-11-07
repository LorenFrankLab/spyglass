"""Motion request, recipe, supplied-estimate and geometry checks."""

from __future__ import annotations

import uuid
from typing import NamedTuple, get_args

import numpy as np

from spyglass.spikesorting.v2._orchestration.types import MotionMode
from spyglass.spikesorting.v2._core.selection_identity import (
    artifact_detection_identity_payload,
    deterministic_id,
)
from spyglass.spikesorting.v2._core.selection_plan import (
    build_recording_selection_plan,
)
from spyglass.spikesorting.v2._recording.geometry import (
    maybe_apply_tetrode_geometry,
    sort_group_planar_positions,
)
from spyglass.spikesorting.v2._recording.source import (
    correction_lineage_mismatch,
)


def motion_request_problem(
    motion_mode, motion_correction_params_name, motion_estimate_id=None
) -> "str | None":
    """Say why a motion mode and recipe name cannot be run together.

    DB-free: it checks the request's shape, not that the named
    ``MotionCorrectionParameters`` row or motion estimate exists.

    Parameters
    ----------
    motion_mode : str
        One of ``"off"``, ``"estimate"``, ``"apply"``.
    motion_correction_params_name : str or None
        The ``MotionCorrectionParameters`` row; required iff ``motion_mode``
        is not ``"off"``.
    motion_estimate_id : uuid.UUID or str, optional
        A saved motion estimate to apply; only valid with ``"apply"``.

    Returns
    -------
    str or None
        The operator-facing problem, or ``None`` for a valid request.
    """
    problem = _motion_mode_problem(motion_mode, motion_correction_params_name)
    if problem is not None or motion_estimate_id is None:
        return problem
    if motion_mode != "apply":
        return (
            f"motion_estimate_id={motion_estimate_id!r} was given with "
            f"motion_mode={motion_mode!r}; only motion_mode='apply' applies "
            "a saved estimate. Pass motion_mode='apply', or drop the id."
        )
    try:
        uuid.UUID(str(motion_estimate_id))
    except ValueError:
        return (
            f"motion_estimate_id={motion_estimate_id!r} is not a UUID; pass "
            "the motion_estimate_id of a saved MotionEstimate."
        )
    return None


def _motion_mode_problem(
    motion_mode, motion_correction_params_name
) -> "str | None":
    """The mode / recipe-name half of :func:`motion_request_problem`."""
    modes = get_args(MotionMode)
    if motion_mode not in modes:
        return (
            f"unknown motion_mode {motion_mode!r}; expected one of "
            f"{list(modes)}."
        )
    if motion_mode == "off":
        if motion_correction_params_name is not None:
            return (
                "motion_correction_params_name="
                f"{motion_correction_params_name!r} was given with "
                "motion_mode='off', which runs no motion stage. Pass "
                "motion_mode='estimate' or 'apply' to use the recipe, or drop "
                "the name."
            )
        return None
    if motion_correction_params_name is None:
        return (
            f"motion_mode={motion_mode!r} requires "
            "motion_correction_params_name, a MotionCorrectionParameters row "
            "(e.g. 'dredge_fast_v1'; see MotionCorrectionParameters())."
        )
    return None


class MotionRecipe(NamedTuple):
    """A ``MotionCorrectionParameters`` recipe with its two stage recipes.

    Attributes
    ----------
    recipe : dict
        The ``MotionCorrectionParameters`` row.
    estimation_params : dict
        The named ``MotionEstimationParameters`` row's ``params`` blob.
    interpolation_params : dict
        The named ``MotionInterpolationParameters`` row's ``params`` blob.
    resolved_estimation : dict
        ``estimation_params`` resolved against the installed SpikeInterface
        (``_motion.estimation.resolve_estimation_params``).
    """

    recipe: dict
    estimation_params: dict
    interpolation_params: dict
    resolved_estimation: dict


def resolve_motion_recipe(motion_correction_params_name: str) -> MotionRecipe:
    """Fetch a motion-correction recipe and resolve its estimation recipe.

    Parameters
    ----------
    motion_correction_params_name : str
        The ``MotionCorrectionParameters`` row.

    Returns
    -------
    MotionRecipe

    Raises
    ------
    ValueError
        If the row is missing, or its estimation recipe does not resolve
        against the installed SpikeInterface.
    """
    from spyglass.spikesorting.v2._motion.estimation import (
        resolve_estimation_params,
    )
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectionParameters,
        MotionEstimationParameters,
        MotionInterpolationParameters,
    )

    rows = (
        MotionCorrectionParameters
        & {"motion_correction_params_name": motion_correction_params_name}
    ).fetch(as_dict=True)
    if len(rows) != 1:
        raise ValueError(
            "MotionCorrectionParameters row "
            f"{motion_correction_params_name!r} is missing. Run "
            "initialize_v2_defaults() (it ships 'dredge_v1' and "
            "'dredge_fast_v1'), or insert the recipe first."
        )
    recipe = rows[0]
    estimation_name = recipe["motion_estimation_params_name"]
    estimation_params = (
        MotionEstimationParameters
        & {"motion_estimation_params_name": estimation_name}
    ).fetch1("params")
    interpolation_params = (
        MotionInterpolationParameters
        & {
            "motion_interpolation_params_name": recipe[
                "motion_interpolation_params_name"
            ]
        }
    ).fetch1("params")
    try:
        resolved = resolve_estimation_params(estimation_params)
    except ValueError as exc:
        raise ValueError(
            f"MotionEstimationParameters row {estimation_name!r} (named by "
            f"motion recipe {motion_correction_params_name!r}) does not "
            f"resolve against the installed SpikeInterface: {exc}"
        ) from exc
    return MotionRecipe(
        recipe, estimation_params, interpolation_params, resolved
    )


def sorter_motion_correction_problem(
    sorter: str, params, sorter_params_name: str
) -> "str | None":
    """Say why a sorter row cannot sort a motion-corrected recording.

    Uses the same check as ``SortingSelection.insert_selection``.

    Parameters
    ----------
    sorter, params, sorter_params_name
        The ``SorterParameters`` row.

    Returns
    -------
    str or None
        The problem, naming the row and the key to turn off, or ``None``.
    """
    from spyglass.spikesorting.v2._params.sorter import (
        reject_internal_motion_correction,
    )

    try:
        reject_internal_motion_correction(
            sorter, params, sorter_params_name=sorter_params_name
        )
    except ValueError as exc:
        return f"motion_mode='apply' sorts a motion-corrected recording: {exc}"
    return None


def _concat_member_mismatch(concat_key: dict, expected: list[dict]) -> list:
    """Describe how a concatenation's frozen members differ from a run's.

    Parameters
    ----------
    concat_key : dict
        ``{"concat_recording_id": ...}`` of the saved concatenation.
    expected : list of dict
        The run's members: ``member_index``, ``recording_id`` and
        ``artifact_detection_id`` (``None`` for an unmasked member).

    Returns
    -------
    list of str
        One description per differing member set or field; empty when they
        agree.
    """
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )

    def _id(value):
        return None if value is None else uuid.UUID(str(value))

    fields = ("recording_id", "artifact_detection_id")
    frozen = {
        int(row["member_index"]): row
        for row in (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch("member_index", *fields, as_dict=True)
    }
    wanted = {int(row["member_index"]): row for row in expected}
    if set(frozen) != set(wanted):
        return [
            f"its concatenation has members {sorted(frozen)}, but this "
            f"run's has {sorted(wanted)}"
        ]
    return [
        f"member {index}: its {name} {_id(frozen[index][name])} != the run's "
        f"{_id(wanted[index][name])}"
        for index in sorted(wanted)
        for name in fields
        if _id(frozen[index][name]) != _id(wanted[index][name])
    ]


def expected_source_ids(
    bundle,
    *,
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    manual_excluded_times,
) -> tuple:
    """Preview the recording and artifact-detection ids a run would mint.

    Uses the insert's own id builders. The sort-group, electrode and
    preprocessing rows must exist.

    Parameters
    ----------
    bundle : _PipelinePreset
        The preset (its preprocessing and artifact rows).
    nwb_file_name, sort_group_id, interval_list_name, team_name
        The recording's selection fields.
    manual_excluded_times : list
        The normalized manual exclusions of this recording.

    Returns
    -------
    recording_id : uuid.UUID
    artifact_detection_id : uuid.UUID or None
        ``None`` when the preset runs no artifact detection.
    """
    from spyglass.spikesorting.v2.recording import resolve_recording_input_hash

    recording_id = build_recording_selection_plan(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_list_name,
            "preprocessing_params_name": bundle.preprocessing_params_name,
            "team_name": team_name,
        },
        recording_input_hash=resolve_recording_input_hash(
            nwb_file_name, sort_group_id, bundle.preprocessing_params_name
        ),
    ).recording_id
    if bundle.artifact_detection_params_name is None:
        return recording_id, None
    return recording_id, deterministic_id(
        "artifact_detection",
        artifact_detection_identity_payload(
            artifact_detection_params_name=bundle.artifact_detection_params_name,
            recording_id=recording_id,
            manual_excluded_times=manual_excluded_times,
        ),
    )


def supplied_motion_estimate_problem(
    motion_estimate_id,
    motion_recipe: MotionRecipe,
    *,
    source_lineage=None,
    concat_source: "dict | None" = None,
    concat_members: "list[dict] | None" = None,
) -> "str | None":
    """Say why a saved motion estimate cannot be applied by a run.

    The estimate must be populated, use the recipe's estimation row, have an
    unchanged source ``content_hash``, and match the run's source: its source
    and mask (``source_lineage``), or for an unbuilt concat, the same group,
    preprocessing and members (``concat_source``, ``concat_members``).

    Parameters
    ----------
    motion_estimate_id : uuid.UUID or str
        The ``MotionEstimate`` to apply.
    motion_recipe : MotionRecipe
        The run's resolved ``MotionCorrectionParameters`` recipe.
    source_lineage : SourceLineage, optional
        The run's source and artifact detection.
    concat_source : dict, optional
        ``session_group_owner``, ``session_group_name`` and
        ``preprocessing_params_name`` of a concat run.
    concat_members : list of dict, optional
        The concat run's expected members, each ``member_index``,
        ``recording_id`` and ``artifact_detection_id`` (``None`` unmasked),
        compared with the estimate's concatenation's ``MemberSnapshot``.

    Returns
    -------
    str or None
        Every mismatch, each naming the estimate's value and the run's, or
        ``None`` when the estimate can be applied.
    """
    from spyglass.spikesorting.v2.motion import (
        _SOURCE_TABLES,
        MotionEstimate,
        MotionEstimateSelection,
        _live_source_row,
    )

    key = {"motion_estimate_id": uuid.UUID(str(motion_estimate_id))}
    if not (MotionEstimate & key):
        return (
            f"motion_estimate_id {key['motion_estimate_id']} is not a "
            "populated MotionEstimate. Save the estimate first "
            "(MotionEstimateSelection.insert_selection, then "
            "MotionEstimate.populate)."
        )
    problems = []
    estimation_name, source_content_hash = (
        MotionEstimateSelection & key
    ).fetch1("motion_estimation_params_name", "source_content_hash")
    recipe_estimation = motion_recipe.recipe["motion_estimation_params_name"]
    if estimation_name != recipe_estimation:
        problems.append(
            f"it was estimated with MotionEstimationParameters "
            f"{estimation_name!r}, but recipe "
            f"{motion_recipe.recipe['motion_correction_params_name']!r} "
            f"names {recipe_estimation!r}"
        )
    lineage = MotionEstimateSelection.resolve_source(key)
    # A recomputed source's traces differ from the estimated ones.
    if _SOURCE_TABLES[lineage.kind] & lineage.key:
        try:
            _live_source_row(
                lineage,
                key["motion_estimate_id"],
                source_content_hash,
            )
        except ValueError as exc:
            problems.append(str(exc))
    if source_lineage is not None:
        problems.extend(
            correction_lineage_mismatch(source_lineage, lineage, consumer="run")
        )
    if concat_source is not None:
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecordingSelection,
        )

        if lineage.kind != "concatenated_recording":
            problems.append(
                f"it was estimated on recording {lineage.key}, but this run "
                "sorts a concatenation"
            )
        else:
            estimated = (ConcatenatedRecordingSelection & lineage.key).fetch1(
                *concat_source
            )
            for name, value in zip(concat_source, estimated):
                if value != concat_source[name]:
                    problems.append(
                        f"its concatenation has {name} {value!r}, but this "
                        f"run's has {concat_source[name]!r}"
                    )
            if concat_members is not None:
                problems.extend(
                    _concat_member_mismatch(lineage.key, concat_members)
                )
    if not problems:
        return None
    return (
        f"motion estimate {key['motion_estimate_id']} cannot be applied to "
        f"this run: {'; '.join(problems)}. Apply an estimate of this run's "
        "source, artifact mask and recipe."
    )


def motion_geometry_problem_from_contacts(
    channel_ids,
    positions,
    probe_types,
    electrode_group_names,
    probe_shanks,
    resolved_estimation: dict,
) -> "str | None":
    """Say why a sort group's effective geometry cannot be motion-estimated.

    Runs ``check_estimation_eligibility`` on a one-frame stand-in recording
    with the recording stage's planar geometry. Incomplete or collapsed
    geometry returns ``None``; ``sort_group_geometry_problem`` reports it.

    Parameters
    ----------
    channel_ids : sequence of int
        Sort-group electrode ids, sorted, row-aligned to the other inputs.
    positions : array_like
        ``(n, 3)`` ``(rel_x, rel_y, rel_z)`` contact positions (NaN where
        missing).
    probe_types, electrode_group_names, probe_shanks : sequence
        Per-channel probe type, electrode group and probe shank.
    resolved_estimation : dict
        A resolved estimation recipe
        (``_motion.estimation.resolve_estimation_params``).

    Returns
    -------
    str or None
        The estimator's refusal, or ``None`` when the geometry is eligible.
    """
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._motion.estimation import (
        check_estimation_eligibility,
    )

    channel_ids = [int(c) for c in channel_ids]
    source, planar = sort_group_planar_positions(
        positions, probe_types, electrode_group_names
    )
    if source is None:
        return None
    recording = NumpyRecording(
        np.zeros((1, len(channel_ids)), dtype="float32"),
        sampling_frequency=30000.0,
        channel_ids=channel_ids,
    )
    if source == "tetrode_12.5":
        maybe_apply_tetrode_geometry(
            recording,
            tuple(probe_types),
            tuple(electrode_group_names),
            channel_ids,
        )
    else:
        recording.set_channel_locations(planar)
    recording.set_property("group", [str(g) for g in electrode_group_names])
    recording.set_property("probe_shank", [str(s) for s in probe_shanks])
    try:
        check_estimation_eligibility(recording, resolved_estimation)
    except ValueError as exc:
        return str(exc)
    return None


def motion_geometry_problem(
    nwb_file_name: str, sort_group_id: int, resolved_estimation: dict
) -> "str | None":
    """Say why a sort group cannot be motion-estimated with a recipe.

    Applies :func:`motion_geometry_problem_from_contacts` to the group's
    registered contacts; the estimator re-checks the actual recording.

    Parameters
    ----------
    nwb_file_name : str
    sort_group_id : int
    resolved_estimation : dict
        A resolved estimation recipe.

    Returns
    -------
    str or None
        The operator-facing problem, or ``None``.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2._recording.geometry import (
        fetch_sort_group_contact_positions,
        fetch_sort_group_probe_info,
    )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    channel_ids = sorted(
        int(c)
        for c in (
            SortGroupV2.SortGroupElectrode
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
            }
        ).fetch("electrode_id")
    )
    if not channel_ids:
        return None
    positions = fetch_sort_group_contact_positions(nwb_file_name, channel_ids)
    probe_types, group_names = fetch_sort_group_probe_info(
        nwb_file_name, channel_ids
    )
    shanks = (
        Electrode
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": c} for c in channel_ids]
    ).fetch("probe_shank", order_by="electrode_id")
    problem = motion_geometry_problem_from_contacts(
        channel_ids,
        positions,
        probe_types,
        group_names,
        shanks,
        resolved_estimation,
    )
    if problem is None:
        return None
    return (
        f"sort_group_id={int(sort_group_id)} of {nwb_file_name!r} cannot be "
        f"motion-estimated with this recipe: {problem}"
    )
