"""Database-independent value types exchanged by recording stages.

The table module and services share these carrier definitions. Services import
this module directly so their result contracts do not activate a schema.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._storage.staged_outputs import StagedOutputs


class RecordingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Recording.make_fetch`.

    Tri-part dispatch unpacks this positionally into ``make_compute``;
    fields are listed in the order they appear in the compute
    signature.

    Attributes
    ----------
    sort_valid_times : numpy.ndarray
        Requested sort interval ``valid_times``, shape
        ``(n_intervals, 2)`` in seconds.
    raw_valid_times : numpy.ndarray
        Raw data ``valid_times``, shape ``(n_intervals, 2)`` in seconds.
    raw_object_id : str
        NWB object id of the session's raw acquisition ElectricalSeries
        (``Raw.raw_object_id``); pins the compute step to the exact raw
        source the selection lineage points at.
    raw_path : str
        Absolute path of the session's raw NWB (``Nwbfile.get_abs_path``).
    """

    sel: dict
    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    sort_valid_times: np.ndarray
    raw_valid_times: np.ndarray
    preprocessing_params: PreprocessingParamsSchema
    preprocessing_job_kwargs: dict | None
    probe_types: tuple
    electrode_group_names: tuple
    bad_channel_ids: tuple
    raw_object_id: str
    raw_path: str


class RecordingComputed(NamedTuple):
    """Outputs of :meth:`Recording.make_compute`.

    Unpacked positionally into ``make_insert``.
    """

    analysis_file_name: str
    object_id: str
    content_hash: str
    saved_start: float
    saved_end: float
    sampling_frequency: float
    n_channels: int
    duration_s: float
    sel: dict
    sort_valid_times: np.ndarray
    expected_saved_total: float
    n_intended_intervals: int

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


class RecordingArtifactResult(NamedTuple):
    """Outputs of :meth:`Recording._compute_recording_artifact`.

    Internal helper result -- NOT a tri-part contract object, so it is never
    splatted into ``make_*``. ``make_compute`` reads these fields by name to
    build :class:`RecordingComputed`, and ``_rebuild_nwb_artifact`` reads only
    ``content_hash``. Typed/named so the eight values are not threaded through
    brittle positional unpacking. Field order intentionally matches
    ``RecordingComputed``'s first eight fields; consumers read these values
    by name.
    """

    analysis_file_name: str
    object_id: str
    content_hash: str
    saved_start: float
    saved_end: float
    sampling_frequency: float
    n_channels: int
    duration_s: float
