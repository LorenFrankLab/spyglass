"""Complete synthetic provenance for tests of current v2 artifact writers."""

from __future__ import annotations

SORTING_ID = "00000000-0000-0000-0000-000000000001"
RECORDING_ID = "00000000-0000-0000-0000-000000000002"
CURATION_UUID = "00000000-0000-0000-0000-000000000003"


def sorting_provenance(**overrides):
    values = {
        "sorting_id": SORTING_ID,
        "recording_id": RECORDING_ID,
        "concat_recording_id": None,
        "sorter": "synthetic",
        "sorter_params_name": "synthetic",
        "sorter_params": {},
        "execution_params": {},
        "artifact_detection_id": None,
        "display_waveform_params_name": "synthetic",
        "effective_random_seed": 0,
        "spikeinterface_version": "test",
        "sorter_version": None,
        "analyzer_spikeinterface_version": "test",
        "statistics_spans": [[0, 100]],
    }
    values.update(overrides)
    return values


def curation_header(**overrides):
    values = {
        "sorting_id": SORTING_ID,
        "curation_id": 0,
        "curation_uuid": CURATION_UUID,
        "parent_curation_id": -1,
        "curation_source": "manual",
        "merges_applied": False,
        "description": "synthetic curation",
    }
    values.update(overrides)
    return values
