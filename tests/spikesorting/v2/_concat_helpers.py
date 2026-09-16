"""Explicit unmasked selections for tests unrelated to artifact detection."""


def select_unmasked_concat(key):
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        SessionGroup,
    )

    indices = (SessionGroup.Member & key).fetch("member_index")
    return ConcatenatedRecordingSelection.insert_selection(
        key, artifact_detection_ids={int(index): None for index in indices}
    )
