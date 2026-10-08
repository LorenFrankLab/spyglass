"""Test helpers for UnitMatch selections (no fixtures, no module-level DB)."""

from __future__ import annotations

from pathlib import Path


def reforge_selection(
    pk,
    *,
    input_edits=None,
    recording_edits=None,
    drop_recordings=(),
    extra_recordings=(),
    rehash=False,
    master_edits=None,
):
    """Re-insert a selection's rows directly, with edits, bypassing insert_inputs.

    Deletes the selection (master and parts), then inserts the master with
    ``allow_direct_insert`` and the edited parts -- the shape a raw insert or
    an out-of-band edit leaves behind.

    Parameters
    ----------
    pk : dict
        ``{"unitmatch_id": ...}`` of a selection made by ``insert_inputs``.
    input_edits : dict, optional
        ``{input_index: {field: value}}`` applied to ``Input`` rows.
    recording_edits : dict, optional
        ``{(input_index, recording_index): {field: value}}`` applied to
        ``InputRecording`` rows.
    drop_recordings : iterable of (int, int), optional
        ``(input_index, recording_index)`` rows to leave out.
    extra_recordings : iterable of dict, optional
        Additional ``InputRecording`` rows (without ``unitmatch_id``).
    rehash : bool, optional
        Store the ``input_set_hash`` of the edited parts on the master, so
        only checks other than the hash can catch the edit.
    master_edits : dict, optional
        Fields to override on the master row (applied after ``rehash``).

    Returns
    -------
    dict
        The inserted master row.
    """
    from spyglass.spikesorting.v2._matching.graph import input_set_hash
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    master = (UnitMatchSelection & pk).fetch1()
    inputs = (UnitMatchSelection.Input & pk).fetch(
        as_dict=True, order_by="input_index"
    )
    recordings = (UnitMatchSelection.InputRecording & pk).fetch(
        as_dict=True, order_by=("input_index", "recording_index")
    )
    for row in inputs:
        row.update((input_edits or {}).get(int(row["input_index"]), {}))
    kept = []
    for row in recordings:
        position = (int(row["input_index"]), int(row["recording_index"]))
        if position in set(drop_recordings):
            continue
        row.update((recording_edits or {}).get(position, {}))
        kept.append(row)
    kept += [
        {**row, "unitmatch_id": master["unitmatch_id"]}
        for row in extra_recordings
    ]
    if rehash:
        master["input_set_hash"] = input_set_hash(inputs, kept)
    master.update(master_edits or {})
    (UnitMatchSelection & pk).super_delete(warn=False)
    UnitMatchSelection.insert1(master, allow_direct_insert=True)
    UnitMatchSelection.Input.insert(inputs, allow_direct_insert=True)
    UnitMatchSelection.InputRecording.insert(kept, allow_direct_insert=True)
    return master


def install_fixture_pairer(
    monkeypatch,
    *,
    matcher_name: str,
    matcher_params_name: str,
    pairs: list[list[int]],
    probability: float = 0.99,
    seen_unit_ids: list[list[int]] | None = None,
    read_bundles: bool = False,
    later_first: bool = False,
    fed: list | None = None,
):
    """Register a lightweight matcher for DB tests.

    By default bundle extraction is stubbed and the matcher emits every listed
    ``(unit_a, unit_b)`` pair. With ``read_bundles=True`` the real
    shared waveform extraction runs, the matcher reads each session's bundle
    ``cluster_group.tsv`` (appending its unit ids to ``seen_unit_ids``), and it
    emits only the listed pairs whose units are both in the bundles -- so the
    pairs follow what the bundles contain, as UnitMatchPy's loader does.
    With ``later_first=True`` each listed pair is emitted with the SECOND fed
    session as side a (``(unit in second, unit in first)``), so side a does
    not follow feed order. ``fed``, when given, collects ``(bundle directory
    name, sorting_id)`` for each session in the order the matcher received
    them.
    """
    from pydantic import BaseModel, ConfigDict, Field

    from spyglass.spikesorting.v2 import matcher_protocol as mp
    from spyglass.spikesorting.v2._matching.waveforms import (
        WaveformInputPreparer,
    )
    from spyglass.spikesorting.v2.matcher_protocol import (
        MatchPair,
        register_matcher,
    )
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    class _FixtureMatcherParams(BaseModel):
        """Params schema for a test-only matcher."""

        model_config = ConfigDict(extra="forbid")
        tracked_unit_threshold: float = 0.5
        max_strict_nodes: int = 2000
        probability: float = 0.99
        pairs: list = Field(default_factory=list)
        schema_version: int = 1

    def _bundle_unit_ids(session_input) -> set[int]:
        lines = (
            (Path(session_input.bundle_dir) / "cluster_group.tsv")
            .read_text()
            .splitlines()
        )
        return {int(line.split("\t")[0]) for line in lines[1:]}

    class _FixturePairer:
        """Emits the listed (unit_a, unit_b) pairs."""

        name = matcher_name

        def match(self, session_inputs, params):
            if fed is not None:
                fed.extend(
                    (
                        Path(session_input.bundle_dir).name,
                        str(session_input.curation_key["sorting_id"]),
                    )
                    for session_input in session_inputs
                )
            first, second = session_inputs[0], session_inputs[1]
            if later_first:
                first, second = second, first
            left = first.curation_key
            right = second.curation_key
            listed = params.get("pairs", [])
            if read_bundles:
                left_ids, right_ids = (
                    _bundle_unit_ids(session_input)
                    for session_input in (first, second)
                )
                if seen_unit_ids is not None:
                    seen_unit_ids.extend([sorted(left_ids), sorted(right_ids)])
                listed = [
                    (pair_a, pair_b)
                    for pair_a, pair_b in listed
                    if pair_a in left_ids and pair_b in right_ids
                ]
            return [
                MatchPair(
                    session_a_sorting_id=str(left["sorting_id"]),
                    session_a_curation_id=int(left["curation_id"]),
                    unit_a_id=int(pair_a),
                    session_b_sorting_id=str(right["sorting_id"]),
                    session_b_curation_id=int(right["curation_id"]),
                    unit_b_id=int(pair_b),
                    match_probability=float(params.get("probability", 0.99)),
                )
                for pair_a, pair_b in listed
            ]

    def _noop_extract(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        if seen_unit_ids is not None:
            seen_unit_ids.append([int(u) for u in sorting.get_unit_ids()])
        return []

    preparer = WaveformInputPreparer()
    if not read_bundles:
        monkeypatch.setattr(preparer, "extract", _noop_extract)

    saved = (
        dict(mp._MATCHER_REGISTRY),
        dict(mp._SCHEMA_REGISTRY),
        dict(mp._PREPARER_REGISTRY),
    )
    try:
        register_matcher(
            _FixturePairer(), _FixtureMatcherParams, input_preparer=preparer
        )
        MatcherParameters().insert1(
            {
                "matcher_params_name": matcher_params_name,
                "matcher": matcher_name,
                "params": {"pairs": pairs, "probability": probability},
            },
            skip_duplicates=True,
        )
    except Exception:
        restore_matcher_registry(saved)
        raise
    return saved


def restore_matcher_registry(saved_registry) -> None:
    """Undo a test-only matcher registration."""
    from spyglass.spikesorting.v2 import matcher_protocol as mp

    saved_matchers, saved_schemas, saved_preparers = saved_registry
    mp._MATCHER_REGISTRY.clear()
    mp._MATCHER_REGISTRY.update(saved_matchers)
    mp._SCHEMA_REGISTRY.clear()
    mp._SCHEMA_REGISTRY.update(saved_schemas)
    mp._PREPARER_REGISTRY.clear()
    mp._PREPARER_REGISTRY.update(saved_preparers)
