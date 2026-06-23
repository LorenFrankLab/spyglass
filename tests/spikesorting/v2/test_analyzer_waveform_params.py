"""Tracked analyzer-waveform parameters + region-resolved display recipe.

Covers the ``AnalyzerWaveformParameters`` Lookup (schema validation, shipped
region rows, path-safe name guard), the source-preprocessing -> display-recipe
resolver (single-recording and concat source parts), and that a populated sort
records its resolved display recipe and rebuilds the same cache folder.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from spyglass.spikesorting.v2._params.analyzer_waveform import (
    AnalyzerWaveformParamsSchema,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    CORTEX_DISPLAY_WAVEFORMS,
    CORTEX_PREPROC,
    HIPPOCAMPUS_DISPLAY_WAVEFORMS,
    HIPPOCAMPUS_METRIC_WAVEFORMS,
    HIPPOCAMPUS_PREPROC,
    waveform_params_for_preprocessing,
)


@contextlib.contextmanager
def _planted_source_sort(selection_table, id_attr, source_part, preproc_name):
    """Plant a SortingSelection + source-part + stub upstream selection row.

    Resolver / make_fetch tests need a sort whose source carries a given
    preprocessing recipe, without a full populate. This builds the upstream
    ``selection_table`` row from its live heading (the ``id_attr`` PK + the
    ``preprocessing_params_name`` set; numeric columns -> 0, the rest ->
    "bypass") and the ``SortingSelection`` master + ``source_part`` link, all
    with FK checks off, then yields the ``sorting_id`` and tears the rows down
    on exit. Works for both source kinds via (``selection_table``, ``id_attr``,
    ``source_part``): RecordingSelection/recording_id/RecordingSource or
    ConcatenatedRecordingSelection/concat_recording_id/ConcatenatedRecordingSource.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import SortingSelection

    id_value = uuid.uuid4()
    sid = uuid.uuid4()
    heading = selection_table().heading
    row = {}
    for name in heading.names:
        if name == id_attr:
            row[name] = id_value
        elif name == "preprocessing_params_name":
            row[name] = preproc_name
        elif heading.attributes[name].numeric:
            row[name] = 0
        else:
            row[name] = "bypass"
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        selection_table.insert1(row, allow_direct_insert=True)
        SortingSelection.insert1(
            {
                "sorting_id": sid,
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
            },
            allow_direct_insert=True,
        )
        source_part.insert1(
            {"sorting_id": sid, id_attr: id_value}, allow_direct_insert=True
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")
    try:
        yield sid
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (SortingSelection & {"sorting_id": sid}).delete_quick()
            (selection_table & {id_attr: id_value}).delete_quick()
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")

# --------------------------------------------------------------------------- #
# Pydantic schema (DB-free)
# --------------------------------------------------------------------------- #


def test_analyzer_waveform_params_schema_rejects_extra():
    """An unknown key raises (``extra='forbid'``)."""
    with pytest.raises(Exception):
        AnalyzerWaveformParamsSchema(unexpected_key=1)


def test_analyzer_waveform_params_schema_rejects_purpose_whiten_mismatch():
    """display must be unwhitened; metric must be whitened."""
    with pytest.raises(Exception):
        AnalyzerWaveformParamsSchema(purpose="display", whiten=True)
    with pytest.raises(Exception):
        AnalyzerWaveformParamsSchema(purpose="metric", whiten=False)


def test_analyzer_waveform_params_schema_rejects_nonpositive_bounds():
    """A zero/negative window or sub-unit subsample raises.

    A zero/negative extraction window or ``max_spikes_per_unit < 1`` would
    reach ``analyzer.compute(...)`` and produce a meaningless analyzer; the
    field bounds (``ms_before/ms_after gt=0``, ``max_spikes_per_unit ge=1``)
    must reject them.
    """
    for bad in (
        {"ms_before": 0.0},
        {"ms_before": -0.5},
        {"ms_after": 0.0},
        {"ms_after": -1.0},
        {"max_spikes_per_unit": 0},
        {"max_spikes_per_unit": -10},
    ):
        with pytest.raises(Exception):
            AnalyzerWaveformParamsSchema(**bad)


# --------------------------------------------------------------------------- #
# Region map (pure function)
# --------------------------------------------------------------------------- #


def test_waveform_params_for_preprocessing_literal_region_names():
    """Region recipes map to their (display, metric) pairs; others fall back.

    Asserts the LITERAL recipe names (not the module constants the mapping is
    built from) so the test pins the actual region behavior rather than being a
    tautology against the same constants.
    """
    assert waveform_params_for_preprocessing("franklab_hippocampus_2026_06") == (
        "franklab_hippocampus_actual_waveforms",
        "franklab_hippocampus_metric_waveforms",
    )
    assert waveform_params_for_preprocessing("franklab_cortex_2026_06") == (
        "franklab_cortex_actual_waveforms",
        "franklab_cortex_metric_waveforms",
    )
    # The schema-default "default" recipe and any unknown / multi-region recipe
    # fall back to the wide cortex pair -- never a silently mixed window.
    assert waveform_params_for_preprocessing("default") == (
        "franklab_cortex_actual_waveforms",
        "franklab_cortex_metric_waveforms",
    )
    assert waveform_params_for_preprocessing("some_custom_recipe") == (
        "franklab_cortex_actual_waveforms",
        "franklab_cortex_metric_waveforms",
    )


# --------------------------------------------------------------------------- #
# Lookup table (DB)
# --------------------------------------------------------------------------- #


def test_analyzer_waveform_params_default_rows(dj_conn):
    """``insert_default`` ships the four region display/metric rows."""
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()
    rows = {
        r["waveform_params_name"]: r["params"]
        for r in AnalyzerWaveformParameters.fetch(as_dict=True)
    }
    for name in (
        "franklab_hippocampus_actual_waveforms",
        "franklab_hippocampus_metric_waveforms",
        "franklab_cortex_actual_waveforms",
        "franklab_cortex_metric_waveforms",
    ):
        assert name in rows, f"missing default row {name!r}"

    hippo_display = rows["franklab_hippocampus_actual_waveforms"]
    assert hippo_display["ms_before"] == 0.5
    assert hippo_display["ms_after"] == 0.5
    assert hippo_display["max_spikes_per_unit"] == 20000
    assert hippo_display["whiten"] is False
    assert hippo_display["purpose"] == "display"

    hippo_metric = rows["franklab_hippocampus_metric_waveforms"]
    assert hippo_metric["whiten"] is True
    assert hippo_metric["purpose"] == "metric"

    cortex_display = rows["franklab_cortex_actual_waveforms"]
    assert cortex_display["ms_before"] == 1.0
    assert cortex_display["ms_after"] == 2.0
    assert cortex_display["max_spikes_per_unit"] == 20000


def test_analyzer_waveform_params_rejects_unsafe_name(dj_conn):
    """A non-path-safe name is rejected (it is embedded in a folder name)."""
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    with pytest.raises(ValueError, match="path-safe"):
        AnalyzerWaveformParameters().insert1(
            {
                "waveform_params_name": "bad/name",
                "params": AnalyzerWaveformParamsSchema().model_dump(),
            }
        )


def test_analyzer_waveform_params_insert_guards(dj_conn):
    """The insert path validates the blob and rejects duplicate content.

    The blob is Pydantic-validated through ``validate_lookup_rows`` (an invalid
    purpose/whiten pair is rejected at insert, not just at schema-construction
    time), and a second NAME for an already-shipped blob forks provenance and
    is rejected unless ``allow_duplicate_params=True``.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()

    # A contradictory blob is rejected THROUGH the table insert (not only at
    # schema construction): build a valid display dict, then corrupt it.
    bad_blob = AnalyzerWaveformParamsSchema().model_dump()
    bad_blob["whiten"] = True  # display + whiten=True -> invalid pair
    with pytest.raises(Exception):
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "bad_pair", "params": bad_blob}
        )

    # A new name for the cortex display blob forks provenance -> rejected.
    cortex_blob = (
        AnalyzerWaveformParameters
        & {"waveform_params_name": "franklab_cortex_actual_waveforms"}
    ).fetch1("params")
    with pytest.raises(DuplicateParameterContentError):
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "cortex_clone", "params": cortex_blob}
        )
    # ...unless the escape hatch is set.
    try:
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "cortex_clone", "params": cortex_blob},
            allow_duplicate_params=True,
        )
        assert (
            AnalyzerWaveformParameters
            & {"waveform_params_name": "cortex_clone"}
        )
    finally:
        (
            AnalyzerWaveformParameters
            & {"waveform_params_name": "cortex_clone"}
        ).delete_quick()


def test_display_waveform_params_name_not_in_sorting_identity(dj_conn):
    """The display recipe is a secondary Sorting attr, never sort identity.

    ``sorting_id`` is content-addressed from (recording, sorter,
    sorter_params, artifact_detection); the region-resolved display window is
    NOT an input, so it cannot perturb idempotency.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    assert "display_waveform_params_name" in Sorting().heading.names
    assert (
        "display_waveform_params_name" not in Sorting().primary_key
    )
    assert (
        "display_waveform_params_name"
        not in SortingSelection().heading.names
    )


# --------------------------------------------------------------------------- #
# Source-preprocessing -> display-recipe resolver (concat source part)
# --------------------------------------------------------------------------- #


def test_concat_display_recipe_resolved_from_concat_preprocessing(dj_conn):
    """A concat-backed sort resolves its recipe from the concat selection.

    Parent-phase concat populate is gated, so this drives the resolver with
    direct source-part rows (FK checks off) rather than a populated
    ``ConcatenatedRecording``: the resolver must read the
    ``ConcatenatedRecordingSelection -> PreprocessingParameters`` FK, NOT a
    member ``RecordingSelection``.
    """
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    PreprocessingParameters.insert_default()
    SorterParameters.insert_default()
    with _planted_source_sort(
        ConcatenatedRecordingSelection,
        "concat_recording_id",
        SortingSelection.ConcatenatedRecordingSource,
        HIPPOCAMPUS_PREPROC,
    ) as sid:
        preproc = SortingSelection.resolve_source_preprocessing_params_name(
            {"sorting_id": sid}
        )
        assert preproc == HIPPOCAMPUS_PREPROC
        display, metric = waveform_params_for_preprocessing(preproc)
        assert display == HIPPOCAMPUS_DISPLAY_WAVEFORMS
        assert metric == HIPPOCAMPUS_METRIC_WAVEFORMS


@pytest.mark.parametrize(
    ("preproc_name", "expected_display"),
    [
        ("franklab_hippocampus_2026_06", "franklab_hippocampus_actual_waveforms"),
        ("franklab_cortex_2026_06", "franklab_cortex_actual_waveforms"),
        # custom / unknown recipe -> wide cortex fallback
        ("default", "franklab_cortex_actual_waveforms"),
    ],
)
def test_recording_source_display_recipe_resolved_by_region(
    dj_conn, preproc_name, expected_display
):
    """A single-recording sort resolves its display recipe from its source
    ``RecordingSelection.preprocessing_params_name`` (region).

    Drives the resolver with direct ``RecordingSource`` rows (FK checks off) so
    it does not depend on a full populate: hippocampus -> the 0.5/0.5 display
    row, cortex -> the 1.0/2.0 display row, any other recipe -> the wide cortex
    fallback. Pins the actual region behavior the generic populated-sort
    integration (which only exercises the fallback) does not.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    PreprocessingParameters.insert_default()
    SorterParameters.insert_default()
    with _planted_source_sort(
        RecordingSelection,
        "recording_id",
        SortingSelection.RecordingSource,
        preproc_name,
    ) as sid:
        preproc = SortingSelection.resolve_source_preprocessing_params_name(
            {"sorting_id": sid}
        )
        assert preproc == preproc_name
        display, _metric = waveform_params_for_preprocessing(preproc)
        assert display == expected_display


def test_make_fetch_resolves_hippocampus_display_blob(dj_conn):
    """``Sorting.make_fetch`` resolves AND fetches the hippocampus 0.5/0.5 blob.

    The build-window unit tests prove ``build_analyzer`` honors a 0.5/0.5 blob,
    and the resolver tests prove the hippocampus recipe NAME resolves; this
    closes the gap between them by driving the real ``make_fetch`` for a
    hippocampus-source sort and asserting the resolved ``display_waveform_params``
    blob it threads to ``make_compute`` is the 0.5/0.5 window (not the cortex
    fallback the generic populated-sort integration exercises).
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    PreprocessingParameters.insert_default()
    SorterParameters.insert_default()
    AnalyzerWaveformParameters.insert_default()
    with _planted_source_sort(
        RecordingSelection,
        "recording_id",
        SortingSelection.RecordingSource,
        HIPPOCAMPUS_PREPROC,
    ) as sid:
        fetched = Sorting().make_fetch({"sorting_id": sid})
        assert (
            fetched.display_waveform_params_name
            == "franklab_hippocampus_actual_waveforms"
        )
        # The fetched blob (threaded into make_compute -> _build_analyzer) is
        # the hippocampus 0.5/0.5 window, not the cortex fallback.
        assert fetched.display_waveform_params["ms_before"] == 0.5
        assert fetched.display_waveform_params["ms_after"] == 0.5
        assert fetched.display_waveform_params["max_spikes_per_unit"] == 20000


def test_fetch_waveform_params_missing_row_raises(dj_conn):
    """Resolution is strict: a missing tracked row raises a clear error.

    The analyzer's waveform parameters must be recorded in the database
    (provenance); resolving an uninstalled recipe must fail loudly with an
    actionable message, never silently fall back to a hardcoded / catalog
    default.
    """
    from spyglass.spikesorting.v2._sorting_analyzer import (
        fetch_waveform_params,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    # Ensure the row is absent, then assert the strict failure.
    missing = "definitely_not_installed_recipe"
    (
        AnalyzerWaveformParameters & {"waveform_params_name": missing}
    ).delete_quick()
    with pytest.raises(ValueError, match="initialize_v2_defaults"):
        fetch_waveform_params(missing)


# --------------------------------------------------------------------------- #
# Integration: a populated sort records + rebuilds its display recipe
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_sorting_records_display_waveform_params(populated_sorting):
    """make_fetch resolves + make_insert stores the display recipe; the
    cache-miss rebuild reads the STORED name, never re-resolving."""
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sid = populated_sorting["sorting_id"]
    row = (Sorting & populated_sorting).fetch1()
    if int(row["n_units"]) == 0:
        pytest.skip("zero-unit smoke sort: no analyzer to assert")

    # The smoke sort uses the 'default' preprocessing recipe, which is not in
    # the region map -> the wide cortex display fallback.
    stored_name = row["display_waveform_params_name"]
    assert stored_name == CORTEX_DISPLAY_WAVEFORMS

    # make_fetch resolves the row (DB I/O lives here, not in make_compute).
    fetched = Sorting().make_fetch(dict(populated_sorting))
    assert fetched.display_waveform_params_name == stored_name
    assert isinstance(fetched.display_waveform_params, dict)
    assert fetched.display_waveform_params["ms_before"] == 1.0
    assert fetched.display_waveform_params["ms_after"] == 2.0

    # The on-disk analyzer folder is named by the stored recipe.
    folder = analyzer_path(sid, stored_name)
    assert folder.exists()

    # Cache-miss rebuild resolves the SAME folder from the stored name.
    import shutil

    import spikeinterface as si

    shutil.rmtree(folder)
    assert not folder.exists()
    analyzer = Sorting().get_analyzer(dict(populated_sorting))
    assert isinstance(analyzer, si.SortingAnalyzer)
    assert folder.exists()


@pytest.mark.slow
def test_rebuild_reads_stored_window_never_re_resolves(
    populated_sorting, monkeypatch
):
    """A cache-miss rebuild reads the STORED recipe, never re-resolving region.

    The contract is that the region-resolved window is computed once at sort
    time, persisted on ``Sorting.display_waveform_params_name``, and never
    re-derived from the recipe catalog on later rebuilds (otherwise a later
    recipe-map change would silently shift an existing sort's window and its
    ``peak_amplitude_uv``). The existing happy-path test can't distinguish
    "read the stored field" from "re-resolved to the same answer" because the
    map returns the same name. Here the sort-time resolver is monkeypatched to
    RAISE, so any rebuild that re-resolves blows up loudly; a correct rebuild
    (reading the stored field) never calls it.
    """
    import shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2 import sorting as sorting_module
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sid = populated_sorting["sorting_id"]
    row = (Sorting & populated_sorting).fetch1()
    if int(row["n_units"]) == 0:
        pytest.skip("zero-unit smoke sort: no analyzer to rebuild")
    stored_name = row["display_waveform_params_name"]

    def _must_not_re_resolve(*args, **kwargs):
        raise AssertionError(
            "analyzer rebuild re-resolved the region window instead of reading "
            "the stored Sorting.display_waveform_params_name"
        )

    monkeypatch.setattr(
        sorting_module,
        "waveform_params_for_preprocessing",
        _must_not_re_resolve,
    )

    folder = analyzer_path(sid, stored_name)
    shutil.rmtree(folder)
    assert not folder.exists()

    # Rebuild must succeed WITHOUT calling the (now-raising) region resolver.
    analyzer = Sorting().get_analyzer(dict(populated_sorting))
    assert isinstance(analyzer, si.SortingAnalyzer)
    # Rebuilt under the originally-persisted recipe; the stored field is intact.
    assert folder.exists()
    assert (Sorting & populated_sorting).fetch1(
        "display_waveform_params_name"
    ) == stored_name
