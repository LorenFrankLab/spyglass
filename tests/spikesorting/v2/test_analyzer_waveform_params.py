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
from spyglass.spikesorting.v2._core.recipe_catalog import (
    CORTEX_DISPLAY_WAVEFORMS,
    HIPPOCAMPUS_DISPLAY_WAVEFORMS,
    HIPPOCAMPUS_METRIC_WAVEFORMS,
    HIPPOCAMPUS_PREPROC,
    waveform_params_for_preprocessing,
)


def _stub_row(table, id_attr, id_value, **fixed):
    """Build a row of ``table`` from its live heading.

    The ``id_attr`` PK is ``id_value``, columns named in ``fixed`` take their
    given values, numeric columns -> 0 and the rest -> "bypass".
    """
    heading = table().heading
    row = {}
    for name in heading.names:
        if name == id_attr:
            row[name] = id_value
        elif name in fixed:
            row[name] = fixed[name]
        elif heading.attributes[name].numeric:
            row[name] = 0
        else:
            row[name] = "bypass"
    return row


@contextlib.contextmanager
def _planted_source_sort(
    selection_table, id_attr, source_part, preproc_name, trace_table=None
):
    """Plant a SortingSelection + source-part + stub upstream selection row.

    Gives resolver tests a sort with a chosen preprocessing recipe without
    a populate. Rows are inserted with FK checks off and removed on exit;
    yields the ``sorting_id`` ``insert_selection`` would derive. Works for
    single-recording and concat sources. ``trace_table``, when given, also
    gets a stub row.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
        sorting_identity_payload,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    sorter = "mountainsort5"
    sorter_params_name = "franklab_30khz_ms5_2026_06"
    id_value = uuid.uuid4()
    sid = deterministic_id(
        "sorting",
        sorting_identity_payload(
            sorter=sorter,
            sorter_params_name=sorter_params_name,
            **{id_attr: id_value},
        ),
    )
    row = _stub_row(
        selection_table,
        id_attr,
        id_value,
        preprocessing_params_name=preproc_name,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        selection_table.insert1(row, allow_direct_insert=True)
        if trace_table is not None:
            trace_table.insert1(
                _stub_row(trace_table, id_attr, id_value),
                allow_direct_insert=True,
            )
        SortingSelection.insert1(
            {
                "sorting_id": sid,
                "sorter": sorter,
                "sorter_params_name": sorter_params_name,
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
            if trace_table is not None:
                (trace_table & {id_attr: id_value}).delete_quick()
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
    """A zero/negative window or ``max_spikes_per_unit < 1`` raises."""
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
    """Region recipes map to their (display, metric) pairs.

    Uses literal names, not the module constants, to avoid a tautology.
    """
    assert waveform_params_for_preprocessing(
        "franklab_hippocampus_2026_06"
    ) == (
        "franklab_hippocampus_actual_waveforms",
        "franklab_hippocampus_metric_waveforms",
    )
    assert waveform_params_for_preprocessing("franklab_cortex_2026_06") == (
        "franklab_cortex_actual_waveforms",
        "franklab_cortex_metric_waveforms",
    )


def test_waveform_params_for_preprocessing_falls_back_to_cortex(dj_conn):
    """Recipes with no region mapping or recorded windows use the cortex pair."""
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

    A second name for an existing blob is rejected unless
    ``allow_duplicate_params=True``.
    """
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

    AnalyzerWaveformParameters.insert_default()

    # Rejected at table insert, not only at schema construction.
    bad_blob = AnalyzerWaveformParamsSchema().model_dump()
    bad_blob["whiten"] = True  # display + whiten=True -> invalid pair
    with pytest.raises(Exception):
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "bad_pair", "params": bad_blob}
        )

    cortex_blob = (
        AnalyzerWaveformParameters
        & {"waveform_params_name": "franklab_cortex_actual_waveforms"}
    ).fetch1("params")
    with pytest.raises(DuplicateParameterContentError):
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "cortex_clone", "params": cortex_blob}
        )
    try:
        AnalyzerWaveformParameters().insert1(
            {"waveform_params_name": "cortex_clone", "params": cortex_blob},
            allow_duplicate_params=True,
        )
        assert AnalyzerWaveformParameters & {
            "waveform_params_name": "cortex_clone"
        }
    finally:
        (
            AnalyzerWaveformParameters
            & {"waveform_params_name": "cortex_clone"}
        ).delete_quick()


def test_display_waveform_params_name_not_in_sorting_identity(dj_conn):
    """The display recipe is a secondary Sorting attr, never sort identity.

    It is not an input to the content-addressed ``sorting_id``.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    assert "display_waveform_params_name" in Sorting().heading.names
    assert "display_waveform_params_name" not in Sorting().primary_key
    assert (
        "display_waveform_params_name" not in SortingSelection().heading.names
    )


# --------------------------------------------------------------------------- #
# Source-preprocessing -> display-recipe resolver (concat source part)
# --------------------------------------------------------------------------- #


def test_concat_display_recipe_resolved_from_concat_preprocessing(dj_conn):
    """A concat-backed sort resolves its recipe from the concat selection.

    It reads ``ConcatenatedRecordingSelection.preprocessing_params_name``,
    not a member ``RecordingSelection``.
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
        (
            "franklab_hippocampus_2026_06",
            "franklab_hippocampus_actual_waveforms",
        ),
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

    Hippocampus -> 0.5/0.5, cortex -> 1.0/2.0, anything else -> the wide
    cortex fallback.
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


def test_make_fetch_resolves_hippocampus_display_blob(dj_conn, monkeypatch):
    """``Sorting.make_fetch`` resolves AND fetches the hippocampus 0.5/0.5 blob.

    The blob passed to ``make_compute`` is the 0.5/0.5 window, not the
    cortex fallback.
    """
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
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
    # The planted Recording has no file; this test never reads traces.
    monkeypatch.setattr(
        SortingSelection,
        "ensure_effective_traces",
        staticmethod(lambda traces: "planted-traces.nwb"),
    )
    with _planted_source_sort(
        RecordingSelection,
        "recording_id",
        SortingSelection.RecordingSource,
        HIPPOCAMPUS_PREPROC,
        trace_table=Recording,
    ) as sid:
        fetched = Sorting().make_fetch({"sorting_id": sid})
        assert (
            fetched.display_waveform_params_name
            == "franklab_hippocampus_actual_waveforms"
        )
        assert fetched.display_waveform_params["ms_before"] == 0.5
        assert fetched.display_waveform_params["ms_after"] == 0.5
        assert fetched.display_waveform_params["max_spikes_per_unit"] == 20000


def test_fetch_waveform_params_missing_row_raises(dj_conn):
    """Resolution is strict: a missing tracked row raises a clear error.

    Never falls back to a hardcoded default.
    """
    from spyglass.spikesorting.v2._sorting.analyzer import fetch_waveform_params
    from spyglass.spikesorting.v2.sorting import AnalyzerWaveformParameters

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
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sid = populated_sorting["sorting_id"]
    row = (Sorting & populated_sorting).fetch1()
    if int(row["n_units"]) == 0:
        pytest.skip("zero-unit smoke sort: no analyzer to assert")

    # 'default' is not in the region map -> cortex fallback.
    stored_name = row["display_waveform_params_name"]
    assert stored_name == CORTEX_DISPLAY_WAVEFORMS

    fetched = Sorting().make_fetch(dict(populated_sorting))
    assert fetched.display_waveform_params_name == stored_name
    assert isinstance(fetched.display_waveform_params, dict)
    assert fetched.display_waveform_params["ms_before"] == 1.0
    assert fetched.display_waveform_params["ms_after"] == 2.0

    folder = analyzer_path(sid, stored_name)
    assert folder.exists()

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

    A recipe-map change must not shift an existing sort's window. The
    resolver is patched to raise, so re-resolving fails the test.
    """
    import shutil

    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting import fetch as _sorting_fetch
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
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
        _sorting_fetch,
        "waveform_params_for_preprocessing",
        _must_not_re_resolve,
    )

    folder = analyzer_path(sid, stored_name)
    shutil.rmtree(folder)
    assert not folder.exists()

    analyzer = Sorting().get_analyzer(dict(populated_sorting))
    assert isinstance(analyzer, si.SortingAnalyzer)
    # Rebuilt under the originally-persisted recipe; the stored field is intact.
    assert folder.exists()
    assert (Sorting & populated_sorting).fetch1(
        "display_waveform_params_name"
    ) == stored_name


def test_sparsity_params_default_matches_si_and_validates_methods():
    """``sparsity`` defaults to SI's radius/100 um; method fields are checked.

    The default reproduces ``create_sorting_analyzer(sparse=True)``, and
    every effective value is recorded in the row.
    """
    import pydantic

    from spyglass.spikesorting.v2._params.analyzer_waveform import (
        ANALYZER_WAVEFORM_SCHEMA_VERSION,
        AnalyzerWaveformParamsSchema,
        SparsityParams,
    )

    assert ANALYZER_WAVEFORM_SCHEMA_VERSION == 3
    default = AnalyzerWaveformParamsSchema().model_dump()["sparsity"]
    assert default == {
        "method": "radius",
        "radius_um": 100.0,
        "num_channels": None,
        "num_spikes_for_sparsity": 100,
        "ms_before": 1.0,
        "ms_after": 2.5,
    }
    assert SparsityParams().si_create_kwargs() == {
        "sparse": True,
        "method": "radius",
        "num_spikes_for_sparsity": 100,
        "ms_before": 1.0,
        "ms_after": 2.5,
        "radius_um": 100.0,
    }
    # A recipe that omits ``sparsity`` validates to the same default, and
    # polarity is not a recipe field (it comes from the sorter).
    without_sparsity = {
        k: v
        for k, v in AnalyzerWaveformParamsSchema().model_dump().items()
        if k != "sparsity"
    }
    assert (
        AnalyzerWaveformParamsSchema.model_validate(without_sparsity).sparsity
        == SparsityParams()
    )
    with pytest.raises(pydantic.ValidationError, match="peak_sign"):
        SparsityParams(peak_sign="pos")
    assert SparsityParams(method="dense").si_create_kwargs() == {
        "sparse": False
    }
    assert (
        SparsityParams(
            method="best_channels", num_channels=4
        ).si_create_kwargs()["num_channels"]
        == 4
    )
    with pytest.raises(pydantic.ValidationError, match="requires num_channels"):
        SparsityParams(method="best_channels")
    with pytest.raises(
        pydantic.ValidationError, match="does not use num_channels"
    ):
        SparsityParams(method="radius", num_channels=3)
    with pytest.raises(pydantic.ValidationError, match="every channel"):
        SparsityParams(method="dense", radius_um=50.0)
    with pytest.raises(pydantic.ValidationError):
        SparsityParams(method="closest_channels")
