"""Routing tests for the ``v2.visualization`` facade (db_unit, monkeypatched).

These exercise the key -> recording/analyzer/metric routing without populating a
sort: SI widget/exporter functions, the raw and curation analyzer resolvers /
``Recording.get_recording`` / ``CurationEvaluation.get_metrics`` /
``CurationEvaluation.get_suggested_merge_groups`` and the curation resolver are
monkeypatched with fakes, and the assertions pin which analyzer (display vs
whitened) and which extensions each helper touches. The ``db_unit`` mark is for
the schema-class imports (Docker MySQL only); nothing here populates. Real
matplotlib rendering against the MEArec fixture lives in the ``slow`` tests.
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from spyglass.spikesorting.v2 import visualization as ssviz  # noqa: E402
from spyglass.spikesorting.v2._visualization import (  # noqa: E402
    MissingDisplayExtensionError,
)


class _FakeAnalyzer:
    """SortingAnalyzer stand-in: tracks present extensions, mutable."""

    def __init__(self, present=()):
        self.present = set(present)

    def has_extension(self, name):
        return name in self.present


_FACADE_FUNCTIONS = (
    "available_visualizations",
    "plot_recording_traces",
    "plot_recording_probe_map",
    "plot_sorting_summary",
    "plot_unit_summary",
    "plot_waveforms",
    "plot_spikes_on_traces",
    "plot_unit_locations",
    "plot_metrics",
    "plot_si_quality_metrics",
    "plot_si_template_metrics",
    "plot_suggested_merges",
    "export_si_report",
    "export_to_phy",
)


# --------------------------------------------------------------------------
# Surface / delegate structure (no DB connection, no SI calls)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_visualization_facade_exports_expected_helpers():
    """The facade exposes every documented helper; the catalog describes them."""
    for name in _FACADE_FUNCTIONS:
        assert hasattr(ssviz, name), f"facade missing {name}"
        assert callable(getattr(ssviz, name))
    # The sorting -> recording key convenience is also part of the surface.
    assert callable(ssviz.recording_key_for_sorting)
    table = ssviz.available_visualizations()
    assert set(table.columns) == {
        "name",
        "key_type",
        "implementation",
        "backend_default",
        "compute_missing",
        "description",
    }
    # Every plotting/export helper is catalogued with a non-empty description.
    catalogued = set(table["name"])
    assert {
        n for n in _FACADE_FUNCTIONS if n != "available_visualizations"
    } <= (catalogued)
    assert all(d for d in table["description"])


@pytest.mark.db_unit
def test_table_delegates_call_facade_if_present(dj_conn):
    """Optional table-class methods are thin one-line delegates to the facade.

    Each delegate, where present, calls the matching ``visualization`` function
    and contains no ``spikeinterface`` import or duplicate key/analyzer routing
    of its own.
    """
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    delegates = {
        Recording.plot_traces: "plot_recording_traces",
        Recording.plot_probe_map: "plot_recording_probe_map",
        Sorting.plot_summary: "plot_sorting_summary",
        Sorting.plot_unit_summary: "plot_unit_summary",
        Sorting.plot_waveforms: "plot_waveforms",
        Sorting.plot_spikes_on_traces: "plot_spikes_on_traces",
        Sorting.plot_unit_locations: "plot_unit_locations",
        Sorting.export_si_report: "export_si_report",
        Sorting.export_to_phy: "export_to_phy",
        CurationEvaluation.plot_metrics: "plot_metrics",
        CurationEvaluation.plot_si_quality_metrics: "plot_si_quality_metrics",
        CurationEvaluation.plot_si_template_metrics: "plot_si_template_metrics",
        CurationEvaluation.plot_suggested_merges: "plot_suggested_merges",
    }
    for method, facade_name in delegates.items():
        source = inspect.getsource(method)
        assert f"visualization.{facade_name}(" in source, (
            f"{method.__qualname__} must delegate to "
            f"visualization.{facade_name}"
        )
        assert (
            "import spikeinterface" not in source
        ), f"{method.__qualname__} must not import spikeinterface"
        # No second routing implementation: delegates resolve nothing themselves.
        assert "get_analyzer" not in source
        assert "get_recording" not in source


# --------------------------------------------------------------------------
# Recording widgets -> saved preprocessed recording, no analyzer
# --------------------------------------------------------------------------


@pytest.mark.db_unit
def test_recording_plot_traces_calls_si_widget(dj_conn, monkeypatch):
    """Facade loads the saved recording and calls SI ``plot_traces``."""
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.recording import Recording

    sentinel = object()
    captured = {}
    monkeypatch.setattr(Recording, "get_recording", lambda self, key: sentinel)

    def _fake(*, recording, backend, **kwargs):
        captured.update(recording=recording, backend=backend, kwargs=kwargs)
        return "TRACES"

    monkeypatch.setattr(sw, "plot_traces", _fake)
    out = ssviz.plot_recording_traces({"recording_id": "r"}, time_range=[0, 1])
    assert out == "TRACES"
    assert captured["recording"] is sentinel
    assert captured["backend"] == "matplotlib"
    assert captured["kwargs"] == {"time_range": [0, 1]}


@pytest.mark.db_unit
def test_recording_plot_traces_exposes_only_saved_recording(dj_conn):
    """No raw toggle is advertised: v2 plots the saved preprocessed recording."""
    assert (
        "raw" not in inspect.signature(ssviz.plot_recording_traces).parameters
    )


@pytest.mark.db_unit
def test_recording_plot_probe_map_calls_si_widget(dj_conn, monkeypatch):
    """Probe map loads the saved recording and calls SI; no analyzer is loaded."""
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.sorting import Sorting

    sentinel = object()
    captured = {}
    monkeypatch.setattr(Recording, "get_recording", lambda self, key: sentinel)

    def _no_analyzer(self, key, waveform_params_name=None):
        raise AssertionError("probe map must not load an analyzer")

    monkeypatch.setattr(Sorting, "get_analyzer", _no_analyzer)

    def _fake(*, recording, backend, **kwargs):
        captured.update(recording=recording, backend=backend)
        return "PROBE"

    monkeypatch.setattr(sw, "plot_probe_map", _fake)
    assert ssviz.plot_recording_probe_map({"recording_id": "r"}) == "PROBE"
    assert captured["recording"] is sentinel
    assert captured["backend"] == "matplotlib"


@pytest.mark.db_unit
def test_recording_key_for_sorting_resolves_single_recording(
    dj_conn, monkeypatch
):
    """A single-recording sort resolves straight to its ``recording_id`` key."""
    from types import SimpleNamespace

    from spyglass.spikesorting.v2.sorting import SortingSelection

    monkeypatch.setattr(
        SortingSelection,
        "resolve_source",
        classmethod(
            lambda cls, key: SimpleNamespace(
                kind="recording", key={"recording_id": "r1"}
            )
        ),
    )
    assert ssviz.recording_key_for_sorting({"sorting_id": "s"}) == {
        "recording_id": "r1"
    }


@pytest.mark.db_unit
def test_recording_key_for_sorting_rejects_concat_source(dj_conn, monkeypatch):
    """A concat-backed sort has no single recording key -> clear error."""
    from types import SimpleNamespace

    from spyglass.spikesorting.v2.sorting import SortingSelection

    monkeypatch.setattr(
        SortingSelection,
        "resolve_source",
        classmethod(
            lambda cls, key: SimpleNamespace(
                kind="concatenated_recording",
                key={"concat_recording_id": "c1"},
            )
        ),
    )
    with pytest.raises(ValueError, match="single-recording"):
        ssviz.recording_key_for_sorting({"sorting_id": "s"})


# --------------------------------------------------------------------------
# Sorting widgets -> display analyzer + extension policy
# --------------------------------------------------------------------------


_REQUEST = {
    "curation_ref": "CURATION",
    "waveform_recipe": "display_recipe",
    "role": "display",
}


def _patch_curation_analyzer(monkeypatch, fake, *, recorder=None):
    """Route every curation-keyed helper at one analyzer fake.

    ``_curation_request`` is stubbed (no DB), and the published resolver
    records each ``(request, extra_extensions)`` call so a test can assert
    which extensions were requested and that only the display role is used.
    ``open_curation_analyzer`` (the export working copy) yields the same fake
    and records too.
    """
    from contextlib import contextmanager

    from spyglass.spikesorting.v2 import _curation_analyzer as resolver

    monkeypatch.setattr(
        ssviz, "_curation_request", lambda curation, *, caller: dict(_REQUEST)
    )
    monkeypatch.setattr(
        ssviz, "_evaluation_curation", lambda key: {"curation_id": 0}
    )

    def _resolve(
        curation_ref, waveform_recipe, role="display", *, extra_extensions=None
    ):
        if recorder is not None:
            recorder.append(
                {
                    "role": role,
                    "extra_extensions": dict(extra_extensions or {}),
                }
            )
        fake.present.update(extra_extensions or {})
        return fake

    @contextmanager
    def _open(
        curation_ref, waveform_recipe, role="display", *, extra_extensions=None
    ):
        if recorder is not None:
            recorder.append(
                {
                    "role": role,
                    "extra_extensions": dict(extra_extensions or {}),
                    "working_copy": True,
                }
            )
        fake.present.update(extra_extensions or {})
        yield fake

    monkeypatch.setattr(resolver, "_resolve_curation_analyzer", _resolve)
    monkeypatch.setattr(resolver, "open_curation_analyzer", _open)
    monkeypatch.setattr(ssviz, "_write_export_provenance", lambda *a, **k: None)


def _requested_extensions(recorder) -> set:
    return {name for call in recorder for name in call["extra_extensions"]}


def _forbid_add_extensions(monkeypatch):
    from spyglass.spikesorting.v2.sorting import Sorting

    def _add(self, key, extensions, **kwargs):
        raise AssertionError(
            f"add_extensions must not be called (got {extensions})"
        )

    monkeypatch.setattr(Sorting, "add_extensions", _add)


@pytest.mark.db_unit
def test_plot_sorting_summary_uses_curation_display_analyzer(
    dj_conn, monkeypatch
):
    """Summary resolves the curation's display analyzer and calls SI."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer(
        [
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
            "template_similarity",
        ]
    )
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    _forbid_add_extensions(monkeypatch)
    captured = {}

    def _fake(analyzer, *, backend, **kwargs):
        captured.update(analyzer=analyzer, backend=backend)
        return "SUMMARY"

    monkeypatch.setattr(sw, "plot_sorting_summary", _fake)
    # SortingSummaryWidget has no matplotlib backend; pass one explicitly.
    assert (
        ssviz.plot_sorting_summary(
            {"sorting_id": "s", "curation_id": 0}, backend="spikeinterface_gui"
        )
        == "SUMMARY"
    )
    assert captured["analyzer"] is fake
    assert captured["backend"] == "spikeinterface_gui"
    # The display role, never the whitened metric recipe; nothing computed.
    assert calls == [{"role": "display", "extra_extensions": {}}]


@pytest.mark.db_unit
def test_unit_level_helpers_reject_bare_sorting_keys(dj_conn, monkeypatch):
    """A bare sorting key is not a curation: refused with the root-curation hint."""
    for call in (
        lambda: ssviz.plot_sorting_summary(
            {"sorting_id": "s"}, backend="spikeinterface_gui"
        ),
        lambda: ssviz.plot_unit_summary({"sorting_id": "s"}, 0),
        lambda: ssviz.plot_waveforms({"sorting_id": "s"}),
        lambda: ssviz.plot_unit_locations({"sorting_id": "s"}),
        lambda: ssviz.export_to_phy({"sorting_id": "s"}, "/tmp/phy"),
        lambda: ssviz.export_si_report({"sorting_id": "s"}, "/tmp/rep"),
    ):
        with pytest.raises(ValueError, match="root_curation"):
            call()


@pytest.mark.db_unit
def test_plot_sorting_summary_requires_explicit_backend(dj_conn):
    """Without a backend it raises (SI's summary widget has no matplotlib path)."""
    with pytest.raises(ValueError, match="no local matplotlib backend"):
        ssviz.plot_sorting_summary({"sorting_id": "s", "curation_id": 0})


@pytest.mark.db_unit
def test_plot_sorting_summary_missing_extensions_read_only_by_default(
    dj_conn, monkeypatch
):
    """Missing display-safe extensions raise by default; no compute happens."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer([])  # only base extensions present
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    _forbid_add_extensions(monkeypatch)

    def _must_not_call(*a, **k):
        raise AssertionError("SI widget must not be called on the error path")

    monkeypatch.setattr(sw, "plot_sorting_summary", _must_not_call)
    with pytest.raises(
        MissingDisplayExtensionError, match="unit_locations"
    ) as exc:
        ssviz.plot_sorting_summary(
            {"sorting_id": "s", "curation_id": 0}, backend="spikeinterface_gui"
        )
    # The absent extensions are exposed structurally, not only in the message.
    assert "unit_locations" in exc.value.missing
    assert _requested_extensions(calls) == set()


@pytest.mark.db_unit
def test_plot_sorting_summary_compute_missing_opt_in(dj_conn, monkeypatch):
    """``compute_missing=True`` requests only the display-safe missing set."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer(["correlograms"])  # missing 3 of the 4 required
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    _forbid_add_extensions(monkeypatch)
    monkeypatch.setattr(
        sw, "plot_sorting_summary", lambda analyzer, **k: "SUMMARY"
    )
    assert (
        ssviz.plot_sorting_summary(
            {"sorting_id": "s", "curation_id": 0},
            compute_missing=True,
            backend="spikeinterface_gui",
        )
        == "SUMMARY"
    )
    assert _requested_extensions(calls) == {
        "spike_amplitudes",
        "unit_locations",
        "template_similarity",
    }


@pytest.mark.db_unit
def test_plot_unit_summary_uses_curation_analyzer(dj_conn, monkeypatch):
    """Unit summary uses the curation analyzer and forwards unit_id + kwargs."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer(["unit_locations"])
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    captured = {}

    def _fake(analyzer, unit_id, *, backend, **kwargs):
        captured.update(
            analyzer=analyzer, unit_id=unit_id, backend=backend, kwargs=kwargs
        )
        return "UNIT"

    monkeypatch.setattr(sw, "plot_unit_summary", _fake)
    out = ssviz.plot_unit_summary(
        {"sorting_id": "s", "curation_id": 0}, 7, sparsity=None
    )
    assert out == "UNIT"
    assert captured["analyzer"] is fake
    assert captured["unit_id"] == 7
    assert {c["role"] for c in calls} == {"display"}


@pytest.mark.db_unit
def test_plot_waveforms_wraps_unit_waveforms(dj_conn, monkeypatch):
    """Spyglass ``plot_waveforms`` wraps SI ``plot_unit_waveforms``."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer([])  # base waveforms/templates suffice; no ensure
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    _forbid_add_extensions(monkeypatch)
    captured = {}

    def _fake(analyzer, *, unit_ids, backend, **kwargs):
        captured.update(analyzer=analyzer, unit_ids=unit_ids)
        return "WF"

    monkeypatch.setattr(sw, "plot_unit_waveforms", _fake)
    assert not hasattr(sw, "plot_waveforms")  # no such SI symbol
    out = ssviz.plot_waveforms(
        {"sorting_id": "s", "curation_id": 0}, unit_ids=[1, 2]
    )
    assert out == "WF"
    assert captured["analyzer"] is fake
    assert captured["unit_ids"] == [1, 2]
    assert _requested_extensions(calls) == set()


@pytest.mark.db_unit
def test_plot_unit_locations_requires_extension_or_opt_in(dj_conn, monkeypatch):
    """``plot_unit_locations`` needs the ``unit_locations`` ext or the opt-in."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer([])
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    monkeypatch.setattr(sw, "plot_unit_locations", lambda analyzer, **k: "LOC")

    # Default: raises naming the missing extension; no compute.
    _forbid_add_extensions(monkeypatch)
    with pytest.raises(MissingDisplayExtensionError, match="unit_locations"):
        ssviz.plot_unit_locations({"sorting_id": "s", "curation_id": 0})
    assert _requested_extensions(calls) == set()

    # Opt-in: requests exactly unit_locations.
    assert (
        ssviz.plot_unit_locations(
            {"sorting_id": "s", "curation_id": 0}, compute_missing=True
        )
        == "LOC"
    )
    assert _requested_extensions(calls) == {"unit_locations"}


# --------------------------------------------------------------------------
# Metric / merge widgets -> routed Spyglass metrics / persisted merges
# --------------------------------------------------------------------------


@pytest.mark.db_unit
def test_curation_evaluation_plot_metrics_uses_spyglass_metrics_by_default(
    dj_conn, monkeypatch
):
    """``plot_metrics`` plots ``get_metrics()``; SI ``plot_quality_metrics`` unused."""
    import pandas as pd
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    metrics = pd.DataFrame(
        {"snr": [3.0, 4.0], "trough_half_width": [0.2, 0.25]},
        index=pd.Index([0, 1], name="unit_id"),
    )
    monkeypatch.setattr(
        CurationEvaluation, "get_metrics", classmethod(lambda cls, key: metrics)
    )

    def _must_not_call(*a, **k):
        raise AssertionError(
            "plot_metrics must not call SI plot_quality_metrics"
        )

    monkeypatch.setattr(sw, "plot_quality_metrics", _must_not_call)
    fig = ssviz.plot_metrics({"sorting_id": "s", "curation_id": 0})
    titles = [ax.get_title() for ax in fig.axes]
    assert "snr" in titles and "trough_half_width" in titles
    matplotlib.pyplot.close(fig)


@pytest.mark.db_unit
def test_plot_metrics_rejects_non_matplotlib_backend(dj_conn):
    """The Spyglass-owned metric plot has no SI backend; non-default rejected."""
    with pytest.raises(ValueError, match="matplotlib"):
        ssviz.plot_metrics({"sorting_id": "s"}, backend="sortingview")


@pytest.mark.db_unit
def test_curation_evaluation_plot_si_quality_metrics_uses_display_analyzer(
    dj_conn, monkeypatch
):
    """The SI-native quality view uses the curation display analyzer."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer(["quality_metrics"])
    requests = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=requests)
    _forbid_add_extensions(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        sw,
        "plot_quality_metrics",
        lambda analyzer, **k: captured.update(analyzer=analyzer) or "QM",
    )
    assert ssviz.plot_si_quality_metrics({"curation_id": 0}) == "QM"
    assert captured["analyzer"] is fake
    assert requests == [{"role": "display", "extra_extensions": {}}]


@pytest.mark.db_unit
def test_curation_evaluation_plot_si_template_metrics_uses_display_analyzer(
    dj_conn, monkeypatch
):
    """The SI-native template-metric view uses the display analyzer only."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer(["template_metrics"])
    requests = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=requests)
    _forbid_add_extensions(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        sw,
        "plot_template_metrics",
        lambda analyzer, **k: captured.update(analyzer=analyzer) or "TM",
    )
    assert ssviz.plot_si_template_metrics({"curation_id": 0}) == "TM"
    assert captured["analyzer"] is fake
    assert requests[0]["role"] == "display"


@pytest.mark.db_unit
def test_si_metric_widgets_require_explicit_compute_for_missing_extensions(
    dj_conn, monkeypatch
):
    """SI metric widgets do not auto-compute; the error points to plot_metrics."""
    import spikeinterface.widgets as sw

    fake = _FakeAnalyzer([])  # quality_metrics absent
    _patch_curation_analyzer(monkeypatch, fake)
    _forbid_add_extensions(monkeypatch)
    monkeypatch.setattr(
        sw,
        "plot_quality_metrics",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not render on the error path")
        ),
    )
    with pytest.raises(MissingDisplayExtensionError) as excinfo:
        ssviz.plot_si_quality_metrics({"curation_id": 0})
    assert "plot_metrics" in str(excinfo.value)


@pytest.mark.db_unit
def test_plot_suggested_merges_uses_persisted_merge_groups(
    dj_conn, monkeypatch
):
    """Wrapper passes persisted merge groups; never recomputes candidates."""
    import spikeinterface.curation as sic
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    # Persisted groups: one real merge (>=2) plus a singleton to be dropped.
    monkeypatch.setattr(
        CurationEvaluation,
        "get_suggested_merge_groups",
        classmethod(lambda cls, key: [[1, 2], [3]]),
    )

    def _must_not_recompute(*a, **k):
        raise AssertionError("plot path must not recompute merge candidates")

    monkeypatch.setattr(sic, "compute_merge_unit_groups", _must_not_recompute)
    fake = _FakeAnalyzer(["spike_amplitudes", "correlograms"])
    _patch_curation_analyzer(monkeypatch, fake)
    _forbid_add_extensions(monkeypatch)
    captured = {}

    def _fake(analyzer, *, potential_merges, backend, **kwargs):
        captured.update(analyzer=analyzer, potential_merges=potential_merges)
        return "MERGES"

    monkeypatch.setattr(sw, "plot_potential_merges", _fake)
    assert ssviz.plot_suggested_merges({"curation_id": 0}) == "MERGES"
    assert captured["analyzer"] is fake
    # Singleton group dropped; only the real >=2 merge is passed.
    assert captured["potential_merges"] == [[1, 2]]


@pytest.mark.db_unit
def test_plot_suggested_merges_errors_when_no_persisted_suggestions(
    dj_conn, monkeypatch
):
    """No persisted >=2 suggestion -> clear error, still no recompute."""
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    monkeypatch.setattr(
        CurationEvaluation,
        "get_suggested_merge_groups",
        classmethod(lambda cls, key: [[3]]),
    )
    with pytest.raises(ValueError, match="never recomputes"):
        ssviz.plot_suggested_merges({"curation_id": 0})


# --------------------------------------------------------------------------
# Exports -> display analyzer
# --------------------------------------------------------------------------


@pytest.mark.db_unit
def test_export_report_uses_working_copy(dj_conn, monkeypatch):
    """``export_si_report`` wraps SI ``export_report`` over a working copy.

    ``compute_missing=False`` requests no extensions; ``True`` requests only
    display-safe report extensions; SI is always called with its own
    ``force_computation=False`` so it never mutates the cache itself.
    """
    import spikeinterface.exporters as sie

    captured = {}
    monkeypatch.setattr(
        sie,
        "export_report",
        lambda analyzer, output_folder, **k: captured.update(
            analyzer=analyzer, folder=output_folder, kwargs=k
        ),
    )

    fake_present = _FakeAnalyzer(["unit_locations"])
    calls = []
    _patch_curation_analyzer(monkeypatch, fake_present, recorder=calls)
    _forbid_add_extensions(monkeypatch)
    ssviz.export_si_report(
        {"sorting_id": "s", "curation_id": 0}, "/tmp/report_a"
    )
    assert captured["analyzer"] is fake_present
    assert captured["kwargs"]["force_computation"] is False
    assert any(c.get("working_copy") for c in calls)
    assert _requested_extensions(calls) == {"unit_locations"}

    # compute_missing=True: requests the display-safe report extensions.
    from spyglass.spikesorting.v2._visualization import (
        REPORT_DISPLAY_EXTENSIONS,
    )

    calls.clear()
    fake_missing = _FakeAnalyzer(["unit_locations"])
    _patch_curation_analyzer(monkeypatch, fake_missing, recorder=calls)
    ssviz.export_si_report(
        {"sorting_id": "s", "curation_id": 0},
        "/tmp/report_b",
        compute_missing=True,
    )
    assert _requested_extensions(calls) == set(REPORT_DISPLAY_EXTENSIONS)
    assert "principal_components" not in _requested_extensions(calls)


@pytest.mark.db_unit
def test_export_report_read_only_requires_unit_locations(dj_conn, monkeypatch):
    """``compute_missing=False`` refuses to run when ``unit_locations`` is absent.

    SI's ``export_report`` computes ``unit_locations`` unconditionally if missing,
    so the read-only path must raise rather than compute -- and must not call
    SI at all.
    """
    import spikeinterface.exporters as sie

    fake = _FakeAnalyzer([])  # unit_locations absent
    _patch_curation_analyzer(monkeypatch, fake)
    _forbid_add_extensions(monkeypatch)

    def _must_not_export(*a, **k):
        raise AssertionError(
            "export_report must not run on the read-only error"
        )

    monkeypatch.setattr(sie, "export_report", _must_not_export)
    with pytest.raises(MissingDisplayExtensionError, match="unit_locations"):
        ssviz.export_si_report(
            {"sorting_id": "s", "curation_id": 0}, "/tmp/report_ro"
        )


@pytest.mark.db_unit
def test_export_to_phy_uses_working_copy(dj_conn, monkeypatch):
    """``export_to_phy`` wraps SI ``export_to_phy`` over a curation working copy.

    PC features default OFF so SI never computes the whitened-metric-only
    ``principal_components`` extension onto the unwhitened display copy; an
    explicit ``compute_pc_features=True`` still passes through.
    """
    import spikeinterface.exporters as sie

    fake = _FakeAnalyzer([])
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    captured = {}
    monkeypatch.setattr(
        sie,
        "export_to_phy",
        lambda analyzer, output_folder, **k: captured.update(
            analyzer=analyzer, folder=output_folder, kwargs=k
        ),
    )
    ssviz.export_to_phy({"sorting_id": "s", "curation_id": 0}, "/tmp/phy")
    assert captured["analyzer"] is fake
    # Display role only, on an owned working copy.
    assert calls == [
        {"role": "display", "extra_extensions": {}, "working_copy": True}
    ]
    # PC features off by default (no principal_components on the display path),
    # and raw SI display-analyzer metric TSVs off by default (the routed
    # CurationEvaluation.get_metrics() stays the single source of official metrics).
    assert captured["kwargs"]["compute_pc_features"] is False
    assert captured["kwargs"]["add_quality_metrics"] is False
    assert captured["kwargs"]["add_template_metrics"] is False

    # Explicit opt-ins are honored.
    ssviz.export_to_phy(
        {"sorting_id": "s", "curation_id": 0},
        "/tmp/phy",
        compute_pc_features=True,
        add_quality_metrics=True,
    )
    assert captured["kwargs"]["compute_pc_features"] is True
    assert captured["kwargs"]["add_quality_metrics"] is True


# --------------------------------------------------------------------------
# Cross-cutting invariants
# --------------------------------------------------------------------------


@pytest.mark.db_unit
def test_no_widget_uses_metric_analyzer_by_default(dj_conn, monkeypatch):
    """Every analyzer-backed helper resolves a display-role analyzer."""
    import spikeinterface.exporters as sie
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    fake = _FakeAnalyzer(
        [
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
            "template_similarity",
            "quality_metrics",
            "template_metrics",
        ]
    )
    calls = []
    _patch_curation_analyzer(monkeypatch, fake, recorder=calls)
    _forbid_add_extensions(monkeypatch)
    monkeypatch.setattr(
        CurationEvaluation,
        "get_suggested_merge_groups",
        classmethod(lambda cls, key: [[1, 2]]),
    )
    for name in (
        "plot_sorting_summary",
        "plot_unit_summary",
        "plot_spikes_on_traces",
        "plot_unit_locations",
        "plot_unit_waveforms",
        "plot_quality_metrics",
        "plot_template_metrics",
        "plot_potential_merges",
    ):
        monkeypatch.setattr(sw, name, lambda *a, **k: None)
    monkeypatch.setattr(sie, "export_report", lambda *a, **k: None)
    monkeypatch.setattr(sie, "export_to_phy", lambda *a, **k: None)

    ckey = {"curation_id": 0}
    curation = {"sorting_id": "s", "curation_id": 0}
    ssviz.plot_sorting_summary(curation, backend="spikeinterface_gui")
    ssviz.plot_unit_summary(curation, 0)
    ssviz.plot_waveforms(curation)
    ssviz.plot_spikes_on_traces(curation)
    ssviz.plot_unit_locations(curation)
    ssviz.plot_si_quality_metrics(ckey)
    ssviz.plot_si_template_metrics(ckey)
    ssviz.plot_suggested_merges(ckey)
    ssviz.export_si_report(curation, "/tmp/r")
    ssviz.export_to_phy(curation, "/tmp/p")

    assert calls, "expected curation analyzer loads"
    assert {call["role"] for call in calls} == {"display"}


@pytest.mark.db_unit
def test_backend_policy_default_and_opt_in(dj_conn, monkeypatch):
    """Default backend is matplotlib; an explicit backend passes through."""
    import spikeinterface.widgets as sw

    from spyglass.spikesorting.v2.recording import Recording

    # matplotlib is the default for the helpers whose SI widget supports it.
    for name in (
        "plot_recording_traces",
        "plot_recording_probe_map",
        "plot_unit_summary",
        "plot_waveforms",
        "plot_spikes_on_traces",
        "plot_unit_locations",
        "plot_si_quality_metrics",
        "plot_si_template_metrics",
    ):
        sig = inspect.signature(getattr(ssviz, name))
        assert sig.parameters["backend"].default == "matplotlib"
    # The two widgets SI offers no matplotlib backend for default honestly:
    # sorting summary requires an explicit backend, potential-merges is ipywidgets.
    assert (
        inspect.signature(ssviz.plot_sorting_summary)
        .parameters["backend"]
        .default
        is None
    )
    assert (
        inspect.signature(ssviz.plot_suggested_merges)
        .parameters["backend"]
        .default
        == "ipywidgets"
    )

    # An explicit backend reaches SI unchanged (opt-in only).
    monkeypatch.setattr(Recording, "get_recording", lambda self, key: object())
    captured = {}
    monkeypatch.setattr(
        sw,
        "plot_traces",
        lambda *, recording, backend, **k: captured.update(backend=backend),
    )
    ssviz.plot_recording_traces({"recording_id": "r"}, backend="sortingview")
    assert captured["backend"] == "sortingview"


@pytest.mark.db_unit
def test_schema_modules_do_not_import_visualization_eagerly(dj_conn):
    """No populate/make path can publish: the facade is imported only lazily.

    The schema modules expose visualization only through method-local imports
    (the delegate one-liners), so importing them -- and therefore the populate
    path -- never pulls the SI-widget facade in at module load.
    """
    import spyglass.spikesorting.v2.metric_curation as mc
    import spyglass.spikesorting.v2.recording as rec
    import spyglass.spikesorting.v2.sorting as srt

    for module in (rec, srt, mc):
        assert not hasattr(
            module, "visualization"
        ), f"{module.__name__} imports visualization at module level"
