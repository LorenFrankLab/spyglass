"""End-to-end UX smoke test: a scientist's first hour on spike sorting v2.

The release gate for the single-session user path. It runs the exact first-hour
sequence -- ``initialize_v2_defaults`` -> sort group -> ``preflight_v2_pipeline``
-> ``run_v2_pipeline`` -> ``summarize_curation`` -> downstream spike-time fetch
-- programmatically against the ``mearec_polymer_smoke`` fixture, and also
executes the user notebook (``notebooks/10_Spike_SortingV2.ipynb``)
cell-by-cell against the same fixture.

The test sets up its OWN clean session (rather than reusing the package-scoped
``populated_sorting``, which is pre-populated) so the ``computed`` -> ``reused``
idempotency assertions are deterministic. The one expensive MountainSort5 sort
is paid once in a module-scoped fixture and shared. It skips cleanly when the
smoke fixture (or, for the notebook-execution test, ``jupytext``) is absent
locally; CI provides them.

The pipeline discovery/orchestration helpers are safe to import at module
scope; table imports stay lazy inside the helpers or inside the tests so
collection does not open a DataJoint connection before the MySQL container is
ready.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spyglass.spikesorting.v2._pipeline_run import _STAGE_STATUSES
from spyglass.spikesorting.v2.pipeline import (
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_sort_groups,
    describe_units,
    plot_sort_group_geometry,
    preflight_v2_pipeline,
    run_v2_pipeline,
)
from tests.spikesorting.v2._ingest_helpers import (
    configure_v2_run_inputs,
    copy_and_insert_nwb,
)

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
# notebooks/ lives at the repo root: tests/spikesorting/v2/ -> parents[3].
_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[3]
    / "notebooks"
    / "10_Spike_SortingV2.ipynb"
)
_TEAM = "ux_smoke_team"
_INTERVAL = "raw data valid times"
_PIPELINE_PRESET = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"

_STABLE_KEYS = (
    "pipeline_preset",
    "recording_id",
    "artifact_detection_id",
    "sorting_id",
    "curation_id",
    "merge_id",
    "n_units",
)
_STATUS_KEYS = (
    "recording_status",
    "artifact_detection_status",
    "sorting_status",
    "curation_status",
)
_STAGES = ("recording", "artifact_detection", "sorting", "curation")
# Keys that legitimately differ between two identical runs.
_VOLATILE_KEYS = {"stage_seconds", *_STATUS_KEYS}
_SORT_GROUP_COLUMNS = [
    "nwb_file_name",
    "sort_group_id",
    "n_electrodes",
    "electrode_ids",
    "electrode_group_names",
    "probe_shanks",
    "brain_regions",
    "bad_channel_count",
    "reference_mode",
    "reference_electrode_id",
]


@pytest.fixture(scope="module")
def ux_session(dj_conn):
    """Ingest the smoke fixture under a UX-isolated session and clean it.

    Distinct ``dest_name`` so this session's rows never collide with the
    package-scoped ``populated_sorting`` or other v2 modules. Cleaned up front
    so the first ``run_v2_pipeline`` genuinely computes every stage.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    nwb_file_name = copy_and_insert_nwb(
        _FIXTURE_PATH, dest_name="mearec_ux_smoke.nwb"
    )
    _clean_session_v2({"nwb_file_name": nwb_file_name})
    return nwb_file_name


@pytest.fixture(scope="module")
def first_hour(ux_session):
    """The first-hour path executed once: preflight (pre-run) + first run.

    Module-scoped so the single expensive MountainSort5 sort is shared across
    every assertion below. ``report_before`` is computed BEFORE the run (so the
    preflight -> run_summary ID round-trip is a genuine prediction on rows that do
    not yet exist); ``run_summary`` is the fresh, all-``computed`` run.
    """
    inputs = configure_v2_run_inputs(
        ux_session,
        _TEAM,
        interval_list_name=_INTERVAL,
        team_description="ux smoke",
    )
    report_before = preflight_v2_pipeline(
        **inputs, pipeline_preset=_PIPELINE_PRESET
    )
    run_summary = run_v2_pipeline(**inputs, pipeline_preset=_PIPELINE_PRESET)
    return {
        "inputs": inputs,
        "report_before": report_before,
        "run_summary": run_summary,
        "nwb_file_name": ux_session,
    }


@pytest.mark.slow
@pytest.mark.integration
def test_ux_smoke_first_hour(first_hour):
    """The whole first-hour path runs end-to-end with the expected results.

    This is the "does a scientist's first hour work?" gate: defaults ->
    sort-group -> preflight -> pipeline -> summary -> fetch, asserting at each
    step.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.curation import CurationV2

    # 1. initialize_v2_defaults is idempotent (callable twice, no error).
    initialize_v2_defaults()
    initialize_v2_defaults()

    # 2. describe_pipeline_presets returns the catalog including the tested preset.
    catalog = describe_pipeline_presets()
    assert _PIPELINE_PRESET in set(catalog["pipeline_preset"])

    # 2b. describe_sort_groups lets users inspect the scientific grouping
    #     context before committing to a sort_group_id.
    sort_groups = describe_sort_groups(first_hour["nwb_file_name"])
    assert list(sort_groups.columns) == _SORT_GROUP_COLUMNS
    assert sort_groups["sort_group_id"].tolist() == sorted(
        sort_groups["sort_group_id"]
    )
    assert len(sort_groups) == 4
    assert set(sort_groups["n_electrodes"]) == {32}
    assert first_hour["inputs"]["sort_group_id"] in set(
        sort_groups["sort_group_id"]
    )
    selected_group = sort_groups.loc[
        sort_groups["sort_group_id"] == first_hour["inputs"]["sort_group_id"]
    ].iloc[0]
    assert selected_group["n_electrodes"] == len(
        selected_group["electrode_ids"]
    )
    assert selected_group["n_electrodes"] > 0
    assert selected_group["electrode_group_names"]
    assert selected_group["brain_regions"]
    assert selected_group["reference_mode"] in {
        "none",
        "global_median",
        "specific",
    }
    assert isinstance(selected_group["bad_channel_count"], (int, np.integer))

    report = first_hour["report_before"]
    run_summary = first_hour["run_summary"]

    # 3. preflight passed and produced the expected_ids.
    assert report.ok is True, report.errors
    assert set(report.expected_ids) == {
        "recording_id",
        "artifact_detection_id",
        "sorting_id",
    }

    # 4. run_summary carries all stable + additive keys with the right types,
    #    and the fresh run computed every stage.
    for key in (*_STABLE_KEYS, *_STATUS_KEYS, "stage_seconds", "warnings"):
        assert key in run_summary, f"missing run_summary key {key!r}"
    assert set(run_summary["stage_seconds"]) == set(_STAGES)
    assert all(isinstance(v, float) for v in run_summary["stage_seconds"].values())
    assert isinstance(run_summary["warnings"], list)
    for key in _STATUS_KEYS:
        assert run_summary[key] in _STAGE_STATUSES
    assert all(run_summary[k] == "computed" for k in _STATUS_KEYS), {
        k: run_summary[k] for k in _STATUS_KEYS
    }

    # 5. summarize_curation accepts the run_summary directly and agrees with the
    #    minimal curation key.
    summary = CurationV2.summarize_curation(run_summary)
    assert summary["merge_id"] == run_summary["merge_id"]
    assert summary["curation_id"] == run_summary["curation_id"]
    assert summary["sorting_id"] == run_summary["sorting_id"]
    assert summary["n_units"] == run_summary["n_units"]
    minimal_key = {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["curation_id"],
    }
    assert CurationV2.summarize_curation(minimal_key) == summary

    # 6. the sort resolves downstream and yields sane per-unit spike arrays.
    spike_times = SpikeSortingOutput().get_spike_times(
        {"merge_id": run_summary["merge_id"]}
    )
    assert isinstance(spike_times, list)
    assert len(spike_times) == run_summary["n_units"]
    for arr in spike_times:
        assert isinstance(arr, np.ndarray)
        assert np.all(np.isfinite(arr))


@pytest.mark.slow
@pytest.mark.integration
def test_ux_smoke_preflight_predicts_ids(first_hour):
    """Preflight's ``expected_ids`` equal the PKs the run_summary returns.

    Ties Phases 2 + 3 together at the user surface: the IDs were predicted
    before the run (the session was cleaned, so they did not yet exist), and
    the run produced exactly those PKs. ``curation_id`` is intentionally not
    predicted (it is assigned by ``insert_curation``, not content-addressed).
    """
    report = first_hour["report_before"]
    run_summary = first_hour["run_summary"]
    for id_key in ("recording_id", "artifact_detection_id", "sorting_id"):
        assert (
            report.expected_ids[id_key]["id"] == run_summary[id_key]
        ), f"preflight mispredicted {id_key}"
        assert (
            report.expected_ids[id_key]["exists"] is False
        ), f"{id_key} unexpectedly existed before the run"


@pytest.mark.slow
@pytest.mark.integration
def test_ux_smoke_idempotent(first_hour):
    """A second identical run is idempotent.

    Equal run_summary modulo ``stage_seconds`` / ``*_status`` (now ``reused``),
    inserting no duplicate selection rows.
    """
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    first_run_summary = first_hour["run_summary"]
    inputs = first_hour["inputs"]

    counts_before = [
        len(RecordingSelection()),
        len(ArtifactDetectionSelection()),
        len(SortingSelection()),
    ]
    second_run_summary = run_v2_pipeline(
        **inputs, pipeline_preset=_PIPELINE_PRESET
    )
    counts_after = [
        len(RecordingSelection()),
        len(ArtifactDetectionSelection()),
        len(SortingSelection()),
    ]
    assert counts_after == counts_before, "re-run inserted duplicate rows"
    assert all(second_run_summary[k] == "reused" for k in _STATUS_KEYS), {
        k: second_run_summary[k] for k in _STATUS_KEYS
    }
    stable_first = {
        k: v for k, v in first_run_summary.items() if k not in _VOLATILE_KEYS
    }
    stable_second = {
        k: v for k, v in second_run_summary.items() if k not in _VOLATILE_KEYS
    }
    assert stable_second == stable_first


@pytest.mark.slow
@pytest.mark.integration
def test_user_notebook_executes(first_hour):
    """The user notebook runs cell-by-cell against the smoke fixture.

    Reads ``notebooks/10_Spike_SortingV2.ipynb`` with ``jupytext`` and executes
    its code cells in order, in-process, so the test's already-bootstrapped DB
    connection and the configured session are visible. The notebook's session
    parameters are injected (overriding the placeholder values) right after its
    ``parameters``-tagged cell; the pipeline rows already exist from
    ``first_hour``, so the notebook's ``run_v2_pipeline`` reuses them (fast).
    Skips cleanly when ``jupytext`` is absent.
    """
    jupytext = pytest.importorskip("jupytext")

    inputs = first_hour["inputs"]
    overrides = {
        "nwb_file_name": inputs["nwb_file_name"],
        "team_name": inputs["team_name"],
        "interval_list_name": inputs["interval_list_name"],
        # The fixture has multiple sort groups; the notebook now requires an
        # explicit sort_group_id when there is more than one (no positional
        # default). Inject the same group first_hour sorted so the reused-row
        # merge_id assertion below holds.
        "sort_group_id": inputs["sort_group_id"],
        "pipeline_preset": _PIPELINE_PRESET,
    }

    notebook = jupytext.read(str(_NOTEBOOK_PATH))
    namespace: dict = {}
    saw_parameters = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)
        if "parameters" in (cell.metadata.get("tags") or []):
            # Replace the user-facing placeholder session with the test one
            # so the remaining cells run against the ingested fixture.
            namespace.update(overrides)
            saw_parameters = True

    assert saw_parameters, "notebook is missing its 'parameters'-tagged cell"
    assert (
        namespace["run_summary"]["merge_id"]
        == first_hour["run_summary"]["merge_id"]
    )
    assert isinstance(namespace["spike_times"], list)


@pytest.mark.slow
@pytest.mark.integration
def test_describe_units_reports_sort_time_quality(first_hour):
    """describe_units returns a per-unit sort-time snapshot for a real sort."""
    from spyglass.spikesorting.v2._pipeline_reporting import _UNIT_COLUMNS

    run_summary = first_hour["run_summary"]
    units = describe_units(run_summary["sorting_id"])

    assert list(units.columns) == _UNIT_COLUMNS
    assert len(units) == run_summary["n_units"]
    assert run_summary["n_units"] > 0  # the smoke sort finds units
    assert (units["n_spikes"] > 0).all()
    assert (units["firing_rate_hz"] > 0).all()
    assert units["peak_electrode_id"].dtype.kind in "iu"
    assert units["brain_region"].map(bool).all()
    # firing_rate uses ONE shared denominator (the sort's observed seconds), so
    # n_spikes / firing_rate is the same for every unit -- the property that
    # makes the rate honest for an artifact-masked sort.
    denom = units["n_spikes"] / units["firing_rate_hz"]
    assert np.allclose(denom, denom.iloc[0])


@pytest.mark.slow
@pytest.mark.integration
def test_describe_pipeline_preset_unpacks_values(first_hour):
    """describe_pipeline_preset resolves a preset to its live parameter values."""
    from spyglass.spikesorting.v2.pipeline import _PIPELINE_PRESETS

    # first_hour installed the default Lookup rows via configure_v2_run_inputs.
    detail = describe_pipeline_preset(_PIPELINE_PRESET)

    assert {"preset", "preprocessing", "artifact_detection", "sorter"} <= set(
        detail["stage"]
    )
    preset_rows = detail[detail["stage"] == "preset"].set_index("key")["value"]
    assert preset_rows["sorter"] == _PIPELINE_PRESETS[_PIPELINE_PRESET].sorter
    assert (
        preset_rows["threshold_units"]
        == _PIPELINE_PRESETS[_PIPELINE_PRESET].threshold_units
    )
    # a known preprocessing knob is unpacked with its dotted key + value
    preproc_rows = detail[detail["stage"] == "preprocessing"]
    preproc_keys = set(preproc_rows["key"])
    assert any(k.startswith("bandpass_filter.") for k in preproc_keys)
    assert preproc_rows["params_schema_version"].notna().all()
    assert "job_kwargs" in detail.columns


@pytest.mark.database
def test_curation_label_options(dj_conn):
    """CurationV2.label_options returns the canonical labels in display order."""
    from spyglass.spikesorting.v2.curation import CurationV2

    assert CurationV2.label_options() == [
        "accept",
        "mua",
        "noise",
        "artifact",
        "reject",
    ]


@pytest.mark.database
def test_describe_units_absent_sorting_id_raises(dj_conn):
    """An unknown sorting_id raises a clear, actionable error (not opaque DJ)."""
    import uuid

    with pytest.raises(ValueError, match="is not in Sorting"):
        describe_units(uuid.uuid4())


@pytest.mark.database
def test_describe_sort_groups_empty(dj_conn):
    """A session with no SortGroupV2 rows returns an empty typed table."""
    sort_groups = describe_sort_groups("not_an_ingested_session_.nwb")
    assert list(sort_groups.columns) == _SORT_GROUP_COLUMNS
    assert sort_groups.empty


@pytest.mark.slow
@pytest.mark.integration
def test_plot_sort_group_geometry_geometry_view(first_hour):
    """The sort-group geometry view renders one contact collection per group."""
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned = plot_sort_group_geometry(first_hour["nwb_file_name"], ax=ax)
    assert returned is ax

    group_collections = [
        collection
        for collection in ax.collections
        if str(collection.get_label()).startswith("sort_group_id ")
    ]
    assert len(group_collections) == 4
    assert sum(len(c.get_offsets()) for c in group_collections) == 128
    assert ax.get_xlabel() == "Probe rel_x (um)"
    assert ax.get_ylabel() == "Probe rel_y (um)"
    assert ax.get_legend() is not None
    plt.close(fig)


@pytest.mark.database
def test_plot_sort_group_geometry_empty(dj_conn):
    """A session with no SortGroupV2 rows produces a clear empty axes."""
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned = plot_sort_group_geometry("not_an_ingested_session_.nwb", ax=ax)
    assert returned is ax
    assert ax.texts
    assert "No SortGroupV2 rows" in ax.texts[0].get_text()
    plt.close(fig)


def test_plot_sort_group_geometry_multi_probe_offset(monkeypatch):
    """Multiple probes are laid out side-by-side (disjoint x) with a warning.

    ``Probe.Electrode`` rel_x/rel_y are per-probe frames, so two probes whose
    contacts share the same raw rel_x would coincide without an offset. The
    smoke fixture is single-probe, so the multi-probe layout is exercised by
    feeding ``plot_sort_group_geometry`` synthetic two-probe geometry rows (no DB).
    """
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from spyglass.spikesorting.v2 import pipeline as pl

    def _fake_rows(nwb_file_name):
        # Two single-column probes, both at rel_x=0 -> identical raw frames.
        rows = []
        for probe_id, sort_group_id in (("probeA", 0), ("probeB", 1)):
            for k in range(4):
                rows.append(
                    {
                        "sort_group_id": sort_group_id,
                        "electrode_id": sort_group_id * 100 + k,
                        "probe_id": probe_id,
                        "bad_channel": "False",
                        "is_reference": False,
                        "coordinate_source": "probe",
                        "plot_x": 0.0,
                        "plot_y": -20.0 * k,
                    }
                )
        return rows

    # ``_sort_group_geometry_rows`` lives in (and is looked up from)
    # ``_pipeline_geometry``; patch it there so the relocated
    # ``plot_sort_group_geometry`` sees the fake rows.
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._pipeline_geometry."
        "_sort_group_geometry_rows",
        _fake_rows,
    )
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match="probes present"):
        pl.plot_sort_group_geometry("any_.nwb", ax=ax)

    group_collections = [
        collection
        for collection in ax.collections
        if str(collection.get_label()).startswith("sort_group_id ")
    ]
    assert len(group_collections) == 2
    xranges = sorted(
        (
            min(point[0] for point in collection.get_offsets()),
            max(point[0] for point in collection.get_offsets()),
        )
        for collection in group_collections
    )
    # Disjoint x-intervals despite identical raw rel_x -> the per-probe offset
    # was applied (probes no longer overlap).
    assert xranges[0][1] < xranges[1][0]
    assert "offset per probe" in ax.get_xlabel()
    plt.close(fig)
