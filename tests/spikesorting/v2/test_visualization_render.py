"""Real matplotlib rendering / export smoke tests for the visualization facade.

Integration-tier: these populate the shared MEArec sort
(``populated_sorting``) and render genuine SpikeInterface figures / write a
real local report folder against its DISPLAY analyzer and saved preprocessed
recording, under the Agg backend. They complement the fast, monkeypatched
routing tests in ``test_visualization_facade.py`` by proving the wrappers
actually drive SI end to end.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from spyglass.spikesorting.v2 import visualization as ssviz  # noqa: E402


def _recording_key(sort_pk):
    # Exercise the real facade resolver (source-aware) end to end rather than
    # re-querying the source part table here.
    return ssviz.recording_key_for_sorting(sort_pk)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_trace_and_probe_map_render(populated_sorting):
    """Trace + probe-map widgets render off the saved preprocessed recording."""
    from spyglass.spikesorting.v2.recording import Recording

    recording_key = _recording_key(populated_sorting)

    traces = ssviz.plot_recording_traces(recording_key, time_range=[0.0, 0.1])
    assert traces is not None
    plt.close("all")

    # SI's probe-map widget defaults to a 2D axis; a 3D probe geometry (the
    # MEArec fixture sets rel_z) needs a 3D axis, supplied through the wrapper's
    # kwargs passthrough -- the thin wrapper forwards ``ax`` straight to SI.
    recording = Recording().get_recording(recording_key)
    probe_ndim = recording.get_probegroup().probes[0].ndim
    if probe_ndim == 3:
        ax = plt.figure().add_subplot(projection="3d")
        probe = ssviz.plot_recording_probe_map(recording_key, ax=ax)
    else:
        probe = ssviz.plot_recording_probe_map(recording_key)
    assert probe is not None
    plt.close("all")


def _root_curation(populated_sorting):
    """Resolve (creating if needed) the smoke sort's root curation ref."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef

    root = CurationV2 & {**populated_sorting, "parent_curation_id": -1}
    if not root:
        CurationV2.insert_curation(sorting_key=populated_sorting, labels={})
        root = CurationV2 & {**populated_sorting, "parent_curation_id": -1}
    return CurationRef.from_key(root.fetch1("KEY"))


@pytest.mark.slow
@pytest.mark.integration
def test_unit_summary_renders_with_compute_missing(populated_sorting):
    """A per-unit summary renders once its display ``unit_locations`` is computed."""
    from spyglass.spikesorting.v2.sorting import Sorting

    unit_ids = list(Sorting().get_analyzer(populated_sorting).unit_ids)
    if not unit_ids:
        pytest.skip("zero-unit smoke sort; no unit summary to render")

    fig = ssviz.plot_unit_summary(
        _root_curation(populated_sorting), unit_ids[0], compute_missing=True
    )
    assert fig is not None
    plt.close("all")


@pytest.mark.slow
@pytest.mark.integration
def test_unit_locations_renders_with_compute_missing(populated_sorting):
    """``plot_unit_locations`` renders end to end against the display analyzer.

    The opt-in computes the display ``unit_locations`` extension if the
    package-shared analyzer does not already carry it, then renders. The
    read-only-by-default error path is pinned deterministically by the
    monkeypatched unit test (a shared on-disk analyzer can't reliably present a
    *fresh* extension here, so this integration check stays on the render path).
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    if not list(Sorting().get_analyzer(populated_sorting).unit_ids):
        pytest.skip("zero-unit smoke sort")

    fig = ssviz.plot_unit_locations(
        _root_curation(populated_sorting), compute_missing=True
    )
    assert fig is not None
    plt.close("all")


@pytest.mark.slow
@pytest.mark.integration
def test_local_report_export_writes_folder(populated_sorting, tmp_path):
    """``export_si_report(compute_missing=True)`` writes a local report folder."""
    from spyglass.spikesorting.v2.sorting import Sorting

    if not list(Sorting().get_analyzer(populated_sorting).unit_ids):
        pytest.skip("zero-unit smoke sort; nothing to report")

    output_folder = tmp_path / "si_report"
    ssviz.export_si_report(
        _root_curation(populated_sorting), output_folder, compute_missing=True
    )
    assert output_folder.is_dir()
    assert (output_folder / "spyglass_provenance.json").exists()
    # SI writes a per-unit figure folder and a unit list; assert the folder is
    # non-empty rather than pinning SI's exact filenames.
    assert any(output_folder.iterdir())
    plt.close("all")


@pytest.mark.slow
@pytest.mark.integration
def test_export_to_phy_writes_folder_off_display_analyzer(
    populated_sorting, tmp_path
):
    """``export_to_phy`` writes a Phy folder from a curation working copy.

    PC features default off, so SI does not compute the whitened-metric-only
    ``principal_components`` extension; the export drives SI end to end,
    produces ``params.py``, records provenance, and leaves the published
    display analyzer untouched by SI's export-time extensions.
    """
    import json

    from spyglass.spikesorting.v2.sorting import Sorting

    if not list(Sorting().get_analyzer(populated_sorting).unit_ids):
        pytest.skip("zero-unit smoke sort; nothing to export")

    root = _root_curation(populated_sorting)
    published = Sorting().get_analyzer(populated_sorting)
    before = set(published.get_saved_extension_names())
    output_folder = tmp_path / "phy"
    ssviz.export_to_phy(root, output_folder)
    assert (output_folder / "params.py").exists()
    provenance = json.loads(
        (output_folder / "spyglass_provenance.json").read_text()
    )
    assert provenance["curation_uuid"] == str(root.curation_uuid)
    assert provenance["unit_ids"] == [
        int(u) for u in Sorting().get_analyzer(populated_sorting).unit_ids
    ]
    # SI computes template_similarity / spike_amplitudes for the export; they
    # land on the owned working copy, never on the published cache.
    after = set(
        Sorting().get_analyzer(populated_sorting).get_saved_extension_names()
    )
    assert after == before
    # compute_pc_features defaults to False, so SI writes no PC-feature arrays
    # (and computes no principal_components onto the display analyzer).
    assert not (output_folder / "pc_features.npy").exists()
    plt.close("all")
