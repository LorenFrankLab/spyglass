"""Clusterless waveform features (``UnitWaveformFeatures.make``) for v2.

- v2 (``CurationV2``) sources run under SI >= 0.101 from a fresh
  ``SortingAnalyzer``, keyed by the true NWB unit_id.
- The legacy guard test needs SI < 0.101; each env skips the other's tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si
from packaging.version import Version

# Lives under v2/ for the v2 conftest's no-op ``mini_insert`` and ``safemode``.
# v2 test helpers are imported lazily: the legacy SI 0.99 job collects this
# file for the guard test, and those imports could fail there.

_SI_IS_LEGACY = Version(si.__version__) < Version("0.101")
_skip_if_legacy = pytest.mark.skipif(
    _SI_IS_LEGACY,
    reason=f"v2 SI-0.104 path needs SI>=0.101 (have {si.__version__})",
)
_skip_unless_legacy = pytest.mark.skipif(
    not _SI_IS_LEGACY,
    reason=f"legacy v0/v1 guard test needs SI<0.101 (have {si.__version__})",
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_PATH = _FIXTURE_DIR / "mearec_polymer_smoke.nwb"
# Same probe as the smoke fixture (safe to co-ingest); its 24 ground-truth
# units yield >= 2 MS5 units per shank.
_POLYMER_60S_PATH = _FIXTURE_DIR / "mearec_polymer_128ch_60s.nwb"
_TEAM = "v2_wave_team"
_AMP_PARAM = "v2_wave_amplitude"


@pytest.fixture(scope="module")
def wave_session(dj_conn, smoke_nwb):
    """Ingest the MEArec polymer smoke fixture (fast clusterless cases)."""
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}
    _reset({"nwb_file_name": nwb_file_name})


@pytest.fixture(scope="module")
def polymer_60s_session(dj_conn):
    """Ingest the 60s polymer fixture (24 ground-truth units).

    The sparse-unit-id test needs several units to merge; the 4 s smoke shank
    yields one.
    """
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _POLYMER_60S_PATH.exists():
        pytest.skip(f"Fixture {_POLYMER_60S_PATH.name} not found.")
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    yield {"nwb_file_name": nwb_file_name}
    _reset({"nwb_file_name": nwb_file_name})


def _amplitude_param():
    """Ensure (and return) a low-overhead single-job amplitude param row."""
    from spyglass.decoding.v1.waveform_features import WaveformFeaturesParams

    WaveformFeaturesParams().insert1(
        {
            "features_param_name": _AMP_PARAM,
            "params": {
                "waveform_extraction_params": {
                    "ms_before": 0.5,
                    "ms_after": 0.5,
                    "max_spikes_per_unit": None,
                    "n_jobs": 1,
                    "chunk_duration": "1000s",
                },
                "waveform_features_params": {
                    "amplitude": {
                        "peak_sign": "neg",
                        "estimate_peak_time": False,
                    }
                },
            },
        },
        skip_duplicates=True,
    )
    return {"features_param_name": _AMP_PARAM}


def _drop_feature_rows():
    """Prompt-free removal of this module's UnitWaveformFeatures rows.

    Otherwise the session cleanup cascades into decoding and hits
    ``cautious_delete``'s confirmation prompt (EOF under pytest).
    """
    from spyglass.decoding.v1.waveform_features import (
        UnitWaveformFeatures,
        UnitWaveformFeaturesSelection,
    )

    sel = {"features_param_name": _AMP_PARAM}
    (UnitWaveformFeatures & sel).delete_quick()
    (UnitWaveformFeaturesSelection & sel).delete_quick()


def _reset(session):
    """Drop feature rows then every v2 row for the session (prompt-free)."""
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    _drop_feature_rows()
    _clean_session_v2(session)


def _sort_group_id(session):
    from spyglass.spikesorting.v2.recording import SortGroupV2

    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=session["nwb_file_name"])
    return int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])


def _run_ms5(session):
    """Clean + run the MountainSort5 pipeline preset; return its run summary."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    _reset(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": _TEAM, "team_description": "v2 waveform tests"},
        skip_duplicates=True,
    )
    sort_group_id = _sort_group_id(session)
    return run_v2_pipeline(
        nwb_file_name=session["nwb_file_name"],
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=_TEAM,
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
        curation_description="v2 waveform ms5",
    )


def _run_clusterless(session, *, tuned):
    """Run a clusterless sort end-to-end; return (sort_pk, recording_id).

    ``tuned=True`` inserts the low-threshold smoke-fixture row that finds
    peaks (one unit); ``tuned=False`` uses the shipped 100 uV default that
    finds zero peaks (zero units).
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    _reset(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": _TEAM, "team_description": "v2 waveform tests"},
        skip_duplicates=True,
    )
    sort_group_id = _sort_group_id(session)

    if tuned:
        from tests.spikesorting.v2._smoke_constants import (
            SMOKE_CLUSTERLESS_PARAM_NAME,
            SMOKE_CLUSTERLESS_PARAMS,
        )

        params_name = SMOKE_CLUSTERLESS_PARAM_NAME
        SorterParameters().insert1(
            {
                "sorter": "clusterless_thresholder",
                "sorter_params_name": params_name,
                "params": dict(SMOKE_CLUSTERLESS_PARAMS),
                "params_schema_version": 4,
                "job_kwargs": None,
            },
            skip_duplicates=True,
        )
    else:
        params_name = "default"  # shipped 100 uV clusterless default

    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": session["nwb_file_name"],
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": _TEAM,
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    art_pk = RecordingArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    RecordingArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )
    Sorting.populate(sort_pk, reserve_jobs=False)
    return sort_pk, rec_pk["recording_id"]


def _curation_merge_id(sort_pk, **curation_kwargs):
    """Insert a CurationV2 row and return its merge_id."""
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    curation_pk = CurationV2.insert_curation(
        sorting_key={"sorting_id": sort_pk["sorting_id"]},
        **curation_kwargs,
    )
    return (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")


def _populate_features(merge_id, param_pk):
    """Insert selection + populate; return the populated table & nwb dict."""
    from spyglass.decoding.v1.waveform_features import (
        UnitWaveformFeatures,
        UnitWaveformFeaturesSelection,
    )

    sel = {"spikesorting_merge_id": merge_id, **param_pk}
    UnitWaveformFeaturesSelection().insert1(sel, skip_duplicates=True)
    UnitWaveformFeatures.populate(sel, reserve_jobs=False)
    tbl = UnitWaveformFeatures & sel
    assert len(tbl) == 1, "make() did not populate exactly one row"
    return tbl, tbl.fetch_nwb()[0]


def _units_dataframe(nwb_row):
    """Return the written units feature DataFrame (NWB object)."""
    return nwb_row["object_id"]


@pytest.mark.usefixtures("dj_conn")
def test_analyzer_waveform_accessor_zero_unit_contract():
    """The empty (``analyzer=None``) accessor only accepts a zero-unit sorting.

    A non-empty sorting is rejected at construction. The zero-unit accessor's
    ``get_waveforms`` raises ``RuntimeError``.
    """
    from spyglass.decoding.v1.waveform_features import (
        _AnalyzerWaveformAccessor,
    )

    class _FakeSorting:
        def __init__(self, n_units):
            self._n = n_units

        def get_num_units(self):
            return self._n

        def get_unit_ids(self):
            return list(range(self._n))

    acc = _AnalyzerWaveformAccessor(sorting=_FakeSorting(0))
    with pytest.raises(RuntimeError, match="zero-unit accessor"):
        acc.get_waveforms(0)

    with pytest.raises(ValueError, match="zero-unit sorting"):
        _AnalyzerWaveformAccessor(sorting=_FakeSorting(3))


@pytest.mark.usefixtures("dj_conn")
@pytest.mark.database
@pytest.mark.db_unit
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("extensions_present", [False, True])
def test_location_accessor_preserves_sparse_units_segments_and_coordinates(
    n_dimensions, extensions_present
):
    """Equal-sized units still keep their distinct coordinates and spike order."""
    from spyglass.decoding.v1.waveform_features import (
        _AnalyzerWaveformAccessor,
    )

    fields = [(axis, "float64") for axis in ("x", "y", "z")[:n_dimensions]]
    locations = {
        0: {
            2: np.array(
                [tuple(row[:n_dimensions]) for row in [(1, 2, 3), (4, 5, 6)]],
                dtype=fields,
            ),
            7: np.array(
                [
                    tuple(row[:n_dimensions])
                    for row in [(40, 50, 60), (70, 80, 90)]
                ],
                dtype=fields,
            ),
        },
        1: {
            2: np.array([tuple((7, 8, 9)[:n_dimensions])], dtype=fields),
            7: np.array([tuple((100, 110, 120)[:n_dimensions])], dtype=fields),
        },
    }
    expected = {
        2: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[:, :n_dimensions],
        7: np.array([[40, 50, 60], [70, 80, 90], [100, 110, 120]])[
            :, :n_dimensions
        ],
    }
    reads, computations = [], []

    class LocationExtension:
        def get_data(self, *, outputs):
            reads.append(outputs)
            return locations

    class Analyzer:
        def __init__(self):
            self.present = (
                {"templates", "spike_locations"}
                if extensions_present
                else set()
            )

        def has_extension(self, name):
            return name in self.present

        def compute(self, name):
            computations.append(name)
            self.present.add(name)

        def get_extension(self, name):
            if name == "waveforms":
                return None  # location access does not read waveform arrays
            assert name == "spike_locations" and name in self.present
            return LocationExtension()

    accessor = _AnalyzerWaveformAccessor(sorting=object(), analyzer=Analyzer())
    # Request out of unit-id order, then repeat a unit. Positional lookups,
    # sorted/flattened coordinate fields, or a second extraction all fail.
    for unit_id in (7, 2, 7):
        actual = accessor.get_spike_locations(unit_id)
        np.testing.assert_array_equal(actual, expected[unit_id])
        assert actual.dtype.kind == "f"
    assert reads == ["by_unit"]
    assert computations == (
        [] if extensions_present else ["templates", "spike_locations"]
    )


@pytest.mark.usefixtures("dj_conn")
def test_v2_dispatch_table_name_resolution():
    """``make`` routes only a ``CurationV2`` source to the v2 path.

    The tier prefix is stripped (``curation_v2`` -> ``CurationV2``) and the
    match is exact, so ``CurationV1`` or ``MetricCurationV2`` do not match.
    """
    from datajoint.utils import to_camel_case

    assert to_camel_case("curation_v2") == "CurationV2"
    for non_v2 in ("curation_v1", "metric_curation_v2"):
        assert to_camel_case(non_v2) != "CurationV2", non_v2


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_v2_clusterless_runs_under_si0104(wave_session):
    """A v2 clusterless ``merge_id`` runs ``make`` under SI 0.104.

    The single unit's amplitude array is ``(n_spikes, n_channels)`` and aligns
    1:1 with the written ``spike_times``.
    """
    param_pk = _amplitude_param()
    sort_pk, _ = _run_clusterless(wave_session, tuned=True)
    try:
        from spyglass.spikesorting.v2.sorting import Sorting

        n_units = int((Sorting & sort_pk).fetch1("n_units"))
        assert (
            n_units == 1
        ), f"tuned clusterless should find one unit; got {n_units}"

        merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)
        _, nwb_row = _populate_features(merge_id, param_pk)
        feature_df = _units_dataframe(nwb_row)

        assert "amplitude" in feature_df.columns, "amplitude column missing"
        assert list(feature_df.index) == [0], (
            "clusterless writes a single unit_id=0; got "
            f"{list(feature_df.index)}"
        )
        unit = feature_df.loc[0]
        amps = np.asarray(unit["amplitude"])
        spike_times = np.asarray(unit["spike_times"])
        assert (
            amps.ndim == 2
        ), f"amplitude must be (n_spikes, n_ch); got {amps.shape}"
        assert amps.shape[0] == spike_times.shape[0], (
            "amplitude rows must align 1:1 with spike_times: "
            f"{amps.shape[0]} amps vs {spike_times.shape[0]} spikes"
        )
        assert amps.shape[0] > 0 and np.isfinite(amps).all(), (
            "expected finite per-spike amplitudes for a populated "
            "clusterless unit"
        )
        # peak_sign="neg": some amplitudes must be negative, or the sign or
        # uV scaling is wrong.
        assert (amps < 0).any(), (
            "neg peak_sign should yield some negative (uV) amplitudes; "
            "all-nonnegative suggests a sign/scale error"
        )

        # With estimate_peak_time=False each amplitude is the recording trace
        # at the spike's frame; recompute it from the recording directly.
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        recording = SpikeSortingOutput.get_recording({"merge_id": merge_id})
        sorting = SpikeSortingOutput.get_sorting({"merge_id": merge_id})
        # The oracle indexes segment 0 only; make() concatenates segments.
        assert recording.get_num_segments() == 1, (
            "amplitude oracle assumes a single-segment recording; got "
            f"{recording.get_num_segments()} segments"
        )
        unit_id = sorting.get_unit_ids()[0]
        frames = np.asarray(sorting.get_unit_spike_train(unit_id))
        traces = recording.get_traces(return_in_uV=True)
        expected_center = traces[frames, :]
        assert expected_center.shape == amps.shape, (
            "recomputed center samples must match the stored amplitude shape: "
            f"{expected_center.shape} vs {amps.shape}"
        )
        # SI zero-pads a window that runs off the recording edge, so edge
        # spikes have amplitude 0. Compare exact values for interior spikes
        # only (_AMP_PARAM: ms_before == ms_after == 0.5 ms).
        n_samples = traces.shape[0]
        fs = recording.get_sampling_frequency()
        margin = int(np.ceil(0.5e-3 * fs))
        interior = (frames >= margin) & (frames < n_samples - margin)
        assert interior.sum() >= 0.8 * frames.size, (
            f"too few interior spikes ({int(interior.sum())}/{frames.size}) "
            "to validate the amplitude oracle"
        )
        np.testing.assert_allclose(
            amps[interior],
            expected_center[interior],
            rtol=1e-3,
            atol=1e-2,
            err_msg=(
                "stored clusterless amplitude does not equal the recording "
                "trace at interior spike frames; make() mis-scales, reorders "
                "channels, or misaligns spikes vs amplitudes"
            ),
        )
        boundary = ~interior
        if boundary.any():
            assert np.all(amps[boundary] == 0.0), (
                "a near-edge spike whose waveform window runs off the "
                "recording was expected to be zero-padded by SI; got a "
                f"nonzero amplitude at frames {frames[boundary].tolist()}"
            )
    finally:
        _reset(wave_session)


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_v2_sparse_unit_ids(polymer_60s_session):
    """Features are keyed by the true unit_id, not a positional index.

    Merging all units gives one survivor with a fresh ``max(id)+1`` id; its
    feature row uses that id and aligns 1:1 with its spike train. One unit
    keeps the ``amplitude`` column rectangular, since
    ``_write_waveform_features_to_nwb`` cannot write ragged multi-unit columns.
    """
    param_pk = _amplitude_param()
    run_summary = _run_ms5(polymer_60s_session)
    try:
        if run_summary["n_units"] < 2:
            pytest.skip(
                f"MS5 produced {run_summary['n_units']} unit(s) on the 60s "
                "polymer fixture; need >=2 to merge into a non-positional "
                "survivor id."
            )

        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
        from spyglass.spikesorting.v2.sorting import Sorting

        sort_pk = {"sorting_id": run_summary["sorting_id"]}
        source_ids = sorted(
            int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id")
        )
        merged_id = max(source_ids) + 1  # CurationV2 apply_merge convention

        # Child of the root curation that run_v2_pipeline inserted.
        merge_id = _curation_merge_id(
            sort_pk,
            parent_curation_id=run_summary["root_curation_id"],
            merge_groups=[source_ids],
            apply_merge=True,
        )

        applied_ids = sorted(
            int(u)
            for u in SpikeSortingOutput()
            .get_sorting({"merge_id": merge_id})
            .get_unit_ids()
        )
        assert applied_ids == [merged_id], (
            f"expected single merged survivor id [{merged_id}]; got "
            f"{applied_ids}"
        )
        assert (
            merged_id != 0
        ), "test is only meaningful if the surviving id is non-positional"

        _, nwb_row = _populate_features(merge_id, param_pk)
        feature_df = _units_dataframe(nwb_row)

        assert sorted(feature_df.index) == [merged_id], (
            "feature row must be keyed by the true (non-positional) unit_id; "
            f"got {sorted(feature_df.index)} vs [{merged_id}]"
        )
        row = feature_df.loc[merged_id]
        amps = np.asarray(row["amplitude"])
        spikes = np.asarray(row["spike_times"])
        assert amps.ndim == 2 and amps.shape[0] == spikes.shape[0], (
            f"unit {merged_id}: amplitude {amps.shape} must align 1:1 with "
            f"{spikes.shape[0]} spikes -- positional mis-keying breaks this"
        )
    finally:
        _reset(polymer_60s_session)


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_zero_unit_v2(wave_session):
    """A zero-unit v2 curation yields an empty-but-valid features row."""
    param_pk = _amplitude_param()
    sort_pk, _ = _run_clusterless(wave_session, tuned=False)
    try:
        from spyglass.spikesorting.v2.sorting import Sorting

        n_units = int((Sorting & sort_pk).fetch1("n_units"))
        assert (
            n_units == 0
        ), f"shipped clusterless default should find zero units; got {n_units}"

        merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)
        tbl, nwb_row = _populate_features(merge_id, param_pk)
        feature_df = _units_dataframe(nwb_row)
        assert len(feature_df) == 0, (
            "zero-unit curation must write an empty units feature table; got "
            f"{len(feature_df)} rows"
        )

        # Two empty sequences, not ``zip(*[])``'s ``()``, so
        # ClusterlessDecodingV1.fetch_spike_data can unpack them.
        spike_times, spike_features = tbl.fetch_data()
        assert spike_times == [] and spike_features == [], (
            "zero-unit fetch_data must yield ([], []); got "
            f"({spike_times!r}, {spike_features!r})"
        )
    finally:
        _reset(wave_session)


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_v2_full_waveform_and_spike_location(
    wave_session,
):
    """v2 supports both ``full_waveform`` and ``spike_location``.

    - ``full_waveform``: ``(n_spikes, n_time*n_channels)`` per unit.
    - ``spike_location``: ``(n_spikes, 2 or 3)`` per unit.

    Both align 1:1 with ``spike_times``.
    """
    from spyglass.decoding.v1.waveform_features import (
        UnitWaveformFeatures,
        UnitWaveformFeaturesSelection,
        WaveformFeaturesParams,
    )

    fw_param = "v2_wave_full_waveform"
    loc_param = "v2_wave_spike_location"
    _extract = {
        "ms_before": 0.5,
        "ms_after": 0.5,
        "max_spikes_per_unit": None,
        "n_jobs": 1,
        "chunk_duration": "1000s",
    }
    full_extract = {**_extract, "ms_before": 0.25, "ms_after": 0.75}
    WaveformFeaturesParams().insert(
        [
            {
                "features_param_name": fw_param,
                "params": {
                    "waveform_extraction_params": full_extract,
                    "waveform_features_params": {"full_waveform": {}},
                },
            },
            {
                "features_param_name": loc_param,
                "params": {
                    "waveform_extraction_params": _extract,
                    "waveform_features_params": {
                        "amplitude": {
                            "peak_sign": "neg",
                            "estimate_peak_time": False,
                        },
                        "spike_location": {},
                    },
                },
            },
        ],
        skip_duplicates=True,
    )

    sort_pk, _ = _run_clusterless(wave_session, tuned=True)
    try:
        merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)

        # full_waveform: supported, shape (n_spikes, n_time*n_channels)
        fw_sel = {
            "spikesorting_merge_id": merge_id,
            "features_param_name": fw_param,
        }
        UnitWaveformFeaturesSelection().insert1(fw_sel, skip_duplicates=True)
        UnitWaveformFeatures.populate(fw_sel, reserve_jobs=False)
        feature_df = _units_dataframe(
            (UnitWaveformFeatures & fw_sel).fetch_nwb()[0]
        )
        assert (
            "full_waveform" in feature_df.columns
        ), "full_waveform column missing"
        row = feature_df.loc[0]
        fw = np.asarray(row["full_waveform"])
        spikes = np.asarray(row["spike_times"])
        assert fw.ndim == 2 and fw.shape[0] == spikes.shape[0], (
            f"full_waveform {fw.shape} must align 1:1 with "
            f"{spikes.shape[0]} spikes"
        )
        assert fw.shape[1] > 1, (
            "full_waveform should flatten n_time*n_channels into axis 1; got "
            f"{fw.shape}"
        )

        # Read the source artifact directly. Every interior row must contain
        # its complete, asymmetric trace window in time-then-channel order;
        # shape alone also passes for zeros, transposed windows, or reordered
        # spike rows. The location recipe above keeps its original window.
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        recording = SpikeSortingOutput.get_recording({"merge_id": merge_id})
        sorting = SpikeSortingOutput.get_sorting({"merge_id": merge_id})
        assert recording.get_num_segments() == 1
        frames = sorting.get_unit_spike_train(0)
        np.testing.assert_array_equal(spikes, recording.get_times()[frames])
        fs = recording.get_sampling_frequency()
        before = int(full_extract["ms_before"] * fs / 1000)
        after = int(full_extract["ms_after"] * fs / 1000)
        assert before != after
        traces = recording.get_traces(return_in_uV=True)
        interior = (frames >= before) & (frames < len(traces) - after)
        assert interior.sum() >= 0.8 * len(frames)
        expected = np.array(
            [
                traces[frame - before : frame + after].reshape(-1)
                for frame in frames[interior]
            ]
        )
        np.testing.assert_allclose(fw[interior], expected, rtol=1e-3, atol=1e-2)
        if (~interior).any():
            np.testing.assert_array_equal(
                fw[~interior], np.zeros_like(fw[~interior])
            )

        # spike_location: supported for v2; writes a (n_spikes, 2 or 3) column
        # per unit aligned 1:1 with spike_times, alongside amplitude.
        loc_sel = {
            "spikesorting_merge_id": merge_id,
            "features_param_name": loc_param,
        }
        UnitWaveformFeaturesSelection().insert1(loc_sel, skip_duplicates=True)
        UnitWaveformFeatures.populate(loc_sel, reserve_jobs=False)
        loc_df = _units_dataframe(
            (UnitWaveformFeatures & loc_sel).fetch_nwb()[0]
        )
        assert "spike_location" in loc_df.columns, "spike_location missing"
        assert "amplitude" in loc_df.columns, "amplitude missing"
        loc_row = loc_df.loc[0]
        loc = np.asarray(loc_row["spike_location"])
        n_spikes = np.asarray(loc_row["spike_times"]).shape[0]
        assert (
            loc.ndim == 2 and loc.shape[0] == n_spikes
        ), f"spike_location {loc.shape} must align 1:1 with {n_spikes} spikes"
        assert loc.shape[1] in (
            2,
            3,
        ), f"spike_location must be 2D or 3D coordinates; got {loc.shape}"
    finally:
        for p in (fw_param, loc_param):
            (UnitWaveformFeatures & {"features_param_name": p}).delete_quick()
            (
                UnitWaveformFeaturesSelection & {"features_param_name": p}
            ).delete_quick()
        _reset(wave_session)


@pytest.mark.db_unit
def test_spike_location_rejects_legacy_waveform_extractor_clearly(dj_conn):
    """``spike_location`` is v2-only under modern SI.

    ``db_unit``: importing ``waveform_features`` activates schemas, so
    ``dj_conn`` must be ready first.
    """
    from spyglass.decoding.v1.waveform_features import (
        WAVEFORM_FEATURE_FUNCTIONS,
    )

    class LegacyWaveformExtractor:
        """WaveformExtractor-shaped object without v2 spike locations."""

        def get_waveforms(self, unit_id):  # pragma: no cover - not called
            raise AssertionError("spike_location must not read waveforms")

    with pytest.raises(NotImplementedError, match="SortingAnalyzer"):
        WAVEFORM_FEATURE_FUNCTIONS["spike_location"](
            LegacyWaveformExtractor(), unit_id=1
        )


@_skip_unless_legacy
@pytest.mark.slow
def test_unit_waveform_features_v0v1_guard_unchanged_under_legacy():
    """Under SI 0.99 the legacy-SI guard is a no-op, so v0/v1 ``make`` runs.

    The v1 populate itself is tested in
    ``tests/spikesorting/v1/test_waveform_features_legacy.py``.
    """
    from spyglass.spikesorting._legacy_runtime import (
        _require_legacy_si_environment,
    )

    # No raise under SI < 0.101 -> v0/v1 make() proceeds for supported rows.
    assert (
        _require_legacy_si_environment("v1 UnitWaveformFeatures.make") is None
    )


@_skip_if_legacy
@pytest.mark.slow
@pytest.mark.integration
def test_fetch_waveform_v2_rejects_max_spikes_per_unit(wave_session):
    """A finite ``max_spikes_per_unit`` fails before the analyzer is built
    (clusterless needs every spike)."""
    from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures

    sort_pk, _ = _run_clusterless(wave_session, tuned=True)
    merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)
    with pytest.raises(ValueError, match="max_spikes_per_unit"):
        UnitWaveformFeatures._fetch_waveform_v2(
            {"merge_id": merge_id}, {"max_spikes_per_unit": 10}
        )


@_skip_if_legacy
@pytest.mark.slow
@pytest.mark.integration
def test_fetch_waveform_v2_is_disk_backed_and_cleaned(wave_session):
    """The clusterless analyzer is disk-backed (``binary_folder``) and
    its TemporaryDirectory is removed once the accessor is dropped."""
    import gc
    from pathlib import Path

    from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures

    sort_pk, _ = _run_clusterless(wave_session, tuned=True)
    merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)

    acc = UnitWaveformFeatures._fetch_waveform_v2({"merge_id": merge_id}, {})
    tmp_path = acc._tmpdir.name
    assert (
        acc._analyzer.format == "binary_folder"
    ), f"clusterless analyzer must be disk-backed; got {acc._analyzer.format}"
    assert Path(tmp_path).exists()
    # Waveforms are readable from disk during the accessor's lifetime.
    uid = acc.sorting.get_unit_ids()[0]
    assert acc.get_waveforms(uid).shape[0] > 0
    # Dropping the accessor removes the scratch folder.
    del acc
    gc.collect()
    assert not Path(
        tmp_path
    ).exists(), (
        "TemporaryDirectory should be removed after the accessor is collected"
    )


@_skip_if_legacy
@pytest.mark.database
def test_clusterless_waveform_buffer_is_disk_backed(dj_conn):
    """Every-spike, every-channel extraction must not be resident in RAM.

    SI extracts to disk only for ``format="binary_folder"``; zarr copies the
    full array into RAM before writing.
    """
    import spikeinterface.full as si

    from spyglass.decoding.v1.waveform_features import (
        _build_clusterless_waveform_accessor,
    )

    recording, sorting = si.generate_ground_truth_recording(
        durations=[5.0], num_units=2, num_channels=4, seed=0
    )
    accessor = _build_clusterless_waveform_accessor(
        recording,
        sorting,
        ms_before=0.3,
        ms_after=0.7,
        job_kwargs={"n_jobs": 1},
    )

    buffer = accessor._waveforms.data["waveforms"]
    assert isinstance(buffer, np.memmap), type(buffer)
    fs = recording.get_sampling_frequency()
    assert accessor.nbefore == int(0.3 * fs / 1000.0)
    n_spikes = sum(
        len(sorting.get_unit_spike_train(u)) for u in sorting.unit_ids
    )
    assert buffer.shape == (
        n_spikes,
        accessor.nbefore + int(0.7 * fs / 1000.0),
        4,
    )
    for unit_id in sorting.unit_ids:
        assert len(accessor.get_waveforms(unit_id)) == len(
            sorting.get_unit_spike_train(unit_id)
        )
