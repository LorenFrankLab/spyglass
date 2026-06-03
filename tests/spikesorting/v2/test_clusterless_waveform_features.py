"""v2 (SI 0.104) clusterless waveform-feature extraction.

``UnitWaveformFeatures.make`` is the clusterless-decoding input. It must run
for v2 (``CurationV2``) sources under SpikeInterface 0.104 -- without the
legacy-SI guard and without the removed ``si.extract_waveforms`` -- producing
per-spike amplitudes from a freshly built ``SortingAnalyzer``, keyed by the
true NWB unit_id. The legacy v0/v1 path stays guarded and unchanged under
SI 0.99.

Tests dispatch on the installed SI version: the v2 tests need SI >= 0.101 and
skip under the legacy env; the legacy guard test needs SI < 0.101 and skips
under the modern env (so this phase merges independently of the legacy job).

All tests are ``slow``: each runs a full MEArec sort plus an analyzer build
against the Docker MySQL container.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si
from packaging.version import Version

# The v2 conftest (tests/spikesorting/v2/conftest.py) provides the autouse
# no-op ``mini_insert`` (skips the repo-wide minirec Box download/ingestion)
# and the boolean ``safemode`` override this module's super_delete cleanup
# relies on -- so this file lives here rather than under tests/decoding/.
#
# NOTE: imports of the v2 test infrastructure (``copy_and_insert_nwb``,
# ``_clean_session_v2``) are deliberately LAZY (inside the helpers/fixtures
# below). This module also runs in the legacy SI 0.99 job for the guard test
# (collected as a single explicit path, NOT the whole v2 dir); importing the
# heavy v2 test modules at collection time could fail under the legacy
# environment, whereas the legacy test itself touches none of them.

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
# 60s polymer fixture: same probe geometry as the smoke fixture (identical
# probe_json in the fixtures manifest), so co-ingesting both in one module is
# conflict-free, and its 24 ground-truth units yield >=2 MS5 units per shank.
_POLYMER_60S_PATH = _FIXTURE_DIR / "mearec_polymer_128ch_60s.nwb"
_TEAM = "v2_wave_team"
_AMP_PARAM = "v2_wave_amplitude"


@pytest.fixture(scope="module")
def wave_session(dj_conn):
    """Ingest the MEArec polymer smoke fixture (fast clusterless cases)."""
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture {_FIXTURE_PATH.name} not found.")
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}


@pytest.fixture(scope="module")
def polymer_60s_session(dj_conn):
    """Ingest the 60s polymer fixture (24 ground-truth units).

    Used by the sparse-unit-id test: the 4-second smoke shank yields only a
    single MS5 unit, too few to form a merge-applied (sparse) id set, whereas a
    60s shank reliably yields several. Shares the smoke fixture's probe, so
    co-ingestion does not collide on the probe/electrode tables.
    """
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    if not _POLYMER_60S_PATH.exists():
        pytest.skip(f"Fixture {_POLYMER_60S_PATH.name} not found.")
    nwb_file_name = copy_and_insert_nwb(_POLYMER_60S_PATH)
    yield {"nwb_file_name": nwb_file_name}


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

    The downstream ``UnitWaveformFeatures`` row makes ``_clean_session_v2``'s
    merge-master ``super_delete`` cascade into the decoding schema, which trips
    ``cautious_delete``'s confirmation prompt (EOF under pytest). Dropping the
    feature + selection rows first (``delete_quick`` -> no cascade, no prompt)
    removes that downstream dependency so the session cleanup runs cleanly.
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
    from tests.spikesorting.v2.test_single_session_pipeline import (
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
    """Clean + run the MountainSort5 preset; return its run manifest."""
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
        preset="franklab_tetrode_mountainsort5",
        description="v2 waveform ms5",
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
        ArtifactDetection,
        ArtifactSelection,
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
            "preproc_params_name": "default_franklab",
            "team_name": _TEAM,
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)

    art_pk = ArtifactSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)

    sort_pk = SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": params_name,
            "artifact_id": art_pk["artifact_id"],
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


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_v2_clusterless_runs_under_si0104(wave_session):
    """A v2 clusterless ``merge_id`` runs ``make`` under SI 0.104.

    No ``_require_legacy_si_environment`` raise, no ``extract_waveforms``
    error: features come from a freshly built ``SortingAnalyzer``. The single
    clusterless unit's amplitude array is ``(n_spikes, n_channels)`` and aligns
    1:1 with the written ``spike_times``.
    """
    param_pk = _amplitude_param()
    sort_pk, _ = _run_clusterless(wave_session, tuned=True)

    from spyglass.spikesorting.v2.sorting import Sorting

    n_units = int((Sorting & sort_pk).fetch1("n_units"))
    assert n_units == 1, (
        f"tuned clusterless should find one unit; got {n_units}"
    )

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
    assert amps.ndim == 2, (
        f"amplitude must be (n_spikes, n_ch); got {amps.shape}"
    )
    assert amps.shape[0] == spike_times.shape[0], (
        "amplitude rows must align 1:1 with spike_times: "
        f"{amps.shape[0]} amps vs {spike_times.shape[0]} spikes"
    )
    assert amps.shape[0] > 0 and np.isfinite(amps).all(), (
        "expected finite per-spike amplitudes for a populated clusterless unit"
    )
    _reset(wave_session)


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_v2_sparse_unit_ids(polymer_60s_session):
    """Features are keyed by the true unit_id, not a positional index.

    A merge-applied v2 curation assigns the survivor a fresh ``max(id)+1`` id,
    so the surviving unit_id is non-positional (``!= 0``). The written feature
    row must be keyed by that true id, and its amplitude array must align 1:1
    with that unit's own spike train. Merging ALL units into a single survivor
    keeps the written ``amplitude`` column rectangular (one unit), isolating the
    true-id keying from the pre-existing ragged-multi-unit-column write
    limitation in ``_write_waveform_features_to_nwb`` (which never bit
    clusterless decoding -- the thresholder always emits exactly one unit).
    """
    param_pk = _amplitude_param()
    manifest = _run_ms5(polymer_60s_session)
    if manifest["n_units"] < 2:
        pytest.skip(
            f"MS5 produced {manifest['n_units']} unit(s) on the 60s polymer "
            "fixture; need >=2 to merge into a non-positional survivor id."
        )

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = {"sorting_id": manifest["sorting_id"]}
    source_ids = sorted(
        int(u) for u in (Sorting.Unit & sort_pk).fetch("unit_id")
    )
    merged_id = max(source_ids) + 1  # CurationV2 apply_merge convention

    # Merge ALL source units into one. ``run_v2_pipeline`` already inserted the
    # root (parent_curation_id=-1) curation; this merge-applied curation is a
    # child of it. The single survivor carries the fresh non-positional id.
    merge_id = _curation_merge_id(
        sort_pk,
        parent_curation_id=manifest["curation_id"],
        merge_groups=[source_ids],
        apply_merge=True,
    )

    applied_ids = sorted(
        int(u) for u in SpikeSortingOutput().get_sorting(
            {"merge_id": merge_id}
        ).get_unit_ids()
    )
    assert applied_ids == [merged_id], (
        f"expected single merged survivor id [{merged_id}]; got {applied_ids}"
    )
    assert merged_id != 0, (
        "test is only meaningful if the surviving id is non-positional"
    )

    _, nwb_row = _populate_features(merge_id, param_pk)
    feature_df = _units_dataframe(nwb_row)

    assert sorted(feature_df.index) == [merged_id], (
        "feature row must be keyed by the true (non-positional) unit_id; got "
        f"{sorted(feature_df.index)} vs [{merged_id}]"
    )
    row = feature_df.loc[merged_id]
    amps = np.asarray(row["amplitude"])
    spikes = np.asarray(row["spike_times"])
    assert amps.ndim == 2 and amps.shape[0] == spikes.shape[0], (
        f"unit {merged_id}: amplitude {amps.shape} must align 1:1 with "
        f"{spikes.shape[0]} spikes -- positional mis-keying would break this"
    )
    _reset(polymer_60s_session)


@_skip_if_legacy
@pytest.mark.slow
def test_unit_waveform_features_zero_unit_v2(wave_session):
    """A zero-unit v2 curation yields an empty-but-valid features row."""
    param_pk = _amplitude_param()
    sort_pk, _ = _run_clusterless(wave_session, tuned=False)

    from spyglass.spikesorting.v2.sorting import Sorting

    n_units = int((Sorting & sort_pk).fetch1("n_units"))
    assert n_units == 0, (
        f"shipped clusterless default should find zero units; got {n_units}"
    )

    merge_id = _curation_merge_id(sort_pk, parent_curation_id=-1)
    tbl, nwb_row = _populate_features(merge_id, param_pk)
    feature_df = _units_dataframe(nwb_row)
    assert len(feature_df) == 0, (
        "zero-unit curation must write an empty units feature table; got "
        f"{len(feature_df)} rows"
    )

    # ``fetch_data`` must return two empty sequences (not the bare ``()`` that
    # ``zip(*[])`` produces), so the public consumer
    # ``ClusterlessDecodingV1.fetch_spike_data`` -- which unpacks the result
    # into ``spike_times, spike_waveform_features`` -- does not raise on an
    # all-zero-unit feature set.
    spike_times, spike_features = tbl.fetch_data()
    assert spike_times == [] and spike_features == [], (
        "zero-unit fetch_data must yield ([], []); got "
        f"({spike_times!r}, {spike_features!r})"
    )
    _reset(wave_session)


@_skip_unless_legacy
@pytest.mark.slow
def test_unit_waveform_features_v0v1_unchanged_under_legacy():
    """Under SI 0.99 the legacy-SI guard is a no-op, so v0/v1 ``make`` runs.

    This phase scoped the guard to the v0/v1 branches only; under the legacy
    env that guard must NOT raise (otherwise the v0/v1 extraction path would
    be broken). The full v0/v1 ``UnitWaveformFeatures.make`` is exercised by
    the existing clusterless-decoding tests in this same legacy job; here we
    pin the guard-version contract make() depends on.
    """
    from spyglass.spikesorting._legacy_runtime import (
        _require_legacy_si_environment,
    )

    # No raise under SI < 0.101 -> v0/v1 make() proceeds exactly as before.
    assert (
        _require_legacy_si_environment("v1 UnitWaveformFeatures.make") is None
    )
