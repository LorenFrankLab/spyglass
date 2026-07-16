"""Producer provenance on computed v2 rows: effective seed + library versions.

Provenance is SECONDARY, never identity: the seed actually used and the
producing library/sorter versions are recorded on the computed rows, but a
fixed input still mints the same ``sorting_id`` / ``unitmatch_id``.

The ``resolve_effective_seed`` tests here are hermetic (pure helper, no DB);
the populate-level provenance + parity tests use the DB fixtures.
"""

from __future__ import annotations

import importlib.metadata
import logging

import datajoint as dj
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2.utils import (
    _resolved_job_kwargs,
    _warn_ambient_seed_once,
    resolve_effective_seed,
)


def test_resolve_effective_seed_defaults_to_zero(restore_custom_config):
    """No per-row blob and no ambient seed resolves to the default 0."""
    dj.config["custom"].pop("spikesorting_v2_job_kwargs", None)
    assert resolve_effective_seed(None) == 0
    assert resolve_effective_seed({"n_jobs": 2}) == 0


def test_resolve_effective_seed_per_row_overrides_ambient(
    restore_custom_config, caplog
):
    """A per-row seed wins over an ambient ``dj.config`` seed and never warns."""
    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"random_seed": 7}
    _warn_ambient_seed_once.cache_clear()
    with caplog.at_level(logging.WARNING, logger="spyglass"):
        effective = resolve_effective_seed({"random_seed": 3})
    assert effective == 3
    ambient_warnings = [
        r
        for r in caplog.records
        if "random_seed" in r.getMessage().lower()
        and "ambient" in r.getMessage().lower()
    ]
    assert ambient_warnings == []


def test_resolve_effective_seed_ambient_seed_warns_once(
    restore_custom_config, caplog
):
    """An ambient-only seed is used (stored) but emits the ambient warning once."""
    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"random_seed": 7}
    _warn_ambient_seed_once.cache_clear()
    with caplog.at_level(logging.WARNING, logger="spyglass"):
        first = resolve_effective_seed(None)
        second = resolve_effective_seed({"n_jobs": 2})
    assert first == 7
    assert second == 7
    ambient_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "random_seed" in r.getMessage().lower()
        and "ambient" in r.getMessage().lower()
    ]
    assert len(ambient_warnings) == 1


def test_resolve_effective_seed_rejects_ambient_when_flagged(
    restore_custom_config,
):
    """With ``reject_ambient_seed=True`` (the sort-identity boundary), an
    ambient-only seed RAISES instead of warning: a Sorting must never be created
    with a seed that is absent from its identity (``sorter_params_name``), or a
    later ambient-seed change would silently reuse it. A per-row seed (which IS
    part of identity) is still accepted, and no ambient seed is a no-op."""
    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"random_seed": 7}
    _warn_ambient_seed_once.cache_clear()
    with pytest.raises(ValueError, match="SorterParameters"):
        resolve_effective_seed(None, reject_ambient_seed=True)
    with pytest.raises(ValueError, match="SorterParameters"):
        resolve_effective_seed({"n_jobs": 2}, reject_ambient_seed=True)
    assert (
        resolve_effective_seed({"random_seed": 3}, reject_ambient_seed=True)
        == 3
    )
    dj.config["custom"].pop("spikesorting_v2_job_kwargs", None)
    assert resolve_effective_seed(None, reject_ambient_seed=True) == 0


def test_resolve_effective_seed_matches_resolved_job_kwargs(
    restore_custom_config,
):
    """The stored value equals what the seed sites consume: the helper bottoms
    out on the same ``_resolved_job_kwargs`` merge the dispatch reads, so the
    stored seed cannot drift from the used seed."""
    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"random_seed": 7}
    for blob in (None, {"random_seed": 3}, {"n_jobs": 2}, {"random_seed": 0}):
        assert resolve_effective_seed(blob) == int(
            _resolved_job_kwargs(blob).get("random_seed", 0)
        )


def test_resolve_effective_seed_rejects_non_integer(restore_custom_config):
    """A non-integer OR negative random_seed is rejected BEFORE compute, so the
    stored seed can never diverge from the raw value the seed sites consume
    (silent int()-coercion) nor defer to a later SI/NumPy RNG failure. Negative
    mirrors the UnitMatch bundle-seed schema's seed >= 0."""
    import numpy as np

    dj.config["custom"].pop("spikesorting_v2_job_kwargs", None)
    for bad in ("7", 7.9, 7.0, True, -1, -5):
        with pytest.raises(
            ValueError, match="random_seed must be a non-negative integer"
        ):
            resolve_effective_seed({"random_seed": bad})
    # Clean non-negative Python and numpy integers are accepted (seed-equivalent).
    assert resolve_effective_seed({"random_seed": 0}) == 0
    assert resolve_effective_seed({"random_seed": 7}) == 7
    assert resolve_effective_seed({"random_seed": np.int64(5)}) == 5


# --- sorter -> distribution mapping (hermetic) ----------------------------


def test_sorter_distribution_version_maps_known_and_none():
    """The sorter package version is the installed distribution version for
    sorters that ship as their own package, and ``None`` for SI-internal
    sorters and the in-process clusterless thresholder (whose producing version
    is ``spikeinterface_version``, recorded separately) -- never a wrong guess.
    """
    from spyglass.spikesorting.v2._sorting_dispatch import (
        sorter_distribution_version,
    )

    assert sorter_distribution_version(
        "mountainsort5"
    ) == importlib.metadata.version("mountainsort5")
    # SI-internal sorters + in-process clusterless have no separate package.
    assert sorter_distribution_version("spykingcircus2") is None
    assert sorter_distribution_version("tridesclous2") is None
    assert sorter_distribution_version("clusterless_thresholder") is None


# --- populate-level provenance (DB) ---------------------------------------


def _setup_clusterless_smoke_sort(session_name):
    """Ingest the smoke fixture and insert an (unpopulated) clusterless sort.

    Returns the ``Sorting`` selection key with recording + artifact already
    populated, so the caller controls the seed-sensitive ``Sorting.populate``.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetection,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAM_NAME,
        SMOKE_CLUSTERLESS_PARAMS,
    )

    from .conftest import _DOWNSTREAM_FIXTURE_PATH

    if not _DOWNSTREAM_FIXTURE_PATH.exists():
        pytest.skip("Generated MEArec smoke fixture not found.")

    nwb_file_name = copy_and_insert_nwb(
        _DOWNSTREAM_FIXTURE_PATH, dest_name=session_name
    )
    session_key = {"nwb_file_name": nwb_file_name}

    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_prov_team", "team_description": "v2 provenance"},
        skip_duplicates=True,
    )
    SorterParameters().insert1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 4,
            "job_kwargs": None,
        },
        skip_duplicates=True,
    )

    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_prov_team",
        }
    )
    Recording.populate(rec_pk, reserve_jobs=False)
    art_pk = ArtifactDetectionSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "artifact_detection_params_name": "none",
        }
    )
    ArtifactDetection.populate(art_pk, reserve_jobs=False)
    return SortingSelection.insert_selection(
        {
            "recording_id": rec_pk["recording_id"],
            "sorter": "clusterless_thresholder",
            "sorter_params_name": SMOKE_CLUSTERLESS_PARAM_NAME,
            "artifact_detection_id": art_pk["artifact_detection_id"],
        }
    )


def test_sorting_row_records_effective_seed_and_versions(populated_sorting):
    """A populated mountainsort5 ``Sorting`` row records the producing library
    versions and the effective seed (0 with no per-row / ambient seed). The
    sorter version is the installed ``mountainsort5`` distribution version."""
    from spyglass.spikesorting.v2.sorting import Sorting

    row = (Sorting & populated_sorting).fetch1()
    assert row["spikeinterface_version"] == si.__version__
    assert row["sorter_version"] == importlib.metadata.version("mountainsort5")
    assert row["effective_random_seed"] == 0


def test_ambient_seed_rejected_then_records_clusterless_provenance(
    restore_custom_config,
):
    """A seed-dependent sort REJECTS an ambient-only ``random_seed`` at
    populate: the clusterless smoke sort whitens (which samples random chunks
    and is seed-dependent), so an ambient seed changes the output WITHOUT
    changing ``sorting_id`` and must instead live in the ``SorterParameters``
    row. (The unit-level reject is pinned by
    ``test_resolve_effective_seed_rejects_ambient_when_flagged``; this covers the
    integration path via ``Sorting.populate``.) With the ambient seed cleared,
    the same sort populates and records clusterless provenance:
    ``sorter_version`` is None (in-process, no distribution version),
    ``spikeinterface_version`` is recorded, and the effective seed defaults to
    0."""
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = _setup_clusterless_smoke_sort("mearec_provenance_ambient.nwb")

    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"random_seed": 7}
    with pytest.raises(ValueError, match="random_seed to live in the"):
        Sorting.populate(sort_pk, reserve_jobs=False)
    # The rejected sort wrote no row (the seed check runs first in make_compute).
    assert len(Sorting & sort_pk) == 0

    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {}
    Sorting.populate(sort_pk, reserve_jobs=False)
    row = (Sorting & sort_pk).fetch1()
    assert row["sorter_version"] is None
    assert row["spikeinterface_version"] == si.__version__
    assert row["effective_random_seed"] == 0


# --- never-identity parity ------------------------------------------------


def test_sorting_id_unchanged_after_provenance_columns():
    """The sorting_id derivation excludes the provenance columns, so a fixed
    sorting selection identity still mints the pre-change deterministic id."""
    from spyglass.spikesorting.v2._selection_identity import (
        deterministic_id,
        sorting_identity_payload,
    )

    payload = sorting_identity_payload(
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        recording_id="11111111-1111-5111-8111-111111111111",
        artifact_detection_id="22222222-2222-5222-8222-222222222222",
    )
    assert (
        str(deterministic_id("sorting", payload))
        == "7c2c8e91-5ee7-53c4-977e-b34af0b5a9c8"
    )


def test_unitmatch_id_unchanged_after_provenance_columns():
    """The unitmatch_id derivation excludes the provenance columns -- and the
    new bundle params enter identity via matcher_params_name, NOT the
    deterministic payload -- so a fixed selection identity still mints the
    pre-change id."""
    from spyglass.spikesorting.v2._selection_identity import deterministic_id

    identity = {
        "session_group_owner": "team_a",
        "session_group_name": "day1",
        "matcher_params_name": "unitmatch_default",
        "curation_set_hash": "0" * 64,
    }
    assert (
        str(deterministic_id("unitmatch", identity))
        == "9da8859b-d341-5bae-8fbb-90602d7a2a39"
    )


@pytest.mark.usefixtures("dj_conn")
def test_provenance_columns_are_secondary_not_identity():
    """Every producer-provenance column is a SECONDARY attribute -- never in a
    primary key -- so it can never fork a deterministic id."""
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_matching import UnitMatch

    sorting_pk = set(Sorting.primary_key)
    for col in (
        "effective_random_seed",
        "spikeinterface_version",
        "sorter_version",
    ):
        assert col in Sorting.heading.secondary_attributes
        assert col not in sorting_pk

    unitmatch_pk = set(UnitMatch.primary_key)
    for col in (
        "spikeinterface_version",
        "matcher_backend",
        "matcher_backend_version",
    ):
        assert col in UnitMatch.heading.secondary_attributes
        assert col not in unitmatch_pk
