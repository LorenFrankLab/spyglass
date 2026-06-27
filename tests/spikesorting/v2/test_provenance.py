"""Producer provenance on computed v2 rows: effective seed + library versions.

Provenance is SECONDARY, never identity: the seed actually used and the
producing library/sorter versions are recorded on the computed rows, but a
fixed input still mints the same ``sorting_id`` / ``unitmatch_id``.

The ``resolve_effective_seed`` tests here are hermetic (pure helper, no DB);
the populate-level provenance + parity tests use the DB fixtures.
"""

from __future__ import annotations

import logging

import datajoint as dj
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
