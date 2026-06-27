"""Hermetic (no-DB) tests for the matcher parameter schema."""

from __future__ import annotations

import pytest


def test_unitmatch_schema_defaults_round_trip():
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    dumped = UnitMatchParamsSchema().model_dump()
    assert dumped["match_threshold"] == 0.5
    assert dumped["tracked_unit_threshold"] == 0.5
    assert dumped["max_strict_nodes"] == 2000
    assert dumped["schema_version"] == 1


def test_unitmatch_schema_forbids_unknown_field():
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError):
        UnitMatchParamsSchema(typo_field=1)


@pytest.mark.parametrize(
    "field, value",
    [
        ("match_threshold", 1.5),
        ("tracked_unit_threshold", -0.1),
        ("max_strict_nodes", 0),
    ],
)
def test_unitmatch_schema_rejects_out_of_range(field, value):
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError):
        UnitMatchParamsSchema(**{field: value})


def test_unitmatch_schema_has_bundle_params():
    """The waveform-bundle params are identity-bearing fields on the schema, so
    they enter the named, content-addressed params blob (and therefore the
    matcher_params_name -> unitmatch_id identity) rather than being silent
    extract_unitmatch_bundle function defaults. Defaults match the prior
    literals so the shipped default row is unchanged."""
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    dumped = UnitMatchParamsSchema().model_dump()
    assert dumped["ms_before"] == 1.5
    assert dumped["ms_after"] == 1.5
    assert dumped["max_spikes_per_unit"] == 100
    assert dumped["seed"] == 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("ms_before", 0.0),
        ("ms_after", -1.0),
        ("max_spikes_per_unit", 0),
        ("seed", -1),
    ],
)
def test_unitmatch_schema_rejects_bad_bundle_params(field, value):
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema

    with pytest.raises(ValidationError):
        UnitMatchParamsSchema(**{field: value})


def test_bundle_compute_kwargs_seed_is_authoritative():
    """The bundle ``seed`` field is authoritative: a stray ``random_seed`` in
    the SI job kwargs (an ambient dj.config seed, or a leaked value) is stripped
    and IGNORED, never allowed to override the identity-bearing seed -- else the
    stored identity would disagree with the seed actually used."""
    from spyglass.spikesorting.v2._unitmatch_backend import (
        _bundle_compute_kwargs,
    )

    seed, kwargs = _bundle_compute_kwargs(
        5, {"random_seed": 9, "n_jobs": 2}
    )
    assert seed == 5
    assert "random_seed" not in kwargs
    assert kwargs == {"n_jobs": 2}
    # None job_kwargs is fine and leaves the seed untouched.
    assert _bundle_compute_kwargs(3, None) == (3, {})


@pytest.mark.usefixtures("dj_conn")
def test_bundle_seed_override_rejected_at_insert():
    """A random_seed in the job_kwargs blob is a second, non-identity seed the
    bundle extractor must ignore, so it is rejected at insert rather than
    silently dropped (which would mislead the user). A job_kwargs blob without
    random_seed inserts fine."""
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    params = UnitMatchParamsSchema().model_dump()
    with pytest.raises(ValueError, match="random_seed"):
        MatcherParameters().insert1(
            {
                "matcher_params_name": "um_bad_seed",
                "matcher": "unitmatch",
                "params": params,
                "params_schema_version": 1,
                "job_kwargs": {"random_seed": 7, "n_jobs": 2},
            }
        )
    # The same row without random_seed in job_kwargs is accepted.
    MatcherParameters().insert1(
        {
            "matcher_params_name": "um_ok_jobkwargs",
            "matcher": "unitmatch",
            "params": params,
            "params_schema_version": 1,
            "job_kwargs": {"n_jobs": 2},
        },
        allow_duplicate_params=True,
    )
    assert MatcherParameters & {"matcher_params_name": "um_ok_jobkwargs"}


@pytest.mark.usefixtures("dj_conn")
def test_named_bundle_params_stored_distinctly():
    """Two named MatcherParameters rows that differ only in a bundle field are
    stored as distinct, content-addressed identities (the bundle window is part
    of the named params blob, not a silent function default)."""
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.unit_matching import MatcherParameters

    MatcherParameters.insert_default()
    wide = UnitMatchParamsSchema(ms_before=2.0).model_dump()
    MatcherParameters().insert1(
        {
            "matcher_params_name": "unitmatch_wide",
            "matcher": "unitmatch",
            "params": wide,
            "params_schema_version": 1,
            "job_kwargs": None,
        }
    )
    default_params = (
        MatcherParameters & {"matcher_params_name": "unitmatch_default"}
    ).fetch1("params")
    wide_params = (
        MatcherParameters & {"matcher_params_name": "unitmatch_wide"}
    ).fetch1("params")
    assert default_params["ms_before"] == 1.5
    assert wide_params["ms_before"] == 2.0


@pytest.mark.usefixtures("dj_conn")
def test_bundle_params_reach_extract(monkeypatch):
    """The named bundle params actually reach extract_unitmatch_bundle in the
    matcher compute path -- they are no longer silent function defaults."""
    from spyglass.spikesorting.v2 import (
        _unitmatch_backend,
        matcher_protocol,
        unit_matching,
    )
    from spyglass.spikesorting.v2._params.matcher import UnitMatchParamsSchema
    from spyglass.spikesorting.v2.curation import CurationV2

    captured = {}

    def fake_extract(session_dir, recording, sorting, **kwargs):
        captured.update(kwargs)

    class _DummySorting:
        def select_units(self, ids):
            return self

    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", fake_extract
    )
    monkeypatch.setattr(
        CurationV2, "get_recording", staticmethod(lambda key: object())
    )
    monkeypatch.setattr(
        CurationV2, "get_sorting", staticmethod(lambda key: _DummySorting())
    )
    monkeypatch.setattr(
        matcher_protocol,
        "get_matcher",
        lambda name: type("_M", (), {"match": lambda self, si, p: []})(),
    )

    params = UnitMatchParamsSchema(ms_before=2.0, seed=4).model_dump()
    member_plan = [
        {
            "member_index": 0,
            "sorting_id": "s0",
            "curation_id": 0,
            "matchable_unit_ids": [1, 2],
            "recording_date": "2026-01-01T00:00:00+00:00",
        },
        {
            "member_index": 1,
            "sorting_id": "s1",
            "curation_id": 0,
            "matchable_unit_ids": [3, 4],
            "recording_date": "2026-01-02T00:00:00+00:00",
        },
    ]
    unit_matching.UnitMatch._extract_and_match(
        member_plan, "unitmatch", params, {}
    )
    assert captured["ms_before"] == 2.0
    assert captured["seed"] == 4
    assert captured["max_spikes_per_unit"] == 100
