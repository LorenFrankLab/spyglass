"""Validated parameter schema for the cross-session matcher parameter table.

One schema per registered matcher backend (dispatched by the matcher registry
in ``matcher_protocol``). ``UnitMatchParamsSchema`` validates the ``params``
blob of a ``MatcherParameters`` row whose ``matcher == "unitmatch"``.

The matcher's own internal knobs (UnitMatch's ``default_params``) are resolved
inside the backend; the fields here are the v2-owned controls that the tables
read: the pair-probability cutoffs and the strict tracked-unit graph cap.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UnitMatchParamsSchema(BaseModel):
    """Validated ``params`` for a ``MatcherParameters`` row using UnitMatch.

    Attributes
    ----------
    match_threshold : float
        Pair-probability cutoff applied to UnitMatch's output probability when
        deciding which pairs are matches. Default ``0.5``.
    tracked_unit_threshold : float
        Pair-probability cutoff used to seed edges in the ``TrackedUnit`` graph.
        Default ``0.5``.
    max_strict_nodes : int
        Hard upper bound on the graph size submitted to strict
        ``networkx.find_cliques``. Exceeding it raises
        ``TrackedUnitBudgetExceededError``. Default ``2000``.
    ms_before, ms_after : float
        Symmetric waveform window (ms) for the per-session UnitMatch bundle.
        Identity-bearing: they change the dense templates the matcher compares,
        so they live in the named params blob (not as silent
        ``extract_unitmatch_bundle`` function defaults). Default ``1.5`` each.
    max_spikes_per_unit : int
        Random-spike cap per unit per cross-validation half when building the
        bundle. Identity-bearing. Default ``100``.
    seed : int
        Seed for the bundle's random-spike subsample. Identity-bearing and
        authoritative -- a ``random_seed`` in ``job_kwargs`` never overrides it.
        Default ``0``.
    schema_version : int
        Bumped on breaking field changes; rows insert at the current version.
    """

    model_config = ConfigDict(extra="forbid")

    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tracked_unit_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_strict_nodes: int = Field(default=2000, ge=1)
    ms_before: float = Field(default=1.5, gt=0.0)
    ms_after: float = Field(default=1.5, gt=0.0)
    max_spikes_per_unit: int = Field(default=100, ge=1)
    seed: int = Field(default=0, ge=0)
    schema_version: int = 1
