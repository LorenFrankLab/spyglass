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
    schema_version : int
        Bumped on breaking field changes; rows insert at the current version.
    """

    model_config = ConfigDict(extra="forbid")

    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tracked_unit_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_strict_nodes: int = Field(default=2000, ge=1)
    schema_version: int = 1
