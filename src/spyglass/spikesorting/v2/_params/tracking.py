"""Backend-independent controls for deriving tracked units from pair matches."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrackingParamsSchema(BaseModel):
    """Shared fields that matcher recipes may include for tracked-unit grouping.

    Flat fields of the matcher recipe blob; a backend may inherit this schema.
    Recipes without them use these defaults. Grouping is a strict greedy
    maximal-clique partition, independent of the matcher backend.
    """

    model_config = ConfigDict(extra="forbid")

    tracked_unit_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_strict_nodes: int = Field(default=2000, ge=1)


def resolve_tracking_params(matcher_params: dict) -> TrackingParamsSchema:
    """Validate grouping controls separately from backend inference parameters."""
    return TrackingParamsSchema.model_validate(
        {
            field: matcher_params[field]
            for field in TrackingParamsSchema.model_fields
            if field in matcher_params
        }
    )
