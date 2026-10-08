"""Backend-independent controls for deriving tracked units from pair matches."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrackingParamsSchema(BaseModel):
    """Shared fields that matcher recipes may include for tracked-unit grouping.

    These retain their established flat parameter names so existing immutable
    recipes keep their content and identities. A backend can inherit this
    schema to validate overrides at insert time. Recipes without these fields
    use the same defaults when deriving tracked units.

    Grouping currently uses the strict greedy maximal-clique partition. It is
    independent of the algorithm that produced the pair probabilities.
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
