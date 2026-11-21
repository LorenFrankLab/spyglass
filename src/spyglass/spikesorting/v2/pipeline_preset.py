"""Saved pipeline presets: a named parameter row for each stage.

``clone_pipeline_preset`` and ``register_pipeline_preset`` save presets here,
so they last beyond one Python session. ``initialize_v2_defaults`` adds the
shipped presets.
"""

import datajoint as dj

from spyglass.spikesorting.v2._core.recipe_catalog import (
    pipeline_preset_specs,
)
from spyglass.spikesorting.v2._orchestration.presets import _PipelinePreset
from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
from spyglass.spikesorting.v2.metric_curation import (
    AutoCurationRules,
    QualityMetricParameters,
)
from spyglass.spikesorting.v2.recording import PreprocessingParameters
from spyglass.spikesorting.v2.sorting import SorterParameters
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v2_pipeline_preset")

#: ``created_by`` of the shipped presets.
SHIPPED_PRESET_CREATOR = "spyglass"


@schema
class PipelinePreset(SpyglassMixin, dj.Manual):
    """A named pipeline preset for ``run_v2_pipeline``.

    Runs are keyed by the parameter rows, not the preset name, so deleting a
    preset leaves its results in place. Add presets with
    ``clone_pipeline_preset`` or ``register_pipeline_preset``.
    """

    definition = """
    pipeline_preset: varchar(200)
    ---
    -> PreprocessingParameters
    -> SorterParameters
    -> QualityMetricParameters
    -> AutoCurationRules
    -> [nullable] ArtifactDetectionParameters  # NULL: no artifact detection
    probe_type="": varchar(32)
    target_region="": varchar(32)
    sampling_rate_hz=null: int
    sorter_family="": varchar(64)
    adjacency_radius_um=null: double
    recommendation_status="": varchar(32)
    intended_use="": varchar(1024)
    threshold_units="": varchar(256)
    notes="": varchar(4000)
    created_by: varchar(128)  # dj.config["database.user"]; "spyglass" if shipped
    created_at=CURRENT_TIMESTAMP: timestamp
    """

    @classmethod
    def _default_rows(cls) -> list[dict]:
        """The shipped presets as rows, without ``created_at``."""
        return [
            {
                "pipeline_preset": name,
                **_PipelinePreset(**spec).model_dump(),
                "created_by": SHIPPED_PRESET_CREATOR,
            }
            for name, spec in pipeline_preset_specs().items()
        ]

    @classmethod
    def insert_default(cls) -> None:
        """Insert the shipped presets; existing names are skipped.

        Like ``SorterParameters.insert_default``, skips (logged at INFO) a
        preset whose sorter row is absent because the sorter is not installed.
        """
        rows = []
        for row in cls._default_rows():
            sorter_key = {
                "sorter": row["sorter"],
                "sorter_params_name": row["sorter_params_name"],
            }
            if SorterParameters & sorter_key:
                rows.append(row)
            else:
                logger.info(
                    "PipelinePreset.insert_default: skipping %r; no "
                    "SorterParameters row %s.",
                    row["pipeline_preset"],
                    sorter_key,
                )
        cls.insert(rows, skip_duplicates=True)
