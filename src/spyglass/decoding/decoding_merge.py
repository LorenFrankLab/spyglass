from itertools import chain
from pathlib import Path

import datajoint as dj

from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1  # noqa: F401
from spyglass.decoding.v1.sorted_spikes import (
    SortedSpikesDecodingV1,
)  # noqa: F401
from spyglass.settings import config
from spyglass.utils import SpyglassMixin, _Merge, logger

schema = dj.schema("decoding_merge")


@schema
class DecodingOutput(_Merge, SpyglassMixin):
    definition = """
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class ClusterlessDecodingV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        ---
        -> ClusterlessDecodingV1
        """

    class SortedSpikesDecodingV1(SpyglassMixin, dj.Part):
        definition = """
        -> master
        ---
        -> SortedSpikesDecodingV1
        """

    def cleanup(self):
        """Remove any decoding outputs that are not in the merge table"""
        logger.info("Cleaning up decoding outputs")
        table_results_paths = list(
            chain(
                *[
                    part_parent_table.fetch("results_path").tolist()
                    for part_parent_table in self.merge_get_parent(
                        multi_source=True
                    )
                ]
            )
        )
        for path in Path(config["SPYGLASS_ANALYSIS_DIR"]).glob("**/*.nc"):
            if str(path) not in table_results_paths:
                logger.info(f"Removing {path}")
                path.unlink()

        table_model_paths = list(
            chain(
                *[
                    part_parent_table.fetch("classifier_path").tolist()
                    for part_parent_table in self.merge_get_parent(
                        multi_source=True
                    )
                ]
            )
        )
        for path in Path(config["SPYGLASS_ANALYSIS_DIR"]).glob("**/*.pkl"):
            if str(path) not in table_model_paths:
                logger.info(f"Removing {path}")
                path.unlink()
