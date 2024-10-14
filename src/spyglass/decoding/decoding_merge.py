from itertools import chain
from pathlib import Path

import datajoint as dj
import numpy as np
from non_local_detector.visualization.figurl_1D import create_1D_decode_view
from non_local_detector.visualization.figurl_2D import create_2D_decode_view

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

    class ClusterlessDecodingV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ClusterlessDecodingV1
        """

    class SortedSpikesDecodingV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> SortedSpikesDecodingV1
        """

    def cleanup(self, dry_run=False):
        """Remove any decoding outputs that are not in the merge table"""
        if dry_run:
            logger.info("Dry run, not removing any files")
        else:
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
                if not dry_run:
                    try:
                        path.unlink(missing_ok=True)  # Ignore FileNotFoundError
                    except PermissionError:
                        logger.warning(f"Unable to remove {path}, skipping")

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
                if not dry_run:
                    try:
                        path.unlink()
                    except (PermissionError, FileNotFoundError):
                        logger.warning(f"Unable to remove {path}, skipping")

    @classmethod
    def fetch_results(cls, key):
        """Fetch the decoding results for a given key."""
        return cls().merge_restrict_class(key).fetch_results()

    @classmethod
    def fetch_model(cls, key):
        """Fetch the decoding model for a given key."""
        return cls().merge_restrict_class(key).fetch_model()

    @classmethod
    def fetch_environments(cls, key):
        """Fetch the decoding environments for a given key."""
        restr_parent = cls().merge_restrict_class(key)
        decoding_selection_key = restr_parent.fetch1("KEY")
        return restr_parent.fetch_environments(decoding_selection_key)

    @classmethod
    def fetch_position_info(cls, key):
        """Fetch the decoding position info for a given key."""
        restr_parent = cls().merge_restrict_class(key)
        decoding_selection_key = restr_parent.fetch1("KEY")
        return restr_parent.fetch_position_info(decoding_selection_key)

    @classmethod
    def fetch_linear_position_info(cls, key):
        """Fetch the decoding linear position info for a given key."""
        restr_parent = cls().merge_restrict_class(key)
        decoding_selection_key = restr_parent.fetch1("KEY")
        return restr_parent.fetch_linear_position_info(decoding_selection_key)

    @classmethod
    def fetch_spike_data(cls, key, filter_by_interval=True):
        """Fetch the decoding spike data for a given key."""
        restr_parent = cls().merge_restrict_class(key)
        decoding_selection_key = restr_parent.fetch1("KEY")
        return restr_parent.fetch_spike_data(
            decoding_selection_key, filter_by_interval=filter_by_interval
        )

    @classmethod
    def create_decoding_view(cls, key, head_direction_name="head_orientation"):
        """Create a decoding view for a given key."""
        results = cls.fetch_results(key)
        posterior = (
            results.squeeze()
            .acausal_posterior.unstack("state_bins")
            .drop_sel(state=["Local", "No-Spike"], errors="ignore")
            .sum("state")
        )
        posterior /= posterior.sum("position")
        env = cls.fetch_environments(key)[0]

        if "x_position" in results.coords:
            position_info, position_variable_names = cls.fetch_position_info(
                key
            )
            # Not 1D
            bin_size = (
                np.nanmedian(np.diff(np.unique(results.x_position.values))),
                np.nanmedian(np.diff(np.unique(results.y_position.values))),
            )
            return create_2D_decode_view(
                position_time=position_info.index,
                position=position_info[position_variable_names],
                interior_place_bin_centers=env.place_bin_centers_[
                    env.is_track_interior_.ravel(order="C")
                ],
                place_bin_size=bin_size,
                posterior=posterior,
                head_dir=position_info[head_direction_name],
            )
        else:
            return create_1D_decode_view(
                posterior=posterior,
                linear_position=cls.fetch_linear_position_info(key),
            )
