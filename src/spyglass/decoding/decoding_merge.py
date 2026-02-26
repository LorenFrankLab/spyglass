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

    def _fetch_registered_paths(self, attr):
        """Fetch a filepath attribute from all part parents, skipping missing."""
        paths = []
        for tbl in self.merge_get_parent(multi_source=True):
            try:
                paths.extend(tbl.fetch(attr).tolist())
            except FileNotFoundError:
                pass
        return paths

    def cleanup(self, dry_run=False):
        """Remove any decoding outputs that are not in the merge table"""
        if dry_run:
            self._info_msg("Dry run, not removing any files")
        else:
            self._info_msg("Cleaning up decoding outputs")
        table_results_paths = self._fetch_registered_paths("results_path")
        for path in Path(config["SPYGLASS_ANALYSIS_DIR"]).glob("**/*.nc"):
            if str(path) not in table_results_paths:
                self._info_msg(f"Removing {path}")
                if not dry_run:
                    try:
                        path.unlink(missing_ok=True)  # Ignore FileNotFoundError
                    except PermissionError:
                        logger.warning(f"Unable to remove {path}, skipping")

        table_model_paths = self._fetch_registered_paths("classifier_path")
        for path in Path(config["SPYGLASS_ANALYSIS_DIR"]).glob("**/*.pkl"):
            if str(path) not in table_model_paths:
                self._info_msg(f"Removing {path}")
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
    def create_decoding_view(
        cls, key, head_direction_name="head_orientation", interval_idx=None
    ):
        """Create a decoding view for a given key.

        Parameters
        ----------
        key : dict
            Key identifying the decoding output
        head_direction_name : str, optional
            Name of head direction column, by default "head_orientation"
        interval_idx : int, optional
            If specified, only visualize this interval (0-indexed).
            If None (default), visualize all intervals together.

        Returns
        -------
        view
            Figurl visualization view (1D or 2D depending on decoder)
        """
        results = cls.fetch_results(key)

        # Filter to specific interval if requested
        if interval_idx is not None:
            if "interval_labels" in results.coords:
                results = results.where(
                    results.interval_labels == interval_idx, drop=True
                )
            else:
                logger.warning(
                    f"interval_idx={interval_idx} specified but results do not "
                    "have 'interval_labels' coordinate. Ignoring interval_idx."
                )

        posterior = (
            results.acausal_posterior.unstack("state_bins")
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
