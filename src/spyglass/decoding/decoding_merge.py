import inspect
from itertools import chain
from pathlib import Path

import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
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
    def _get_source_class(cls, key):
        if cls._source_class_dict is None:
            cls._source_class_dict = {}
            module = inspect.getmodule(cls)
            for part_name in cls.parts():
                part_name = to_camel_case(part_name.split("__")[-1].strip("`"))
                part = getattr(module, part_name)
                cls._source_class_dict[part_name] = part

        source = (cls & key).fetch1("source")
        return cls._source_class_dict[source]

    @classmethod
    def load_results(cls, key):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return (source_class & decoding_selection_key).load_results()

    @classmethod
    def load_model(cls, key):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return (source_class & decoding_selection_key).load_model()

    @classmethod
    def load_environments(cls, key):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return source_class.load_environments(decoding_selection_key)

    @classmethod
    def load_position_info(cls, key):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return source_class.load_position_info(decoding_selection_key)

    @classmethod
    def load_linear_position_info(cls, key):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return source_class.load_linear_position_info(decoding_selection_key)

    @classmethod
    def load_spike_data(cls, key, filter_by_interval=True):
        decoding_selection_key = cls.merge_get_parent(key).fetch1("KEY")
        source_class = cls._get_source_class(key)
        return source_class.load_linear_position_info(
            decoding_selection_key, filter_by_interval=filter_by_interval
        )

    @classmethod
    def create_decoding_view(cls, key, head_direction_name="head_orientation"):
        results = cls.load_results(key)
        posterior = results.acausal_posterior.unstack("state_bins").sum("state")
        env = cls.load_environments(key)[0]

        if "x_position" in results.coords:
            position_info, position_variable_names = cls.load_position_info(key)
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
            (
                position_info,
                position_variable_names,
            ) = cls.load_linear_position_info(key)
            return create_1D_decode_view(
                posterior=posterior,
                linear_position=position_info["linear_position"],
                ref_time_sec=position_info.index[0],
            )
