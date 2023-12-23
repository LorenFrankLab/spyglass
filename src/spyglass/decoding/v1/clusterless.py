"""Pipeline for decoding the animal's mental position and some category of interest
from unclustered spikes and spike waveform features. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import datajoint as dj
import pandas as pd
from non_local_detector.models import ContFragClusterlessClassifier
from non_local_detector.models.base import ClusterlessDetector

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.decoding.v1.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.decoding.v1.waveform_features import (
    UnitWaveformFeatures,
)  # noqa: F401
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.utils import SpyglassMixin

schema = dj.schema("decoding_clusterless_v1")


@schema
class DecodingParameters(SpyglassMixin, dj.Lookup):
    """Parameters for decoding the animal's mental position and some category of interest"""

    definition = """
    classifier_param_name : varchar(80) # a name for this set of parameters
    ---
    classifier_params :   BLOB        # initialization parameters for model
    estimate_parameters_kwargs : BLOB # keyword arguments for estimate_parameters
    """
    definition = """
    features_param_name : varchar(80) # a name for this set of parameters
    ---
    params : longblob # the parameters for the waveform features
    """

    # contents = [
    #     [
    #         "contfrag_clusterless",
    #         vars(ContFragClusterlessClassifier()),
    #         dict(),
    #     ],
    # ]

    # @classmethod
    # def insert_default(cls):
    #     cls.insert(cls.contents, skip_duplicates=True)

    # def insert(self, keys, **kwargs):
    #     pass

    def insert1(self, key, **kwargs):
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        return restore_classes(super().fetch1(*args, **kwargs))


@schema
class UnitWaveformFeaturesGroup(SpyglassMixin, dj.Manual):
    definition = """
    waveform_features_group_name: varchar(80)
    """

    class UnitFeatures(SpyglassMixin, dj.Part):
        definition = """
        -> UnitWaveformFeaturesGroup
        -> UnitWaveformFeatures
        """

    def create_group(self, group_name: str, keys: list[dict]):
        self.insert1(
            {"waveform_features_group_name": group_name}, skip_duplicates=True
        )
        for key in keys:
            self.UnitFeatures.insert1(
                {
                    **key,
                    "waveform_features_group_name": group_name,
                },
                skip_duplicates=True,
            )


@schema
class PositionGroup(SpyglassMixin, dj.Manual):
    definition = """
    position_group_name: varchar(80)
    ----
    position_variables = NULL: longblob # list of position variables to decode
    """

    class Position(SpyglassMixin, dj.Part):
        definition = """
        -> PositionGroup
        -> PositionOutput.proj(pos_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        keys: list[dict],
        position_variables: list[str] = ["position_x", "position_y"],
    ):
        self.insert1(
            {
                "position_group_name": group_name,
                "position_variables": position_variables,
            },
            skip_duplicates=True,
        )
        for key in keys:
            self.Position.insert1(
                {
                    **key,
                    "position_group_name": group_name,
                },
                skip_duplicates=True,
            )


@schema
class ClusterlessDecodingSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> UnitWaveformFeaturesGroup
    -> PositionGroup
    -> DecodingParameters
    # -> IntervalList.proj(encoding_interval='interval_list_name')
    # -> IntervalList.proj(decoding_interval='interval_list_name')
    """


@schema
class ClusterlessDecodingV1(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessDecodingSelection
    ---
    results_path: varchar(255) # path to the results file
    """

    def make(self, key):
        selection_params = (ClusterlessDecodingSelection & key).fetch1()

        # Get model parameters
        decoding_params = (
            DecodingParameters
            & {
                "classifier_param_name": selection_params[
                    "classifier_param_name"
                ]
            }
        ).fetch1()

        estimate_parameters_kwargs = decoding_params[
            "estimate_parameters_kwargs"
        ]
        classifier_params = decoding_params["classifier_params"]

        # Get position data
        position_group_key = {
            "position_group_name": selection_params["position_group_name"]
        }
        position_variable_names = (PositionGroup & position_group_key).fetch1(
            "position_variables"
        )

        position_info = []
        for pos_merge_id in (PositionGroup.Position & position_group_key).fetch(
            "pos_merge_id"
        ):
            position_info.append(
                IntervalPositionInfo._data_to_df(
                    PositionOutput.fetch_nwb({"merge_id": pos_merge_id})[0],
                    prefix="",
                    add_frame_ind=True,
                )
            )
        position_info = pd.concat(position_info, axis=0).dropna()

        # Get the waveform features for the selected units
        waveform_keys = (
            (
                UnitWaveformFeaturesGroup.UnitFeatures
                & {
                    "waveform_features_group_name": selection_params[
                        "waveform_features_group_name"
                    ]
                }
            )
        ).fetch("KEY")

        spike_times, spike_waveform_features = (
            UnitWaveformFeatures & waveform_keys
        ).fetch_data()

        # Get the encoding and decoding intervals

        # Decode
        classifier = ClusterlessDetector(**classifier_params)
        results = classifier.estimate_parameters(
            position_time=position_info.index.to_numpy(),
            position=position_info[position_variable_names].to_numpy(),
            spike_times=spike_times,
            spike_waveform_features=spike_waveform_features,
            time=position_info.index.to_numpy(),
            **estimate_parameters_kwargs,
        )

        # Insert results
        # in future use https://github.com/rly/ndx-xarray
        # from ndx_xarray import ExternalXarrayDataset
        # xr_dset2 = ExternalXarrayDataset(name="test_xarray2", description="test description", path=xr_path)
        # nwbfile.add_analysis(xr_dset2)

        results_path = "results.nc"
        classifier.save_results(
            results,
            results_path,
        )
        key["results_path"] = results_path

        self.insert1(key)
