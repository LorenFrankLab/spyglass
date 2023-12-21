"""Pipeline for decoding the animal's mental position and some category of interest
from unclustered spikes and spike waveform features. See [1] for details.

References
----------
[1] Denovellis, E. L. et al. Hippocampal replay of experience at real-world
speeds. eLife 10, e64505 (2021).

"""

import datajoint as dj
from non_local_detector import ContFragClusterlessClassifier
from spyglass.utils import SpyglassMixin

from spyglass.decoding.v1.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)

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

    def insert_default(self):
        self.insert1(
            {
                "classifier_param_name": "contfrag_clusterless",
                "classifier_params": vars(ContFragClusterlessClassifier()),
                "estimate_parameters_kwargs": dict(),
            },
            skip_duplicates=True,
        )

    def insert1(self, key, **kwargs):
        super().insert1(convert_classes_to_dict(key), **kwargs)

    def fetch1(self, *args, **kwargs):
        return restore_classes(super().fetch1(*args, **kwargs))


@schema
class DecodingElectrodeSelection(dj.Manual):
    definition = """

    -> ElectrodeWaveformFeaturesGroup

    """

    class DecodingElectrode(dj.Part):
        definition = """
        -> DecodingElectrodeSelection
        -> UnitWaveformFeatures
        """


@schema
class ClusterlessDecodingV1(dj.Computed):
    definition = """
    -> DecodingSelection
    -> PositionOutput.proj(pos_merge_id='merge_id')
    -> DecodingParameters
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)
    """

    def make(self, key):
        pass
