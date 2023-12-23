import datajoint as dj

from spyglass.decoding.v1.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("decoding_core_v1")


@schema
class DecodingParameters(SpyglassMixin, dj.Lookup):
    """Parameters for decoding the animal's mental position and some category of interest"""

    definition = """
    decoding_param_name : varchar(80)  # a name for this set of parameters
    ---
    decoding_params : BLOB             # initialization parameters for model
    decoding_kwargs : BLOB             # additional keyword arguments
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
