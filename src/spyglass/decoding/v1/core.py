import datajoint as dj
import numpy as np
import pandas as pd
from non_local_detector import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector import __version__ as non_local_detector_version

from spyglass.common.common_session import Session  # noqa: F401
from spyglass.decoding.v1.dj_decoder_conversion import (
    convert_classes_to_dict,
    restore_classes,
)
from spyglass.position.position_merge import PositionOutput  # noqa: F401
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("decoding_core_v1")


@schema
class DecodingParameters(SpyglassMixin, dj.Lookup):
    """Params for decoding mental position and some category of interest"""

    definition = """
    decoding_param_name : varchar(80)  # a name for this set of parameters
    ---
    decoding_params : BLOB             # initialization parameters for model
    decoding_kwargs = NULL : BLOB      # additional keyword arguments
    """

    pk = "decoding_param_name"
    sk = "decoding_params"

    contents = [
        {
            pk: f"contfrag_clusterless_{non_local_detector_version}",
            sk: ContFragClusterlessClassifier(),
        },
        {
            pk: f"nonlocal_clusterless_{non_local_detector_version}",
            sk: NonLocalClusterlessDetector(),
        },
        {
            pk: f"contfrag_sorted_{non_local_detector_version}",
            sk: ContFragSortedSpikesClassifier(),
        },
        {
            pk: f"nonlocal_sorted_{non_local_detector_version}",
            sk: NonLocalSortedSpikesDetector(),
        },
    ]

    @classmethod
    def insert_default(cls):
        cls.insert(cls.contents, skip_duplicates=True)

    def insert(self, rows, *args, **kwargs):
        for row in rows:
            row["decoding_params"] = convert_classes_to_dict(
                vars(row["decoding_params"])
            )
        super().insert(rows, *args, **kwargs)

    def fetch(self, *args, **kwargs):
        rows = super().fetch(*args, **kwargs)
        if len(rows) > 0 and len(rows[0]) > 1:
            content = []
            for (
                decoding_param_name,
                decoding_params,
                decoding_kwargs,
            ) in rows:
                content.append(
                    (
                        decoding_param_name,
                        restore_classes(decoding_params),
                        decoding_kwargs,
                    )
                )
        else:
            content = rows
        return content

    def fetch1(self, *args, **kwargs):
        row = super().fetch1(*args, **kwargs)
        row["decoding_params"] = restore_classes(row["decoding_params"])
        return row


@schema
class PositionGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    position_group_name: varchar(80)
    ----
    position_variables = NULL: longblob # list of position variables to decode
    upsample_rate = NULL: float # upsampling rate for position data (Hz)
    """

    class Position(SpyglassMixinPart):
        definition = """
        -> PositionGroup
        -> PositionOutput.proj(pos_merge_id='merge_id')
        """

    def create_group(
        self,
        nwb_file_name: str,
        group_name: str,
        keys: list[dict],
        position_variables: list[str] = ["position_x", "position_y"],
        upsample_rate: float = np.nan,
    ):
        group_key = {
            "nwb_file_name": nwb_file_name,
            "position_group_name": group_name,
        }
        if self & group_key:
            raise ValueError(
                f"Group {nwb_file_name}: {group_name} already exists",
                "please delete the group before creating a new one",
            )
        self.insert1(
            {
                **group_key,
                "position_variables": position_variables,
                "upsample_rate": upsample_rate,
            },
            skip_duplicates=True,
        )
        for key in keys:
            self.Position.insert1(
                {
                    **key,
                    **group_key,
                },
                skip_duplicates=True,
            )

    def fetch_position_info(
        self, key: dict = None, min_time: float = None, max_time: float = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """fetch position information for decoding

        Parameters
        ----------
        key : dict, optional
            restriction to a single entry in PositionGroup, by default None
        min_time : float, optional
            restrict position information to times greater than min_time,
            by default None
        max_time : float, optional
            restrict position information to times less than max_time,
            by default None

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            position information and names of position variables
        """
        if key is None:
            key = {}
        key = (self & key).fetch1("KEY")
        position_variable_names = (self & key).fetch1("position_variables")

        position_info = []
        upsample_rate = (self & key).fetch1("upsample_rate")
        for pos_merge_id in (self.Position & key).fetch("pos_merge_id"):
            if not np.isnan(upsample_rate):
                position_info.append(
                    self._upsample(
                        (
                            PositionOutput & {"merge_id": pos_merge_id}
                        ).fetch1_dataframe(),
                        upsampling_sampling_rate=upsample_rate,
                        position_variable_names=position_variable_names,
                    )
                )
            else:
                position_info.append(
                    (
                        PositionOutput & {"merge_id": pos_merge_id}
                    ).fetch1_dataframe()
                )

        if min_time is None:
            min_time = min([df.index.min() for df in position_info])
        if max_time is None:
            max_time = max([df.index.max() for df in position_info])
        position_info = pd.concat(position_info, axis=0).loc[min_time:max_time]

        return position_info, position_variable_names

    @staticmethod
    def _upsample(
        position_df: pd.DataFrame,
        upsampling_sampling_rate: float,
        upsampling_interpolation_method: str = "linear",
        position_variable_names: list[str] = None,
    ) -> pd.DataFrame:
        """upsample position data to a fixed sampling rate

        Parameters
        ----------
        position_df : pd.DataFrame
            dataframe containing position data
        upsampling_sampling_rate : float
            sampling rate to upsample to
        upsampling_interpolation_method : str, optional
            pandas method for interpolation, by default "linear"
        position_variable_names : list[str], optional
            names of position variables of focus, for which nan values will not be
            interpolated, by default None includes all columns

        Returns
        -------
        pd.DataFrame
            upsampled position data
        """

        upsampling_start_time = position_df.index[0]
        upsampling_end_time = position_df.index[-1]

        n_samples = (
            int(
                np.ceil(
                    (upsampling_end_time - upsampling_start_time)
                    * upsampling_sampling_rate
                )
            )
            + 1
        )
        new_time = np.linspace(
            upsampling_start_time, upsampling_end_time, n_samples
        )
        new_index = pd.Index(
            np.unique(np.concatenate((position_df.index, new_time))),
            name="time",
        )

        # Find NaN intervals
        nan_intervals = {}
        if position_variable_names is None:
            position_variable_names = position_df.columns
        for column in position_variable_names:
            is_nan = position_df[column].isna().to_numpy().astype(int)
            st = np.where(np.diff(is_nan) == 1)[0] + 1
            en = np.where(np.diff(is_nan) == -1)[0]
            if is_nan[0]:
                st = np.insert(st, 0, 0)
            if is_nan[-1]:
                en = np.append(en, len(is_nan) - 1)
            st = position_df.index[st].to_numpy()
            en = position_df.index[en].to_numpy()
            nan_intervals[column] = list(zip(st, en))

        # upsample and interpolate
        position_df = (
            position_df.reindex(index=new_index)
            .interpolate(method=upsampling_interpolation_method)
            .reindex(index=new_time)
        )

        # Fill NaN intervals
        for column, intervals in nan_intervals.items():
            for st, en in intervals:
                position_df[column][st:en] = np.nan

        return position_df
