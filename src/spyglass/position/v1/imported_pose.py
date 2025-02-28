import datajoint as dj
import ndx_pose
import numpy as np
import pandas as pd
import pynwb

from spyglass.common import IntervalList, Nwbfile
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_valid_intervals,
)

schema = dj.schema("position_v1_imported_pose")


@schema
class ImportedPose(SpyglassMixin, dj.Manual):
    """
    Table to ingest pose data generated prior to spyglass.
    Each entry corresponds to on ndx_pose.PoseEstimation object in an NWB file.
    PoseEstimation objects should be stored in nwb.processing.behavior
    Assumptions:
    - Single skeleton object per PoseEstimation object
    """

    _nwb_table = Nwbfile

    definition = """
    -> IntervalList
    ---
    pose_object_id: varchar(80) # unique identifier for the pose object
    skeleton_object_id: varchar(80) # unique identifier for the skeleton object
    """

    class BodyPart(SpyglassMixin, dj.Part):
        definition = """
        -> master
        part_name: varchar(80)
        ---
        part_object_id: varchar(80)
        """

    def make(self, key):
        self.insert_from_nwbfile(key["nwb_file_name"])

    def insert_from_nwbfile(self, nwb_file_name):
        file_path = Nwbfile().get_abs_path(nwb_file_name)
        interval_keys = []
        master_keys = []
        part_keys = []
        with pynwb.NWBHDF5IO(file_path, mode="r") as io:
            nwb = io.read()
            behavior_module = nwb.get_processing_module("behavior")

            # Loop through all the PoseEstimation objects in the behavior module
            for name, obj in behavior_module.data_interfaces.items():
                if not isinstance(obj, ndx_pose.PoseEstimation):
                    continue

                # use the timestamps from the first body part to define valid times
                timestamps = list(obj.pose_estimation_series.values())[
                    0
                ].timestamps[:]
                sampling_rate = estimate_sampling_rate(
                    timestamps, filename=nwb_file_name
                )
                valid_intervals = get_valid_intervals(
                    timestamps,
                    sampling_rate=sampling_rate,
                    min_valid_len=sampling_rate,
                )
                interval_key = {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": f"pose_{name}_valid_intervals",
                    "valid_times": valid_intervals,
                    "pipeline": "ImportedPose",
                }
                interval_keys.append(interval_key)

                # master key
                master_key = {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": interval_key["interval_list_name"],
                    "pose_object_id": obj.object_id,
                    "skeleton_object_id": obj.skeleton.object_id,
                }
                master_keys.append(master_key)

                # part keys
                for part, part_obj in obj.pose_estimation_series.items():
                    part_key = {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": interval_key[
                            "interval_list_name"
                        ],
                        "part_name": part,
                        "part_object_id": part_obj.object_id,
                    }
                    part_keys.append(part_key)

        IntervalList().insert(interval_keys, skip_duplicates=True)
        self.insert(master_keys)
        self.BodyPart().insert(part_keys)

    def fetch_pose_dataframe(self, key=None):
        """Fetch pose data as a pandas DataFrame

        Parameters
        ----------
        key : dict
            Key to fetch pose data for

        Returns
        -------
        pd.DataFrame
            DataFrame containing pose data
        """
        key = key or dict()
        key = (self & key).fetch1("KEY")
        pose_estimations = (
            (self & key).fetch_nwb()[0]["pose"].pose_estimation_series
        )

        index = None
        pose_df = {}
        body_parts = list(pose_estimations.keys())
        index = pd.Index(
            pose_estimations[body_parts[0]].timestamps[:], name="time"
        )
        for body_part in pose_estimations.keys():
            bp_data = pose_estimations[body_part].data
            part_df = {
                "video_frame_ind": np.nan,
                "x": bp_data[:, 0],
                "y": bp_data[:, 1],
                "likelihood": pose_estimations[body_part].confidence[:],
            }

            pose_df[body_part] = pd.DataFrame(part_df, index=index)

        pose_df
        return pd.concat(pose_df, axis=1)

    def fetch_skeleton(self, key=None):
        key = key or dict()
        query = self & key
        if not len(query) == 1:
            raise ValueError(
                "fetch_skeleton can only be called on a single entry"
            )
        skeleton = query.fetch_nwb()[0]["skeleton"]
        nodes = skeleton.nodes[:]
        int_edges = skeleton.edges[:]
        named_edges = [[nodes[i], nodes[j]] for i, j in int_edges]
        return {"nodes": nodes, "edges": named_edges}
