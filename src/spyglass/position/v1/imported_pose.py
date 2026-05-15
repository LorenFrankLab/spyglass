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
        self.insert_from_nwbfile(key["nwb_file_name"])  # pragma: no cover

    def insert_from_nwbfile(self, nwb_file_name, import_to_v2=False, **kwargs):
        """Ingest all ndx-pose PoseEstimation objects from a registered NWB.

        Parameters
        ----------
        nwb_file_name : str
            Spyglass-registered NWB filename (must exist in Nwbfile table).
        import_to_v2 : bool, optional
            When True, also register skeleton graph(s) from the NWB in the V2
            ``Skeleton`` table.  ndx-pose files hold pose *results*, not
            trained model weights, so only the skeleton metadata belongs in V2.
            By default False.
        **kwargs
            Forwarded to ``dj.Table.insert`` (e.g. ``skip_duplicates``).
        """
        file_path = Nwbfile().get_abs_path(nwb_file_name)
        interval_keys = []
        master_keys = []
        part_keys = []
        with pynwb.NWBHDF5IO(file_path, mode="r") as io:
            nwb = io.read()
            pose_objects = [
                obj
                for obj in nwb.objects.values()
                if isinstance(obj, ndx_pose.PoseEstimation)
            ]

            # Loop through all the PoseEstimation objects in the behavior module
            for obj in pose_objects:
                name = obj.name
                # use the timestamps from the first body part to define valid times
                timestamps = list(obj.pose_estimation_series.values())[
                    0
                ].get_timestamps()
                sampling_rate = estimate_sampling_rate(
                    timestamps, filename=nwb_file_name
                )
                valid_intervals = get_valid_intervals(
                    timestamps,
                    sampling_rate=sampling_rate,
                    min_valid_len=sampling_rate,
                    warn=not self._test_mode,
                )
                interval_pk = {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": f"pose_{name}_valid_intervals",
                }
                interval_keys.append(
                    {
                        **interval_pk,
                        "valid_times": valid_intervals,
                        "pipeline": "ImportedPose",
                    }
                )
                master_keys.append(
                    {
                        **interval_pk,
                        "pose_object_id": obj.object_id,
                        "skeleton_object_id": obj.skeleton.object_id,
                    }
                )
                part_keys.extend(
                    [
                        {
                            **interval_pk,
                            "part_name": part,
                            "part_object_id": part_obj.object_id,
                        }
                        for part, part_obj in obj.pose_estimation_series.items()
                    ]
                )

        with self._safe_context():
            IntervalList().insert(interval_keys, **kwargs)
            self.insert(master_keys, **kwargs)
            self.BodyPart().insert(part_keys, **kwargs)

        if import_to_v2:
            self._import_to_v2_pipeline(file_path, nwb_file_name)

    def _import_to_v2_pipeline(self, file_path, nwb_file_name):
        """Register skeleton from an ndx-pose NWB in the V2 Skeleton table.

        ndx-pose NWBs contain skeleton graphs that are tool-agnostic and belong
        in ``Skeleton``.  This is the only V2 table populated by
        ``import_to_v2=True``; model/inference metadata is not created because
        ndx-pose files hold pose *results*, not trained model weights.

        Errors are logged as warnings so that the primary ``ImportedPose``
        ingestion is never rolled back by a V2 failure.

        Parameters
        ----------
        file_path : str or Path
            Absolute path to the ndx-pose NWB file.
        nwb_file_name : str
            Spyglass-registered NWB filename (used only for logging).
        """
        import warnings

        import ndx_pose
        import pynwb
        from pynwb import NWBHDF5IO

        from spyglass.position.v2.train import Skeleton

        try:
            with NWBHDF5IO(str(file_path), mode="r") as io:
                nwbf = io.read()
                behavior = nwbf.processing.get("behavior")
                if behavior is None:
                    return
                for obj in nwbf.objects.values():
                    if not isinstance(obj, ndx_pose.Skeleton):
                        continue
                    bodyparts = list(obj.nodes)
                    edges_arr = obj.edges
                    edges = []
                    if edges_arr is not None:
                        for edge in edges_arr:
                            edges.append(
                                (
                                    bodyparts[int(edge[0])],
                                    bodyparts[int(edge[1])],
                                )
                            )
                    try:
                        Skeleton().insert1(
                            {"bodyparts": bodyparts, "edges": edges},
                            accept_default=True,
                            skip_duplicates=True,
                        )
                    except Exception as sk_exc:  # noqa: BLE001
                        warnings.warn(
                            f"V2 Skeleton insert failed for '{obj.name}' "
                            f"in '{nwb_file_name}': {sk_exc}",
                            stacklevel=4,
                        )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"V2 skeleton registration failed for '{nwb_file_name}': "
                f"{exc}. ImportedPose entries were inserted successfully.",
                stacklevel=3,
            )

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
        _ = self.ensure_single_entry()
        key = key or self.fetch1("KEY")
        query = self & key
        if len(query) != 1:
            raise ValueError(f"Key selected {len(query)} entries: {query}")
        key = query.fetch1("KEY")
        pose_estimations = (
            (self & key).fetch_nwb()[0]["pose"].pose_estimation_series
        )

        index = None
        pose_df = {}
        body_parts = list(pose_estimations.keys())
        index = pose_estimations[body_parts[0]].get_timestamps()
        for body_part in body_parts:
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
        _ = self.ensure_single_entry()
        key = key or self.fetch1("KEY")
        skeleton = (self & key).fetch_nwb()[0]["skeleton"]
        nodes = skeleton.nodes[:]
        int_edges = skeleton.edges[:]
        named_edges = [[nodes[i], nodes[j]] for i, j in int_edges]
        return {"nodes": nodes, "edges": named_edges}
