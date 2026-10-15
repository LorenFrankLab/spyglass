import datajoint as dj
import ndx_pose
import numpy as np
import pandas as pd

from spyglass.common import IntervalList, Nwbfile
from spyglass.utils.dj_mixin import SpyglassIngestion
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_valid_intervals,
)

schema = dj.schema("position_v1_imported_pose")


def _named_edges(bodyparts, edges_arr):
    """Convert integer-index skeleton edges to name pairs.

    Parameters
    ----------
    bodyparts : list of str
        Ordered node names; edge indices refer to positions in this list.
    edges_arr : array-like or None
        Iterable of ``(i, j)`` integer index pairs, or None when the
        skeleton defines no edges.

    Returns
    -------
    list of tuple of str
        ``(bodyparts[i], bodyparts[j])`` for each edge; empty when
        ``edges_arr`` is None.
    """
    if edges_arr is None:
        return []
    return [
        (bodyparts[int(edge[0])], bodyparts[int(edge[1])]) for edge in edges_arr
    ]


@schema
class ImportedPose(SpyglassIngestion, dj.Manual):
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

    class BodyPart(SpyglassIngestion, dj.Part):
        definition = """
        -> master
        part_name: varchar(80)
        ---
        part_object_id: varchar(80)
        """

        table_key_to_obj_attr = {"self": {"part_object_id": "object_id"}}

    _source_nwb_object_type = ndx_pose.PoseEstimation

    table_key_to_obj_attr = {
        "self": {"pose_object_id": "object_id"},
        "skeleton": {"skeleton_object_id": "object_id"},
    }

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Generate the interval, the pose entry, and one entry per body part.

        The interval comes first: it is this table's parent, derived from the
        timestamps of the object's first body part.
        """
        nwb_file_name = base_key["nwb_file_name"]

        # use the timestamps from the first body part to define valid times
        timestamps = list(nwb_obj.pose_estimation_series.values())[
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
            "interval_list_name": f"pose_{nwb_obj.name}_valid_intervals",
        }
        part_attr = self.BodyPart.table_key_to_obj_attr["self"]

        return {
            IntervalList: [
                {
                    **interval_pk,
                    "valid_times": valid_intervals,
                    "pipeline": "ImportedPose",
                }
            ],
            **super().generate_entries_from_nwb_object(nwb_obj, interval_pk),
            self.BodyPart: [
                dict(
                    interval_pk,
                    part_name=part,
                    **{k: getattr(part_obj, v) for k, v in part_attr.items()},
                )
                for part, part_obj in nwb_obj.pose_estimation_series.items()
            ],
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "ImportedPose.make is deprecated. Use insert_from_nwbfile."
        )

    def insert_from_nwbfile(
        self, nwb_file_name, config=None, dry_run=False, import_to_v2=False
    ):
        """Ingest all ndx-pose PoseEstimation objects from a registered NWB.

        Parameters
        ----------
        nwb_file_name : str
            Spyglass-registered NWB filename (must exist in Nwbfile table).
        config : dict, optional
            A configuration dictionary to supplement NWB data. Default None.
        dry_run : bool, optional
            If True, do not insert into the database, just return the
            entries that would be inserted. Default False.
        import_to_v2 : bool, optional
            When True, also register skeleton graph(s) from the NWB in the V2
            ``Skeleton`` table.  ndx-pose files hold pose *results*, not
            trained model weights, so only the skeleton metadata belongs in V2.
            By default False.
        """
        entries = super().insert_from_nwbfile(nwb_file_name, config, dry_run)

        if entries and not dry_run and import_to_v2:
            file_path = Nwbfile().get_abs_path(nwb_file_name)
            self._import_to_v2_pipeline(file_path, nwb_file_name)

        return entries

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
        from pynwb import NWBHDF5IO

        try:
            with NWBHDF5IO(str(file_path), mode="r") as io:
                nwbf = io.read()
                if nwbf.processing.get("behavior") is None:
                    return
                skeletons = [
                    obj
                    for obj in nwbf.objects.values()
                    if isinstance(obj, ndx_pose.Skeleton)
                ]
                for skeleton_obj in skeletons:
                    self._insert_v2_skeleton(skeleton_obj, nwb_file_name)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"V2 skeleton registration failed for '{nwb_file_name}': "
                f"{exc}. ImportedPose entries were inserted successfully.",
                stacklevel=3,
            )

    @staticmethod
    def _insert_v2_skeleton(skeleton_obj, nwb_file_name):
        """Register one ndx-pose Skeleton in the V2 ``Skeleton`` table.

        A failed insert is logged as a warning rather than raised, so that a
        single bad skeleton does not abort registration of the others or roll
        back the primary ``ImportedPose`` ingestion.

        Parameters
        ----------
        skeleton_obj : ndx_pose.Skeleton
            Skeleton graph object read from the ndx-pose NWB file.
        nwb_file_name : str
            Spyglass-registered NWB filename (used only for logging).
        """
        import warnings

        from spyglass.position.v2.train import Skeleton

        bodyparts = list(skeleton_obj.nodes)
        edges = _named_edges(bodyparts, skeleton_obj.edges)
        try:
            Skeleton().insert1(
                {"bodyparts": bodyparts, "edges": edges},
                accept_default=True,
                skip_duplicates=True,
            )
        except Exception as sk_exc:  # noqa: BLE001
            warnings.warn(
                f"V2 Skeleton insert failed for '{skeleton_obj.name}' "
                f"in '{nwb_file_name}': {sk_exc}",
                stacklevel=4,
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
