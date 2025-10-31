import copy

import datajoint as dj
import pandas as pd
import pynwb

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassIngestion, SpyglassMixin, logger

schema = dj.schema("spikesorting_imported")


@schema
class ImportedSpikeSorting(SpyglassIngestion, dj.Imported):
    definition = """
    -> Session
    ---
    object_id: varchar(40)
    """

    _nwb_table = Nwbfile

    class Annotations(SpyglassMixin, dj.Part):
        definition = """
        -> ImportedSpikeSorting
        id: int # unit id, corresponds to dataframe index of unit in NWB file
        ---
        label = Null: longblob # list of string labels for the unit
        annotations: longblob # dict of other annotations (e.g. metrics)
        """

    # SpyglassIngestion properties
    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "object_id": "object_id",
            }
        }

    @property
    def _source_nwb_object_type(self):
        return pynwb.misc.Units

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Override to get units from nwb_file.units."""
        if not getattr(nwb_file, "units", None):
            logger.warn("No units found in NWB file")
            return []
        return [nwb_file.units]

    def insert_from_nwbfile(
        self,
        nwb_file_name: str,
        config: dict = None,
        dry_run: bool = False,
    ):
        """Override base method to add merge table integration."""
        # Call the base implementation first
        result = super().insert_from_nwbfile(nwb_file_name, config, dry_run)

        if dry_run or not result or self not in result:
            return result

        # Add merge table integration
        orig_key = {"nwb_file_name": nwb_file_name}

        from spyglass.spikesorting.spikesorting_merge import (  # noqa: F401
            SpikeSortingOutput,
        )

        part_name = SpikeSortingOutput._part_name(self.table_name)
        SpikeSortingOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

        return result

    def make(self, key):
        """Legacy make method - replaced by insert_from_nwbfile.

        Kept for backward compatibility during migration.
        """
        # Call the new SpyglassIngestion method
        from spyglass.common.common_usage import ActivityLog

        ActivityLog().deprecate_log(
            self, "ImportedSpikesorting.make", alt="insert_from_nwbfile"
        )

        return self.insert_from_nwbfile(key["nwb_file_name"])

    # ------------ Placeholder methods for merge table integration ------------
    @classmethod
    def get_recording(cls, key):
        """Placeholder for merge table to call on all sources."""
        raise NotImplementedError(
            "Imported spike sorting does not have a `get_recording` method"
        )

    @classmethod
    def get_sorting(cls, key):
        """Placeholder for merge table to call on all sources."""
        raise NotImplementedError(
            "Imported spike sorting does not have a `get_sorting` method"
        )

    # --------------------------- Annotation methods ---------------------------
    def add_annotation(
        self, key, id, label=[], annotations={}, merge_annotations=False
    ):
        """Manually add annotations to the spike sorting output

        Parameters
        ----------
        key : dict
            restriction key for ImportedSpikeSorting
        id : int
            unit id
        label : List[str], optional
            list of str labels for the unit, by default None
        annotations : dict, optional
            dictionary of other annotation values for unit, by default None
        merge_annotations : bool, optional
            whether to merge with existing annotations, by default False
        """
        if isinstance(label, str):
            label = [label]
        query = self & key
        if not len(query) == 1:
            raise ValueError(
                f"ImportedSpikeSorting key must be unique. Found: {query}"
            )
        unit_key = {**key, "id": id}
        annotation_query = ImportedSpikeSorting.Annotations & unit_key
        if annotation_query and not merge_annotations:
            raise ValueError(
                f"Unit already has annotations: {annotation_query}"
            )
        elif annotation_query:
            existing_annotations = annotation_query.fetch1()
            existing_annotations["label"] += label
            existing_annotations["annotations"].update(annotations)
            self.Annotations.update1(existing_annotations)
        else:
            self.Annotations.insert1(
                dict(unit_key, label=label, annotations=annotations),
                skip_duplicates=True,
            )

    def make_df_from_annotations(self):
        """Convert the annotations part table into a dataframe that can be
        concatenated to the spikes dataframe in the nwb file."""
        df = []
        for id, label, annotations in zip(
            *self.Annotations.fetch("id", "label", "annotations")
        ):
            df.append(
                dict(
                    id=id,
                    label=label,
                    **annotations,
                )
            )
        df = pd.DataFrame(df)
        df.set_index("id", inplace=True)
        return df

    def fetch_nwb(self, *attrs, **kwargs):
        """Fetch the nwb and add annotations to the spike dfs returned"""
        # get the original nwbs
        nwbs = super().fetch_nwb(*attrs, **kwargs)
        # for each nwb, get the annotations and add them to the spikes dataframe
        for i, key in enumerate(self.fetch("KEY")):
            if not ImportedSpikeSorting.Annotations & key:
                continue
            # make the annotation_df
            annotation_df = (self & key).make_df_from_annotations()
            # concatenate the annotations to the spikes dataframe in the returned nwb
            nwbs[i]["object_id"] = pd.concat(
                [nwbs[i]["object_id"], annotation_df], axis="columns"
            )
        return nwbs
