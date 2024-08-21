import copy

import datajoint as dj
import pandas as pd
import pynwb

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_imported")


@schema
class ImportedSpikeSorting(SpyglassMixin, dj.Imported):
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

    def make(self, key):
        """Make without transaction

        Allows populate_all_common to work within a single transaction."""
        orig_key = copy.deepcopy(key)

        nwb_file_abs_path = Nwbfile.get_abs_path(key["nwb_file_name"])

        with pynwb.NWBHDF5IO(
            nwb_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            if not nwbfile.units:
                logger.warn("No units found in NWB file")
                return

        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )  # noqa: F401

        key["object_id"] = nwbfile.units.object_id

        self.insert1(key, skip_duplicates=True, allow_direct_insert=True)

        part_name = SpikeSortingOutput._part_name(self.table_name)
        SpikeSortingOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

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
        """class method to fetch the nwb and add annotations to the spike dfs returned"""
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
