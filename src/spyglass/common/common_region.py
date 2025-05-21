import datajoint as dj

from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("common_region")


@schema
class BrainRegion(SpyglassMixin, dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name: varchar(200)             # the name of the brain region
    subregion_name=NULL: varchar(200)     # subregion name
    subsubregion_name=NULL: varchar(200)  # subregion within subregion
    """

    @classmethod
    def fetch_add(
        cls,
        region_name: str,
        subregion_name: str = None,
        subsubregion_name: str = None,
    ):
        """Return the region ID for names. If no match, add to the BrainRegion.

        The combination of (region_name, subregion_name, subsubregion_name) is
        effectively unique, then.

        Parameters
        ----------
        region_name : str
            The name of the brain region.
        subregion_name : str, optional
            The name of the subregion within the brain region.
        subsubregion_name : str, optional
            The name of the subregion within the subregion.

        Returns
        -------
        region_id : int
            The index of the region in the BrainRegion table.
        """
        key = dict(
            region_name=region_name,
            subregion_name=subregion_name,
            subsubregion_name=subsubregion_name,
        )
        query = BrainRegion & key
        if not query:
            cls.insert1(key)
            query = BrainRegion & key
        return query.fetch1("region_id")
