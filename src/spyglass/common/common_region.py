import datajoint as dj

schema = dj.schema("common_region")


@schema
class BrainRegion(dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name: varchar(200)             # the name of the brain region
    subregion_name=NULL: varchar(200)     # subregion name
    subsubregion_name=NULL: varchar(200)  # subregion within subregion
    """

    # TODO consider making (region_name, subregion_name, subsubregion_name) a primary key
    # subregion_name='' and subsubregion_name='' will be necessary but that seems OK

    @classmethod
    def fetch_add(cls, region_name, subregion_name=None, subsubregion_name=None):
        """Return the region ID for the given names, and if no match exists, first add it to the BrainRegion table.

        The combination of (region_name, subregion_name, subsubregion_name) is effectively unique, then.

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
        key = dict()
        key["region_name"] = region_name
        key["subregion_name"] = subregion_name
        key["subsubregion_name"] = subsubregion_name
        query = BrainRegion & key
        if not query:
            cls.insert1(key)
            query = BrainRegion & key
        return query.fetch1("region_id")
