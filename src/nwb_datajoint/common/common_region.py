import datajoint as dj

schema = dj.schema("common_region")


@schema
class BrainRegion(dj.Lookup):
    definition = """
    region_id: smallint auto_increment
    ---
    region_name=NULL: varchar(80)        # the name of the brain region
    subregion_name=NULL: varchar(80)     # subregion name
    subsubregion_name=NULL: varchar(80)  # subregion within subregion
    """

    @staticmethod
    def fetch_add(region_name):
        """Return the region ID for the given region name, and if it does not exist, first add the region."""
        key = dict()
        key['region_name'] = region_name
        query = BrainRegion & key
        if len(query) == 0:
            BrainRegion.insert1(key)
            query = BrainRegion & key
        return query.fetch1('region_id')
