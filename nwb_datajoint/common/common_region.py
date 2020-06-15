import datajoint as dj

schema = dj.schema("common_region", locals())


@schema
class BrainRegion(dj.Lookup):
    definition = """
    region_id  :  smallint auto_increment
    ---
    region_name='': varchar(80)      # the name of the brain region
    subregion_name='': varchar(80)   # subregion name
    subsubregion_name='': varchar(80) #subregion within subregion
    """
