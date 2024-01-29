"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as table chains, which will be used to
determine which features are used, how often, and by whom. This will help 
plan future development of Spyglass.
"""

import datajoint as dj

schema = dj.schema("common_usage")


@schema
class CautiousDelete(dj.Manual):
    definition = """
    id: int auto_increment
    ---
    dj_user: varchar(64)
    duration: float
    origin: varchar(64)
    restriction: varchar(64)
    merge_deletes = null: blob
    """
