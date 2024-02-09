"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
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
    restriction: varchar(255)
    merge_deletes = null: blob
    """


@schema
class InsertError(dj.Manual):
    definition = """
    id: int auto_increment
    ---
    dj_user: varchar(64)
    connection_id: int     # MySQL CONNECTION_ID()
    nwb_file_name: varchar(64)
    table: varchar(64)
    error_type: varchar(64)
    error_message: varchar(255)
    error_raw = null: blob
    """
