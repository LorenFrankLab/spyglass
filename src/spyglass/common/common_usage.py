"""A schema to store the usage of advanced Spyglass features.

Records show usage of features such as cautious delete and fault-permitting
insert, which will be used to
determine which features are used, how often, and by whom. This will help
plan future development of Spyglass.
"""

import datajoint as dj

from spyglass.utils import logger

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


@schema
class ActivityLog(dj.Manual):
    """A log of suspected low-use features worth deprecating."""

    definition = """
    id: int auto_increment
    ---
    function: varchar(64)
    dj_user: varchar(64)
    timestamp=CURRENT_TIMESTAMP: timestamp
    """

    @classmethod
    def log(cls, name):
        logger.warning(f"Function scheduled for deprecation: {name}")
        cls.insert1(dict(dj_user=dj.config["database.user"], function=name))
