import datajoint as dj
from .common_session import Session

schema = dj.schema("common_dio", locals())


@schema
class Digitalio(dj.Manual):
    definition = """
    -> Session
    dio_label: varchar(80)  # the label for this digital IO port
    ---
    input_port: enum('True', 'False') # is this an input port
    nwb_object_id: varchar(255) # the object identifier for these data
    """
