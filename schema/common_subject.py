# Test of automatic datajoint schema generation from NWB file
import datajoint as dj

schema = dj.schema("common_subject")


@schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(80)
    ---
    age: varchar(80)
    description: varchar(80)
    genotype: varchar(80)
    sex: enum('M', 'F', 'U')
    species: varchar(80)
    """
