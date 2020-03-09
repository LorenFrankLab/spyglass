# Frank lab schema: lab name and members
import datajoint as dj

schema = dj.schema("common_lab", locals())


@schema
class LabMember(dj.Lookup):
    definition = """
    lab_member_name: varchar(80)
    ---
    first_name: varchar(80)
    last_name: varchar(80)
    """


@schema
class Institution(dj.Manual):
    definition = """
    institution_name: varchar(80)
    ---
    """


@schema
class Lab(dj.Manual):
     definition = """
    lab_name: varchar(80)
    ---
    """



