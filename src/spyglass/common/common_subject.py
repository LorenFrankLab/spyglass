import datajoint as dj

from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("common_subject")


@schema
class Subject(SpyglassMixin, dj.Manual):
    definition = """
    subject_id: varchar(80)
    ---
    age = NULL: varchar(200)
    description = NULL: varchar(2000)
    genotype = NULL: varchar(2000)
    sex = "U": enum("M", "F", "U")
    species = NULL: varchar(200)
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Get the subject info from the NWBFile, insert into the Subject."""
        sub = nwbf.subject
        if sub is None:
            logger.warn("No subject metadata found.\n")
            return
        subject_dict = dict()
        subject_dict["subject_id"] = sub.subject_id
        subject_dict["age"] = sub.age
        subject_dict["description"] = sub.description
        subject_dict["genotype"] = sub.genotype
        if sub.sex in ("Male", "male", "M", "m"):
            sex = "M"
        elif sub.sex in ("Female", "female", "F", "f"):
            sex = "F"
        else:
            sex = "U"
        subject_dict["sex"] = sex
        subject_dict["species"] = sub.species
        cls.insert1(subject_dict, skip_duplicates=True)
