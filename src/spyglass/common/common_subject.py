import datajoint as dj

schema = dj.schema("common_subject")


@schema
class Subject(dj.Manual):
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
    def insert_from_nwbfile(cls, nwbf, config):
        """Get the subject information from the NWBFile and insert it into the Subject table."""
        sub = nwbf.subject
        if sub is None and "Subject" not in config:
            print('No subject metadata found.\n')
            return

        subject_dict = dict()
        if sub is not None:
            subject_dict['subject_id'] = sub.subject_id
            subject_dict['age'] = sub.age
            subject_dict['description'] = sub.description
            subject_dict['genotype'] = sub.genotype
            if sub.sex in ('Male', 'male', 'M', 'm'):
                sex = 'M'
            elif sub.sex in ('Female', 'female', 'F', 'f'):
                sex = 'F'
            else:
                sex = 'U'
            subject_dict['sex'] = sex
            subject_dict['species'] = sub.species

        if "Subject" in config:
            if "subject_id" not in subject_dict:
                assert "subject_id" in config["Subject"], "Subject.subject_id is required if the file has no Subject"
            subject_dict.update(config["Subject"])  # overwrite from config file
        cls.insert1(subject_dict, skip_duplicates=True)
