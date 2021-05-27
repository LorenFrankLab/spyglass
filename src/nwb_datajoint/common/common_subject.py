import datajoint as dj

schema = dj.schema("common_subject")


@schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(80)
    ---
    age = NULL: varchar(80)
    description = NULL: varchar(80)
    genotype = NULL: varchar(80)
    sex = 'U': enum('M', 'F', 'U')
    species = NULL: varchar(80)
    """

    def insert_from_nwbfile(self, nwbf):
        """Get the subject information from the NWBFile and insert it into the Subject table."""
        sub = nwbf.subject
        subject_dict = dict()
        subject_dict['subject_id'] = sub.subject_id
        subject_dict['age'] = sub.age
        subject_dict['description'] = sub.description
        subject_dict['genotype'] = sub.genotype
        if sub.sex == 'Male':
            sex = 'M'
        elif sub.sex == 'Female':
            sex = 'F'
        else:
            sex = 'U'
        subject_dict['sex'] = sex
        subject_dict['species'] = sub.species
        self.insert1(subject_dict, skip_duplicates=True)
