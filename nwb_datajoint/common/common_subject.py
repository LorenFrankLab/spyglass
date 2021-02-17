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

    def insert_from_nwbfile(self, nwbf):
        """Get the subject information from the NWBFile and insert it into the Subject table."""
        sub = nwbf.subject
        subject_dict = dict()
        subject_dict['subject_id'] = sub.subject_id
        if sub.age is None:
            subject_dict['age'] = 'unknown'
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
