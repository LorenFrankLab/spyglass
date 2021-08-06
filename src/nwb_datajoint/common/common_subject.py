import datajoint as dj

schema = dj.schema('common_subject')


@schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(200)
    ---
    age = NULL: varchar(200)
    description = NULL: varchar(2000)
    genotype = NULL: varchar(2000)
    sex = 'U': enum('M', 'F', 'U')
    species = NULL: varchar(200)
    """
    # TODO increase length of description field (and maybe all fields, really)

    UNKNOWN = 'UNKNOWN'

    @classmethod
    def initialize(cls):
        # initialize with an unknown subject for use when NWB file does not contain a subject
        # TODO: move to initialization script so it doesn't get called every time
        cls.insert1(dict(subject_id=cls.UNKNOWN), skip_duplicates=True)

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Get the subject information from the NWBFile and insert it into the Subject table."""
        # cls.initialize()
        sub = nwbf.subject
        if sub is None:
            print('No subject metadata found.\n')
            return
        subject_dict = dict()
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
        cls.insert1(subject_dict, skip_duplicates=True)
