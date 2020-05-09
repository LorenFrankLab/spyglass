# Test of automatic datajoint schema generation from NWB file
import datajoint as dj
import pynwb

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

    def __init__(self, *args):
        super().__init__(*args)  # call the base implementation

    def insert_from_nwb(self, nwb_file_name):
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            io.close()
            return
        #get the subject information and create a dictionary from it
        sub = nwbf.subject
        subject_dict = dict()
        subject_dict['subject_id'] = sub.subject_id
        if sub.age is None:
            subject_dict['age'] = 'unknown'

        subject_dict['description'] = sub.description
        subject_dict['genotype'] = sub.genotype
        sex = 'U'
        if (sub.sex == 'Male'):
            sex = 'M'
        elif (sub.sex == 'Female'):
            sex = 'F'
        subject_dict['sex'] = sex
        subject_dict['species'] = sub.species
        self.insert1(subject_dict,skip_duplicates=True)
        io.close()


