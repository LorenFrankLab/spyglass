import datajoint as dj
import warnings

from .errors import PopulateException

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
            print("No subject metadata found.\n")
            return None

        subject_dict = dict()
        if sub is not None:
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

        if "Subject" in config:
            if "subject_id" not in subject_dict:
                assert (
                    "subject_id" in config["Subject"]
                ), "Subject.subject_id is required if the file has no Subject"
            subject_dict.update(config["Subject"])  # overwrite from config file

        cls._add_subject(subject_dict)

        return subject_dict["subject_id"]

    @classmethod
    def _add_subject(cls, subject_dict):
        """Prompt the user to create a Subject entity using the given data if the entity does not already exist.

        If the entity does not exist and the user chooses to create a new entity, then the entity will be created.

        If the entity does not exist and the user chooses not to create a new entity, then a PopulateException will
        be raised.

        If the entity already exists and it differs from the given data, then a warning will be raised and
        nothing else will happen.

        If the entity already exists and it equals the given data, then nothing will happen.
        """
        subject_id = subject_dict["subject_id"]
        if subject_id not in cls.fetch("subject_id"):
            print(
                f"The subject ID '{subject_id}' was not found in the database. "
                f"Existing subject IDs: {cls.fetch('subject_id').tolist()}.\n"
                "Please ensure that the Subject you want to create does not already "
                "exist in the database under a different name or spelling. "
                "If you want to use an existing Subject in the database, "
                "please specify that Subject's subject ID in the config YAML or "
                "change the subject ID in the NWB file. "
                "Entering 'N' will raise an exception."
            )
            val = input(
                f"Do you want to create a new Subject with properties: {subject_dict}? (y/N)"
            )
            if val.lower() in ["y", "yes"]:
                cls.insert1(subject_dict)
            else:
                raise PopulateException(
                    f"User chose not to create a new Subject with properties '{subject_dict}'."
                )
        else:
            if subject_dict not in cls:
                existing_subject = (cls & {"subject_id": subject_id}).fetch1()
                warnings.warn(
                    f"Subject exists in database with subject ID '{subject_id}' but its properties:\n"
                    f"{existing_subject}\n"
                    "differ from the properties read from the file:\n"
                    f"{subject_dict}"
                )
        return subject_dict
