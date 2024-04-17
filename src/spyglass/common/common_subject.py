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
    def insert_from_nwbfile(cls, nwbf, config):
        """Get the subject info from the NWBFile, insert into the Subject.

        Parameters
        ----------
        nwbf: pynwb.NWBFile
            The NWB file with subject information.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        subject_id : string
            The id of the subject found in the NWB or config file, or None.
        """
        if "Subject" not in config and nwbf.subject is None:
            logger.warn("No subject metadata found.\n")
            return None

        conf = config["Subject"][0] if "Subject" in config else dict()
        sub = (
            nwbf.subject
            if nwbf.subject is not None
            else type("DefaultObject", (), {})()
        )

        subject_dict = dict()
        subject_dict["subject_id"] = (
            conf["subject_id"]
            if "subject_id" in conf
            else sub.subject_id if hasattr(sub, "subject_id") else None
        )
        subject_dict["age"] = (
            conf["age"]
            if "age" in conf
            else sub.age if hasattr(sub, "age") else None
        )
        subject_dict["description"] = (
            conf["description"]
            if "description" in conf
            else sub.description if hasattr(sub, "description") else None
        )
        subject_dict["genotype"] = (
            conf["genotype"]
            if "genotype" in conf
            else sub.genotype if hasattr(sub, "genotype") else None
        )
        sex = (
            conf["sex"]
            if "sex" in conf
            else sub.sex if hasattr(sub, "sex") else None
        )
        if sex in ("Male", "male", "M", "m"):
            sex = "M"
        elif sex in ("Female", "female", "F", "f"):
            sex = "F"
        else:
            sex = "U"
        subject_dict["sex"] = sex
        subject_dict["species"] = (
            conf["species"]
            if "species" in conf
            else sub.species if hasattr(sub, "species") else None
        )
        cls.insert1(subject_dict, skip_duplicates=True)
        return subject_dict["subject_id"]
