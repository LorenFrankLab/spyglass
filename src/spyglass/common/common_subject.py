import datajoint as dj
from pynwb import NWBFile

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
    def insert_from_nwbfile(cls, nwbf: NWBFile, config: dict = None):
        """Get the subject info from the NWBFile, insert into the Subject.

        Parameters
        ----------
        nwbf: pynwb.NWBFile
            The NWB file with subject information.
        config : dict, optional
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        subject_id : string
            The id of the subject found in the NWB or config file, or None.
        """
        config = config or dict()
        if "Subject" not in config and nwbf.subject is None:
            logger.warning("No subject metadata found.\n")
            return None

        conf = config["Subject"][0] if "Subject" in config else dict()
        sub = (
            nwbf.subject
            if nwbf.subject is not None
            else type("DefaultObject", (), {})()
        )
        subject_dict = {
            field: conf.get(field, getattr(sub, field, None))
            for field in [
                "subject_id",
                "age",
                "description",
                "genotype",
                "species",
                "sex",
            ]
        }
        if (sex := subject_dict["sex"][0].upper()) in ("M", "F"):
            subject_dict["sex"] = sex
        else:
            subject_dict["sex"] = "U"

        cls.insert1(subject_dict, skip_duplicates=True)
        return subject_dict["subject_id"]
