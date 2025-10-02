import datajoint as dj
import pynwb

from spyglass.utils import SpyglassIngestion, logger

schema = dj.schema("common_subject")


@schema
class Subject(SpyglassIngestion, dj.Manual):
    definition = """
    subject_id: varchar(80)
    ---
    age = NULL: varchar(200)
    description = NULL: varchar(2000)
    genotype = NULL: varchar(2000)
    sex = "U": enum("M", "F", "U")
    species = NULL: varchar(200)
    """
    _expected_duplicates = True

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "subject_id": "subject_id",
                "age": "age",
                "description": "description",
                "genotype": "genotype",
                "species": self.standardized_sex_string,
            }
        }

    @property
    def _source_nwb_object_type(self):
        return pynwb.file.Subject

    @staticmethod
    def standardized_sex_string(subject, warn=True):
        """Takes subject.sex and returns 'M', 'F', or 'U'."""
        sex_field = getattr(subject, "sex", "U") or "U"
        if (sex := sex_field[0].upper()) in ("M", "F", "U"):
            return sex
        elif warn:
            logger.info(
                f"Unrecognized sex identifier {sex_field}, setting to 'U'"
            )
        return "U"

    def _adjust_key_for_entry(self, key):
        """Fill in any NULL values in the key with defaults."""
        # Avoids triggering 'accept_divergence' on reinsert
        ret = key.copy()
        ret["sex"] = self.standardized_sex_string(key, warn=False)
        return super()._adjust_key_for_entry(ret)
