import datajoint as dj
from spyglass.lfp.v1 import LFPV1, ImportedLFPV1

schema = dj.schema("lfp")


@schema
class LFPOutput(dj.Manual):
    definition = """
    lfp_id: uuid
    ---
    stream: varchar(40)
    """

    class LFPV1(dj.Part):
        definition = """
        -> LFPOutput
        ---
        -> LFPV1
        """

    class ImportedLFP(dj.Part):
        definition = """
        -> LFPOutput
        ---
        -> ImportedLFPV1
        """

    @staticmethod
    def get_lfp_object(key: dict):
        """Returns the lfp object corresponding to the key

        Parameters
        ----------
        key : dict
            A dictionary containing some combination of
                                    uuid,
                                    nwb_file_name,
                                    lfp_electrode_group_name,
                                    interval_list_name,
                                    fir_filter_name

        Returns
        -------
        lfp_object
            The entry or entries in the LFPOutput part table that corresponds to the key
        """
        # first check if this returns anything from the LFP table
        lfp_object = LFPOutput.LFP & key
        if lfp_object is not None:
            return LFPV1 & lfp_object.fetch("KEY")
        else:
            return ImportedLFPV1 & (LFPOutput.ImportedLFP & key).fetch("KEY")
