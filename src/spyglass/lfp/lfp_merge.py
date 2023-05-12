import datajoint as dj
from spyglass.lfp.v1 import LFP, ImportedLFP

schema = dj.schema("lfp")


@schema
class LFPOutput(dj.Manual):
    definition = """
    lfp_id: uuid
    """

    class LFP(dj.Part):
        definition = """
        -> LFPOutput
        -> LFP
        """

    class ImportedLFP(dj.Part):
        definition = """
        -> LFPOutput
        -> ImportedLFP
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
            return LFP & lfp_object.fetch("KEY")
        else:
            return ImportedLFP & (LFPOutput.ImportedLFP & key).fetch("KEY")
