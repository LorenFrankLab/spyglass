import uuid


def generate_nwb_uuid(nwb_file_name: str, initial: str, len_uuid: int = 6):
    """Generates a unique identifier related to an NWB file.

    Parameters
    ----------
    nwb_file_name : str
        _description_
    initial : str
        R if recording; A if artifact; S if sorting etc
    len_uuid : int
        how many digits of uuid4 to keep
    """
    uuid4 = str(uuid.uuid4())
    nwb_uuid = nwb_file_name + "_" + initial + "_" + uuid4[:len_uuid]
    return nwb_uuid
