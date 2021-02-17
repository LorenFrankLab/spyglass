import os


def check_env():
    """Check whether environment variables have been set properly.
    Raise an exception if not.
    """
    base_dir()
    kachery_storage_dir()


def base_dir():
    """Get the base directory from $NWB_DATAJOINT_BASE_DIR

    Returns:
        str: The base directory
    """
    p = os.getenv('NWB_DATAJOINT_BASE_DIR', None)
    assert p is not None, '''
    You must set the NWB_DATAJOINT_BASE_DIR environment variable.
    You MUST also set the $KACHERY_STORAGE_DIR environment variable
    to be equal to $NWB_DATAJOINT_BASE_DIR/kachery-storage
    '''
    return p


def kachery_storage_dir():
    """Get the kachery storage directory from $KACHERY_STORAGE_DIR
    And verifies that it is equal to $NWB_DATAJOINT_BASE_DIR/kachery-storage
    Raise an exception if not.

    Returns:
        str: The kachery storage directory
    """
    base = base_dir()
    p = os.getenv('KACHERY_STORAGE_DIR', None)
    assert p is not None, '''
    You must set the KACHERY_STORAGE_DIR environment variable.
    And it should be equal to $NWB_DATAJOINT_BASE_DIR/kachery-storage
    '''

    assert p == os.path.join(base, 'kachery-storage'), f'''
    Although KACHERY_STORAGE_DIR is set, it is not equal to $NWB_DATAJOINT_BASE_DIR/kachery-storage

    Current values:
    NWB_DATAJOINT_BASE_DIR={base}
    KACHERY_STORAGE_DIR={p}

    You must update these variables before proceeding
    '''
    if not os.path.exists(p):
        os.mkdir(p)
    return p
