import os


def check_env():
    """Check whether environment variables have been set properly.
    Raise an exception if not.
    """
    base_dir()


def base_dir():
    """Get the base directory from $SPYGLASS_BASE_DIR

    Returns:
        str: The base directory
    """
    p = os.getenv("SPYGLASS_BASE_DIR", None)
    assert (
        p is not None
    ), """
    You must set the SPYGLASS_BASE_DIR environment variable.
    """
    return p
