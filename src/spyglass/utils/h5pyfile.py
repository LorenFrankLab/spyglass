import os


# If SPYGLASS_LINDI is set to 1, use the lindi library to open NWB files.
# Otherwise, use h5py. Note that even in LINDI mode, if the file path does not
# end with ".lindi.json", then H5pyFile is equivalent to h5py.File. Thus, the
# difference only matters when opening LINDI files.

USE_LINDI = os.environ.get("SPYGLASS_LINDI", "0") == "1"

if USE_LINDI:
    from lindi import File as H5pyFile  # noqa: F401
else:
    from h5py import File as H5pyFile  # noqa: F401
