try:
    # Try to get H5pyFile from lindi
    # This is the same as h5py.File when file path does not end with .lindi.json
    from lindi import File as H5pyFile  # noqa: F401
except ImportError:
    # Fall back to h5py (no support for .lindi.json files)
    from h5py import File as H5pyFile  # noqa: F401
