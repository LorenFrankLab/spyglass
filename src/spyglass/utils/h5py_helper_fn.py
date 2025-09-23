from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np


def is_float16_dtype(
    dt: np.dtype, *, include_subarray_base: bool = False
) -> bool:
    """Return True if dtype is a plain float16 (any endianness).

    Parameters
    ----------
    dt : numpy.dtype
        Dtype to test.
    include_subarray_base : bool, optional
        If True, treat subarray dtypes with base float16 as float16.
        If False (default), exclude subarray dtypes.

    Returns
    -------
    bool
        True if dtype is float16 under the chosen policy; otherwise False.

    Notes
    -----
    Excludes:
    - Object/region references
    - Variable-length (vlen) types
    - Compound dtypes
    - Subarray dtypes unless `include_subarray_base=True`
    """
    # Exclude references and vlen types
    if h5py.check_dtype(ref=dt) is not None:
        return False
    if h5py.check_dtype(vlen=dt) is not None:
        return False

    # Exclude compound
    if dt.names is not None:
        return False

    # Handle subarray dtypes (e.g., (M, N)<f2)
    if dt.subdtype is not None:
        if not include_subarray_base:
            return False
        base, _ = dt.subdtype
        dt = np.dtype(base)

    # Match binary16 (IEEE 754 half, <f2/>f2/=f2)
    return dt.kind == "f" and dt.itemsize == 2


def find_float16_datasets(
    file_path: str,
    *,
    include_dimension_scales: bool = False,
    include_subarray_base: bool = False,
) -> List[str]:
    """List absolute HDF5 paths of datasets stored as float16.

    Parameters
    ----------
    file_path : str
        Path to the HDF5/NWB file.
    include_dimension_scales : bool, optional
        If False (default), skip datasets that are HDF5 dimension scales
        (CLASS='DIMENSION_SCALE').
    include_subarray_base : bool, optional
        If True, include subarray dtypes whose base is float16.

    Returns
    -------
    list of str
        Absolute dataset paths with dtype float16 under the chosen policy.
    """
    hits: List[str] = []

    def _visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return

        if not is_float16_dtype(
            obj.dtype, include_subarray_base=include_subarray_base
        ):
            return

        if not include_dimension_scales:
            cls = obj.attrs.get("CLASS", None)
            if (
                isinstance(cls, (bytes, bytearray))
                and cls == b"DIMENSION_SCALE"
            ) or (isinstance(cls, str) and cls == "DIMENSION_SCALE"):
                return

        path = f"/{name}" if not name.startswith("/") else name
        hits.append(path)

    with h5py.File(file_path, "r") as f:
        f.visititems(_visit)

    return hits


def convert_dataset_type(file: h5py.File, dataset_path: str, target_dtype: str):
    """Convert a dataset to a different dtype 'in place' (-ish).

    Parameters
    ----------
    file : h5py.File
        Open HDF5 file handle with write access.
    dataset_path : str
        Absolute path of the dataset to convert.
    target_dtype : str
        Target dtype (e.g., 'float32', 'int16', etc.)
    """
    dset = file[dataset_path]
    data = dset[()]  # loads into memory
    attrs = dict(dset.attrs.items())
    creation_kwargs = dict(
        chunks=dset.chunks,
        compression=dset.compression,
        compression_opts=dset.compression_opts,
        shuffle=dset.shuffle,
        fletcher32=dset.fletcher32,
        scaleoffset=dset.scaleoffset,
        fillvalue=dset.fillvalue,
    )

    del file[dataset_path]
    new_dset = file.create_dataset(
        dataset_path,
        data=np.asarray(data, dtype=target_dtype),
        dtype=target_dtype,
        **creation_kwargs,
    )
    for k, v in attrs.items():
        new_dset.attrs[k] = v


def find_dynamic_tables_missing_id(nwb_path: str | Path) -> List[str]:
    """Return DynamicTable paths that do not contain an 'id' dataset.

    The check is intentionally minimal:
    a DynamicTable group is considered GOOD if and only if the key 'id'
    exists directly under that group. Otherwise, it's BAD and included
    in the returned list.

    Parameters
    ----------
    nwb_path : str | Path
        Path to the NWB (HDF5) file.

    Returns
    -------
    List[str]
        Sorted list of absolute HDF5 paths to DynamicTable groups that
        are missing the 'id' key.

    Notes
    -----
    A group is detected as a DynamicTable if its attribute
    ``neurodata_type`` equals ``"DynamicTable"`` (bytes or str).
    """
    nwb_path = Path(nwb_path)

    def _attr_str(obj: h5py.Group, key: str) -> Optional[str]:
        if key not in obj.attrs:
            return None
        val = obj.attrs[key]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        return str(val)

    bad_paths: List[str] = []

    with h5py.File(nwb_path, "r") as f:

        def _visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Group):
                ndt = _attr_str(obj, "neurodata_type")
                if ndt == "DynamicTable":
                    # Minimal check: only presence of 'id' key
                    if "id" not in obj.keys():
                        bad_paths.append(f"/{name}")

        f.visititems(_visitor)

    bad_paths.sort()
    return bad_paths


def add_id_column_to_table(file: h5py.File, dataset_path: str):
    """Add an 'id' column to a DynamicTable dataset if it doesn't exist.

    The 'id' column will be populated with sequential integers starting from 0.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    dataset_path : str
        The path to the DynamicTable dataset within the HDF5 file.

    Raises
    ------
    ValueError
        If the specified dataset_path does not exist or is not a DynamicTable.
    """
    if dataset_path not in file:
        raise ValueError(
            f"Dataset path '{dataset_path}' does not exist in the file."
        )

    obj = file[dataset_path]
    if (
        "neurodata_type" not in obj.attrs
        or obj.attrs["neurodata_type"] != "DynamicTable"
    ):
        raise ValueError(
            f"The group at '{dataset_path}' is not a DynamicTable."
        )

    if "id" in obj.keys():
        print(
            f"'id' column already exists in '{dataset_path}'. No action taken."
        )
        return

    # Determine the number of rows from an existing column
    sample_column = obj[list(obj.keys())[0]]
    n_rows = sample_column.shape[0]
    # Create the 'id' dataset
    data = np.arange(n_rows, dtype=np.int64)
    id_ds = obj.create_dataset(
        "id", data=data, compression="gzip", compression_opts=4
    )
    id_ds.attrs["neurodata_type"] = "ElementIdentifiers"
    id_ds.attrs["namespace"] = "core"

    # Good practice to include object_id if missing
    if "object_id" not in id_ds.attrs:
        id_ds.attrs["object_id"] = str(uuid.uuid4())
