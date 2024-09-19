from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Union

import h5py
import numpy as np


class NwbfileHasher:
    def __init__(self, path: Union[str, Path], data_limit: int = 4095):
        """Hashes the contents of an NWB file, limiting to partial data.

        In testing, chunking the data for large datasets caused false positives
        in the hash comparison, and some datasets may be too large to store in
        memory. This method limits the data to the first N elements to avoid
        this issue, and may not be suitable for all datasets.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the NWB file.
        data_limit : int, optional
            Limit of data to hash for large datasets, by default 4095.
        """
        self.file = h5py.File(path, "r")
        self.data_limit = data_limit

    def collect_names(self, file):
        """Collects all object names in the file."""

        def collect_items(name, obj):
            items_to_process.append((name, obj))

        items_to_process = []
        file.visititems(collect_items)
        items_to_process.sort(key=lambda x: x[0])
        return items_to_process

    def serialize_attr_value(self, value):
        """Serializes an attribute value into bytes for hashing.

        Setting all numpy array types to string avoids false positives.

        Parameters
        ----------
        value : Any
            Attribute value.

        Returns
        -------
        bytes
            Serialized bytes of the attribute value.
        """
        if isinstance(value, np.ndarray):
            return value.astype(str).tobytes()
        elif isinstance(value, (str, int, float)):
            return str(value).encode()
        return repr(value).encode()  # For other data types, use repr

    @cached_property
    def hash(self) -> str:
        """Hashes the NWB file contents, limiting to partal data where large."""
        hashed = md5("".encode())
        for name, obj in self.collect_names(self.file):
            if isinstance(obj, h5py.Dataset):  # hash the dataset name and shape
                hashed.update(str(obj.shape).encode())
                hashed.update(str(obj.dtype).encode())
                partial_data = (  # full if scalar dataset, else use data_limit
                    obj[()] if obj.shape == () else obj[: self.data_limit]
                )
                hashed.update(self.serialize_attr_value(partial_data))
            for attr_key in sorted(obj.attrs):
                attr_value = obj.attrs[attr_key]
                hashed.update(attr_key.encode())
                hashed.update(self.serialize_attr_value(attr_value))
        self.file.close()
        return hashed.hexdigest()
