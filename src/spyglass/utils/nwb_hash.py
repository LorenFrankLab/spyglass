import atexit
from hashlib import md5
from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 4095


def hash_directory(directory_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
    """Generate a hash of the contents of a directory, recursively.

    Searches though all files in the directory and subdirectories, hashing
    the contents of files. nwb files are hashed with the NwbfileHasher class.

    Parameters
    ----------
    directory_path : str
        Path to the directory to hash.
    batch_size : int, optional
        Limit of data to hash for large files, by default 4095.
    """
    hash_obj = md5()

    for file_path in sorted(Path(directory_path).rglob("*")):
        if not file_path.is_file():  # Only hash files, not directories
            continue
        if file_path.suffix == ".nwb":
            hasher = NwbfileHasher(file_path, batch_size=batch_size)
            hash_obj.update(hasher.hash.encode())
            continue
        with file_path.open("rb") as f:
            while chunk := f.read(batch_size):
                hash_obj.update(chunk)
        # update with the rel path to for same file in diff dirs
        hash_obj.update(str(file_path.relative_to(directory_path)).encode())

    return hash_obj.hexdigest()  # Return the hex digest of the hash


class NwbfileHasher:
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        verbose: bool = True,
    ):
        """Hashes the contents of an NWB file, limiting to partial data.

        In testing, chunking the data for large datasets caused false positives
        in the hash comparison, and some datasets may be too large to store in
        memory. This method limits the data to the first N elements to avoid
        this issue, and may not be suitable for all datasets.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the NWB file.
        batch_size  : int, optional
            Limit of data to hash for large datasets, by default 4095.
        verbose : bool, optional
            Display progress bar, by default True.
        """
        self.file = h5py.File(path, "r")
        atexit.register(self.cleanup)

        self.batch_size = batch_size
        self.verbose = verbose
        self.hashed = md5("".encode())
        self.hash = self.compute_hash()

        self.cleanup()
        atexit.unregister(self.cleanup)

    def cleanup(self):
        self.file.close()

    def collect_names(self, file):
        """Collects all object names in the file."""

        def collect_items(name, obj):
            items_to_process.append((name, obj))

        items_to_process = []
        file.visititems(collect_items)
        items_to_process.sort(key=lambda x: x[0])
        return items_to_process

    def serialize_attr_value(self, value: Any):
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
            return value.astype(str).tobytes()  # Try with and without `str`
        elif isinstance(value, (str, int, float)):
            return str(value).encode()
        return repr(value).encode()  # For other data types, use repr

    def hash_dataset(self, dataset: h5py.Dataset):
        _ = self.hash_shape_dtype(dataset)

        if dataset.shape == ():
            self.hashed.update(self.serialize_attr_value(dataset[()]))
            return

        size = dataset.shape[0]
        start = 0

        while start < size:
            end = min(start + self.batch_size, size)
            self.hashed.update(self.serialize_attr_value(dataset[start:end]))
            start = end

    def hash_shape_dtype(self, obj: [h5py.Dataset, np.ndarray]) -> str:
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return
        self.hashed.update(str(obj.shape).encode() + str(obj.dtype).encode())

    def compute_hash(self) -> str:
        """Hashes the NWB file contents, limiting to partal data where large."""
        # Dev note: fallbacks if slow: 1) read_direct_chunk, 2) read from offset

        for name, obj in tqdm(
            self.collect_names(self.file),
            desc=self.file.filename.split("/")[-1].split(".")[0],
            disable=not self.verbose,
        ):
            self.hashed.update(name.encode())
            for attr_key in sorted(obj.attrs):
                attr_value = obj.attrs[attr_key]
                _ = self.hash_shape_dtype(attr_value)
                self.hashed.update(attr_key.encode())
                self.hashed.update(self.serialize_attr_value(attr_value))

            if isinstance(obj, h5py.Dataset):
                _ = self.hash_dataset(obj)
            elif isinstance(obj, h5py.SoftLink):
                # TODO: Check that this works
                self.hashed.update(obj.path.encode())
            elif isinstance(obj, h5py.Group):
                for k, v in obj.items():
                    self.hashed.update(k.encode())
                    self.hashed.update(self.serialize_attr_value(v))
            else:
                raise TypeError(
                    f"Unknown object type: {type(obj)}\n"
                    + "Please report this an issue on GitHub."
                )

        return self.hashed.hexdigest()
