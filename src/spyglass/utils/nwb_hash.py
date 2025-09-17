import atexit
import json
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import pynwb
from hdmf.build import TypeMap
from hdmf.spec import NamespaceCatalog
from pynwb.spec import NWBDatasetSpec, NWBGroupSpec, NWBNamespace
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 4095
IGNORED_KEYS = ["version", "source_script"]
PRECISION_LOOKUP = dict(ProcessedElectricalSeries=4)


def get_file_namespaces(file_path: Union[str, Path]) -> dict:
    """Get all namespace versions from an NWB file.

    WARNING: This function falsely reports core <= 2.6.0 as 2.6.0-alpha

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the NWB file.
    """
    catalog = NamespaceCatalog(NWBGroupSpec, NWBDatasetSpec, NWBNamespace)
    pynwb.NWBHDF5IO.load_namespaces(catalog, file_path)
    name_cat = TypeMap(catalog).namespace_catalog

    return {
        ns_name: name_cat.get_namespace(ns_name).get("version", None)
        for ns_name in name_cat.namespaces
    }


class DirectoryHasher:
    def __init__(
        self,
        directory_path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        keep_obj_hash: bool = False,
        verbose: bool = False,
    ):
        """Generate a hash of the contents of a directory, recursively.

        Searches though all files in the directory and subdirectories, hashing
        the contents of files. nwb files are hashed with the NwbfileHasher
        class. JSON files are hashed by encoding the contents, ignoring
        specific keys, like 'version'. All other files are hashed by reading
        the file in chunks.

        If the contents of a json file is otherwise the same, but the 'version'
        value is different, we assume that the dependency change had no effect
        on the data and ignore the difference.

        Parameters
        ----------
        directory_path : str
            Path to the directory to hash.
        batch_size : int, optional
            Limit of data to hash for large files, by default 4095.
        keep_obj_hash : bool, optional
            Default false. If true, keep cache the hash of each file.
        verbose : bool, optional
            Display progress bar, by default False.
        """
        self.dir_path = Path(directory_path)
        if not self.dir_path.exists():
            raise FileNotFoundError(f"Dir does not exist: {self.dir_path}")
        if not self.dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a dir: {self.dir_path}")

        self.batch_size = int(batch_size)
        self.keep_obj_hash = bool(keep_obj_hash)
        self.cache = {}
        self.verbose = bool(verbose)
        self.hashed = md5("".encode())
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Hashes the contents of the directory, recursively."""
        all_files = [f for f in sorted(self.dir_path.rglob("*")) if f.is_file()]

        for file_path in all_files:
            if file_path.suffix == ".nwb":
                this_hash = NwbfileHasher(
                    file_path, batch_size=self.batch_size
                ).hash.encode()
            elif file_path.suffix in [".json", ".jsonl"]:
                this_hash = self.json_encode(file_path)
            elif file_path.suffix in [".npy", ".npz"]:
                this_hash = self.npy_encode(file_path)
            else:
                this_hash = self.chunk_encode(file_path)

            self.hashed.update(this_hash)

            # update with the rel path to for same file in diff dirs
            rel_path = str(file_path.relative_to(self.dir_path))
            self.hashed.update(rel_path.encode())

            if self.keep_obj_hash:
                self.cache[rel_path] = this_hash

        return self.hashed.hexdigest()  # Return the hex digest of the hash

    def npy_encode(self, file_path: Path) -> str:
        """Encode the contents of a numpy file for hashing."""
        data = np.load(file_path, allow_pickle=True).tobytes()
        return md5(data).hexdigest().encode()

    def chunk_encode(self, file_path: Path) -> str:
        """Encode the contents of a file in chunks for hashing."""
        this_hash = md5("".encode())
        with file_path.open("rb") as f:
            while chunk := f.read(self.batch_size):
                this_hash.update(chunk)
        return this_hash.hexdigest().encode()

    def json_encode(self, file_path: Path) -> str:
        """Encode the contents of a json file for hashing.

        Ignores the predetermined keys in the IGNORED_KEYS list.
        """
        with file_path.open("r") as f:
            file_data = json.load(f, object_hook=self.pop_version)

        return json.dumps(file_data, sort_keys=True).encode()

    def pop_version(self, data: Union[dict, list]) -> Union[dict, list]:
        """Recursively remove banned keys from any nested dicts/lists."""
        if isinstance(data, dict):
            return {
                k: self.pop_version(v)
                for k, v in data.items()
                if k not in IGNORED_KEYS
            }
        elif isinstance(data, list):
            return [self.pop_version(item) for item in data]
        else:
            return data


class NwbfileHasher:
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        precision_lookup: Union[int, Dict[str, int]] = PRECISION_LOOKUP,
        keep_obj_hash: bool = False,
        keep_file_open: bool = False,
        verbose: bool = False,
    ):
        """Hashes the contents of an NWB file.

        Iterates through all objects in the NWB file, hashing the names, attrs,
        and data of each object. Ignores NWB specifications, and only considers
        NWB version.

        Uses a batch size to limit the amount of data hashed at once for large
        datasets. Rounds data to n decimal places for specific dataset names,
        as provided in the data_rounding dict.

        Version numbers stored in '/general/source_script' are ignored.

        Keeps each object hash as a dictionary, if keep_obj_hash is True. This
        is useful for debugging, but not recommended for large files.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the NWB file.
        batch_size  : int, optional
            Limit of data to hash for large datasets, by default 4095.
        precision_lookup : Union[int, Dict[str, int]], optional
            For dict, round data to n decimal places for specific datasets
            ({dataset_name: n}). If int, round all datasets to this precision.
            Default is to round ProcessedElectricalSeries to 4 significant
            digits via np.round(chunk, n).
        keep_obj_hash : bool, optional
            Keep the hash of each object in the NWB file, by default False.
        verbose : bool, optional
            Display progress bar, by default True.
        """
        self.path = Path(path)
        self.file = h5py.File(path, "r")
        atexit.register(self.cleanup)

        if precision_lookup is None:
            precision_lookup = PRECISION_LOOKUP
        if isinstance(precision_lookup, int):  # same precision for all datasets
            precision_lookup = {k: precision_lookup for k in PRECISION_LOOKUP}

        self.precision = precision_lookup
        self.batch_size = batch_size
        self.verbose = verbose
        self.keep_obj_hash = keep_obj_hash
        self.objs = {}
        self.hashed = md5("".encode())
        self.hash = self.compute_hash()

        if not keep_file_open:
            self.cleanup()
            atexit.unregister(self.cleanup)

    def cleanup(self):
        self.file.close()

    def collect_names(self, file):
        """Collects all object names in the file."""

        def collect_items(name, obj):
            if "specifications" in name:
                return  # Ignore specifications, because we hash namespaces
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
            return value.astype(str).tobytes()  # must be 'astype(str)'
        elif isinstance(value, (str, int, float)):
            return str(value).encode()
        return repr(value).encode()  # For other, use repr

    def is_roundable(self, data) -> bool:
        """Check if data is roundable."""
        if isinstance(data, np.ndarray):
            return np.issubdtype(data.dtype, np.number)
        return isinstance(data, (float, int, np.number))

    def hash_dataset(self, dataset: h5py.Dataset):
        if dataset.name in IGNORED_KEYS:
            return  # Ignore source script

        this_hash = md5(self.hash_shape_dtype(dataset))

        if dataset.shape == ():
            raw_scalar = str(dataset[()])
            this_hash.update(self.serialize_attr_value(raw_scalar))
            return

        dataset_name = dataset.parent.name.split("/")[-1]
        precision = self.precision.get(dataset_name, None)

        size = dataset.shape[0]
        start = 0

        while start < size:
            end = min(start + self.batch_size, size)
            data = dataset[start:end]
            if precision and self.is_roundable(data):
                data = np.round(data, precision)
            this_hash.update(self.serialize_attr_value(data))
            start = end

        return this_hash.hexdigest()

    def hash_shape_dtype(self, obj: Union[h5py.Dataset, np.ndarray]) -> str:
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return "".encode()
        return str(obj.shape).encode() + str(obj.dtype).encode()

    @cached_property
    def namespaces(self) -> dict:
        """Encoded string of all NWB namespace specs."""
        return get_file_namespaces(self.path)

    @cached_property
    def namespaces_str(self) -> str:
        """String representation of all NWB namespace specs."""
        return json.dumps(self.namespaces, sort_keys=True).encode()

    def add_to_cache(self, name: str, obj: Any, digest: str = None):
        """Add object to the cache.

        Centralizes conditional logic for adding objects to the cache.
        """
        if self.keep_obj_hash:
            self.objs[name] = (obj, digest)

    def compute_hash(self) -> str:
        """Hashes the NWB file contents."""
        self.hashed.update(self.namespaces_str)

        self.add_to_cache("namespaces", self.namespaces, None)

        for name, obj in tqdm(
            self.collect_names(self.file),
            desc=self.file.filename.split("/")[-1].split(".")[0],
            disable=not self.verbose,
        ):
            this_hash = md5(name.encode())

            for attr_key in sorted(obj.attrs):
                if attr_key in IGNORED_KEYS:
                    continue
                attr_value = obj.attrs[attr_key]
                this_hash.update(self.hash_shape_dtype(attr_value))
                this_hash.update(attr_key.encode())
                this_hash.update(self.serialize_attr_value(attr_value))

            if isinstance(obj, h5py.Dataset):
                _ = self.hash_dataset(obj)
            elif isinstance(obj, h5py.SoftLink):
                this_hash.update(obj.path.encode())
            elif isinstance(obj, h5py.Group):
                for k, v in obj.items():
                    this_hash.update(k.encode())
                    obj_value = self.serialize_attr_value(v)
                    this_hash.update(obj_value)
                    self.add_to_cache(
                        f"{name}/k", v, md5(obj_value).hexdigest()
                    )
            else:
                raise TypeError(
                    f"Unknown object type: {type(obj)}\n"
                    + "Please report this an issue on GitHub."
                )

            this_digest = this_hash.hexdigest()
            self.hashed.update(this_digest.encode())

            self.add_to_cache(name, obj, this_digest)

        return self.hashed.hexdigest()
