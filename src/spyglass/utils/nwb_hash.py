import atexit
import json
import re
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
IGNORED_KEYS = [
    "version",
    "object_id",  # TODO: remove
]
PRECISION_LOOKUP = dict(ProcessedElectricalSeries=8)


class DirectoryHasher:
    def __init__(
        self,
        directory_path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
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
        """

        self.dir_path = Path(directory_path)
        self.batch_size = batch_size
        self.verbose = verbose
        self.hashed = md5("".encode())
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Hashes the contents of the directory, recursively."""
        all_files = [f for f in sorted(self.dir_path.rglob("*")) if f.is_file()]

        for file_path in tqdm(all_files, disable=not self.verbose):
            if file_path.suffix == ".nwb":
                hasher = NwbfileHasher(file_path, batch_size=batch_size)
                self.hashed.update(hasher.hash.encode())
            elif file_path.suffix == ".json":
                self.hashed.update(self.json_encode(file_path))
            else:
                self.chunk_encode(file_path)

            # update with the rel path to for same file in diff dirs
            rel_path = str(file_path.relative_to(self.dir_path))
            self.hashed.update(rel_path.encode())

            if self.verbose:
                print(f"{file_path.name}: {self.hased.hexdigest()}")

        return self.hashed.hexdigest()  # Return the hex digest of the hash

    def chunk_encode(self, file_path: Path) -> str:
        """Encode the contents of a file in chunks for hashing."""
        with file_path.open("rb") as f:
            while chunk := f.read(self.batch_size):
                self.hashed.update(chunk)

    def json_encode(self, file_path: Path) -> str:
        """Encode the contents of a json file for hashing.

        Ignores the 'version' key(s) in the json file.
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
        precision_lookup: Dict[str, int] = PRECISION_LOOKUP,
        source_script_version: bool = False,
        verbose: bool = True,
    ):
        """Hashes the contents of an NWB file.

        Iterates through all objects in the NWB file, hashing the names, attrs,
        and data of each object. Ignores NWB specifications, and only considers
        NWB version.

        Uses a batch size to limit the amount of data hashed at once for large
        datasets. Rounds data to n decimal places for specific dataset names,
        as provided in the data_rounding dict.

        Version numbers stored in '/general/source_script' are ignored by
        default per the source_script_version flag.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the NWB file.
        batch_size  : int, optional
            Limit of data to hash for large datasets, by default 4095.
        data_rounding : Dict[str, int], optional
            Round data to n decimal places for specific datasets (i.e.,
            {dataset_name: n}). Default is to round ProcessedElectricalSeries
            to 10 significant digits via np.round(chunk, n).
        source_script_version : bool, optional
            Include version numbers from the source_script in the hash, by
            default False. If false, uses regex pattern to censor version
            numbers from this field.
        verbose : bool, optional
            Display progress bar, by default True.
        """
        self.path = Path(path)
        self.file = h5py.File(path, "r")
        atexit.register(self.cleanup)

        self.source_ver = source_script_version
        self.precision = precision_lookup
        self.batch_size = batch_size
        self.verbose = verbose
        self.hashed = md5("".encode())
        self.hash = self.compute_hash()

        self.cleanup()
        atexit.unregister(self.cleanup)

    def cleanup(self):
        self.file.close()

    def remove_version(self, key: str) -> bool:
        version_pattern = (
            r"\d+\.\d+\.\d+"  # Major.Minor.Patch
            + r"(?:-alpha|-beta|a\d+)?"  # Optional alpha or beta, -alpha
            + r"(?:\.dev\d+)?"  # Optional dev build, .dev01
            + r"(?:\+[a-z0-9]{9})?"  # Optional commit hash, +abcdefghi
            + r"(?:\.d\d{8})?"  # Optional date, dYYYYMMDD
        )
        return re.sub(version_pattern, "VERSION", key)

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

    def hash_dataset(self, dataset: h5py.Dataset):
        _ = self.hash_shape_dtype(dataset)

        if dataset.shape == ():
            raw_scalar = str(dataset[()])
            if "source_script" in dataset.name and not self.source_ver:
                raw_scalar = self.remove_version(raw_scalar)
            self.hashed.update(self.serialize_attr_value(raw_scalar))
            return

        dataset_name = dataset.parent.name.split("/")[-1]
        precision = self.precision.get(dataset_name, None)

        size = dataset.shape[0]
        start = 0

        while start < size:
            end = min(start + self.batch_size, size)
            data = dataset[start:end]
            if precision:
                data = np.round(data, precision)
            self.hashed.update(self.serialize_attr_value(data))
            start = end

    def hash_shape_dtype(self, obj: [h5py.Dataset, np.ndarray]) -> str:
        if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
            return
        self.hashed.update(str(obj.shape).encode() + str(obj.dtype).encode())

    @property
    def all_namespaces(self) -> bytes:
        """Encoded string of all NWB namespace specs."""
        catalog = NamespaceCatalog(NWBGroupSpec, NWBDatasetSpec, NWBNamespace)
        pynwb.NWBHDF5IO.load_namespaces(catalog, self.path)
        name_cat = TypeMap(catalog).namespace_catalog
        ret = ""
        for ns_name in name_cat.namespaces:
            ret += ns_name
            ret += name_cat.get_namespace(ns_name)["version"]
        return ret.encode()

    def compute_hash(self) -> str:
        """Hashes the NWB file contents, limiting to partal data where large."""
        # Dev note: fallbacks if slow: 1) read_direct_chunk, 2) read from offset

        self.hashed.update(self.all_namespaces)

        for name, obj in tqdm(
            self.collect_names(self.file),
            desc=self.file.filename.split("/")[-1].split(".")[0],
            disable=not self.verbose,
        ):
            self.hashed.update(name.encode())

            for attr_key in sorted(obj.attrs):
                if attr_key in IGNORED_KEYS:
                    continue
                attr_value = obj.attrs[attr_key]
                _ = self.hash_shape_dtype(attr_value)
                self.hashed.update(attr_key.encode())
                self.hashed.update(self.serialize_attr_value(attr_value))

            if isinstance(obj, h5py.Dataset):
                _ = self.hash_dataset(obj)
            elif isinstance(obj, h5py.SoftLink):
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
