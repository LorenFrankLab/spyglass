"""Helper methods for comparing pynwb objects."""

import atexit
from json import loads as json_loads
from pathlib import Path

import h5py
import numpy as np
from yaml import safe_load as yaml_safe_load

from spyglass.utils.nwb_hash import IGNORED_KEYS


def sort_dict(d) -> dict:
    return dict(sorted(d.items()))


class H5pyComparator:  # pragma: no cover # informational, not functional
    """Compare two objects by treating them as dictionaries.

    Designed to compare two h5py objects, but can be used with any objects.
    By default, the comparison is run when the object is created and traverse
    embedded dictionaries and lists to compare values, printing differences.
    """

    def __init__(self, old, new, line_limit=80, run=True):
        """Initialize the comparator with two objects."""

        self.open_files = []
        self.found_diff = False
        atexit.register(self.cleanup)

        self.inputs = (old.__repr__(), new.__repr__())
        self.old = self.obj_to_dict(old)
        self.new = self.obj_to_dict(new)
        self.line_limit = line_limit

        if run:
            label = getattr(old, "stem", type(old))
            print(f"Compare: {label}")
            self.run()

        self.cleanup()
        atexit.unregister(self.cleanup)

    def __repr__(self):
        old, new = self.inputs
        return f"{self.__class__.__name__}({old}, {new})"

    def cleanup(self):
        """Close all open files."""
        for file in self.open_files:
            file.close()
        self.open_files = []

    def run(self):
        """Rerun the comparison."""
        self.compare_dicts(self.old, self.new)
        if not self.found_diff:
            print("\tNo differences")

    def unpack_scalar(self, obj):
        """Unpack a scalar from an h5py dataset."""
        if isinstance(obj, (int, float)):
            return dict(scalar=obj)
        if hasattr(obj, "shape") and obj.shape == ():
            obj = str(obj[()])
        return json_loads(obj) if "{" in obj else dict(scalar=obj)

    def assemble_dict(self, obj):
        """Assemble a dictionary from an h5py group."""
        ret = dict()
        for k, v in obj.items():
            if isinstance(v, h5py.Dataset):
                ret[k] = self.unpack_scalar(v)
            elif isinstance(v, h5py.Group):
                ret[k] = self.assemble_dict(v)
            else:
                ret[k] = v
        return ret

    def obj_to_dict(self, obj):
        """Convert an h5py object to a dictionary."""
        if obj in [None, "", [], {}, set()]:
            return dict(empty=True)
        if isinstance(obj, (Path, str)) and Path(obj).exists():
            return self.obj_to_dict(self.open_file(Path(obj)))
        elif isinstance(obj, Path):
            return dict(missing_path=str(obj))
        if isinstance(obj, dict):
            return {k: self.obj_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (float, str, int, h5py.Dataset)):
            return self.unpack_scalar(obj)
        if isinstance(obj, h5py.Group):
            return self.assemble_dict(obj)
        if isinstance(obj, np.ndarray):
            return self.numpy_to_dict(obj)
        if isinstance(obj, bytes):
            return self.unpack_scalar(obj.decode())
        if isinstance(obj, (list, tuple)):
            return {
                f"iter_{i}": (
                    x if isinstance(x, (int, float)) else self.obj_to_dict(x)
                )
                for i, x in enumerate(obj)
            }
        cache_attr = getattr(obj, "cache", None)  # Handle DirectoryHasher
        if isinstance(cache_attr, dict) and cache_attr:
            return cache_attr
        return json_loads(obj)

    def open_file(self, path):
        if path.suffix == ".h5":
            file = h5py.File(path, "r")
            self.open_files.append(file)
            return file
        if path.suffix == ".nwb":
            return f"pointer to {path}"
        if path.suffix == ".json":
            return json_loads(path.read_text())
        if path.suffix == ".yaml":
            return yaml_safe_load(path.read_text())
        if path.suffix in ["npy", "npz"]:
            return np.load(path)
        return dict(unrecognized_file_type=path.suffix)

    def numpy_to_dict(self, obj):
        """Convert a numpy object to a dictionary."""
        if obj.dtype.names:
            return {k: self.numpy_to_dict(obj[k]) for k in obj.dtype.names}
        elif getattr(obj, "ndim", 0) == 1:
            return obj.tolist()
        return [self.numpy_to_dict(x) for x in obj]

    def sort_list_of_dicts(self, obj):
        """Sort a list of dictionaries."""
        return sorted(
            obj,
            key=lambda x: sorted(x.keys() if isinstance(x, dict) else str(x)),
        )

    def compare_dict_values(self, key, oval, nval, level, iteration):
        """Compare values of a specific key in two dictionaries."""
        next_level = f"{level} {key}".replace("kwargs ", "")
        if isinstance(oval, dict):
            self.compare_dicts(oval, nval, next_level, iteration + 1)
        elif isinstance(oval, list):
            self.compare_lists(oval, nval, next_level, iteration)
        elif oval != nval:
            self.found_diff = True
            show = f"\n\t{oval} != {nval}"[: self.line_limit]
            print(f"{level} {iteration}: vals differ for {key}{show}")

    def compare_lists(self, old_list, new_list, level, iteration):
        """Compare two lists of dictionaries."""
        old_sorted = self.sort_list_of_dicts(old_list)
        new_sorted = self.sort_list_of_dicts(new_list)
        if not len(old_sorted) == len(new_sorted):
            print(f"{iteration} {level}: list length differ")
            print(f"\t{len(old_sorted)} != {len(new_sorted)}")
        for o, n in zip(old_sorted, new_sorted):
            iteration += 1
            if isinstance(o, dict):
                self.compare_dicts(o, n, level, iteration)
            elif o != n:
                self.found_diff = True
                print(f"{iteration} {level}: list val differ")
                print(f"\t{str(o)[:self.line_limit]}")
                print(f"\t{str(n)[:self.line_limit]}")

    def compare_dicts(self, old, new, level="", iteration=0):
        """Compare two dictionaries."""
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            if key in IGNORED_KEYS:
                continue
            if key not in old:
                print(f"{iteration} {level}: old missing key: {key[:]}")
                print(f"\tNew val: {new[key]}"[: self.line_limit])
                self.found_diff = True
                continue
            if key not in new:
                self.found_diff = True
                print(f"{iteration} {level}: new missing key: {key}")
                print(f"\tOld val: {old[key]}"[: self.line_limit])
                continue
            self.compare_dict_values(key, old[key], new[key], level, iteration)
