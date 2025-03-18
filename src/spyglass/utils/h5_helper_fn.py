"""Helper methods for comparing pynwb objects."""

from json import loads as json_loads
from pathlib import Path

import h5py
import numpy as np
from yaml import safe_load as yaml_safe_load


def sort_dict(d) -> dict:
    return dict(sorted(d.items()))


class H5pyComparator:
    """Compare two objects by treating them as dictionaries.

    Designed to compare two h5py objects, but can be used with any objects.
    By default, the comparison is run when the object is created and traverse
    embedded dictionaries and lists to compare values, printing differences.
    """

    def __init__(self, old, new, line_limit=80, run=True):
        self.inputs = (old.__repr__(), new.__repr__())
        self.old = self.obj_to_dict(old)
        self.new = self.obj_to_dict(new)
        self.line_limit = line_limit
        if run:
            self.compare_dicts(self.old, self.new)

    def __repr__(self):
        old, new = self.inputs
        return f"{self.__class__.__name__}({old}, {new})"

    def run(self):
        """Rerun the comparison."""
        self.compare_dicts(self.old, self.new)

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
        if not obj:
            return dict(empty=True)
        if isinstance(obj, (Path, str)) and Path(obj).exists():
            return self.obj_to_dict(self.open_file(Path(obj)))
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
            return {"iterable": self.obj_to_dict(x) for x in obj}
        return json_loads(obj)

    def open_file(self, path):
        if path.suffix == ".h5":
            return h5py.File(path, "r")
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
            show = f"\n\t{oval} != {nval}"[: self.line_limit]
            print(f"{level} {iteration}: vals differ for {key}{show}")

    def compare_lists(self, old_list, new_list, level, iteration):
        """Compare two lists of dictionaries."""
        old_sorted = self.sort_list_of_dicts(old_list)
        new_sorted = self.sort_list_of_dicts(new_list)
        for o, n in zip(old_sorted, new_sorted):
            iteration += 1
            if isinstance(o, dict):
                self.compare_dicts(o, n, level, iteration)
            elif o != n:
                print(f"{iteration} {level}: list val differ")
                print(f"\t{str(o)[:self.line_limit]}")
                print(f"\t{str(n)[:self.line_limit]}")

    def compare_dicts(self, old, new, level="", iteration=0):
        """Compare two dictionaries."""
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            if key not in old:
                print(f"{iteration} {level}: old missing key: {key}")
                continue
            if key not in new:
                print(f"{iteration} {level}: new missing key: {key}")
                continue
            self.compare_dict_values(key, old[key], new[key], level, iteration)
