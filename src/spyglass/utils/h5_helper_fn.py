"""Helper methods for comparing pynwb objects."""

from json import loads as json_loads

import h5py


class H5pyComparator:
    def __init__(self, old, new, line_limit=80):
        self.old = self.obj_to_dict(old)
        self.new = self.obj_to_dict(new)
        self.line_limit = line_limit
        self.compare_dicts(self.old, self.new)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.old}, {self.new})"

    def run(self):
        """Rerun the comparison."""
        self.compare_dicts(self.old, self.new)

    def unpack_scalar(self, obj):
        """Unpack a scalar from an h5py dataset."""
        if isinstance(obj, (int, float, str)):
            return dict(scalar=obj)
        str_obj = str(obj[()])
        if "{" not in str_obj:
            return dict(scalar=str_obj)
        return json_loads(str_obj)

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
        if isinstance(obj, dict):
            return {k: self.obj_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (float, str, int, h5py.Dataset)):
            return self.unpack_scalar(obj)
        if isinstance(obj, h5py.Group):
            return self.assemble_dict(obj)
        return json_loads(obj)

    def sort_list_of_dicts(self, obj):
        """Sort a list of dictionaries."""
        return sorted(
            obj,
            key=lambda x: sorted(x.keys() if isinstance(x, dict) else str(x)),
        )

    def compare_dict_values(self, key, oval, nval, level, iteration):
        """Compare values of a specific key in two dictionaries."""
        if oval != nval:
            print(f"{level} {iteration}: dict val differ for {key}")
        if isinstance(oval, dict):
            self.compare_dicts(oval, nval, f"{level} {key}", iteration + 1)
        elif isinstance(oval, list):
            self.compare_lists(oval, nval, f"{level} {key}", iteration)

    def compare_lists(self, old_list, new_list, level, iteration):
        """Compare two lists of dictionaries."""
        old_sorted = self.sort_list_of_dicts(old_list)
        new_sorted = self.sort_list_of_dicts(new_list)
        for o, n in zip(old_sorted, new_sorted):
            iteration += 1
            if isinstance(o, dict):
                self.compare_dicts(o, n, level, iteration)
            elif o != n:
                print(f"{level} {iteration}: list val differ")
                print(f"\t{str(o)[:self.line_limit]}")
                print(f"\t{str(n)[:self.line_limit]}")

    def compare_dicts(self, old, new, level="", iteration=0):
        """Compare two dictionaries."""
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            if key not in old:
                print(f"{level} {iteration}: old missing key: {key}")
                continue
            if key not in new:
                print(f"{level} {iteration}: new missing key: {key}")
                continue
            self.compare_dict_values(key, old[key], new[key], level, iteration)
