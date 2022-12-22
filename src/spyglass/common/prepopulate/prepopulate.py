import os
import pathlib
import yaml
import sys

# from .probe import create_probes


def prepopulate():
    base_dir = os.getenv("SPYGLASS_BASE_DIR", None)
    assert (
        base_dir is not None
    ), "You must set SPYGLASS_BASE_DIR or provide the base_dir argument"

    yaml_path = pathlib.Path(base_dir) / "add_entries.yaml"
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, "r") as stream:
        d = yaml.safe_load(stream)

    for table_name, table_entries in d.items():
        cls = getattr(sys.modules["spyglass.common"], table_name)
        for entry_dict in table_entries:
            # test whether an entity with the primary key(s) already exists in the table
            primary_key_values = {
                k: v for k, v in entry_dict.items() if k in cls.primary_key
            }
            if primary_key_values not in cls.fetch(*cls.primary_key, as_dict=True):
                print(f"Prepopulating table {cls.__name__} with data {entry_dict}")
                cls.insert1(entry_dict, skip_duplicates=True)

    # create_probes()
