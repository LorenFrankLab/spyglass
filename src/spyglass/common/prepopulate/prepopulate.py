import logging
import os
import pathlib
import yaml
import sys

# from .probe import create_probes


def prepopulate_default():
    base_dir = os.getenv("SPYGLASS_BASE_DIR", None)
    assert (
        base_dir is not None
    ), "You must set SPYGLASS_BASE_DIR or provide the base_dir argument"

    yaml_path = pathlib.Path(base_dir) / "add_entries.yaml"
    populate_from_yaml(yaml_path)

    # create_probes()


def populate_from_yaml(yaml_path: str):
    """Populate"""
    if not os.path.exists(yaml_path):
        raise ValueError(f"There is no file found with the path: {yaml_path}")
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
                print(
                    f"Prepoulate: Prepopulating table {cls.__name__} with data {entry_dict}"
                )
                cls.insert1(entry_dict)
            else:
                logging.info(
                    f"Prepoulate: Entry in {cls.__name__} with primary keys {primary_key_values} already exists."
                )
