import datajoint as dj
import logging
import os
import pathlib
import yaml
import sys


def prepopulate_default():
    """Prepopulate the database with the default values in SPYGLASS_BASE_DIR/entries.yaml."""
    base_dir = os.getenv("SPYGLASS_BASE_DIR", None)
    assert (
        base_dir is not None
    ), "You must set SPYGLASS_BASE_DIR or provide the base_dir argument"

    yaml_path = pathlib.Path(base_dir) / "entries.yaml"
    if os.path.exists(yaml_path):
        populate_from_yaml(yaml_path)


def populate_from_yaml(yaml_path: str):
    """Populate the database from specially formatted YAML files."""
    if not os.path.exists(yaml_path):
        raise ValueError(f"There is no file found with the path: {yaml_path}")
    with open(yaml_path, "r") as stream:
        d = yaml.safe_load(stream)

    for table_name, table_entries in d.items():
        table_cls = _get_table_cls(table_name)
        for entry_dict in table_entries:
            # test whether an entity with the primary key(s) already exists in the table
            if not issubclass(table_cls, (dj.Manual, dj.Lookup, dj.Part)):
                raise ValueError(
                    f"Prepopulate YAML ('{yaml_path}') contains table '{table_name}' that cannot be "
                    "prepopulated. Only Manual and Lookup tables can be prepopulated."
                )
            if hasattr(table_cls, "fetch_add"):
                # if the table has defined a fetch_add method, use that instead of insert1. this is useful for
                # tables where the primary key is an ID that auto-increments.
                # first check whether an entry exists with the same information.
                query = table_cls & entry_dict
                if not query:
                    print(
                        f"Populate: Populating table {table_cls.__name__} with data {entry_dict} using fetch_add."
                    )
                    table_cls.fetch_add(**entry_dict)
                continue

            primary_key_values = {
                k: v for k, v in entry_dict.items() if k in table_cls.primary_key
            }
            if not primary_key_values:
                print(
                    f"Populate: No primary key provided in data {entry_dict} for table {table_cls.__name__}"
                )
                continue
            if primary_key_values not in table_cls.fetch(
                *table_cls.primary_key, as_dict=True
            ):
                print(
                    f"Populate: Populating table {table_cls.__name__} with data {entry_dict} using insert1."
                )
                table_cls.insert1(entry_dict)
            else:
                logging.info(
                    f"Populate: Entry in {table_cls.__name__} with primary keys {primary_key_values} already exists."
                )


def _get_table_cls(table_name):
    """Get the spyglass.common class associated with a given table name. Also works for part tables one level deep."""
    if "." in table_name:  # part table
        master_table_name = table_name[0 : table_name.index(".")]
        part_table_name = table_name[table_name.index(".") + 1 :]
        master_table_cls = getattr(sys.modules["spyglass.common"], master_table_name)
        part_table_cls = getattr(master_table_cls, part_table_name)
        return part_table_cls
    else:
        return getattr(sys.modules["spyglass.common"], table_name)
