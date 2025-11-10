"""Helper functions for manipulating information from DataJoint fetch calls."""

import inspect
import math
import multiprocessing.pool
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Type, Union
from uuid import uuid4

import datajoint as dj
import h5py
import numpy as np
from datajoint.table import Table
from datajoint.user_tables import TableMeta, UserTable

from spyglass.utils.logging import logger
from spyglass.utils.nwb_helper_fn import file_from_dandi, get_nwb_file

STR_DTYPE = h5py.special_dtype(vlen=str)

# Tables that should be excluded from the undirected graph when finding paths
# for TableChain objects and searching for an upstream key.
PERIPHERAL_TABLES = [
    "`common_interval`.`interval_list`",
    "`common_nwbfile`.`__analysis_nwbfile_kachery`",
    "`common_nwbfile`.`__nwbfile_kachery`",
    "`common_nwbfile`.`analysis_nwbfile_kachery_selection`",
    "`common_nwbfile`.`analysis_nwbfile_kachery`",
    "`common_nwbfile`.`analysis_nwbfile`",
    "`common_nwbfile`.`kachery_channel`",
    "`common_nwbfile`.`nwbfile_kachery_selection`",
    "`common_nwbfile`.`nwbfile_kachery`",
    "`common_nwbfile`.`nwbfile`",
]


def sanitize_unix_name(name: str) -> str:
    """Sanitize a string to be a valid unix name.

    This function replaces any invalid characters with underscores and
    removes leading and trailing whitespace. It warns the user if the name
    has been changed.

    Parameters
    ----------
    name : str
        Name to sanitize.

    Returns
    -------
    str
        Sanitized name.
    """
    invalid_chars = r"[^a-zA-Z0-9_.-]+"
    sanitized_name = re.sub(invalid_chars, "_", name.strip().replace(" ", "_"))
    if sanitized_name != name:
        logger.warning(
            f"Name '{name}' contains invalid characters. "
            + f"Sanitized to '{sanitized_name}'."
        )
    return sanitized_name


def ensure_names(
    table: Union[str, Table, Iterable] = None, force_list: bool = False
) -> Union[str, List[str], None]:
    """Ensure table is a string.

    Parameters
    ----------
    table : Union[str, Table, Iterable], optional
        Table to ensure is a string, by default None. If passed as iterable,
        will ensure all elements are strings.
    force_list : bool, optional
        Force the return to be a list, by default False, only used if input is
        iterable.

    Returns
    -------
    Union[str, List[str], None]
        Table as a string or list of strings.
    """
    # is iterable (list, set, set) but not a table/string
    is_collection = isinstance(table, Iterable) and not isinstance(
        table, (Table, TableMeta, str)
    )
    if force_list and not is_collection:
        return [ensure_names(table)]
    if table is None:
        return None
    if isinstance(table, str):
        return table
    if is_collection:
        return [ensure_names(t) for t in table]
    return getattr(table, "full_table_name", None)


def declare_all_merge_tables():
    """Ensures all merge tables in the spyglass core package are declared.
    - Prevents circular imports
    - Prevents errors from table declaration within a transaction
    - Run during nwb insertion
    """
    from spyglass.decoding.decoding_merge import DecodingOutput  # noqa: F401
    from spyglass.lfp.lfp_merge import LFPOutput  # noqa: F401
    from spyglass.position.position_merge import PositionOutput  # noqa: F401
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
    )  # noqa: F401


def fuzzy_get(index: Union[int, str], names: List[str], sources: List[str]):
    """Given lists of items/names, return item at index or by substring."""
    if isinstance(index, int):
        return sources[index]
    for i, part in enumerate(names):
        if index in part:
            return sources[i]
    return None


def unique_dicts(list_of_dict):
    """Remove duplicate dictionaries from a list."""
    return [dict(t) for t in {tuple(d.items()) for d in list_of_dict}]


def deprecated_factory(classes: list, old_module: str = "") -> list:
    """Creates a list of classes and logs a warning when instantiated

    Parameters
    ---------
    classes : list
        list of tuples containing old_class, new_class

    Returns
    ------
    list
        list of classes that will log a warning when instantiated
    """

    if not isinstance(classes, list):
        classes = [classes]

    ret = [
        _subclass_factory(old_name=c[0], new_class=c[1], old_module=old_module)
        for c in classes
    ]

    return ret[0] if len(ret) == 1 else ret


def _subclass_factory(
    old_name: str, new_class: Type, old_module: str = ""
) -> Type:
    """Creates a subclass with a deprecation warning on __init__

    Old class is a subclass of new class, so it will inherit all of the new
    class's methods. Old class retains its original name and module. Use
    __name__ to get the module name of the caller.

    Usage: OldClass = _subclass_factory('OldClass', __name__, NewClass)
    """

    new_module = new_class().__class__.__module__

    # Define the __call__ method for the new class
    def init_override(self, *args, **kwargs):
        logger.warning(
            "Deprecation: this class has been moved out of "
            + f"{old_module}\n"
            + f"\t{old_name} -> {new_module}.{new_class.__name__}"
            + "\nPlease use the new location."
        )
        return super(self.__class__, self).__init__(*args, **kwargs)

    class_dict = {
        "__module__": old_module or new_class.__class__.__module__,
        "__init__": init_override,
        "_is_deprecated": True,
    }

    return type(old_name, (new_class,), class_dict)


def dj_replace(original_table, new_values, key_column, replace_column):
    """Given the output of a fetch() call from a schema and a 2D array made up
    of (key_value, replace_value) tuples, find each instance of key_value in
    the key_column of the original table and replace the specified
    replace_column with the associated replace_value. Key values must be
    unique.

    Parameters
    ----------
    original_table
        Result of a datajoint .fetch() call on a schema query.
    new_values : list
        List of tuples, each containing (key_value, replace_value).
    replace_column : str
        The name of the column where to-be-replaced values are located.

    Returns
    -------
    original_table
        Structured array of new table entries that can be inserted back into the schema
    """

    # check to make sure the new_values are a list or array of tuples and fix if not
    if isinstance(new_values, tuple):
        tmp = list()
        tmp.append(new_values)
        new_values = tmp

    new_val_array = np.asarray(new_values)
    replace_ind = np.where(
        np.isin(original_table[key_column], new_val_array[:, 0])
    )
    original_table[replace_column][replace_ind] = new_val_array[:, 1]
    return original_table


def get_all_tables_in_stack(stack):
    """Get all classes from a stack of tables."""
    classes = set()
    for frame_info in stack:
        locals_dict = frame_info.frame.f_locals
        for obj in locals_dict.values():
            if not isinstance(obj, UserTable):
                continue  # skip non-tables
            if (name := obj.full_table_name) in PERIPHERAL_TABLES:
                continue  # skip common_nwbfile tables
            classes.add(name)
    return classes


def get_fetching_table_from_stack(stack):
    """Get all classes from a stack of tables."""
    classes = get_all_tables_in_stack(stack)
    if len(classes) > 1:
        classes = None  # predict only one but not sure, so return None
    return next(iter(classes)) if classes else None


def get_nwb_table(query_expression, tbl, attr_name, *attrs, **kwargs):
    """Get the NWB file name and path from the given DataJoint query.

    Parameters
    ----------
    query_expression : query
        A DataJoint query expression (e.g., join, restrict) or a table to call fetch on.
    tbl : table
        DataJoint table to fetch from.
    attr_name : str
        Attribute name to fetch from the table.
    *attrs : list
        Attributes from normal DataJoint fetch call.
    **kwargs : dict
        Keyword arguments from normal DataJoint fetch call.

    Returns
    -------
    nwb_files : list
        List of NWB file names.
    file_path_fn : function
        Function to get the absolute path to the NWB file.
    """
    from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
    from spyglass.utils.dj_mixin import SpyglassMixin

    kwargs["as_dict"] = True  # force return as dictionary
    attrs = attrs or query_expression.heading.names  # if none, all

    which = "analysis" if "analysis" in attr_name else "nwb"
    tbl_map = {  # map to file_name_str and file_path_fn
        "analysis": ["analysis_file_name", AnalysisNwbfile.get_abs_path],
        "nwb": ["nwb_file_name", Nwbfile.get_abs_path],
    }
    file_name_str, file_path_fn = tbl_map[which]

    # logging arg only if instanced table inherits Mixin
    inst = (  # instancing may not be necessary
        query_expression()
        if isinstance(query_expression, type)
        and issubclass(query_expression, dj.Table)
        else query_expression
    )
    arg = dict(log_export=False) if isinstance(inst, SpyglassMixin) else dict()

    nwb_files = (
        query_expression.join(tbl.proj(nwb2load_filepath=attr_name), **arg)
    ).fetch(file_name_str)

    # Disabled #1024
    # if which == "analysis":  # log access of analysis files to log table
    #     AnalysisNwbfile().increment_access(
    #         nwb_files, table=get_fetching_table_from_stack(inspect.stack())
    #     )

    return nwb_files, file_path_fn


def fetch_nwb(query_expression, nwb_master, *attrs, **kwargs):
    """Get an NWB object from the given DataJoint query.

    Parameters
    ----------
    query_expression : query
        A DataJoint query expression (e.g., join, restrict) or a table to call fetch on.
    nwb_master : tuple
        Tuple (table, attr) to get the NWB filepath from.
        i.e. absolute path to NWB file can be obtained by looking up attr column of table
        table is usually Nwbfile or AnalysisNwbfile;
        attr is usually 'nwb_file_abs_path' or 'analysis_file_abs_path'
    *attrs : list
        Attributes from normal DataJoint fetch call.
    **kwargs : dict
        Keyword arguments from normal DataJoint fetch call.

    Returns
    -------
    nwb_objects : list
        List of dicts containing fetch results and NWB objects.
    """
    from spyglass.utils.dj_mixin import SpyglassMixin

    kwargs["as_dict"] = True  # force return as dictionary

    tbl, attr_name = nwb_master
    if "analysis" in attr_name:
        file_name_attr = "analysis_file_name"
    else:
        file_name_attr = "nwb_file_name"

    if not attrs:
        attrs = query_expression.heading.names

    nwb_files, file_path_fn = get_nwb_table(
        query_expression, tbl, attr_name, *attrs, **kwargs
    )

    for file_name in nwb_files:
        file_path = file_path_fn(file_name, from_schema=True)
        if not os.path.exists(file_path):
            # get from kachery/dandi or recompute, store in cache
            get_nwb_file(file_path, query_expression)

    # logging arg only if instanced table inherits Mixin
    inst = (  # instancing may not be necessary
        query_expression()
        if isinstance(query_expression, type)
        and issubclass(query_expression, dj.Table)
        else query_expression
    )
    arg = dict(log_export=False) if isinstance(inst, SpyglassMixin) else dict()
    query_table = query_expression.join(
        tbl.proj(nwb2load_filepath=attr_name), **arg
    )

    # Fix for custom analysis tables #1435
    if attrs and "nwb2load_filepath" not in attrs:
        attrs = list(attrs) + ["nwb2load_filepath"]

    rec_dicts = query_table.fetch(*attrs, **kwargs)
    # get filepath for each. Use datajoint for checksum if local
    for rec_dict in rec_dicts:
        file_path = file_path_fn(rec_dict[file_name_attr])
        if file_from_dandi(file_path):
            # skip the filepath checksum if streamed from Dandi
            rec_dict["nwb2load_filepath"] = file_path
            continue

        # Full dict caused issues with dlc tables using dicts in secondary keys
        rec_only_pk = {k: rec_dict[k] for k in query_table.heading.primary_key}
        rec_dict["nwb2load_filepath"] = (query_table & rec_only_pk).fetch1(
            "nwb2load_filepath"
        )

    if not rec_dicts or not np.any(
        ["object_id" in key for key in rec_dicts[0]]
    ):
        return rec_dicts

    ret = []
    for rec_dict in rec_dicts:
        nwbf = get_nwb_file(rec_dict.pop("nwb2load_filepath"))
        # for each attr that contains substring 'object_id', store key-value:
        # attr name to NWB object
        # remove '_object_id' from attr name
        nwb_objs = {
            id_attr.replace("_object_id", ""): _get_nwb_object(
                nwbf.objects, rec_dict[id_attr]
            )
            for id_attr in attrs
            if "object_id" in id_attr and rec_dict[id_attr] != ""
        }
        ret.append({**rec_dict, **nwb_objs})
    return ret


def _get_nwb_object(objects, object_id):
    """Retrieve NWB object and try to convert to dataframe if possible"""
    try:
        return objects[object_id].to_dataframe()
    except AttributeError:
        return objects[object_id]


def get_child_tables(table):
    """Get all child tables of a given table."""
    table = table() if inspect.isclass(table) else table
    return [
        dj.FreeTable(
            table.connection,
            (
                s
                if not s.isdigit()
                else next(iter(table.connection.dependencies.children(s)))
            ),
        )
        for s in table.children()
    ]


def update_analysis_for_dandi_standard(
    filepath: str,
    age: str = "P4M/P8M",
    resolve_external_table: bool = True,
):
    """Function to resolve common nwb file format errors within the database

    Parameters
    ----------
    filepath : str
        abs path to the file to edit
    age : str, optional
        age to assign animal if missing, by default "P4M/P8M"
    resolve_external_table : bool, optional
        whether to update the external table. Set False if editing file
        outside the database, by default True
    """
    from spyglass.common import LabMember

    LabMember().check_admin_privilege(
        error_message="Admin permissions required to edit existing analysis files"
    )
    file_name = filepath.split("/")[-1]
    # edit the file
    with h5py.File(filepath, "a") as file:
        sex_value = file["/general/subject/sex"][()].decode("utf-8")
        if sex_value not in ["Female", "Male", "F", "M", "O", "U"]:
            raise ValueError(f"Unexpected value for sex: {sex_value}")

        if len(sex_value) > 1:
            new_sex_value = sex_value[0].upper()
            logger.info(
                f"Adjusting subject sex: '{sex_value}' -> '{new_sex_value}'"
            )
            file["/general/subject/sex"][()] = new_sex_value

        # replace subject species value "Rat" with "Rattus norvegicus"
        species_value = file["/general/subject/species"][()].decode("utf-8")
        if species_value == "Rat":
            new_species_value = "Rattus norvegicus"
            logger.info(
                f"Adjusting subject species from '{species_value}' to "
                + f"'{new_species_value}'."
            )
            file["/general/subject/species"][()] = new_species_value

        elif not (
            len(species_value.split(" ")) == 2 or "NCBITaxon" in species_value
        ):
            raise ValueError(
                "Dandi upload requires species either be in Latin binomial form"
                + " (e.g., 'Mus musculus' and 'Homo sapiens') or be a NCBI "
                + "taxonomy link (e.g., "
                + "'http://purl.obolibrary.org/obo/NCBITaxon_280675').\n "
                + f"Please update species value of: {species_value}"
            )

        # add subject age dataset "P4M/P8M"
        if "age" not in file["/general/subject"]:
            new_age_value = age
            logger.info(
                f"Adding missing subject age, set to '{new_age_value}'."
            )
            file["/general/subject"].create_dataset(
                name="age", data=new_age_value, dtype=STR_DTYPE
            )

        # format name to "Last, First"
        experimenter_value = file["/general/experimenter"][:].astype(str)
        new_experimenter_value = dandi_format_names(experimenter_value)
        if experimenter_value != new_experimenter_value:
            new_experimenter_value = new_experimenter_value.astype(STR_DTYPE)
            logger.info(
                f"Adjusting experimenter from {experimenter_value} to "
                + f"{new_experimenter_value}."
            )
            file["/general/experimenter"][:] = new_experimenter_value

    # update the datajoint external store table to reflect the changes
    if resolve_external_table:
        location = "raw" if filepath.endswith("_.nwb") else "analysis"
        _resolve_external_table(filepath, file_name, location)


def dandi_format_names(experimenter: List) -> List:
    """Make names compliant with dandi standard of "Last, First"

    Parameters
    ----------
    experimenter : List
        List of experimenter names

    Returns
    -------
    List
        reformatted list of experimenter names
    """
    for i, name in enumerate(experimenter):
        parts = name.split(" ")
        new_name = " ".join(
            parts[:-1],
        )
        new_name = f"{parts[-1]}, {new_name}"
        experimenter[i] = new_name
    return experimenter


def _resolve_external_table(
    filepath: str, file_name: str, location: str = "analysis"
):
    """Function to resolve database vs. file property discrepancies.

    WARNING: This should only be used when editing file metadata. Can violate
    data integrity if improperly used.

    Parameters
    ----------
    filepath : str
        abs path to the file to edit
    file_name : str
        name of the file to edit
    location : str, optional
        which external table the file is in, current options are
        ["analysis", "raw], by default "analysis"
    """
    from spyglass.common import LabMember
    from spyglass.common.common_nwbfile import AnalysisRegistry
    from spyglass.common.common_nwbfile import schema as common_schema

    LabMember().check_admin_privilege(
        error_message="Please contact database admin to edit database checksums"
    )

    file_restr = f"filepath LIKE '%{file_name}'"

    to_updates = []
    if location == "analysis":  # Update for each custom Analysis external
        for external in AnalysisRegistry().get_externals():
            restr_external = external & file_restr
            if not bool(restr_external):
                continue
            if len(restr_external) > 1:
                raise ValueError(
                    "Multiple entries found in external table for file: "
                    + f"{file_name}, cannot resolve."
                )
            to_updates.append(restr_external)

    elif location == "raw":
        restr_external = common_schema.external["raw"] & file_restr
        to_updates.append(restr_external)

    if not to_updates:
        logger.warning(
            f"No entries found in external tables for file: {file_name}"
        )
        return

    update_vals = dict(
        size=Path(filepath).stat().st_size,
        contents_hash=dj.hash.uuid_from_file(filepath),
    )
    for to_update in to_updates:
        key = to_update.fetch1()
        key.update(update_vals)
        to_update.update1(key)


def make_file_obj_id_unique(nwb_path: str):
    """Make the top-level object_id attribute of the file unique

    Parameters
    ----------
    nwb_path : str
        path to the NWB file

    Returns
    -------
    str
        the new object_id
    """
    from spyglass.common.common_lab import LabMember  # noqa: F401

    LabMember().check_admin_privilege(
        error_message="Admin permissions required to edit existing analysis files"
    )
    new_id = str(uuid4())
    with h5py.File(nwb_path, "a") as f:
        f.attrs["object_id"] = new_id
    location = "raw" if nwb_path.endswith("_.nwb") else "analysis"
    _resolve_external_table(
        nwb_path, nwb_path.split("/")[-1], location=location
    )
    return new_id


def _quick_get_analysis_path(file: str):
    """Get the absolute path to an analysis file on disc without integrity checks.
    For use when scanning large number of files.

    Parameters
    ----------
    file : str
        Name of the analysis file to get the path for.
    Returns
    -------
    str
        Absolute path to the analysis file. Returns None if the file is not found locally.
    """
    from spyglass.common import AnalysisNwbfile
    from spyglass.settings import sg_config

    analysis_dir = sg_config.analysis_dir
    if (path := Path(analysis_dir) / Path(file)).exists():
        return path

    folder = "_".join(file.split("_")[:-1])
    path = Path(analysis_dir) / Path(folder) / Path(file)
    if os.path.exists(path):
        return path
    path = Path(AnalysisNwbfile().get_abs_path(file))
    if path.exists():
        return path

    logger.warning(f"File {file} not found in {analysis_dir}")
    return None


def populate_pass_function(value):
    """Pass function for parallel populate.

    Note: To avoid pickling errors, the table must be passed by class,
        NOT by instance.
    Note: This function must be defined in the global namespace.

    Parameters
    ----------
    value : (table, key, kwargs)
        Class of table to populate, key to populate, and kwargs for populate
    """
    table, key, kwargs = value
    return table.populate(key, **kwargs)


class NonDaemonPool(multiprocessing.pool.Pool):
    """Non-daemonized pool for multiprocessing.

    Used to create a pool of non-daemonized processes, which are required for
    parallel populate operations in DataJoint.
    """

    # Explicitly set the start method to 'fork'
    # Allows the pool to be used in MacOS, where the default start method is 'spawn'
    multiprocessing.set_start_method("fork", force=True)

    def Process(self, *args, **kwds):
        """Return a non-daemonized process."""
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc


def str_to_bool(value) -> bool:
    """Return whether the provided string represents true. Otherwise false."""
    # Due to distutils equivalent depreciation in 3.10
    # Adopted from github.com/PostHog/posthog/blob/master/posthog/utils.py
    if not value:
        return False
    return str(value).lower() in ("y", "yes", "t", "true", "1")


def bytes_to_human_readable(size: int) -> str:
    """Convert a byte size to a human-readable format."""
    msg_template = "{size:.2f} {unit}"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return msg_template.format(size=size, unit=unit)
        size /= 1024

    return msg_template.format(size=size, unit="PB")


def accept_divergence(
    key: str,
    new_value: Any,
    existing_value: Any,
    test_mode: bool = False,
    table_name: Optional[str] = None,
):
    """Prompt to accept divergence in values between existing and new entries

    Parameters
    ----------
    key : str
        Name of the column where the divergence is found
    new_value : Any
        New value to be inserted into the table
    existing_value : Any
        Existing value in the table that is different from the new value
    test_mode : bool, optional
        If True, will not prompt and return False, by default False
    table_name : str, optional
        Name of the table where the divergence is found, by default None
    """
    if test_mode:
        # If get here in test mode, is because want to test failure
        logger.warning(
            "accept_divergence called in test mode, returning False w/o prompt"
        )
        return False
    tbl_msg = ""
    if table_name:  # optional message with table name
        tbl_msg = f" of '{table_name}'"
    response = dj.utils.user_choice(
        f"Existing entry differs in '{key}' column{tbl_msg}.\n"
        + "Accept the existing value of: \n"
        + f"'{existing_value}' \n"
        + "in place of the new value: \n"
        + f"'{new_value}' ?\n"
    )
    return str_to_bool(response)


def _replace_nan_with_default(data_dict, default_value=-1.0):
    """
    Replace NaN values in a dictionary with a default value.

    This is necessary because DataJoint cannot properly format queries
    with NaN values, causing errors during probe insertion/validation.

    Args:
        data_dict: Dictionary that may contain NaN values
        default_value: Value to replace NaN with (default: -1.0)

    Returns:
        Dictionary with NaN values replaced
    """
    if not isinstance(data_dict, dict):
        return data_dict

    result = data_dict.copy()
    for key, value in result.items():
        if isinstance(value, float) and math.isnan(value):
            result[key] = default_value

    return result
