"""Helper functions for manipulating information from DataJoint fetch calls."""

import inspect
import os
from typing import Type

import datajoint as dj
import numpy as np
from datajoint.user_tables import UserTable

from spyglass.utils.dj_chains import PERIPHERAL_TABLES
from spyglass.utils.logging import logger
from spyglass.utils.nwb_helper_fn import get_nwb_file


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
        logger.warn(
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


def get_fetching_table_from_stack(stack):
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
    if len(classes) > 1:
        logger.warn(
            f"Multiple classes found in stack: {classes}. "
            "Please submit a bug report with the snippet used."
        )
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

    kwargs["as_dict"] = True  # force return as dictionary
    attrs = attrs or query_expression.heading.names  # if none, all

    which = "analysis" if "analysis" in attr_name else "nwb"
    tbl_map = {  # map to file_name_str and file_path_fn
        "analysis": ["analysis_file_name", AnalysisNwbfile.get_abs_path],
        "nwb": ["nwb_file_name", Nwbfile.get_abs_path],
    }
    file_name_str, file_path_fn = tbl_map[which]

    # TODO: check that the query_expression restricts tbl - CBroz
    nwb_files = (
        query_expression * tbl.proj(nwb2load_filepath=attr_name)
    ).fetch(file_name_str)

    if which == "analysis":  # log access of analysis files to log table
        AnalysisNwbfile().increment_access(
            nwb_files, table=get_fetching_table_from_stack(inspect.stack())
        )

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
    kwargs["as_dict"] = True  # force return as dictionary

    tbl, attr_name = nwb_master

    if not attrs:
        attrs = query_expression.heading.names

    nwb_files, file_path_fn = get_nwb_table(
        query_expression, tbl, attr_name, *attrs, **kwargs
    )

    for file_name in nwb_files:
        file_path = file_path_fn(file_name)
        if not os.path.exists(file_path):  # retrieve the file from kachery.
            # This also opens the file and stores the file object
            get_nwb_file(file_path)

    rec_dicts = (
        query_expression * tbl.proj(nwb2load_filepath=attr_name)
    ).fetch(*attrs, "nwb2load_filepath", **kwargs)

    if not rec_dicts or not np.any(
        ["object_id" in key for key in rec_dicts[0]]
    ):
        return rec_dicts

    ret = []
    for rec_dict in rec_dicts:
        nwbf = get_nwb_file(rec_dict.pop("nwb2load_filepath"))
        # for each attr that contains substring 'object_id', store key-value: attr name to NWB object
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
