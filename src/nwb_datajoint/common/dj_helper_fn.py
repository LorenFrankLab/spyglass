"""Helper functions for manipulating information from DataJoint fetch calls."""
import inspect
import re

import datajoint as dj
import numpy as np

from .nwb_helper_fn import get_nwb_file


def dj_replace(original_table, new_values, key_column, replace_column):
    """Given the output of a fetch() call from a schema and a 2D array made up of (key_value, replace_value) tuples,
    find each instance of key_value in the key_column of the original table and replace the specified replace_column
    with the associated replace_value.
    Key values must be unique.

    Parameters
    ----------
    original_table
        Result of a datajoint .fetch() call on a schema query.
    new_values : list
        List of tuples, each containing (key_value, replace_value).
    index_column : str
        The name of the column where the key_values are located.
    replace_column : str
        The name of the column where to-be-replaced values are located.

    Returns
    -------
    original_table
        Structured array of new table entries that can be inserted back into the schema
    """

    # check to make sure the new_values are a list or array of tuples and fix if not
    if type(new_values) is tuple:
        tmp = list()
        tmp.append(new_values)
        new_values = tmp

    new_val_array = np.asarray(new_values)
    replace_ind = np.where(
        np.isin(original_table[key_column], new_val_array[:, 0]))
    original_table[replace_column][replace_ind] = new_val_array[:, 1]
    return original_table


def fetch_nwb(query_expression, nwb_master, *attrs, **kwargs):
    """Get an NWB object from the given DataJoint query.

    Parameters
    ----------
    query_expression
        A DataJoint query expression (e.g., join, restrict) or a table to call fetch on.
    nwb_master : tuple
        Tuple (table, attr) to get the NWB filepath from.
    attrs : list
        Attributes from normal DataJoint fetch call.
    kwargs : dict
        Keyword arguments from normal DataJoint fetch call.

    Returns
    -------
    nwb_objects : list
        List of dicts containing fetch results and NWB objects.
    """
    kwargs['as_dict'] = True  # force return as dictionary
    tbl, attr_name = nwb_master

    if not attrs:
        attrs = query_expression.heading.names

    rec_dicts = (query_expression * tbl.proj(nwb2load_filepath=attr_name)
                 ).fetch(*attrs, 'nwb2load_filepath', **kwargs)

    if not rec_dicts or not np.any(['object_id' in key for key in rec_dicts[0]]):
        return rec_dicts

    ret = []
    for rec_dict in rec_dicts:
        nwbf = get_nwb_file(rec_dict.pop('nwb2load_filepath'))
        # for each attr that contains substring 'object_id', store key-value: attr name to NWB object
        # remove '_object_id' from attr name
        nwb_objs = {re.sub('(_?)object_id', '', id_attr): nwbf.objects[rec_dict[id_attr]]
                    for id_attr in attrs if 'object_id' in id_attr and rec_dict[id_attr] != ''}
        ret.append({**rec_dict, **nwb_objs})
    return ret


def get_child_tables(table):
    table = table() if inspect.isclass(table) else table
    return [dj.FreeTable(table.connection, s if not s.isdigit() else next(iter(table.connection.dependencies.children(s))))
            for s in table.children()]
