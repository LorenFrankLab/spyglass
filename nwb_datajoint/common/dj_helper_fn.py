# helper functions for manipulating information from DataJoing fetch calls
import numpy as np
import pynwb
import re

def dj_replace(original_table, new_values, key_column, replace_column):
    '''
    Given the output of a fetch() call from a schema and a 2D array made up of (key_value, replace_value) tuples,
    finds each instance of key_value in the key_column of the original table and replaces the specified replace_column
    with the associated replace_value.
    Key values must be unique.
    :param original_table: result of a datajoint .fetch() call on a schema query
    :param new_values: list of tuples, each containing (key_value, replace_value)
    :param index_column: string - the name of the column where the key_values are located
    :param replace_column: string - the name of the column where to-be-replaced values are located
    :return: structured array of new table entries that can be inserted back into the schema
    '''
 
    # check to make sure the new_values are a list or array of tuples and fix if not
    if type(new_values) is tuple:
        tmp = list()
        tmp.append(new_values)
        new_values = tmp
    
    new_val_array = np.asarray(new_values)
    replace_ind = np.where(np.isin(original_table[key_column], new_val_array[:,0]))
    original_table[replace_column][replace_ind] = new_val_array[:,1]
    return original_table


def fetch_nwb(query_expression, nwb_master, *attrs, **kwargs):
    """
    :param query_expression: a DJ query expression (e.g. join, restrict) or a table to call fetch on
    :param nwb_master: tuple of (table, attr) to get the NWB filepath from
    :param attrs: attrs from normal fetch()
    :param kwargs: kwargs from normal fetch()
    :return: fetched list of dict
    """
    kwargs['as_dict'] = True  # force return as dictionary
    tbl, attr_name = nwb_master
    
    if not attrs:
        attrs = query_expression.heading.names

    rec_dicts = (query_expression * tbl.proj(nwb2load_filepath=attr_name)).fetch(*attrs, 'nwb2load_filepath', **kwargs)
    
    if not rec_dicts or not np.any(['object_id' in key for key in rec_dicts[0]]):
        return rec_dicts
    
    ret = []
    print(rec_dicts)
    for rec_dict in rec_dicts:
        io = pynwb.NWBHDF5IO(rec_dict.pop('nwb2load_filepath'), mode='r')
        nwbf = io.read()
        nwb_objs = {re.sub('(_?)object_id', '', id_attr): nwbf.objects[rec_dict[id_attr]]
                    for id_attr in attrs if 'object_id' in id_attr and rec_dict[id_attr] != ''}                  
        ret.append({**rec_dict, **nwb_objs})
    return ret

 