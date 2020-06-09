# helper functions for manipulating information from DataJoing fetch calls
import numpy as np

def replace(original_table, new_values, key_column, replace_column):
    '''
    Given the output of a fetch() call from a schema and a 2D array made up of [key_value, replace_value] tuples,
    finds each instance of key_value in the key_column of the original table and replaces the specified replace_column
    with the associated replace_value.
    Key values must be unique.
    :param original_table: result of a datajoint .fetch() call on a schema query
    :param new_values: list or array of tuples, each containing [key_value, replace_value]
    :param index_column: string - the name of the column where the key_values are located
    :param replace_column: string - the name of the column where to-be-replaced values are located
    :return: structured array of new table entries that can be inserted back into the schema
    '''
    # sort the new values so we can use search sorted
    new_values = np.sort(new_values,0)

    replace_ind = np.searchsorted(original_table[key_column], new_values[:,0])
    # check to make sure the indeces agree
    if not np.array_equiv(original_table[key_column][replace_ind], new_values[:,0]):
        print(original_table[key_column][replace_ind], new_values[:,0])
        print('Error in replace: indeces for new values do not match indeces in original table')
        return

    original_table[replace_column][replace_ind] = new_values[:,1]
    return original_table