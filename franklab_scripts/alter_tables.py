"""Update the MySQL tables with the latest table definitions in each nwb_datajoint.common class."""

import datajoint as dj
import importlib
import inspect

# NOTE: for some reason, all classes need to be imported first for datajoint to be able to resolve foreign
# key references properly in the code below
from nwb_datajoint.common import *  # noqa: F401,F403


def main():
    module = importlib.import_module('nwb_datajoint.common')
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, (dj.Manual, dj.Lookup, dj.Imported, dj.Computed, dj.Part)):
            print('Updating', obj)
            try:
                # NOTE: datajoint does not allow altering indexes yet
                # this should affect only AnalysisNwbfile
                obj.alter()
            except Exception as e:
                print('ERROR:', e)
            print()


if __name__ == '__main__':
    main()
