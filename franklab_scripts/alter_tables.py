"""Update the MySQL tables with the latest table definitions in each spyglass.common class."""

import importlib
import inspect
import warnings

import datajoint as dj

# NOTE: for some reason, all classes need to be imported first for datajoint to be able to resolve foreign
# key references properly in the code below
from spyglass.common import *  # noqa: F401,F403


def main():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    module = importlib.import_module("spyglass.common")
    for name, cls in inspect.getmembers(module, inspect.isclass):
        update_cls(cls)
        for attrname in dir(cls):
            attr = getattr(cls, attrname)
            if isinstance(attr, type):
                update_cls(attr)


def update_cls(cls):
    if issubclass(
        cls, (dj.Manual, dj.Lookup, dj.Imported, dj.Computed, dj.Part)
    ):
        print("Updating", cls)
        try:
            # NOTE: datajoint does not allow altering indexes yet
            # this should affect only AnalysisNwbfile
            # NOTE: datajoint cannot resolve foreign key references of master from part table yet,
            # so part tables cannot be updated yet
            # NOTE: until https://github.com/datajoint/datajoint-python/issues/943 is resolved,
            # some enum definitions will prompt alter() even when there has not been a change
            # NOTE: datajoint does not allow primary keys to be altered yet
            # this includes changing the length of a varchar primary key
            # (other tables use this primary key, so constraints would probably have to be removed and re-added)
            cls.alter()
        except Exception as e:
            print("ERROR:", e)
        print()


if __name__ == "__main__":
    main()
