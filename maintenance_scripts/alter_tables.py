"""Update the MySQL tables with the latest table definitions.

NOTES:
- import all classes for datajoint to resolve foreign key references
- datajoint does not allow altering primary keys, indexes, or foreign keys
- datajoint cannot resolve foreign key references of master from part table,
    so part tables cannot be updated yet
- until https://github.com/datajoint/datajoint-python/issues/943 is resolved,
    some enum definitions will prompt alter() even without a change
"""

import importlib
import inspect
import warnings

import datajoint as dj

from spyglass.common import *  # noqa: F401, F403
from spyglass.decoding import *  # noqa: F401, F403
from spyglass.lfp import *  # noqa: F401, F403
from spyglass.linearization import *  # noqa: F401, F403
from spyglass.mua import *  # noqa: F401, F403
from spyglass.position import *  # noqa: F401, F403
from spyglass.ripple import *  # noqa: F401, F403
from spyglass.sharing import *  # noqa: F401, F403
from spyglass.spikesorting import *  # noqa: F401, F403
from spyglass.utils.database_settings import SHARED_MODULES


def main():
    """Run `alter` on all tables in the shared modules."""
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    for submodule in SHARED_MODULES:
        update_submodule(submodule)


def update_submodule(submodule):
    """Run `alter` on all tables in `spyglass.submodule`."""
    module = importlib.import_module(f"spyglass.{submodule}")
    for name, cls in inspect.getmembers(module, inspect.isclass):
        update_cls(cls)
        for attrname in dir(cls):
            attr = getattr(cls, attrname)
            if isinstance(attr, type) and is_dj_table(attr):
                update_cls(attr)


def is_dj_table(cls):
    """Return True if `cls` is a DataJoint table class."""
    return issubclass(
        cls, (dj.Manual, dj.Lookup, dj.Imported, dj.Computed, dj.Part)
    )


def update_cls(cls):
    """Run `alter` on `cls`, print all errors."""
    print("Updating", cls)
    try:
        cls.alter()
    except Exception as e:
        print("ERROR:", e)
    print()


if __name__ == "__main__":
    main()
