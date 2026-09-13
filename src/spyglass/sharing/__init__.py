"""Ways of sharing Spyglass files with other people.

Three modules live here: `sharing_kachery`, which is being retired;
`sharing_store`, the DataJoint record of what has been shared through the
broker; and `store_client`, which talks to it.

Every table name is resolved lazily. Declaring a schema opens a database
connection at import time, and `spyglass-store login` has to work *before* a
user has a working DataJoint config — that is the whole point of a one-command
login. Importing `spyglass.sharing.store_client` runs this module first, so an
eager table import would make the login command require the very setup it
exists to precede.
"""

_LAZY_NAMES = {
    "AnalysisNwbfileKachery": "sharing_kachery",
    "AnalysisNwbfileKacherySelection": "sharing_kachery",
    "KacheryZone": "sharing_kachery",
    "share_data_to_kachery": "sharing_kachery",
    "AnalysisFileSelection": "sharing_store",
    "SharedAnalysisFile": "sharing_store",
    "SharedFile": "sharing_store",
    "SharedFileSelection": "sharing_store",
    "most_restrictive": "sharing_store",
    "share_file": "sharing_store",
}

from spyglass.sharing.store_client import (  # noqa: E402, F401
    StoreAuthError,
    StoreClient,
    StoreError,
    StoreForbidden,
    StoreNotConfigured,
    StoreNotFound,
    StoreQuotaExceeded,
    get_client,
)


def __getattr__(name):
    """Resolve a table name on first use, so import alone hits no database.

    Parameters
    ----------
    name : str
        Attribute being looked up on this module.

    Returns
    -------
    Any
        The requested table or function.

    Raises
    ------
    AttributeError
        For any other name, as a module normally would.
    """
    module_name = _LAZY_NAMES.get(name)

    if module_name is not None:
        from importlib import import_module

        module = import_module(f"{__name__}.{module_name}")

        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List both the eager and the lazily-resolved names."""
    return sorted(set(globals()) | set(_LAZY_NAMES))
