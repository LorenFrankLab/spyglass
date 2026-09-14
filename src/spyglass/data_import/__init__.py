"""Reading an NWB file into the database: plan it, then insert it."""

__all__ = ["insert_sessions"]


def __getattr__(name):
    """Resolve `insert_sessions` on first use.

    `ingestion_plan` holds the vocabulary the ingestion mixin speaks, so it is
    imported while `spyglass.utils` is still initializing -- which runs this
    module. Importing `insert_sessions` here would pull in `spyglass.common`,
    and every table it declares, before the mixin those tables inherit from
    exists. Deferring keeps `spyglass.data_import.insert_sessions` working as
    an attribute without that cycle.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from spyglass.data_import.insert_sessions import insert_sessions

    # Importing the submodule binds it here under the same name as the
    # function it holds. Rebinding puts the function back, so that both
    # `from spyglass.data_import import insert_sessions` and attribute access
    # give the callable, as they did when this was a plain import.
    globals()["insert_sessions"] = insert_sessions

    return insert_sessions
