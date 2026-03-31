"""SpyglassSchema: a dj.Schema subclass that survives missing DB connections.

When ``activate()`` fails (no credentials, host unreachable, or user
interruption), ``SpyglassSchema`` catches the error, sets
``_no_connection = True``, and emits a single ``UserWarning`` pointing the
user at the reconnect instructions.  All table classes decorated by an offline
schema receive the same ``_no_connection`` flag so that ``NoConnectionMixin``
can short-circuit every DB-touching method.

Normal connected usage is entirely unchanged; ``SpyglassSchema`` is a
drop-in replacement for ``dj.Schema``.
"""

import warnings

import datajoint as dj
from datajoint.errors import DataJointError, LostConnectionError

_RECONNECT_HINT = (
    "  Tables in this module will return empty results.\n"
    "  To reconnect: dj.conn(reset=True) and re-import the module.\n"
)


class SpyglassSchema(dj.Schema):
    """dj.Schema subclass that degrades gracefully when DB is unreachable.

    Differences from ``dj.Schema``
    --------------------------------
    * ``activate()`` catches ``LostConnectionError``, ``DataJointError``, and
      ``KeyboardInterrupt`` (password prompt cancelled).  On failure it sets
      ``self._no_connection = True`` and emits a single ``UserWarning`` — it
      does NOT raise, so module-level ``schema = SpyglassSchema("name")`` never
      crashes.  The warning is emitted only once per Python session no matter
      how many schemas fail to connect.
    * ``__call__`` (the decorator) propagates ``_no_connection`` to each table
      class it decorates, including Part tables, without touching the DB.

    Parameters
    ----------
    Same as ``dj.Schema``.
    """

    _no_connection: bool = False
    # Class-level flag so the UserWarning fires only once per session even
    # when many schemas are activated without a DB (e.g. a full spyglass import).
    _warned: bool = False

    # ------------------------------------------------------------------
    # activate — intercept connection errors
    # ------------------------------------------------------------------

    def activate(
        self,
        schema_name=None,
        *,
        connection=None,
        create_schema=None,
        create_tables=None,
        add_objects=None,
    ):
        """Activate schema, falling back to offline mode on connection error."""
        try:
            super().activate(
                schema_name,
                connection=connection,
                create_schema=create_schema,
                create_tables=create_tables,
                add_objects=add_objects,
            )
        except (LostConnectionError, DataJointError, KeyboardInterrupt) as err:
            self._offline(schema_name, err)
        except Exception as err:
            # pymysql.err.OperationalError (2003/2006/2013) and similar
            # low-level connection errors are not always wrapped by DataJoint
            # before bubbling up.  Treat any exception whose args contain a
            # known MySQL connection-refused code as a connection failure;
            # re-raise everything else so schema/SQL logic errors still surface.
            code = err.args[0] if err.args else None
            if code not in (2003, 2006, 2013):
                raise
            self._offline(schema_name, err)

    # ------------------------------------------------------------------
    # __call__ — propagate flag to decorated table classes
    # ------------------------------------------------------------------

    def __call__(self, cls, *, context=None):
        """Decorate a table class, or mark it offline if no connection."""
        if not self._no_connection:
            return super().__call__(cls, context=context)

        # Offline path: stamp the flag on the class and any Part sub-classes.
        # Avoid calling _decorate_master / _decorate_table — both require a
        # live connection (Heading construction, SHOW TABLES query, etc.).
        cls._no_connection = True
        cls.database = self.database
        self._stamp_parts(cls)
        return cls

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _offline(self, schema_name, err):
        """Set offline state; emit the UserWarning only once per session."""
        self._no_connection = True
        if schema_name is not None:
            self.database = schema_name
        if not SpyglassSchema._warned:
            SpyglassSchema._warned = True
            warnings.warn(
                f"\n\n[Spyglass] COULD NOT CONNECT to schema '{schema_name}': "
                f"{type(err).__name__}.\n\n{_RECONNECT_HINT}",
                UserWarning,
                stacklevel=3,  # points at the calling module's SpyglassSchema(...)
            )

    @staticmethod
    def _stamp_parts(master_cls):
        """Set _no_connection on any Part inner-classes of *master_cls*."""
        import inspect

        from datajoint.user_tables import Part

        for attr_name in dir(master_cls):
            if not attr_name[0].isupper():
                continue
            attr = getattr(master_cls, attr_name, None)
            if attr is None:
                continue
            if inspect.isclass(attr) and issubclass(attr, Part):
                attr._no_connection = True
                attr.database = master_cls.database
                attr._master = master_cls
