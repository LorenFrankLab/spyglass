"""SpyglassSchema: a dj.Schema subclass that survives missing DB connections.

When ``activate()`` fails (no credentials, host unreachable, or user
interruption), ``SpyglassSchema`` catches the error, sets
``_no_connection = True``, and emits a single ``UserWarning`` pointing the
user at the reconnect instructions.  All table classes decorated by an
offline schema receive the same ``_no_connection`` flag so that
``NoConnectionMixin`` can short-circuit every DB-touching method.

Normal connected usage is entirely unchanged; ``SpyglassSchema`` is a
drop-in replacement for ``dj.Schema``.
"""

import inspect
import warnings

import datajoint as dj
from datajoint.errors import DataJointError, LostConnectionError

_RECONNECT_HINT = (
    "  Tables in this module will return empty results.\n"
    "  To reconnect: dj.conn(reset=True) and re-import the module.\n"
)


class SpyglassSchema(dj.Schema):
    """``dj.Schema`` subclass that degrades gracefully when DB is unreachable.

    Notes
    -----
    Differences from ``dj.Schema``:

    * ``activate()`` catches ``LostConnectionError``, auth-style
      ``DataJointError``, and ``KeyboardInterrupt`` (password prompt
      cancelled).  On failure it sets ``self._no_connection = True`` and
      emits a single ``UserWarning`` — it does NOT raise, so module-level
      ``schema = SpyglassSchema("name")`` never crashes.  The warning
      fires only once per Python session regardless of how many schemas
      fail to connect.
    * ``__call__`` (the decorator) propagates ``_no_connection`` to each
      table class it decorates, including Part tables, without touching
      the DB.

    Parameters
    ----------
    Same as ``dj.Schema``.
    """

    _no_connection: bool = False
    # Class-level flag: UserWarning fires only once per session even when
    # many schemas are activated without a DB (e.g. full spyglass import).
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
        """Activate schema, falling back to offline mode on connection error.

        Parameters
        ----------
        schema_name : str, optional
            Database schema name passed through to ``dj.Schema.activate``.
        connection : datajoint.Connection, optional
            Existing connection object.
        create_schema : bool, optional
            Whether to create the schema if it does not exist.
        create_tables : bool, optional
            Whether to create tables if they do not exist.
        add_objects : dict, optional
            Additional objects to add to the schema context.
        """
        try:
            super().activate(
                schema_name,
                connection=connection,
                create_schema=create_schema,
                create_tables=create_tables,
                add_objects=add_objects,
            )
        except LostConnectionError as err:
            self._offline(schema_name, err)
        except KeyboardInterrupt as err:
            self._offline(schema_name, err)
        except DataJointError as err:
            # DataJointError is broad; only treat it as a connection
            # failure when the message indicates an auth/connectivity
            # problem.  Re-raise schema logic and declare errors so real
            # activation bugs are not silently swallowed.
            msg = str(err).lower()
            if not any(
                phrase in msg
                for phrase in (
                    "access denied",
                    "can't connect",
                    "unknown mysql server host",
                    "lost connection",
                    "connection refused",
                    "no connection",
                )
            ):
                raise
            self._offline(schema_name, err)
        except Exception as err:
            # pymysql.err.OperationalError (2003/2006/2013) and similar
            # low-level connection errors are not always wrapped by
            # DataJoint before bubbling up.  Treat any exception whose
            # first arg is a known MySQL connection-refused code as a
            # connection failure; re-raise everything else so schema/SQL
            # logic errors still surface.
            code = err.args[0] if err.args else None
            if code not in (2003, 2006, 2013):
                raise
            self._offline(schema_name, err)

    # ------------------------------------------------------------------
    # __call__ — propagate flag to decorated table classes
    # ------------------------------------------------------------------

    def __call__(self, cls, *, context=None):
        """Decorate *cls*, propagating offline state when not connected.

        When connected, the caller's frame is captured before delegating
        to ``dj.Schema.__call__``.  DataJoint resolves foreign-key
        references via ``inspect.currentframe().f_back.f_locals``, which
        from inside ``dj.Schema.__call__`` would normally point at the
        decorated module.  Introducing this subclass adds one extra frame,
        so without the explicit ``context`` hand-off DataJoint would see
        ``dj_schema.py`` locals instead of the module's globals and fail
        to resolve sibling table classes.

        Parameters
        ----------
        cls : type
            Table class to decorate.
        context : dict, optional
            Symbol table for FK resolution.  When ``None`` and the schema
            is connected, the caller's frame locals and globals are
            captured automatically so that sibling table references
            resolve correctly.

        Returns
        -------
        type
            The decorated (or offline-stamped) table class.
        """
        if not self._no_connection:
            if context is None:
                # Capture the module-level frame that applied @schema so
                # FK lookups (e.g. -> DataAcquisitionDeviceSystem) work.
                frame = inspect.currentframe().f_back
                context = {**frame.f_globals, **frame.f_locals}
            return super().__call__(cls, context=context)

        # Offline path: stamp the flag on the class and any Part classes.
        # Avoid calling _decorate_master / _decorate_table — both require
        # a live connection (Heading construction, SHOW TABLES query).
        cls._no_connection = True
        cls.database = self.database
        self._stamp_parts(cls)
        return cls

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _offline(self, schema_name, err):
        """Mark schema offline and emit a one-time ``UserWarning``.

        Parameters
        ----------
        schema_name : str
            Schema name stored on ``self.database``.
        err : Exception
            Exception that triggered the offline transition; its type
            name is included in the warning message.
        """
        self._no_connection = True
        if schema_name is not None:
            self.database = schema_name
        if not SpyglassSchema._warned:
            SpyglassSchema._warned = True
            warnings.warn(
                f"\n\n[Spyglass] COULD NOT CONNECT to schema "
                f"'{schema_name}': {type(err).__name__}."
                f"\n\n{_RECONNECT_HINT}",
                UserWarning,
                # stacklevel 3: _offline → activate → dj.Schema.__init__
                # Points at the calling module's SpyglassSchema(...) call.
                stacklevel=3,
            )

    @staticmethod
    def _stamp_parts(master_cls):
        """Set ``_no_connection`` on all Part inner-classes of *master_cls*.

        Parameters
        ----------
        master_cls : type
            Master table class whose Part sub-classes should be stamped.
        """
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
