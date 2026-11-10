"""Common mixin classes for Spyglass topic-specific mixins."""

from functools import cached_property
from re import match as re_match


class BaseMixin:

    # Set once per session, on the first insert or populate that logs the
    # install. Class-level so every table shares it, as
    # SpyglassMixin._fk_validated does.
    _logged_install_mode = False

    def insert(self, *args, **kwargs):
        """Log a dirty Spyglass install before the session's first insert.

        The import-time hook warns but does not write, so that importing
        spyglass never opens a connection. This is where the record is made:
        by the first insert a connection exists, and a process that never
        writes is never logged.

        At most one call per session does any work, so every later insert
        costs one attribute read.
        """
        self._log_install_once()
        return super().insert(*args, **kwargs)

    def _log_install_once(self) -> None:
        """Record a dirty install once per session (best-effort).

        Skipped under test_mode, as the import-time warning is: the suite runs
        from a developer's working tree, which is dirty by definition, so
        logging here would write rows the tests under way did not ask for.
        """
        if BaseMixin._logged_install_mode or self._test_mode:
            return

        # Set on BaseMixin, not self: an instance attribute would shadow the
        # class one and leave every other table unguarded. Set before logging,
        # because the insert it makes re-enters this hook.
        BaseMixin._logged_install_mode = True

        try:  # the import is inside the guard: it declares a schema
            from spyglass.common.common_user import UserEnvironment

            UserEnvironment().log_install()
        except Exception as exc:  # never block the caller's own write
            self._logger.debug(f"dirty env logging failed: {exc}")

    def restrict(self, restriction, *args, **kwargs):
        """Intercept no-op restrict calls to throw warning.

        DataJoint returns the same object (ret is self) only when make_condition
        evaluates the restriction to True — i.e. no shared attributes were found
        and the restriction had no effect.
        """
        ret = super().restrict(restriction, *args, **kwargs)
        if ret is self and restriction is not True and restriction is not False:
            self._warn_msg(
                "Restriction had no effect — no shared attributes found. "
                "Please check your restrict() call."
            )
        return ret

    @cached_property
    def _logger(self):
        """Lazy import of logger to avoid circular imports.

        Used by ...
        - CautiousDeleteMixin
        - PopulateMixin
        - RestrictByMixin
        - ExportMixin
        - AnalysisMixin
        - Merge
        """

        from spyglass.utils import logger

        return logger

    @cached_property
    def _graph_deps(self) -> list:
        """Dependencies for graph search and restriction.

        Used by ...
        - RestrictByMixin
        - CautiousDeleteMixin
        - PopulateMixin (upstream hash)
        """
        from spyglass.utils.dj_graph import RestrGraph  # noqa #F401
        from spyglass.utils.dj_graph import TableChain

        return [TableChain, RestrGraph]

    def _info_msg(self, msg: str) -> None:
        """Log info message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.

        Used by ...
        - AnalysisMixin.copy and .create
        - IngestionMixin._insert_logline
        - Merge._merge_repr
        """
        log = self._logger.debug if self._test_mode else self._logger.info
        log(msg)

    def _warn_msg(self, msg: str) -> None:
        """Log warning message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.
        """
        log = self._logger.debug if self._test_mode else self._logger.warning
        log(msg)

    def _err_msg(self, msg: str) -> None:
        """Log error message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.
        """
        log = self._logger.debug if self._test_mode else self._logger.error
        log(msg)

    @cached_property
    def _test_mode(self) -> bool:
        """Return True if in test mode.

        Avoids circular import. Prevents prompt on delete.

        Note: Using cached property b/c we don't expect test_mode to change
        during runtime, and it avoids repeated lookups. Changing to @property
        wouldn't reload the config. It would just re-fetch from the settings
        module.

        Used by ...
        - BaseMixin._spyglass_version
        - HelpersMixin
        """
        from spyglass.settings import config as sg_config

        return sg_config.get("test_mode", False)

    @cached_property
    def _spyglass_version(self):
        """Get Spyglass version.

        Used by ...
        - ExportMixin
        - AnalysisMixin
        """
        from spyglass import __version__ as sg_version

        ret = ".".join(sg_version.split(".")[:3])  # Ditch commit info

        if self._test_mode:
            return ret[:16] if len(ret) > 16 else ret

        if not bool(re_match(r"^\d+\.\d+\.\d+", ret)):  # Major.Minor.Patch
            raise ValueError(
                f"Spyglass version issues. Expected #.#.#, Got {ret}."
                + "Please try running `hatch build` from your spyglass dir."
            )

        return ret
