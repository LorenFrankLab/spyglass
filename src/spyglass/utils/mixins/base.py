"""Common mixin classes for Spyglass topic-specific mixins."""

from functools import cached_property
from re import match as re_match


class BaseMixin:

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
