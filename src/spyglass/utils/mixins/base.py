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

    @cached_property
    def _test_mode(self) -> bool:
        """Return True if in test mode.

        Avoids circular import. Prevents prompt on delete.

        Used by ...
        - BaseMixin._spyglass_version
        - HelpersMixin
        """
        from spyglass.settings import test_mode

        return test_mode

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
