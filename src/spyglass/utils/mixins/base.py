"""Common mixin classes for Spyglass topic-specific mixins."""

from functools import cached_property


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
