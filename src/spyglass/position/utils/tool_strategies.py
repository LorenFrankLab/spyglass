"""Tool strategy pattern for pose estimation model training.

Provides a pluggable architecture for different pose estimation tools
(DLC, SLEAP, etc.) with consistent interfaces and parameter management.

The interface lives in ``tool_strategy_base`` and each tool's implementation
in ``tool_strategy_<tool>``. All three names stay importable from here, which
is where callers and tests have always taken them from.
"""

from .tool_strategy_base import PoseToolStrategy
from .tool_strategy_dlc import DLCStrategy
from .tool_strategy_sleap import SLEAPStrategy

__all__ = [
    "DLCStrategy",
    "PoseToolStrategy",
    "SLEAPStrategy",
    "ToolStrategyFactory",
]


class ToolStrategyFactory:
    """Factory for creating tool strategy instances."""

    _strategies = {
        "DLC": DLCStrategy,
        "SLEAP": SLEAPStrategy,
    }

    @classmethod
    def create_strategy(cls, tool: str) -> PoseToolStrategy:
        """Create strategy instance for specified tool.

        Parameters
        ----------
        tool : str
            Tool name ("DLC", "SLEAP")

        Returns
        -------
        PoseToolStrategy
            Strategy instance for the tool

        Raises
        ------
        ValueError
            If tool is not supported
        """
        if tool not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported tool: {tool}. Available: {available}"
            )

        return cls._strategies[tool]()

    @classmethod
    def tool_for_suffix(cls, suffix: str) -> str:
        """Which tool claims a project file with this suffix.

        Each strategy declares its own `project_suffixes`, so adding a tool
        does not mean editing a caller's extension map.

        Parameters
        ----------
        suffix : str
            File suffix without the leading dot, any case.

        Returns
        -------
        str or None
            The tool name, or None when no strategy claims it.
        """
        suffix = suffix.lower().lstrip(".")
        for tool, strategy in cls._strategies.items():
            if suffix in strategy.project_suffixes:
                return tool
        return None

    @classmethod
    def get_available_tools(cls) -> list[str]:
        """Get list of available tool names."""
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, tool: str, strategy_class: type) -> None:
        """Register a new tool strategy (for extensibility).

        Parameters
        ----------
        tool : str
            Tool name
        strategy_class : type
            Strategy class implementing PoseToolStrategy
        """
        if not issubclass(strategy_class, PoseToolStrategy):
            raise ValueError(
                "Strategy class must implement PoseToolStrategy interface"
            )
        cls._strategies[tool] = strategy_class
