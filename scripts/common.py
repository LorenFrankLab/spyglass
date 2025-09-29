"""
Common utilities shared between Spyglass scripts.

This module provides shared functionality for quickstart.py and validate_spyglass.py
to improve consistency and reduce code duplication.
"""

from collections import namedtuple
from enum import Enum
from typing import NamedTuple


# Shared color definitions using namedtuple (immutable and functional)
Colors = namedtuple(
    "Colors",
    [
        "HEADER",
        "OKBLUE",
        "OKCYAN",
        "OKGREEN",
        "WARNING",
        "FAIL",
        "ENDC",
        "BOLD",
        "UNDERLINE",
    ],
)(
    HEADER="\033[95m",
    OKBLUE="\033[94m",
    OKCYAN="\033[96m",
    OKGREEN="\033[92m",
    WARNING="\033[93m",
    FAIL="\033[91m",
    ENDC="\033[0m",
    BOLD="\033[1m",
    UNDERLINE="\033[4m",
)

# Disabled colors for no-color mode
DisabledColors = namedtuple(
    "DisabledColors",
    [
        "HEADER",
        "OKBLUE",
        "OKCYAN",
        "OKGREEN",
        "WARNING",
        "FAIL",
        "ENDC",
        "BOLD",
        "UNDERLINE",
    ],
)(*([""] * 9))


# Shared exception hierarchy
class SpyglassSetupError(Exception):
    """Base exception for setup errors."""

    pass


class SystemRequirementError(SpyglassSetupError):
    """System doesn't meet requirements."""

    pass


class EnvironmentCreationError(SpyglassSetupError):
    """Environment creation failed."""

    pass


class DatabaseSetupError(SpyglassSetupError):
    """Database setup failed."""

    pass


# Enums for type-safe choices
class MenuChoice(Enum):
    """User menu choices for installation type."""

    MINIMAL = 1
    FULL = 2
    PIPELINE = 3


class DatabaseChoice(Enum):
    """Database setup choices."""

    DOCKER = 1
    EXISTING = 2
    SKIP = 3


class ConfigLocationChoice(Enum):
    """Configuration file location choices."""

    REPO_ROOT = 1
    CURRENT_DIR = 2
    CUSTOM = 3


class PipelineChoice(Enum):
    """Pipeline-specific installation choices."""

    DLC = 1
    MOSEQ_CPU = 2
    MOSEQ_GPU = 3
    LFP = 4
    DECODING = 5


# Configuration constants
class Config:
    """Centralized configuration constants."""

    DEFAULT_TIMEOUT = 1800  # 30 minutes
    DEFAULT_DB_PORT = 3306
    DEFAULT_ENV_NAME = "spyglass"

    # Timeouts for different operations
    TIMEOUTS = {
        "environment_create": 1800,  # 30 minutes for env creation
        "package_install": 600,  # 10 minutes for packages
        "database_check": 60,  # 1 minute for DB readiness
    }
