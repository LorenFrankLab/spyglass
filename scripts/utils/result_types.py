"""Result type system for explicit error handling.

This module provides Result types as recommended in REVIEW.md to replace
exception-heavy error handling with explicit success/failure contracts.
"""

from typing import TypeVar, Generic, Union, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


T = TypeVar("T")
E = TypeVar("E")


class Severity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ValidationError:
    """Structured validation error with context."""

    message: str
    field: str
    severity: Severity = Severity.ERROR
    recovery_actions: List[str] = None

    def __post_init__(self):
        if self.recovery_actions is None:
            object.__setattr__(self, "recovery_actions", [])


@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result containing a value."""

    value: T
    message: str = ""

    @property
    def is_success(self) -> bool:
        return True

    @property
    def is_failure(self) -> bool:
        return False


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Failed result containing error information."""

    error: E
    message: str
    context: dict = None
    recovery_actions: List[str] = None

    def __post_init__(self):
        if self.context is None:
            object.__setattr__(self, "context", {})
        if self.recovery_actions is None:
            object.__setattr__(self, "recovery_actions", [])

    @property
    def is_success(self) -> bool:
        return False

    @property
    def is_failure(self) -> bool:
        return True


# Type alias for common Result pattern
Result = Union[Success[T], Failure[E]]


# Convenience functions for creating results
def success(value: T, message: str = "") -> Success[T]:
    """Create a successful result."""
    return Success(value, message)


def failure(
    error: E,
    message: str,
    context: dict = None,
    recovery_actions: List[str] = None,
) -> Failure[E]:
    """Create a failed result."""
    return Failure(error, message, context or {}, recovery_actions or [])


def validation_failure(
    field: str,
    message: str,
    severity: Severity = Severity.ERROR,
    recovery_actions: List[str] = None,
) -> Failure[ValidationError]:
    """Create a validation failure result."""
    error = ValidationError(message, field, severity, recovery_actions or [])
    return Failure(error, f"Validation failed for {field}: {message}")


# Common validation result type
ValidationResult = Union[Success[None], Failure[ValidationError]]


def validation_success(message: str = "Validation passed") -> Success[None]:
    """Create a successful validation result."""
    return Success(None, message)


# Helper functions for working with Results
def collect_errors(results: List[Result]) -> List[Failure]:
    """Collect all failures from a list of results."""
    return [r for r in results if r.is_failure]


def all_successful(results: List[Result]) -> bool:
    """Check if all results are successful."""
    return all(r.is_success for r in results)


def first_error(results: List[Result]) -> Optional[Failure]:
    """Get the first error from a list of results, or None if all successful."""
    for result in results:
        if result.is_failure:
            return result
    return None


# System-specific error types
@dataclass(frozen=True)
class SystemRequirementError:
    """System requirement not met."""

    requirement: str
    found: Optional[str]
    minimum: Optional[str]
    suggestion: str


@dataclass(frozen=True)
class DockerError:
    """Docker-related error."""

    operation: str
    docker_available: bool
    daemon_running: bool
    permission_error: bool


@dataclass(frozen=True)
class NetworkError:
    """Network-related error."""

    operation: str
    url: Optional[str]
    timeout: bool
    connection_refused: bool


@dataclass(frozen=True)
class DiskSpaceError:
    """Disk space related error."""

    path: str
    required_gb: float
    available_gb: float
    operation: str


# Convenience type aliases
SystemResult = Union[Success[Any], Failure[SystemRequirementError]]
DockerResult = Union[Success[Any], Failure[DockerError]]
NetworkResult = Union[Success[Any], Failure[NetworkError]]
DiskSpaceResult = Union[Success[Any], Failure[DiskSpaceError]]
