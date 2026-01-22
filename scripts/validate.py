#!/usr/bin/env python3
"""Validate Spyglass installation.

This script checks that Spyglass is properly installed and configured.
It can be run standalone or called by the installer.

Usage:
    python scripts/validate.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import os
import re
import sys
from pathlib import Path
from typing import Callable, NamedTuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


class Check(NamedTuple):
    """Represents a validation check to run.

    Attributes
    ----------
    name : str
        Human-readable name of the check
    func : Callable[[], None]
        Function to execute for this check
    critical : bool
        If True, failure causes validation to fail (default: True)
        If False, failure only produces warning
    """

    name: str
    func: Callable[[], None]
    critical: bool = True


def check_python_version() -> None:
    """Check Python version meets minimum requirement.

    Reads requirement from pyproject.toml to maintain single source of truth.
    Falls back to hardcoded (3, 9) if parsing fails.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If Python version is below minimum requirement
    """
    min_version = get_required_python_version()

    if sys.version_info < min_version:
        raise RuntimeError(
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )

    print(
        f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def get_required_python_version() -> tuple[int, int]:
    """Get required Python version from pyproject.toml.

    This ensures single source of truth for version requirements.
    Falls back to (3, 9) if parsing fails.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        Tuple of (major, minor) version numbers as integers

    Examples
    --------
    >>> major, minor = get_required_python_version()
    >>> print(f"Requires Python {major}.{minor}+")

    Notes
    -----
    INTENTIONAL DUPLICATION: This function is duplicated in both install.py
    and validate.py because validate.py must work standalone before Spyglass
    is installed. Both scripts are designed to run independently without
    importing from each other to avoid path/module complexity.

    If you modify this function, you MUST update it in both files:
    - scripts/install.py
    - scripts/validate.py (this file)

    Future: Consider extracting to scripts/_shared.py if the installer
    becomes a package, but for now standalone scripts are simpler.
    """
    fallback = (3, 10)
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Python 3.10
        except ImportError:
            return fallback

    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        # Parse ">=3.10,<3.13" format
        requires_python = data["project"]["requires-python"]
        match = re.search(r">=(\d+)\.(\d+)", requires_python)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, KeyError, AttributeError, ValueError):
        # Expected errors during parsing - use safe fallback
        pass

    return fallback


def check_conda() -> None:
    """Check conda/mamba is available.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If neither conda nor mamba is found in PATH
    """
    import shutil

    conda_cmd = None
    if shutil.which("mamba"):
        conda_cmd = "mamba"
    elif shutil.which("conda"):
        conda_cmd = "conda"
    else:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            print("✓ Package manager: conda (from CONDA_EXE)")
            return
        raise RuntimeError(
            "conda or mamba not found\n"
            "Install from: https://github.com/conda-forge/miniforge"
        )

    print(f"✓ Package manager: {conda_cmd}")


def check_spyglass_import() -> None:
    """Verify spyglass can be imported.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If spyglass package cannot be imported or version is unavailable
    """
    try:
        import spyglass

        version = getattr(spyglass, "__version__", None)
        if version is None:
            raise RuntimeError(
                "Spyglass version not available. This usually means the package "
                "was not installed properly. Try: pip install -e . (from repo root)"
            )
        print(f"✓ Spyglass version: {version}")
    except ImportError as e:
        raise RuntimeError(f"Cannot import spyglass: {e}")


def check_spyglass_config() -> None:
    """Verify SpyglassConfig integration works.

    This is a non-critical check - warns instead of failing.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Prints warnings for configuration issues but does not raise exceptions.
    """
    try:
        from spyglass.settings import SpyglassConfig

        config = SpyglassConfig()
        print("✓ SpyglassConfig loaded")
        print(f"  Base directory: {config.base_dir}")

        base_dir_path = Path(config.base_dir)
        if not base_dir_path.exists():
            print("  Status: Will be created on first use")
        else:
            print("  Status: Ready")
    except Exception as e:
        print(f"⚠ SpyglassConfig warning: {e}")
        print("  This may not be a critical issue")


def check_database() -> None:
    """Test database connection if configured.

    This is a non-critical check - warns instead of failing.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Prints warnings for database issues but does not raise exceptions.
    """
    try:
        import datajoint as dj

        dj.conn().ping()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"⚠ Database not configured: {e}")
        print("  Configure manually or run: python scripts/install.py --docker")


def main() -> None:
    """Run all validation checks.

    Executes suite of validation checks and reports results. Exits with
    code 0 on success, 1 on failure.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("\n" + "=" * 60)
    print("  Spyglass Installation Validation")
    print("=" * 60 + "\n")

    # Define all validation checks
    checks = [
        Check("Python version", check_python_version, critical=True),
        Check("Conda/Mamba", check_conda, critical=True),
        Check("Spyglass import", check_spyglass_import, critical=True),
        Check("SpyglassConfig", check_spyglass_config, critical=False),
        Check("Database connection", check_database, critical=False),
    ]

    critical_failed = []
    warnings = []

    # Run critical checks
    print("Critical Checks:")
    for check in checks:
        if not check.critical:
            continue
        try:
            check.func()
        except Exception as e:
            print(f"✗ {check.name}: {e}")
            critical_failed.append(check.name)

    # Run optional checks
    print("\nOptional Checks:")
    for check in checks:
        if check.critical:
            continue
        try:
            check.func()
        except Exception as e:
            print(f"⚠ {check.name}: {e}")
            warnings.append(check.name)

    # Summary
    print("\n" + "=" * 60)
    if critical_failed:
        print("✗ Validation failed - installation incomplete")
        print("=" * 60 + "\n")
        print("Failed checks:", ", ".join(critical_failed))
        print("\nThese issues must be fixed before using Spyglass.")
        print("See docs/TROUBLESHOOTING.md for help")
        sys.exit(EXIT_FAILURE)
    elif warnings:
        print("⚠ Validation passed with warnings")
        print("=" * 60 + "\n")
        print("Warnings:", ", ".join(warnings))
        print("\nSpyglass is installed but optional features may not work.")
        print("See docs/TROUBLESHOOTING.md for configuration help")
        sys.exit(EXIT_SUCCESS)  # Exit 0 since installation is functional
    else:
        print("✅ All checks passed!")
        print("=" * 60 + "\n")
        sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
