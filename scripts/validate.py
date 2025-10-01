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

import sys
from pathlib import Path


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


def get_required_python_version() -> tuple:
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
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Python 3.9-3.10
        except ImportError:
            return (3, 9)  # Safe fallback

    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Parse ">=3.9,<3.13" format
        requires_python = data["project"]["requires-python"]
        import re

        match = re.search(r">=(\d+)\.(\d+)", requires_python)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass

    return (3, 9)  # Safe fallback


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
        If spyglass package cannot be imported
    """
    try:
        import spyglass

        version = getattr(spyglass, "__version__", "unknown")
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

        if not config.base_dir.exists():
            print("  Note: Base directory will be created on first use")
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

    checks = [
        ("Python version", check_python_version),
        ("Conda/Mamba", check_conda),
        ("Spyglass import", check_spyglass_import),
        ("SpyglassConfig", check_spyglass_config),
        ("Database connection", check_database),
    ]

    failed = []
    for name, check_fn in checks:
        try:
            check_fn()
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"✗ {len(failed)} check(s) failed")
        print("=" * 60 + "\n")
        print("Failed checks:", ", ".join(failed))
        print("\nSee docs/TROUBLESHOOTING.md for help")
        sys.exit(1)
    else:
        print("✅ All checks passed!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
