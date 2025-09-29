#!/usr/bin/env python3
"""Run pytest tests for Spyglass quickstart scripts.

This script demonstrates how to run tests according to CLAUDE.md conventions.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🧪 {description}")
    print(f"   Command: {' '.join(cmd)}")
    print("   " + "-" * 50)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Errors:\n{result.stderr}", file=sys.stderr)

    return result.returncode

def main():
    """Run various test scenarios."""
    print("=" * 60)
    print("Spyglass Quickstart Test Runner (Pytest)")
    print("=" * 60)

    # Check if pytest is installed
    pytest_check = subprocess.run(["python", "-m", "pytest", "--version"],
                                 capture_output=True, text=True)

    if pytest_check.returncode != 0:
        print("\n❌ pytest is not installed!")
        print("\nTo install pytest:")
        print("  pip install pytest")
        print("\nFor property-based testing, also install:")
        print("  pip install hypothesis")
        return 1

    print(f"\n✅ Using: {pytest_check.stdout.strip()}")

    # Test commands to demonstrate
    test_commands = [
        (["python", "-m", "pytest", "test_quickstart.py", "-v"],
         "Run all quickstart tests (verbose)"),

        (["python", "-m", "pytest", "test_quickstart.py::TestValidation", "-v"],
         "Run validation tests only"),

        (["python", "-m", "pytest", "test_quickstart.py", "-k", "validate"],
         "Run tests matching 'validate'"),

        (["python", "-m", "pytest", "test_quickstart.py", "--collect-only"],
         "Show available tests without running"),
    ]

    print("\n" + "=" * 60)
    print("Example Test Commands")
    print("=" * 60)

    for cmd, description in test_commands:
        print(f"\n📝 {description}")
        print(f"   Command: {' '.join(cmd)}")

    print("\n" + "=" * 60)
    print("Running Basic Validation Tests")
    print("=" * 60)

    # Actually run the validation tests as a demo
    result = run_command(
        ["python", "-m", "pytest", "test_quickstart.py::TestValidation", "-v"],
        "Validation Tests"
    )

    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    print("\n" + "=" * 60)
    print("Additional Testing Resources")
    print("=" * 60)

    print("\nAccording to CLAUDE.md, you can also:")
    print("  • Run with coverage: pytest --cov=spyglass --cov-report=term-missing")
    print("  • Run without Docker: pytest --no-docker")
    print("  • Run without DLC: pytest --no-dlc")
    print("\nFor property-based tests (if hypothesis installed):")
    print("  • pytest test_property_based.py")

    return result

if __name__ == "__main__":
    sys.exit(main())