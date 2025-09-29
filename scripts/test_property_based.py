#!/usr/bin/env python3
"""Property-based tests for Spyglass validation functions.

These tests use the hypothesis library to generate random inputs and verify
that our validation functions behave correctly across all possible inputs.
"""

import sys
from pathlib import Path

# Add the scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

try:
    from hypothesis import given, strategies as st, assume
    from hypothesis.strategies import text, integers
    import pytest

    # Import functions to test
    from quickstart import validate_base_dir
    from ux.validation import validate_port, validate_environment_name

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("Hypothesis not available. Install with: pip install hypothesis")


if HYPOTHESIS_AVAILABLE:

    @given(st.integers(min_value=1, max_value=65535))
    def test_valid_ports_always_pass(port):
        """All valid port numbers should pass validation."""
        result = validate_port(str(port))
        assert result.is_success, f"Port {port} should be valid"
        assert result.value == port

    @given(st.integers().filter(lambda x: x <= 0 or x > 65535))
    def test_invalid_ports_always_fail(port):
        """All invalid port numbers should fail validation."""
        result = validate_port(str(port))
        assert result.is_failure, f"Port {port} should be invalid"

    @given(st.text(min_size=1, max_size=50))
    def test_environment_name_properties(name):
        """Test environment name validation properties."""
        result = validate_environment_name(name)

        # If the name passes validation, it should contain only allowed characters
        if result.is_success:
            # Valid names should contain only letters, numbers, hyphens, underscores
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
            assert all(c in allowed_chars for c in name), f"Valid name {name} contains invalid characters"

            # Valid names should not be empty or just whitespace
            assert name.strip(), f"Valid name should not be empty or whitespace: '{name}'"

            # Valid names should not start with numbers or special characters
            assert name[0].isalpha() or name[0] == '_', f"Valid name should start with letter or underscore: '{name}'"

    @given(st.text(alphabet=['a', 'b', 'c', '1', '2', '3', '_', '-'], min_size=1, max_size=20))
    def test_well_formed_environment_names(name):
        """Test that well-formed environment names behave predictably."""
        # Skip names that start with numbers or hyphens (invalid)
        assume(name[0].isalpha() or name[0] == '_')

        result = validate_environment_name(name)

        # Well-formed names should generally pass
        if result.is_failure:
            # If it fails, it should be for a specific reason we can identify
            error_message = result.message.lower()
            assert any(keyword in error_message for keyword in
                      ['reserved', 'invalid', 'length', 'character']), \
                f"Failure reason should be clear for name '{name}': {result.message}"

    def test_base_directory_validation_properties():
        """Test base directory validation properties."""
        # Test with home directory (should always work)
        home_result = validate_base_dir(Path.home())
        assert home_result.is_success, "Home directory should always be valid"

        # Test that result is always a resolved absolute path
        if home_result.is_success:
            resolved_path = home_result.value
            assert resolved_path.is_absolute(), "Validated path should be absolute"
            assert str(resolved_path) == str(resolved_path.resolve()), "Validated path should be resolved"

    @given(st.text(min_size=1, max_size=10))
    def test_port_string_formats(port_str):
        """Test that port validation handles various string formats correctly."""
        result = validate_port(port_str)

        # If validation succeeds, the string should represent a valid integer
        if result.is_success:
            try:
                port_int = int(port_str)
                assert 1 <= port_int <= 65535, f"Valid port should be in range 1-65535: {port_int}"
                assert result.value == port_int, "Validated port should match parsed integer"
            except ValueError:
                assert False, f"Valid port string should be parseable as integer: '{port_str}'"

    def test_hypothesis_examples():
        """Example-based tests to demonstrate hypothesis usage."""
        if not HYPOTHESIS_AVAILABLE:
            pytest.skip("Hypothesis not available")

        # Example of how hypothesis finds edge cases
        # These should work
        test_valid_ports_always_pass(80)
        test_valid_ports_always_pass(443)
        test_valid_ports_always_pass(65535)

        # These should fail
        test_invalid_ports_always_fail(0)
        test_invalid_ports_always_fail(-1)
        test_invalid_ports_always_fail(65536)

        print("✅ Property-based testing examples work correctly!")


if __name__ == "__main__":
    if HYPOTHESIS_AVAILABLE:
        # Run a few example tests
        test_hypothesis_examples()

        print("\n🧪 Property-based testing setup complete!")
        print("\nTo run these tests:")
        print("  1. Install hypothesis: pip install hypothesis")
        print("  2. Run with pytest: pytest test_property_based.py")
        print("  3. Or run specific tests: pytest test_property_based.py::test_valid_ports_always_pass")
        print("\nBenefits of property-based testing:")
        print("  • Automatically finds edge cases you didn't think of")
        print("  • Tests invariants across large input spaces")
        print("  • Provides better confidence than example-based tests")
    else:
        print("❌ Hypothesis not available. Install with: pip install hypothesis")