# validate_spyglass.py Walkthrough

A comprehensive health check script that validates Spyglass installation and configuration without requiring any user interaction.

## Architecture Overview

The validation script uses functional programming patterns for reliable diagnostics:

- **Result types**: All validation functions return explicit Success/Failure outcomes
- **Pure functions**: Validation logic has no side effects
- **Error categorization**: Systematic classification of issues for targeted recovery
- **Property-based testing**: Hypothesis tests validate edge cases
- **Immutable data structures**: Validation results use frozen dataclasses

## Purpose

The validation script provides a zero-interaction diagnostic tool that checks all aspects of a Spyglass installation to ensure everything is working correctly.

## Usage

```bash
# Basic validation
python scripts/validate_spyglass.py

# Verbose output (show all checks)
python scripts/validate_spyglass.py -v

# Disable colored output
python scripts/validate_spyglass.py --no-color

# Custom config file
python scripts/validate_spyglass.py --config-file /path/to/custom_config.json

# Combined options
python scripts/validate_spyglass.py -v --no-color --config-file ./my_config.json
```

## User Experience

**Zero prompts, zero decisions** - The script runs completely automatically and provides detailed feedback.

### Example Output

```
Spyglass Installation Validator
==================================================

Checking Prerequisites...
  ✓ Python version: Python 3.13.5
  ✓ Operating System: macOS
  ✓ Package Manager: conda found: conda 25.7.0

Checking Spyglass Installation...
  ✗ Spyglass Import: Cannot import spyglass
  ✗ DataJoint: Not installed
  ✗ PyNWB: Not installed

Checking Configuration...
  ✗ DataJoint Config: DataJoint not installed
  ✗ Directory Check: Cannot import SpyglassConfig

Checking Database Connection...
  ⚠ Database Connection: DataJoint not installed

Checking Optional Dependencies...
  ℹ Spike Sorting: Not installed (optional)
  ℹ MountainSort: Not installed (optional)
  ℹ LFP Analysis: Not installed (optional)
  ℹ DeepLabCut: Not installed (optional)
  ℹ Visualization: Not installed (optional)
  ℹ Data Sharing: Not installed (optional)

Validation Summary
==================================================

Total checks: 19
  Passed: 7
  Warnings: 1
  Errors: 5

❌ Validation FAILED

Please address the errors above before proceeding.
See https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/
```

## What It Checks

### 1. Prerequisites (No User Input)

**Implementation:**
- `validate_python_version()` pure function checks version requirements
- `SystemDetector` class verifies Linux/macOS compatibility
- Package manager detection with Result types
- `MINIMUM_PYTHON_VERSION` constant defines requirements

**Validates:**
- Python version ≥3.9
- Operating system compatibility
- Conda/mamba availability

### 2. Spyglass Installation (No User Input)

**Implementation:**
- Core import testing with Result type outcomes
- Dependency validation using pure functions
- Version reporting with error categorization
- `ErrorRecoveryGuide` for missing dependencies

**Validates:**
- `import spyglass` functionality
- Core dependencies (DataJoint, PyNWB, pandas, numpy, matplotlib)
- Package versions and compatibility

### 3. Configuration (No User Input)

**Implementation:**
- `validate_config_file()` function checks DataJoint configuration
- `validate_base_dir()` validates directory structure
- SpyglassConfig integration testing with Result types
- Path safety validation and sanitization

**Validates:**
- DataJoint configuration files
- Spyglass data directory structure
- SpyglassConfig system integration

### 4. Database Connection (No User Input)

**Implementation:**
- `validate_database_connection()` tests connectivity with Result types
- Database table accessibility verification
- Permission checking with detailed error reporting
- `validate_port()` function ensures proper database ports

**Validates:**
- Database connectivity when configured
- Spyglass table accessibility
- Database permissions and configuration

### 5. Optional Dependencies (No User Input)

**Implementation:**
- Pipeline tool validation (spikeinterface, mountainsort4, ghostipy)
- Analysis tool testing (DeepLabCut, JAX, figurl) with Result types
- Integration validation (kachery_cloud)
- `validate_environment_name()` for conda environments

**Validates:**
- Spike sorting tools
- Analysis and visualization packages
- Data sharing integrations

## Exit Codes

- **0**: Success - all checks passed
- **1**: Warning - setup complete but with warnings
- **2**: Failure - critical issues found

## Safety Features

- **Read-only**: Never modifies any files or settings
- **Safe to run**: Can be executed on any system without risk
- **No network calls**: Only checks local installation
- **No sudo required**: Runs with user permissions only

## When to Use

- **After installation** to verify everything works
- **Before starting analysis** to catch configuration issues
- **Troubleshooting** to identify specific problems
- **CI/CD pipelines** to verify environment setup (use `--no-color` flag)
- **Documentation** to show installation proof
- **Regular health checks** to ensure environment hasn't degraded

## Integration with Quickstart

The validator is automatically called by the quickstart script during installation:

```bash
python scripts/quickstart.py
# ... installation process ...
#
# ==========================================
# Running Validation
# ==========================================
#
# ℹ Running comprehensive validation checks...
# ✓ All validation checks passed!
```

This ensures that installations are verified immediately, and any issues are caught early in the setup process.

## Key Classes and Functions

**Core Classes:**
- `ValidationSummary`: Immutable results container
- `SystemValidator`: Core system validation logic
- `InstallationValidator`: Installation-specific checks
- `ErrorRecoveryGuide`: Troubleshooting assistance

**Pure Functions:**
- `validate_python_version()`: Version requirement checks
- `validate_environment_name()`: Environment name validation
- `validate_port()`: Port number validation
- `validate_base_dir()`: Directory validation and safety
- `validate_config_file()`: Configuration file validation
- `validate_database_connection()`: Database connectivity testing

**Result Types:**
- `Success[T]`: Successful validation with details
- `Failure[E]`: Failed validation with error information
- `ValidationResult`: Union type for explicit validation outcomes

**Constants:**
- `MINIMUM_PYTHON_VERSION`: Required Python version
- `SUPPORTED_PLATFORMS`: Compatible operating systems
- `DEFAULT_CONFIG_LOCATIONS`: Standard configuration paths

## Technical Features

**Functional Programming Excellence:**
- Pure functions for all validation logic
- Result types for explicit Success/Failure outcomes
- Immutable data structures for validation results
- Type safety with comprehensive type hints

**Enhanced Validation:**
- Context managers for safe module imports
- Error categorization using `ErrorCategory` enum
- Intelligent dependency detection (required vs optional)
- Real-time progress feedback during checks

**Error Recovery:**
- `ErrorRecoveryGuide` with platform-specific solutions
- Categorized troubleshooting (Docker, Conda, Python, Network, etc.)
- Clear summary statistics with pass/fail counts
- Property-based testing validates edge cases

This architecture provides immediate feedback on installation health without requiring any user decisions or potentially dangerous operations, while maintaining exceptional code quality and comprehensive error handling capabilities.