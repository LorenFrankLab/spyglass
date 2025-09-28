# validate_spyglass.py Walkthrough

A comprehensive health check script that validates Spyglass installation and configuration without requiring any user interaction.

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
- **Python Version**: Ensures Python ≥3.9
- **Operating System**: Verifies Linux/macOS compatibility
- **Package Manager**: Detects mamba/conda availability

### 2. Spyglass Installation (No User Input)
- **Core Import**: Tests `import spyglass`
- **Dependencies**: Checks DataJoint, PyNWB, pandas, numpy, matplotlib
- **Version Information**: Reports installed versions

### 3. Configuration (No User Input)
- **Config Files**: Looks for DataJoint configuration
- **Directory Structure**: Validates Spyglass data directories
- **SpyglassConfig**: Tests configuration system integration

### 4. Database Connection (No User Input)
- **Connectivity**: Tests database connection if configured
- **Table Access**: Verifies Spyglass tables are accessible
- **Permissions**: Checks database permissions

### 5. Optional Dependencies (No User Input)
- **Pipeline Tools**: Checks for spikeinterface, mountainsort4, ghostipy
- **Analysis Tools**: Tests DeepLabCut, JAX, figurl availability
- **Sharing Tools**: Validates kachery_cloud integration

### 6. Color and Display Options
- **Color Support**: Automatically detects terminal capabilities
- **No-Color Mode**: Can disable colors for CI/CD or plain text output
- **Verbose Mode**: Shows all checks (passed and failed) instead of just failures

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

## Technical Features

- **Context Manager**: Uses safe module imports to prevent crashes
- **Error Categorization**: Distinguishes between errors, warnings, and info
- **Dependency Detection**: Intelligently checks for optional vs required packages
- **Progress Feedback**: Shows real-time status of each check
- **Summary Statistics**: Provides clear pass/fail counts at the end

This script provides immediate feedback on installation health without requiring any user decisions or potentially dangerous operations.