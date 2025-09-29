# quickstart.py Walkthrough

An interactive installer that automates Spyglass setup with minimal user input, providing a robust installation experience through functional programming patterns.

## Architecture Overview

The quickstart script uses modern Python patterns for reliability and maintainability:

- **Result types**: All operations return explicit Success/Failure outcomes
- **Factory pattern**: Clean object creation through InstallerFactory
- **Pure functions**: Validation and configuration functions have no side effects
- **Immutable data**: SetupConfig uses frozen dataclasses
- **Named constants**: Clear configuration values replace magic numbers
- **Error recovery**: Comprehensive guidance for troubleshooting issues

## Purpose

The quickstart script handles the complete Spyglass installation process - from environment creation to database configuration - with smart defaults and minimal user interaction.

## Usage

```bash
# Minimal installation (default)
python scripts/quickstart.py

# Full installation with all dependencies
python scripts/quickstart.py --full

# Pipeline-specific installation
python scripts/quickstart.py --pipeline=dlc

# Fully automated (no prompts)
python scripts/quickstart.py --no-database

# Custom data directory
python scripts/quickstart.py --base-dir=/path/to/data
```

## User Experience

**1-3 prompts maximum** - The script automates everything except essential decisions that affect the installation.

## Step-by-Step Walkthrough

### 1. System Detection (No User Input)

```
╔═══════════════════════════════════════╗
║     Spyglass Quickstart Installer    ║
╚═══════════════════════════════════════╝

==========================================
System Detection
==========================================

✓ Operating System: macOS
✓ Architecture: Apple Silicon (M1/M2)
```

**Implementation:**
- `SystemDetector` class identifies OS and architecture
- Platform-specific logic handles macOS/Linux/Windows differences
- Returns `Result[SystemInfo, SystemError]` for explicit handling
- Immutable `SystemInfo` dataclass stores detection results

### 2. Python & Package Manager Check (No User Input)

```
==========================================
Python Check
==========================================

✓ Python 3.13.5 found

==========================================
Package Manager Check
==========================================

✓ Found conda: conda 25.7.0
ℹ Consider installing mamba for faster environment creation:
ℹ   conda install -n base -c conda-forge mamba
```

**Implementation:**
- `validate_python_version()` pure function checks version requirements
- Package manager detection prefers mamba over conda for performance
- Returns `Result[PackageManager, ValidationError]` outcomes
- `MINIMUM_PYTHON_VERSION` constant defines requirements
- Error messages include specific recovery actions

### 3. Installation Type Selection (Interactive Choice)

```
==========================================
Installation Type Selection
==========================================

Choose your installation type:
1) Minimal (core dependencies only)
   ├─ Basic Spyglass functionality
   ├─ Standard data analysis tools
   └─ Fastest installation (~5-10 minutes)

2) Full (all optional dependencies)
   ├─ All analysis pipelines included
   ├─ Spike sorting, LFP, visualization tools
   └─ Longer installation (~15-30 minutes)

3) Pipeline-specific
   ├─ Choose specific analysis pipeline
   ├─ DeepLabCut, Moseq, LFP, or Decoding
   └─ Optimized environment for your workflow

Enter choice (1-3): █
```

**User Decision:** Choose installation type and dependencies.

**If option 3 (Pipeline-specific) is chosen:**
```
Choose your pipeline:
1) DeepLabCut - Pose estimation and behavior analysis
2) Keypoint-Moseq (CPU) - Behavioral sequence analysis
3) Keypoint-Moseq (GPU) - GPU-accelerated behavioral analysis
4) LFP Analysis - Local field potential processing
5) Decoding - Neural population decoding

Enter choice (1-5): █
```

**Implementation:**
- `InstallType` enum provides type-safe installation options
- `UserInterface` class handles interactive prompts with fallbacks
- Command-line flags bypass prompts for automation
- `InstallerFactory` creates appropriate configuration objects
- `SetupConfig` frozen dataclass stores all installation parameters
- Menu displays include time estimates and dependency descriptions

### 4. Environment Selection & Creation (Conditional Prompt)

```
==========================================
Environment Selection
==========================================

ℹ Selected: DeepLabCut pipeline environment
  (or "Standard environment (minimal)" / "Full environment" etc.)

==========================================
Creating Conda Environment
==========================================
```

**If environment already exists:**
```
⚠ Environment 'spyglass' already exists
Do you want to update it? (y/N): █
```

**User Decision:** Update existing environment or keep it unchanged.

**Implementation:**
- `EnvironmentManager` encapsulates all conda operations
- `select_environment_file()` function maps installation types to files
- Pipeline-specific environments (environment_dlc.yml, environment_moseq_*.yml)
- Returns `Result[Environment, CondaError]` for all operations
- Handles existing environments with user confirmation prompts
- `ErrorRecoveryGuide` provides conda-specific troubleshooting

### 5. Dependency Installation (No User Input)

```
==========================================
Installing Additional Dependencies
==========================================

ℹ Installing Spyglass in development mode...
ℹ Installing LFP dependencies...
ℹ Detected M1 Mac, installing pyfftw via conda first...
✓ Additional dependencies installed
```

**Implementation:**
- Development mode installation with `pip install -e .`
- Platform-specific dependency handling through `SystemDetector`
- M1 Mac pyfftw workarounds automatically applied
- `Pipeline` enum determines additional package requirements
- `ErrorCategory` enum classifies installation failures
- `ErrorRecoveryGuide` provides targeted troubleshooting steps

### 6. Database Setup (Interactive Choice)

```
==========================================
Database Setup
==========================================

Choose database setup option:
1) Local Docker database (recommended for beginners)
2) Connect to existing database
3) Skip database setup

Enter choice (1-3): █
```

**User Decision:** How to configure the database.

#### Option 1: Docker Database (No Additional Prompts)
```
ℹ Setting up local Docker database...
ℹ Pulling MySQL image...
✓ Docker database started
✓ Configuration file created at: ./dj_local_conf.json
```

#### Option 2: Existing Database (Additional Prompts)
```
ℹ Configuring connection to existing database...
Database host: █
Database port (3306): █
Database user: █
Database password: █ (hidden input)
```

#### Option 3: Skip Database
```
ℹ Skipping database setup
⚠ You'll need to configure the database manually later
```

### 7. Configuration & Validation (No User Input)

```
ℹ Creating configuration file...
ℹ Using SpyglassConfig official directory structure
✓ Configuration file created at: ./dj_local_conf.json
✓ Data directories created at: ~/spyglass_data

==========================================
Running Validation
==========================================

ℹ Running comprehensive validation checks...
✓ All validation checks passed!
```

**Implementation:**
- DataJoint configuration generation through pure functions
- `validate_base_dir()` ensures directory path safety and accessibility
- Directory structure creation using `DEFAULT_SPYGLASS_DIRS` constants
- Validation system returns `Result[ValidationSummary, ValidationError]`
- Configuration written atomically with backup handling
- Success/failure outcomes guide user through any issues

### 8. Setup Complete (No User Input)

```
==========================================
Setup Complete!
==========================================

Next steps:

1. Activate the Spyglass environment:
   conda activate spyglass

2. Test the installation:
   python -c "from spyglass.settings import SpyglassConfig; print('✓ Integration successful')"

3. Start with the tutorials:
   cd notebooks
   jupyter notebook 01_Concepts.ipynb

4. For help and documentation:
   Documentation: https://lorenfranklab.github.io/spyglass/
   GitHub Issues: https://github.com/LorenFrankLab/spyglass/issues

Configuration Summary:
  Base directory: ~/spyglass_data
  Environment: spyglass
  Database: Configured
  Integration: SpyglassConfig compatible
```

## Command Line Options

### Installation Types (Optional - will prompt if not specified)
- `--minimal`: Core dependencies only
- `--full`: All optional dependencies
- `--pipeline=X`: Specific pipeline (dlc, moseq-cpu, moseq-gpu, lfp, decoding)

**Note:** If none of these flags are provided, the script will interactively prompt you to choose your installation type.

### Automation Options
- `--no-database`: Skip database setup entirely
- `--no-validate`: Skip final validation
- `--base-dir=PATH`: Custom data directory

### Non-Interactive Options
- `--yes`: Auto-accept all prompts without user input
- `--no-color`: Disable colored output
- `--help`: Show all options

### Exit Codes
- `0`: Success - installation completed successfully
- `1`: Error - installation failed or requirements not met
- `130`: Interrupted - user cancelled installation (Ctrl+C)

## User Interaction Summary

### Most Common Experience (2 prompts):
```bash
python scripts/quickstart.py
# Prompt 1: Installation type choice (user picks option 1: Minimal)
# Prompt 2: Database choice (user picks option 1: Docker)
# Result: Minimal installation with Docker database
```

### Fully Automated (0 prompts):
```bash
python scripts/quickstart.py --minimal --no-database --yes
# Result: Minimal environment and dependencies installed, manual database setup needed
```

### Auto-Accept Mode (0 prompts for most operations):
```bash
python scripts/quickstart.py --full --yes
# Automatically accepts: environment updates, default database settings
# Only prompts if absolutely necessary (e.g., database credentials for existing DB)
```

### Pipeline-specific Experience (2-3 prompts):
```bash
python scripts/quickstart.py
# Prompt 1: Installation type choice (user picks option 3: Pipeline-specific)
# Prompt 2: Pipeline choice (user picks DeepLabCut)
# Prompt 3: Database choice (user picks option 1: Docker)
# Result: DeepLabCut environment with Docker database
```

### Maximum Interaction (4+ prompts):
```bash
python scripts/quickstart.py
# Prompt 1: Installation type choice (user picks option 3: Pipeline-specific)
# Prompt 2: Pipeline choice (user picks option varies)
# Prompt 3: Update existing environment? (if environment exists)
# Prompt 4: Database choice (user picks option 2: Existing database)
# Prompt 5-8: Database credentials (host, port, user, password)
```

## What Gets Created

### Files
- `dj_local_conf.json`: DataJoint configuration file
- Conda environment named "spyglass"

### Directories
- Base directory (default: `~/spyglass_data`)
- Subdirectories: `raw/`, `analysis/`, `recording/`, `sorting/`, `tmp/`, `video/`, `waveforms/`

### Services
- Docker MySQL container (if Docker option chosen)
- Port 3306 exposed for database access

## Code Quality Features

**Functional Programming Patterns:**
- Pure functions for validation and configuration logic
- Immutable data structures prevent accidental state changes
- Result types make error handling explicit and composable

**Type Safety:**
- Comprehensive type hints including forward references
- Enum classes for type-safe choices (InstallType, Pipeline, etc.)
- Generic Result types for consistent error handling

**Error Handling:**
- Categorized errors (Docker, Conda, Python, Network, Permissions)
- Platform-specific recovery guidance
- No silent failures - all operations return explicit results

**User Experience:**
- Graceful degradation when optional components fail
- Clear progress indicators and informative error messages
- Minimal prompts with sensible defaults
- Backup warnings before overwriting existing configurations

## Key Classes and Functions

**Core Classes:**
- `SetupConfig`: Immutable configuration container
- `QuickstartOrchestrator`: Main installation coordinator
- `EnvironmentManager`: Conda environment operations
- `UserInterface`: User interaction and display
- `InstallerFactory`: Object creation and configuration
- `ErrorRecoveryGuide`: Troubleshooting assistance

**Pure Functions:**
- `validate_base_dir()`: Path validation and safety checks
- `validate_python_version()`: Version requirement verification
- `select_environment_file()`: Environment file selection logic

**Result Types:**
- `Success[T]`: Successful operation with value
- `Failure[E]`: Failed operation with error details
- `Result[T, E]`: Union type for explicit error handling

**Constants:**
- `DEFAULT_MYSQL_PORT`: Database connection default
- `MINIMUM_PYTHON_VERSION`: Required Python version
- `DEFAULT_SPYGLASS_DIRS`: Standard directory structure

This architecture provides a robust, maintainable installation system that guides users from initial setup to working Spyglass environment with comprehensive error handling and recovery.