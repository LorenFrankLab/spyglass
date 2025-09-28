# quickstart.py Walkthrough

An interactive installer that automates Spyglass setup with minimal user input, transforming the complex manual process into a streamlined experience.

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

**What it does:**
- Detects OS (macOS/Linux/Windows)
- Identifies architecture (x86_64/ARM64)
- Handles platform-specific requirements automatically

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

**What it does:**
- Verifies Python ≥3.9
- Finds conda/mamba (prefers mamba)
- Provides helpful suggestions

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

**What it does:**
- Prompts user for installation type if not specified via command line
- Skipped if user provided `--full`, `--minimal`, or `--pipeline` flags
- Determines which environment file and dependencies to install
- Provides clear descriptions and time estimates for each option

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

**What it does:**
- Selects appropriate environment file based on installation type choice
- Uses specialized environment files for pipelines (environment_dlc.yml, etc.)
- Creates new environment or updates existing one
- Shows progress during installation

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

**What it does:**
- Installs Spyglass in development mode
- Handles platform-specific dependencies (M1 Mac workarounds)
- Installs pipeline-specific packages based on options

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

**What it does:**
- Creates DataJoint configuration file
- Sets up directory structure
- Runs comprehensive validation
- Reports any issues

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

## Safety Features

- **Backup awareness**: Warns before overwriting existing environments
- **Validation**: Runs comprehensive checks after installation
- **Error handling**: Clear error messages with actionable advice
- **Graceful degradation**: Works even if optional components fail
- **User control**: Can skip database setup if needed

This script transforms the complex 200+ line manual setup process into a simple, interactive experience that gets users from zero to working Spyglass installation in under 10 minutes.