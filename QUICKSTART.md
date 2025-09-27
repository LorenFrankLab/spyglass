# Spyglass Quickstart (5 minutes)

Get from zero to analyzing neural data with Spyglass in just a few commands.

## Prerequisites

- **Operating System**: macOS or Linux (Windows support experimental)
- **Python**: Version 3.9 or higher
- **Disk Space**: ~10GB for installation + data storage
- **Package Manager**: [mamba](https://mamba.readthedocs.io/) or [conda](https://docs.conda.io/) (mamba recommended for speed)

If you don't have mamba/conda, install [miniforge](https://github.com/conda-forge/miniforge#install) first.

## Quick Installation (2 commands)

### Option 1: Bash Script (macOS/Linux)
```bash
# Download and run the quickstart script
curl -sSL https://raw.githubusercontent.com/LorenFrankLab/spyglass/master/scripts/quickstart.sh | bash
```

### Option 2: Python Script (Cross-platform)
```bash
# Clone the repository
git clone https://github.com/LorenFrankLab/spyglass.git
cd spyglass

# Run quickstart
python scripts/quickstart.py
```

### Available Options

For customized installations:

```bash
# Minimal installation (default)
python scripts/quickstart.py --minimal

# Full installation with all optional dependencies
python scripts/quickstart.py --full

# Pipeline-specific installations
python scripts/quickstart.py --pipeline=dlc        # DeepLabCut
python scripts/quickstart.py --pipeline=moseq-gpu  # Keypoint-Moseq
python scripts/quickstart.py --pipeline=lfp        # LFP analysis
python scripts/quickstart.py --pipeline=decoding   # Neural decoding

# Skip database setup (configure manually later)
python scripts/quickstart.py --no-database

# Custom data directory
python scripts/quickstart.py --base-dir=/path/to/data
```

## What the quickstart does

1. **Detects your system** - OS, architecture, Python version
2. **Sets up conda environment** - Creates optimized environment for your system
3. **Installs Spyglass** - Development installation with all core dependencies
4. **Configures database** - Sets up local Docker MySQL or connects to existing
5. **Creates directories** - Standard data directory structure
6. **Validates installation** - Runs comprehensive health checks

## Verification

After installation, verify everything works:

```bash
# Activate the environment
conda activate spyglass

# Quick test
python -c "from spyglass.settings import SpyglassConfig; print('✓ Installation successful!')"

# Run full validation
python scripts/validate_spyglass.py -v
```

## Next Steps

### 1. Start with tutorials
```bash
cd notebooks
jupyter notebook 01_Concepts.ipynb
```

### 2. Configure for your data
- Place NWB files in `~/spyglass_data/raw/` (or your custom directory)
- See [Data Import Guide](https://lorenfranklab.github.io/spyglass/latest/notebooks/01_Insert_Data/) for details

### 3. Join the community
- 📖 [Documentation](https://lorenfranklab.github.io/spyglass/)
- 💬 [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)
- 🐛 [Report Issues](https://github.com/LorenFrankLab/spyglass/issues)
- 📧 [Mailing List](https://groups.google.com/g/spyglass-users)

## Common Installation Paths

### Beginners
```bash
python scripts/quickstart.py
# ↳ Minimal install + local Docker database + validation
```

### Position Tracking Researchers
```bash
python scripts/quickstart.py --pipeline=dlc
# ↳ DeepLabCut environment for pose estimation
```

### Electrophysiology Researchers
```bash
python scripts/quickstart.py --full
# ↳ All spike sorting + LFP analysis tools
```

### Existing Database Users
```bash
python scripts/quickstart.py --no-database
# ↳ Skip database setup, configure manually
```

## Troubleshooting

### Permission Errors
```bash
# On macOS, you may need to allow Docker in System Preferences
# On Linux, add your user to the docker group:
sudo usermod -aG docker $USER
```

### Environment Conflicts
```bash
# Remove existing environment and retry
conda env remove -n spyglass
python scripts/quickstart.py
```

### Apple Silicon (M1/M2) Issues
The quickstart automatically handles M1/M2 compatibility, including:
- Installing `pyfftw` via conda before pip packages
- Using ARM64-optimized packages where available

### Network Issues
```bash
# Use offline mode if conda install fails
python scripts/quickstart.py --no-validate
# Then run validation separately when online
python scripts/validate_spyglass.py
```

### Validation Failures
If validation fails:
1. Check the specific error messages
2. Ensure all dependencies installed correctly
3. Verify database connection
4. See [Advanced Setup Guide](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/) for manual configuration

## Advanced Options

For complex setups, see the detailed guides:

- [Manual Installation](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/00_Spyglass_Setup.ipynb)
- [Database Configuration](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/00_DatabaseSetup.ipynb)
- [Environment Files](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/00_Environments.ipynb)
- [Developer Setup](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/00_Development.ipynb)

## What's Included

The quickstart installation gives you:

### Core Framework
- **Spyglass** - Main analysis framework
- **DataJoint** - Database schema and pipeline management
- **PyNWB** - Neurodata Without Borders format support
- **SpikeInterface** - Spike sorting tools
- **NumPy/Pandas/Matplotlib** - Data analysis essentials

### Optional Components (with `--full`)
- **DeepLabCut** - Pose estimation (separate environment)
- **Ghostipy** - LFP analysis tools
- **JAX** - Neural decoding acceleration
- **Figurl** - Interactive visualizations
- **Kachery** - Data sharing platform

### Infrastructure
- **MySQL Database** - Local Docker container or existing server
- **Jupyter** - Interactive notebook environment
- **Pre-configured directories** - Organized data storage

---

**Total installation time**: ~5-10 minutes
**Next tutorial**: [01_Concepts.ipynb](notebooks/01_Concepts.ipynb)
**Need help?** [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)