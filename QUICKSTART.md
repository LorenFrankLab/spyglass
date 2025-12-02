# Spyglass Quickstart

Get from zero to analyzing neural data with Spyglass in just a few commands.

## Choose Your Path

### Joining an Existing Lab? (Recommended)

If you received database credentials from your lab admin, this is you!
The installer will:

- Set up your development environment
- Connect you to your lab's existing database
- Prompt you to change your temporary password
- Configure all necessary directories

**Time**: ~5 minutes | **Database**: Connect to lab's existing database

### Trying Spyglass Locally?

Want to explore Spyglass features without joining a lab?
The installer can:

- Set up a local trial database using Docker
- Create an isolated test environment
- Let you experiment with sample data

**Time**: ~10 minutes | **Database**: Local Docker container
(requires [Docker Desktop](https://docs.docker.com/get-docker/))

---

## Prerequisites

- **Python**: Version 3.9 or higher
- **Disk Space**: ~10GB for installation + data storage
- **Operating System**: macOS or Linux (Windows experimental)
- **Package Manager**: [conda](https://docs.conda.io/) (23.10.0+ recommended) or [mamba](https://mamba.readthedocs.io/)

If you don't have mamba/conda, install [miniforge](https://github.com/conda-forge/miniforge#install) first.

## Installation (2 steps)

### Step 1: Run the installer

```bash
# Clone the repository
git clone https://github.com/LorenFrankLab/spyglass.git
cd spyglass

# Run interactive installer
python scripts/install.py
```

The installer will prompt you to choose:

1. **Installation type**: Minimal (recommended) or Full
2. **Database setup**:
   - **Remote** (recommended for lab members) - Connect to lab's existing database
   - **Docker** - Local trial database for testing
   - **Skip** - Configure manually later

If joining a lab, you'll be prompted to change your password during installation.

### Step 2: Validate installation

```bash
# Activate the environment
conda activate spyglass

# Run validation
python scripts/validate.py -v
```

**That's it!** Setup complete in ~5-10 minutes.

## Next Steps

### Run first tutorial
```bash
cd notebooks
jupyter notebook 01_Concepts.ipynb
```

### Configure for your data
- Place NWB files in `~/spyglass_data/raw/`
- See [Data Import Guide](https://lorenfranklab.github.io/spyglass/latest/notebooks/01_Insert_Data/) for details

### Join community

- **Documentation**: [lorenfranklab.github.io/spyglass](https://lorenfranklab.github.io/spyglass/)
- **Discussions**: [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)
- **Report Issues**: [GitHub Issues](https://github.com/LorenFrankLab/spyglass/issues)

---

## Installation Options

Need something different? The installer supports these options:

```bash
python scripts/install.py --full           # All optional dependencies
python scripts/install.py --pipeline=dlc   # DeepLabCut pipeline
python scripts/install.py --no-database    # Skip database setup
python scripts/install.py --help           # See all options
```

## What Gets Installed

The installer creates:

- **Conda environment** with Spyglass and core dependencies
- **Database connection** (remote lab database OR local Docker container)
- **Data directories** in `~/spyglass_data/`
- **Jupyter environment** for running tutorials

## Troubleshooting

### Installation fails?
```bash
# Remove environment and retry
conda env remove -n spyglass
python scripts/install.py
```

### Validation fails?

1. Check error messages for specific issues
2. If using Docker database, ensure Docker Desktop is running
3. If database connection fails, verify credentials with your lab admin
4. Try skipping database: `python scripts/install.py --no-database`

### Need help?

- Check [Advanced Setup Guide](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/) for manual installation
- Ask questions in [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)

---

**Next tutorial**: [01_Concepts.ipynb](notebooks/01_Concepts.ipynb)
**Full documentation**: [lorenfranklab.github.io/spyglass](https://lorenfranklab.github.io/spyglass/)