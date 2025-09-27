# Spyglass Quickstart (5 minutes)

Get from zero to analyzing neural data with Spyglass in just a few commands.

## Prerequisites

- **Python**: Version 3.9 or higher
- **Disk Space**: ~10GB for installation + data storage
- **Operating System**: macOS or Linux (Windows experimental)
- **Package Manager**: [mamba](https://mamba.readthedocs.io/) or [conda](https://docs.conda.io/) (mamba recommended)

If you don't have mamba/conda, install [miniforge](https://github.com/conda-forge/miniforge#install) first.

## Installation (2 commands)

### 1. Download and run quickstart
```bash
# Clone the repository
git clone https://github.com/LorenFrankLab/spyglass.git
cd spyglass

# Run quickstart (minimal installation)
python scripts/quickstart.py
```

### 2. Validate installation
```bash
# Activate the environment
conda activate spyglass

# Run validation
python scripts/validate_spyglass.py -v
```

**That's it!** Total time: ~5-10 minutes

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
- 📖 [Documentation](https://lorenfranklab.github.io/spyglass/)
- 💬 [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)
- 🐛 [Report Issues](https://github.com/LorenFrankLab/spyglass/issues)

---

## Installation Options

Need something different? The quickstart supports these options:

```bash
python scripts/quickstart.py --full           # All optional dependencies
python scripts/quickstart.py --pipeline=dlc   # DeepLabCut pipeline
python scripts/quickstart.py --no-database    # Skip database setup
python scripts/quickstart.py --help           # See all options
```

## What Gets Installed

The quickstart creates:
- **Conda environment** with Spyglass and core dependencies
- **MySQL database** (local Docker container)
- **Data directories** in `~/spyglass_data/`
- **Jupyter environment** for running tutorials

## Troubleshooting

### Installation fails?
```bash
# Remove environment and retry
conda env remove -n spyglass
python scripts/quickstart.py
```

### Validation fails?
1. Check error messages for specific issues
2. Ensure Docker is running (for database)
3. Try: `python scripts/quickstart.py --no-database`

### Need help?
- Check [Advanced Setup Guide](https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/) for manual installation
- Ask questions in [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)

---

**Next tutorial**: [01_Concepts.ipynb](notebooks/01_Concepts.ipynb)
**Full documentation**: [lorenfranklab.github.io/spyglass](https://lorenfranklab.github.io/spyglass/)