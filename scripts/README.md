# Spyglass Installation Scripts

This directory contains streamlined installation and validation scripts for Spyglass.

## Quick Start

Install Spyglass in one command:

```bash
python scripts/install.py
```

This interactive installer will:
1. Check prerequisites (Python version, conda/mamba)
2. Create conda environment
3. Install Spyglass package
4. Optionally set up local database with Docker
5. Validate installation

## Scripts

### `install.py` - Main Installer

Cross-platform installation script that automates the setup process.

**Interactive Mode:**
```bash
python scripts/install.py
```

**Non-Interactive Mode:**
```bash
# Minimal installation
python scripts/install.py --minimal

# Full installation with database
python scripts/install.py --full --docker

# Custom environment name
python scripts/install.py --env-name my-spyglass

# Custom data directory
python scripts/install.py --base-dir /data/spyglass
```

**Environment Variables:**
```bash
# Set base directory (skips prompt)
export SPYGLASS_BASE_DIR=/data/spyglass
python scripts/install.py
```

**Options:**
- `--minimal` - Install minimal dependencies only (~5 min, ~8 GB)
- `--full` - Install all dependencies (~15 min, ~18 GB)
- `--docker` - Set up local Docker database
- `--remote` - Connect to remote database (interactive or with CLI args)
- `--db-host HOST` - Database host for remote setup (non-interactive)
- `--db-port PORT` - Database port (default: 3306)
- `--db-user USER` - Database user (default: root)
- `--db-password PASS` - Database password (or use SPYGLASS_DB_PASSWORD env var)
- `--skip-validation` - Skip validation checks after installation
- `--env-name NAME` - Custom conda environment name (default: spyglass)
- `--base-dir PATH` - Base directory for data storage
- `--force` - Overwrite existing environment without prompting
- `--dry-run` - Show what would be done without making changes (coming soon)

### `validate.py` - Health Check

Validates that Spyglass is properly installed and configured.

**Usage:**
```bash
python scripts/validate.py
```

**Checks:**
1. Python version ≥3.9
2. Conda/mamba available
3. Spyglass can be imported
4. SpyglassConfig loads correctly
5. Database connection (if configured)

**Exit Codes:**
- `0` - All checks passed
- `1` - One or more checks failed

## Installation Types

### Minimal Installation
- Core dependencies only
- Suitable for basic usage
- Disk space: ~8 GB
- Install time: ~5 minutes

### Full Installation
- All pipeline dependencies
- Includes LFP, position, spikesorting
- Disk space: ~18 GB
- Install time: ~15 minutes

Note: DeepLabCut, Moseq, and Decoding require separate installation.

## Requirements

**System Requirements:**
- Python 3.9 or later
- conda or mamba package manager
- Git (recommended)
- Docker (optional, for local database)

**Platform Support:**
- macOS (Intel & Apple Silicon)
- Linux
- Windows (via WSL or native)

## Database Setup

The installer supports three database setup options:

### Option 1: Docker Compose (Recommended for Local Development)

Automatically set up a local MySQL database using Docker Compose:

```bash
python scripts/install.py --docker  # Auto-uses Compose
```

Or directly:
```bash
docker compose up -d
```

This creates a container named `spyglass-db` with:
- Host: localhost
- Port: 3306
- User: root
- Password: tutorial
- TLS: Disabled
- Persistent storage via Docker volume

**Benefits:**
- One-command setup
- Infrastructure as code (version controlled)
- Easy to customize via `.env` file
- Built-in health checks

**Customization:**
```bash
# Create .env file to customize settings
cp .env.example .env
nano .env  # Edit MYSQL_PORT, MYSQL_ROOT_PASSWORD, etc.
```

See `docker-compose.yml` and `.env.example` in the repository root.

### Option 2: Remote Database

**Interactive mode:**

```bash
python scripts/install.py --remote
```

You'll be prompted to enter:
- Host (e.g., db.example.com)
- Port (default: 3306)
- User (default: root)
- Password (hidden input)
- TLS settings (automatically enabled for non-localhost hosts)

**Non-interactive mode (for automation/CI/CD):**

```bash
# Using CLI arguments
python scripts/install.py --remote \
  --db-host db.lab.edu \
  --db-user myuser \
  --db-password mysecret

# Using environment variable for password (recommended)
export SPYGLASS_DB_PASSWORD=mysecret
python scripts/install.py --remote \
  --db-host db.lab.edu \
  --db-user myuser
```

**Security Notes:**
- Passwords are hidden during interactive input (using `getpass`)
- For automation, use `SPYGLASS_DB_PASSWORD` env var instead of `--db-password`
- TLS is automatically enabled for remote hosts
- Configuration is saved to `~/.datajoint_config.json`
- Use `--force` to overwrite existing configuration

### Option 3: Interactive Choice

Without flags, the installer presents an interactive menu:

```bash
python scripts/install.py

Database setup:
  1. Docker Compose (Recommended) - One-command setup
  2. Remote - Connect to existing database
  3. Skip - Configure later

Choice [1-3]:
```

The installer will auto-detect if Docker Compose is available and recommend it.

### Option 4: Manual Setup

Skip database setup during installation and configure manually later:

```bash
python scripts/install.py --skip-validation
# Then configure manually: see docs/DATABASE.md
```

## Configuration

The installer respects the following configuration priority:

1. **CLI arguments** (highest priority)
   ```bash
   python scripts/install.py --base-dir /custom/path
   ```

2. **Environment variables**
   ```bash
   export SPYGLASS_BASE_DIR=/custom/path
   python scripts/install.py
   ```

3. **Interactive prompts** (lowest priority)
   - Installer will ask for configuration if not provided

## Troubleshooting

### Environment Already Exists

If the installer detects an existing environment:
```
Environment 'spyglass' exists. Overwrite? [y/N]:
```

**Options:**
- Answer `n` to use the existing environment (installation continues)
- Answer `y` to remove and recreate the environment
- Use `--env-name different-name` to create a separate environment
- Use `--force` to automatically overwrite without prompting

### Environment Creation Fails

```bash
# Update conda
conda update conda

# Clear cache
conda clean --all

# Try with mamba (faster)
mamba env create -f environment.yml
```

### Docker Issues

Check Docker is running:
```bash
docker info
```

If Docker is not available:
- Install from https://docs.docker.com/get-docker/
- Or configure database manually (see docs/DATABASE.md)

### Database Connection Fails

Verify configuration:
```bash
# Check config file exists
ls ~/.datajoint_config.json

# Test connection
python -c "import datajoint as dj; dj.conn().ping(); print('✓ Connected')"
```

### Import Errors

Ensure environment is activated:
```bash
conda activate spyglass
python -c "import spyglass; print(spyglass.__version__)"
```

## Development

### Testing the Installer

```bash
# Create test environment
python scripts/install.py --env-name spyglass-test --minimal --skip-validation

# Validate installation
conda activate spyglass-test
python scripts/validate.py

# Clean up
conda deactivate
conda env remove -n spyglass-test
```

### Running Unit Tests

```bash
# Direct testing (bypasses pytest conftest issues)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from install import get_required_python_version, get_conda_command

version = get_required_python_version()
print(f'Python version: {version}')
assert version[0] == 3 and version[1] >= 9

cmd = get_conda_command()
print(f'Conda command: {cmd}')
assert cmd in ['conda', 'mamba']

print('✓ All tests passed')
"
```

## Architecture

### Design Principles

1. **Self-contained** - Minimal dependencies (stdlib only)
2. **Cross-platform** - Works on Windows, macOS, Linux
3. **Single source of truth** - Reads versions from `pyproject.toml`
4. **Explicit configuration** - Clear priority: CLI > env var > prompt
5. **Graceful degradation** - Works even if optional components fail

### Critical Execution Order

The installer must follow this order to avoid circular dependencies:

1. **Prerequisites check** (no spyglass imports)
2. **Create conda environment** (no spyglass imports)
3. **Install spyglass package** (`pip install -e .`)
4. **Setup database** (inline code, no spyglass imports)
5. **Validate** (runs in new environment, CAN import spyglass)

### Why Inline Docker Code?

The installer uses inline Docker operations instead of importing from `spyglass.utils.docker` because:
- Spyglass is not installed yet when the installer runs
- Cannot create circular dependency (installer → spyglass → installer)
- Must be self-contained with stdlib only

The reusable Docker utilities are in `src/spyglass/utils/docker.py` for:
- Testing infrastructure (`tests/container.py`)
- Post-installation database management
- Other spyglass code

## Comparison with Original Setup

| Aspect | Old Setup | New Installer |
|--------|-----------|---------------|
| Steps | ~30 manual | 1 command |
| Time | Hours | 5-15 minutes |
| Lines of code | ~6,000 | ~500 |
| Platforms | Manual per platform | Unified cross-platform |
| Validation | Manual | Automatic |
| Error recovery | Debug manually | Clear messages + guidance |

## Related Files

- `environment-min.yml` - Minimal dependencies
- `environment.yml` - Full dependencies
- `src/spyglass/utils/docker.py` - Reusable Docker utilities
- `tests/setup/test_install.py` - Unit tests
- `pyproject.toml` - Python version requirements (single source of truth)

## Support

For issues:
1. Check validation output: `python scripts/validate.py`
2. See docs/TROUBLESHOOTING.md (coming soon)
3. File issue at https://github.com/LorenFrankLab/spyglass/issues

## License

Same as Spyglass main package.
