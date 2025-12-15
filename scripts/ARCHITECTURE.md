# Spyglass Scripts Architecture

This document describes the execution flow, decision points, and design
rationale for the installation and validation scripts.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Entry Points                            │
├─────────────────────────────────────────────────────────────────────┤
│  python scripts/install.py     Interactive or CLI installation      │
│  python scripts/validate.py    Post-install health check            │
└─────────────────────────────────────────────────────────────────────┘
```

## install.py - Main Installer

### Execution Flow

The installer follows a **strict execution order** to avoid circular
dependencies (Spyglass cannot be imported until after it's installed).

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INSTALLATION FLOW                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PARSE ARGUMENTS                                                  │
│     └─→ CLI flags (--minimal, --docker, etc.)                        │
│     └─→ Environment variables (SPYGLASS_BASE_DIR)                    │
│                                                                      │
│  2. DRY-RUN CHECK ─────────────────────────────────────────────────┐ │
│     └─→ If --dry-run: show plan and exit                          │ │
│                                                                    │ │
│  3. DETERMINE INSTALLATION TYPE                                    │ │
│     ├─→ --minimal flag → environment_min.yml                       │ │
│     ├─→ --full flag → environment.yml                              │ │
│     └─→ Neither → Interactive prompt                               │ │
│                                                                    │ │
│  4. GET BASE DIRECTORY                                             │ │
│     ├─→ --base-dir CLI arg (highest priority)                      │ │
│     ├─→ SPYGLASS_BASE_DIR env var                                  │ │
│     └─→ Interactive prompt → ~/spyglass_data default               │ │
│                                                                    │ │
│  5. CHECK PREREQUISITES  ⚠️ NO SPYGLASS IMPORTS                     │ │
│     ├─→ Python version (from pyproject.toml)                       │ │
│     ├─→ conda/mamba available                                      │ │
│     ├─→ Git available (optional, warns if missing)                 │ │
│     └─→ Disk space sufficient                                      │ │
│                                                                    │ │
│  6. CREATE CONDA ENVIRONMENT  ⚠️ NO SPYGLASS IMPORTS               │ │
│     ├─→ Check if environment exists                                │ │
│     │   ├─→ --force: remove and recreate                           │ │
│     │   └─→ Prompt: overwrite or use existing                      │ │
│     └─→ Run: conda/mamba env create -f environment.yml             │ │
│                                                                    │ │
│  7. INSTALL SPYGLASS PACKAGE                                       │ │
│     └─→ Run: pip install -e . (in new environment)                 │ │
│                                                                    │ │
│  8. DATABASE SETUP  ⚠️ INLINE CODE ONLY (no spyglass imports)      │ │
│     ├─→ --docker flag → Docker Compose setup                       │ │
│     ├─→ --remote flag → Remote database config                     │ │
│     └─→ Neither → Interactive menu                                 │ │
│                                                                    │ │
│  9. VALIDATION  ✅ CAN IMPORT SPYGLASS (runs in new env)           │ │
│     └─→ Runs validate.py in the new conda environment              │ │
│                                                                    │ │
│  10. COMPLETION MESSAGE                                            │ │
│      └─→ Next steps: activate env, run tutorials                   │ │
│                                                                    │ │
└──────────────────────────────────────────────────────────────────────┘
```

### Decision Points

#### 1. Installation Type Selection

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INSTALLATION TYPE DECISION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CLI Input                                                          │
│     │                                                               │
│     ├─→ --minimal ──────────→ Minimal Install                       │
│     │                         • environment_min.yml                 │
│     │                         • ~8 GB packages                      │
│     │                         • ~5 min install time                 │
│     │                         • Core functionality only             │
│     │                                                               │
│     ├─→ --full ─────────────→ Full Install                          │
│     │                         • environment.yml                     │
│     │                         • ~18 GB packages                     │
│     │                         • ~15 min install time                │
│     │                         • All pipelines (LFP, sorting, etc.)  │
│     │                                                               │
│     └─→ (neither) ──────────→ Interactive Prompt                    │
│                               • Shows disk space for each option    │
│                               • Recommends minimal for most users   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2. Database Setup Selection

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATABASE SETUP DECISION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CLI Input                                                          │
│     │                                                               │
│     ├─→ --docker ───────────→ Docker Compose Setup                  │
│     │                         • Checks Docker availability          │
│     │                         • Runs: docker compose up -d          │
│     │                         • Creates ~/.datajoint_config.json    │
│     │                         • Waits for MySQL readiness           │
│     │                                                               │
│     ├─→ --remote ───────────→ Remote Database Setup                 │
│     │   │                                                           │
│     │   ├─→ With --db-host ─→ Non-interactive mode                  │
│     │   │                     Uses CLI args for all config          │
│     │   │                                                           │
│     │   └─→ Without ────────→ Interactive prompts                   │
│     │                         • Host (validates format)             │
│     │                         • Port (validates 1024-65535)         │
│     │                         • User/Password                       │
│     │                         • Offers password change              │
│     │                                                               │
│     └─→ (neither) ──────────→ Interactive Menu                      │
│                               │                                     │
│                               ├─→ [1] Docker (if available)         │
│                               ├─→ [2] Remote database               │
│                               └─→ [3] Skip (configure later)        │
│                                                                     │
│  On Failure:                                                        │
│     └─→ Offer retry with different option                           │
│     └─→ Allow skip to continue installation                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3. Environment Handling

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ENVIRONMENT EXISTS DECISION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Check: Does conda environment already exist?                       │
│     │                                                               │
│     ├─→ No ─────────────────→ Create new environment                │
│     │                                                               │
│     └─→ Yes                                                         │
│         │                                                           │
│         ├─→ --force flag ───→ Remove and recreate                   │
│         │                                                           │
│         └─→ (no flag) ──────→ Prompt user                           │
│             │                                                       │
│             ├─→ "y" ────────→ Remove and recreate                   │
│             │                                                       │
│             └─→ "n" ────────→ Use existing (skip env creation)      │
│                               Continue to package install           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose | Stage |
|----------|---------|-------|
| `main()` | Entry point, argument parsing | Start |
| `run_installation()` | Orchestrates all steps | Start |
| `run_dry_run()` | Shows plan without executing | Optional |
| `determine_installation_type()` | Minimal vs full selection | Step 3 |
| `get_base_directory()` | Resolve data directory | Step 4 |
| `check_prerequisites()` | Verify system requirements | Step 5 |
| `create_conda_environment()` | Create/update conda env | Step 6 |
| `install_spyglass_package()` | Run pip install | Step 7 |
| `setup_database()` | Route to appropriate DB setup | Step 8 |
| `handle_database_setup_interactive()` | Interactive DB menu | Step 8 |
| `handle_database_setup_cli()` | CLI-driven DB setup | Step 8 |
| `setup_database_compose()` | Docker Compose setup | Step 8 |
| `setup_database_remote()` | Remote DB configuration | Step 8 |
| `change_database_password()` | Password change flow | Step 8 |
| `validate_installation()` | Run validate.py in new env | Step 9 |

### Configuration Priority

The installer uses a consistent priority order for all configuration:

```
1. CLI arguments      (highest priority)
2. Environment vars   (middle priority)
3. Interactive prompt (lowest priority, uses defaults)
```

Example for base directory:

```
--base-dir /custom/path     →  /custom/path
SPYGLASS_BASE_DIR=/env/path →  /env/path (if no CLI arg)
(neither)                   →  Prompt with ~/spyglass_data default
```

---

## validate.py - Health Check

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VALIDATION FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CRITICAL CHECKS (must pass)                                        │
│     │                                                               │
│     ├─→ Python version ≥ 3.9                                        │
│     │   └─→ Reads from pyproject.toml (single source of truth)      │
│     │                                                               │
│     ├─→ conda/mamba available                                       │
│     │   └─→ Checks PATH for executables                             │
│     │                                                               │
│     └─→ Spyglass importable                                         │
│         └─→ import spyglass; print version                          │
│                                                                     │
│  OPTIONAL CHECKS (warn only)                                        │
│     │                                                               │
│     ├─→ SpyglassConfig loads                                        │
│     │   └─→ Tests settings.py integration                           │
│     │                                                               │
│     └─→ Database connection                                         │
│         └─→ datajoint.conn().ping()                                 │
│                                                                     │
│  EXIT CODES                                                         │
│     ├─→ 0: All critical checks passed                               │
│     └─→ 1: One or more critical checks failed                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Check Categories

| Check | Critical | Failure Impact |
|-------|----------|----------------|
| Python version | Yes | Cannot run Spyglass |
| Conda/Mamba | Yes | Cannot manage environment |
| Spyglass import | Yes | Package not installed |
| SpyglassConfig | No | Config may need setup |
| Database | No | Can configure later |

---

## Design Principles

### 1. Self-Contained Scripts

Both scripts use **stdlib only** (no external dependencies) until after
Spyglass is installed:

```python
# ✅ OK before installation
import subprocess, sys, json, pathlib

# ❌ NOT OK before installation
import spyglass  # Not installed yet!
import datajoint  # Not installed yet!
```

### 2. Intentional Code Duplication

`get_required_python_version()` is duplicated in both scripts because:

- `validate.py` must work standalone before Spyglass is installed
- Both scripts designed to run independently
- Avoids complex import path management

If modified, **update both files**:

- `scripts/install.py`
- `scripts/validate.py`

### 3. Inline Docker Code

Database setup uses inline Docker code instead of importing from
`spyglass.utils.docker` because Spyglass isn't installed yet when
the database is configured.

```python
# In install.py - inline implementation
def is_docker_available_inline() -> bool:
    # Self-contained, no imports needed
    ...

# In spyglass/utils/docker.py - for post-install use
def is_docker_available() -> bool:
    # Used by tests and other code
    ...
```

### 4. Graceful Degradation

The installer continues even if optional components fail:

```
Docker unavailable     →  Offer remote DB or skip
Database setup fails   →  Offer retry or skip
Validation fails       →  Show warnings, continue
```

### 5. Single Source of Truth

Configuration is read from authoritative sources:

| Setting | Source |
|---------|--------|
| Python version | `pyproject.toml` |
| Directory structure | `config_schema.json` |
| Environment deps | `environment.yml` / `environment_min.yml` |

---

## Error Handling

### Exception Hierarchy

```python
main()
├── KeyboardInterrupt     →  "Installation cancelled"
├── RuntimeError          →  Expected errors (prerequisites, validation)
├── CalledProcessError    →  Subprocess failures (conda, pip)
├── OSError/IOError       →  File system errors
└── ValueError            →  Configuration/validation errors
```

### Recovery Paths

| Error | Recovery |
|-------|----------|
| Disk space | Suggest --minimal or different --base-dir |
| Docker unavailable | Offer remote DB or skip |
| Connection refused | Show connection troubleshooting |
| Permission denied | Check file permissions guidance |
| Env exists | Offer overwrite or use existing |

---

## Testing

### Unit Tests

- `tests/setup/test_install.py` - Installer function tests
- `tests/setup/test_config_schema.py` - Schema consistency tests

### Manual Testing

```bash
# Dry run (shows plan without changes)
python scripts/install.py --dry-run

# Test minimal install
python scripts/install.py --minimal --skip-validation

# Test validation standalone
python scripts/validate.py
```

---

## Related Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python version requirements |
| `config_schema.json` | Directory structure schema |
| `environment.yml` | Full dependencies |
| `environment_min.yml` | Minimal dependencies |
| `docker-compose.yml` | Database infrastructure |
| `example.env` | Docker configuration template |
| `src/spyglass/settings.py` | Runtime config (loads schema) |
| `src/spyglass/utils/docker.py` | Post-install Docker utilities |
