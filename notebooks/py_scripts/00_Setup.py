# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# # Setup
#

# ## Intro
#

# Welcome to [Spyglass](https://lorenfranklab.github.io/spyglass/),
# a [DataJoint](https://github.com/datajoint/datajoint-python/)
# pipeline maintained by the [Frank Lab](https://franklab.ucsf.edu/) at UCSF.
#
# Spyglass will help you take an [NWB](https://www.nwb.org/) file from raw data to
# analysis-ready preprocessed formats using DataJoint to (a) connect to a
# [relational database](https://www.youtube.com/watch?v=q-PMUSC5P5o) (here,
# MySQL), and (b) automate processing steps. To use Spyglass, you'll need to ...
#
# 1. Set up your local environment
# 2. Connect to a database
#
# **Quick Start:** For most users, we recommend using our automated installer:
#
# ```bash
# git clone https://github.com/LorenFrankLab/spyglass.git
# cd spyglass
# python scripts/install.py
# ```
#
# The installer handles environment creation, database setup, and configuration.
# See [QUICKSTART.md](../QUICKSTART.md) for details. The sections below provide
# manual installation instructions for advanced users or troubleshooting.
#

# ## Local environment
#

# Skip this step if you're ...
#
# 1. Running the tutorials on [JupyterHub](https://spyglass.hhmi.2i2c.cloud/)
# 2. A member of the Frank Lab. Instead, ssh to a shared machine and run `scripts/setup_franklab.sh`
# 3. Using the automated installer (`python scripts/install.py`) - it handles this for you
#

# ### Tools
#
# For local use, download and install ...
#
# 1. [Python 3.9](https://wiki.python.org/moin/BeginnersGuide/Download).
# 2. [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
#     as a
#    replacement for conda. Spyglass installation is significantly faster with
#    mamba.
#    ```bash
#    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
#    bash Miniforge3-$(uname)-$(uname -m).sh
#    ```
# 3. [VS Code](https://code.visualstudio.com/docs/python/python-tutorial) with
#    relevant python extensions, including
#    [Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
#    Hold off on selecting your interpreter until after you make the environment
#    with `mamba`.
# 4. [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for
#    downloading the repository, including notebooks.
#
# See [this DataJoint guide](https://datajoint.com/docs/elements/user-guide/) for
# additional details on each of these programs and the role they play in using the
# pipeline.
#
# <details><summary>Suggested VSCode settings</summary>
#
# Within the Spyglass repository, there is a `.vscode` folder with `json` files
# that specify limited settings and extensions intended for developers. The average
# user may benefit from the following fuller sets.
#
# We recommending these incrementally so you get a feel for what each one does
# before adding the next, and to avoid being overwhelmed by changes.
#
# 1. `extensions.json`. By updating this file, you'll add to the 'Recommended'
# section of the extensions tab. Each extension page will provide more information
# on the uses and benefits. Some relevant concepts include...
#     - Linting: Warning of potential problems
#     - Formatting: Auto-adjusting optional coding styles to align across users
#     - Debugger: Progressive running of code. Please search for tutorials
#     - Autocompletion: Prompting for potential options when coding
#
# ```json
# {
#     "recommendations": [
#         // Python Extensions
#         "charliermarsh.ruff", // Fast linter
#         "donjayamanne.python-environment-manager", // Environment manager
#         "kevinrose.vsc-python-indent", // Auto-indent when coding
#         "ms-python.black-formatter", // Opinionated formatting
#         "ms-python.debugpy", // Debugger
#         "ms-python.isort", // Opinionated formatter for imports
#         "ms-python.pylint", // Linter to support a DataJoint-specific linter
#         "ms-python.python", // Language support for Python
#         "ms-python.vscode-pylance", // Additional language support
#         // Jupyter
#         "ms-toolsai.jupyter", // Run notebooks in VSCode
#         "ms-toolsai.jupyter-keymap", // Allow key-bindings
#         "ms-toolsai.jupyter-renderers", // Display images
#         // Autocompletion/Markdown
#         "github.copilot", // Auto-suggest with copilot LLM
#         "github.copilot-chat", // Add chat-box for questions to LLM
#         "visualstudioexptteam.intellicode-api-usage-examples", // Prompt package options
#         "visualstudioexptteam.vscodeintellicode", // Prompt Python-general options
#         "davidanson.vscode-markdownlint", // Linter for markdown
#         "streetsidesoftware.code-spell-checker", // Spell checker
#         // SSH - Work on remote servers - Required for Frank Lab members
#         "ms-vscode-remote.remote-ssh",
#         "ms-vscode-remote.remote-ssh-edit",
#         "ms-vscode.remote-explorer",
#     ],
#     "unwantedRecommendations": []
# }
# ```
#
# 2. `settings.json`. These can be places just in Spyglass, or added to your user
# settings file. Search settings in the command panel (cmd/ctrl+shift+P) to open
# this file directly.
#
# ```json
# {
#     // GENERAL
#     "editor.insertSpaces": true, // tab -> spaces
#     "editor.rulers": [ 80 ], // vertical line at 80
#     "editor.stickyScroll.enabled": true, // Show scope at top
#     "files.associations": { "*.json": "jsonc" }, // Load JSON with comments
#     "files.autoSave": "onFocusChange", // Save on focus change
#     "files.exclude": {  // Hide these in the file viewer
#       "**/__pycache*": true, // Add others with wildcards
#       "**/.ipynb_ch*": true,
#     },
#     "files.trimTrailingWhitespace": true, // Remove extra spaces in lines
#     "git.enabled": true, // use git
#     "workbench.editorAssociations": {  // open file extension as given type
#       "*.ipynb": "jupyter-notebook",
#     },
#     // PYTHON
#     "editor.defaultFormatter": "ms-python.black-formatter", // use black
#     "[python]": {
#         "editor.formatOnSave": true,
#         "editor.defaultFormatter": "ms-python.black-formatter",
#         "editor.codeActionsOnSave": { "source.organizeImports": "always"},
#     },
#     "python.analysis.autoImportCompletions": false, // Disable auto-import
#     "python.languageServer": "Pylance", // Use Pylance
#     "pylint.args": [ // DataJoint linter optional
#         // "--load-plugins=datajoint_linter", // Requires pip installing
#         // "--permit-dj-filepath=y", // Specific to datajoint_linter
#         "--disable=E0401,E0102,W0621,W0401,W0611,W0614"
#     ],
#     // NOTEBOOKS
#     "jupyter.askForKernelRestart": false, // Prevent dialog box on restart
#     "jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"], // IPyWidgets
#     "notebook.output.textLineLimit": 15, // Limit output
#     "notebook.lineNumbers": "on", // Number lines in cells
#     "notebook.formatOnSave.enabled": true, // blackify cells
#     // AUTOCOMPLETION
#     "editor.tabCompletion": "on", // tab over suggestions
#     "github.copilot.editor.enableAutoCompletions": true, // Copilot
#     "cSpell.enabled": true, // Spellcheck
#     "cSpell.language": "en,en-US,companies,python,python-common",
#     "cSpell.maxDuplicateProblems": 2, // Only mention a problem twice
#     "cSpell.spellCheckDelayMs": 500, // Wait 0.5s after save
#     "cSpell.userWords": [ "datajoint", "longblob", ], // Add words
#     "cSpell.enableFiletypes": [
#       "!json", "markdown", "yaml", "python" // disable (!) json, check others
#     ],
#     "cSpell.logLevel": "Warning", // Only show warnings, can turn off
#     // MARKDOWN
#     "[markdown]": { // Use linter and format on save
#     "editor.defaultFormatter": "DavidAnson.vscode-markdownlint",
#         "editor.formatOnSave": true,
#     },
#     "editor.codeActionsOnSave": { "source.fixAll.markdownlint": "explicit" },
#     "rewrap.reformat": true, // allows context-aware rewrapping
#     "rewrap.wrappingColumn": 80, // Align with Black formatter
# }
# ```
#
# The DataJoint linter is available at
# [this repository](https://github.com/CBroz1/datajoint_linter).
#
# </details>

#
# ### Installation
#
# #### Recommended: Automated Installer
#
# The easiest way to install Spyglass is with our automated installer:
#
# ```bash
# git clone https://github.com/LorenFrankLab/spyglass.git
# cd spyglass
# python scripts/install.py
# ```
#
# The installer will:
# - Check prerequisites (Python version, conda/mamba)
# - Create a conda environment with all dependencies
# - Optionally set up a local database with Docker Compose
# - Configure Spyglass directories and settings
# - Validate the installation
#
# See [QUICKSTART.md](../QUICKSTART.md) for full details and options.
#
# #### Manual Installation
#
# For manual installation, in a terminal:
#
# 1. Navigate to your project directory.
# 2. Use `git` to download the Spyglass repository.
# 3. Navigate to the newly downloaded directory.
# 4. Create a `mamba` environment with either `environments/environment.yml` (full) or
#    `environments/environment_min.yml` (minimal, faster install).
# 5. Open this notebook with VSCode
#
# Commands for the steps above:
#
# ```bash
# cd /your/project/directory/             # 1
# git clone https://github.com/LorenFrankLab/spyglass/  # 2
# cd spyglass                             # 3
# mamba env create -f environments/environment_min.yml # 4 (or environments/environment.yml for full)
# code notebooks/00_Setup.ipynb           # 5
# ```
#
# Next, within VSCode,
# [select the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
# that matches your spyglass environment created with `mamba`. To use other Python
# interfaces, be sure to activate the environment: `conda activate spyglass`
#
#
# ### Considerations
#
# 1. Spyglass is also installable via
# [pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>)
# and [pypi](https://pypi.org/project/spyglass-neuro/) with
# `pip install spyglass-neuro`, but downloading from GitHub will also download
# other files, like this tutorial.
# 2. Developers who wish to work on the code base may want to do an editable
# install from within their conda environment: `pip install -e /path/to/spyglass/`
#
# ### Staying up to date
#
# To get the latest updates and bugfixes between releases, you should occasionally
# navigate to your spyglass directory and run ...
#
# ```bash
# # cd /your/project/directory/spyglass/
# git pull # update files from GitHub
# pip install . -U # update install from this directory
# ```
#
# Adding `--no-deps` to this last install command will skip over dependency checks,
# for a faster first pass.
#
# If you encounter issues, please try updating first.

# ### Optional Dependencies
#
# Some pipelines require installation of additional packages.
#
# #### Spike Sorting
#
# The spike sorting pipeline relies on `spikeinterface` and optionally
# `mountainsort4`.
#
# ```bash
# conda activate <your-spyglass-env>
# pip install spikeinterface[full,widgets]
# pip install mountainsort4
# ```
#
# #### LFP
#
# The LFP pipeline uses `ghostipy`.
#
# __WARNING:__ If you are on an M1 Mac, you need to install `pyfftw` via `conda`
# BEFORE installing `ghostipy`:
#
# ```bash
# conda install -c conda-forge pyfftw # for M1 Macs
# pip install ghostipy
# ```
#
# #### Decoding
#
# The Decoding pipeline relies on `jax` to process data with GPUs. Please see
# their conda installation steps
# [here](https://jax.readthedocs.io/en/latest/installation.html#conda-installation).
#
# #### Deep Lab Cut (DLC)
#
# Spyglass provides an environment build for using the DLC pipeline. To create an
# environment with these features, please:
# 1. navigate to your cloned spyglass repo.
# 2. build the environment from the dlc version
# 3. activate the environment to use
#
# ```bash
# # cd /path/to/spyglass # 1
# mamba env create -f environments/environment_dlc.yml # 2
# mamba activate spyglass-dlc # 3
# ```
#
# Alternatively, you can pip install using
# ```bash
# pip install spyglass[dlc]
# ```
#
# #### Keypoint-Moseq
#
# Spyglass provides an environment build for using the Moseq pipeline. To create an
# environment with these features, please:
# 1. navigate to your cloned spyglass repo.
# 2. build the environment from one of the moseq versions
# 3. activate the environment to use
#
# ```bash
# # cd /path/to/spyglass # 1
# mamba env create -f environments/environment_moseq_cpu.yml # 2
# mamba activate spyglass-moseq-cpu # 3
# ```
#
# Alternatively, you can pip install using
# ```bash
# pip install spyglass[moseq-cpu]
# ```
#
# To use a GPU enabled version of the package, replace `cpu` with `gpu` in the above
# commands
#

# ## Database
#

# You have a few options for databases.
#
# 1. **Automated setup** (recommended): Use `python scripts/install.py --docker` or `--remote`
# 2. Connect to an existing database manually
# 3. Run your own database with [Docker Compose](#running-your-own-database)
# 4. JupyterHub (database pre-configured, skip this step)
#
# Your choice above should result in a set of credentials, including host name,
# host port, user name, and password. Note these for the next step.
#
# For detailed database setup instructions, see the [Database Setup Guide](https://lorenfranklab.github.io/spyglass/latest/DATABASE/).
#
# <details><summary>Note for MySQL 8 users, including Frank Lab members</summary>
#
# Using a MySQL 8 server, like the server hosted by the Frank Lab, will
# require DataJoint >= 0.14.2.
#
# </details>
#

# ### Existing Database
#

# Connecting to an existing database will require a user name and password.
# Please contact your database administrator for this information.
#
# For persistent databases with backups, administrators should review our
# documentation on
# [database management](https://lorenfranklab.github.io/spyglass/latest/ForDevelopers/Management).
#

# ### Running your own database with Docker Compose
#

# The easiest way to run a local database is with Docker Compose:
#
# ```bash
# # From the spyglass repository root
# docker compose up -d
# ```
#
# This uses the included `docker-compose.yml` file to create a properly configured
# MySQL container with persistent storage. To customize settings (port, password),
# copy `example.env` to `.env` and edit as needed.
#
# **Or use the installer:**
# ```bash
# python scripts/install.py --docker
# ```
#
# Docker credentials (default):
#
# - Host: `localhost`
# - User: `root`
# - Password: `tutorial`
# - Port: `3306`
#
# **Management commands:**
# ```bash
# docker compose up -d      # Start
# docker compose stop       # Stop (keeps data)
# docker compose down       # Stop and remove container (keeps data)
# docker compose down -v    # Stop and delete all data
# docker compose logs mysql # View logs
# ```
#
# For more details, see the [Database Setup Guide](https://lorenfranklab.github.io/spyglass/latest/DATABASE/).
#

# ### Config
#

# **Note:** If you used `python scripts/install.py`, configuration is handled
# automatically. The section below is for manual configuration or troubleshooting.
#
# Spyglass will load settings from the 'custom' section of your DataJoint config file.
# The code below will generate a config file, but we first need to decide a 'base path'.
# This is generally the parent directory where the data will be stored, with
# subdirectories for `raw`, `analysis`, and other data folders. If they don't exist
# already, they will be created relative to the base path specified.
#
# A temporary directory is one such subfolder (default `base-dir/tmp`) to speed
# up spike sorting. Ideally, this folder should have ~500GB free.
#
# The function below will create a config file (`~/.datajoint_config.json` if global,
# `./dj_local_conf.json` if local).
# See also [DataJoint docs](https://datajoint.com/docs/core/datajoint-python/0.14/quick-start/#connection).
# Local is recommended for the notebooks, as each will start by loading this file.
#
# To point Spyglass to a folder elsewhere (e.g., an external drive for waveform
# data), simply edit the resulting json file.
#

# +
import datajoint as dj
from spyglass.settings import SpyglassConfig
from pathlib import Path

username = "your username"  # replace with username made for you in the spyglass database
initial_password = (
    "your initial password"  # replace with the initial password given to you
)
new_password = (
    "your new password"  # replace with the new password you want to set
)
spyglass_base_dir = "/path/like/stelmo/nwb/"  # replace with a path on your system where spyglass can store NWB files
database_host = "address like: 'localhost' or 'lmf-db.cin.ucsf.edu'"


dj.config.update(
    {
        "database.host": database_host,
        "database.user": username,
        "database.password": initial_password,
        "database.port": 3306,
    }
)

dj.conn()  # connect to the database

# change your password from the initial password
dj.admin.set_password(new_password, update_config=True)

# save the configuration
SpyglassConfig().save_dj_config(
    base_dir=spyglass_base_dir,
    database_user=username,
    database_password=new_password,
    database_host=database_host,  # only list one
    database_port=3306,
    set_password=False,
)

# ensure the configuration is saved for future use
your_config = Path.home() / ".datajoint_config.json"
print(f"Config exists: {your_config.exists()}")
# -

# <details><summary>Legacy config</summary>
#
# Older versions of Spyglass relied exclusively on environment variables for
# config. If `spyglass_dirs` is not found in the config file, Spyglass will look
# for environment variables. These can be set either once in a terminal session,
# or permanently in a unix settings file (e.g., `.bashrc` or `.bash_profile`) in
# your home directory.
#
# ```bash
# export SPYGLASS_BASE_DIR="/stelmo/nwb"
# export SPYGLASS_RECORDING_DIR="$SPYGLASS_BASE_DIR/recording"
# export SPYGLASS_SORTING_DIR="$SPYGLASS_BASE_DIR/sorting"
# export SPYGLASS_VIDEO_DIR="$SPYGLASS_BASE_DIR/video"
# export SPYGLASS_WAVEFORMS_DIR="$SPYGLASS_BASE_DIR/waveforms"
# export SPYGLASS_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
# export KACHERY_CLOUD_DIR="$SPYGLASS_BASE_DIR/.kachery-cloud"
# export KACHERY_TEMP_DIR="$SPYGLASS_BASE_DIR/tmp"
# export DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE"
# ```
#
# To load variables from a `.bashrc` file, run `source ~/.bashrc` in a terminal.
#
# </details>

# ### Managing Files
#
# [`kachery-cloud`](https://github.com/flatironinstitute/kachery-cloud) is a file
# manager for collaborators to share files. This is an optional dependency for
# collaborating teams who don't have direct access to one another's disk space,
# but want to share a MySQL database instance.
# To customize `kachery` file paths, see `dj_local_conf_example.json`.
#
# To set up a new `kachery` instance for your project, contact maintainers
# of this package.

# ### Connecting

# If you used either a local or global save method, we can check the connection
# to the database with ...
#

# +
import datajoint as dj

dj.conn()  # test connection
dj.config  # check config

from spyglass.common import Nwbfile

Nwbfile()
# -

# If you see an error saying `Could not find SPYGLASS_BASE_DIR`, try loading your
# config before importing Spyglass.
#
# ```python
# import datajoint as dj
# dj.config.load('/your/config/path')
#
# from spyglass.common import Session
#
# Session()
#
# # If successful...
# dj.config.save_local() # or global
# ```
#

# ## Troubleshooting
#

# If you encounter issues during setup:
#
# 1. **Run the validation script:** `python scripts/validate.py`
# 2. **Check the troubleshooting guide:** [Troubleshooting](https://lorenfranklab.github.io/spyglass/latest/TROUBLESHOOTING/)
# 3. **Database issues:** See the [Database Setup Guide](https://lorenfranklab.github.io/spyglass/latest/DATABASE/)
# 4. **Ask for help:** [GitHub Discussions](https://github.com/LorenFrankLab/spyglass/discussions)
#

# # Up Next
#

# Next, we'll [introduce some concepts](./01_Concepts.ipynb)
#
