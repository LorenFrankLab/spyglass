# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# # Setup
#

# ## Intro
#

# Welcome to [Spyglass](https://lorenfranklab.github.io/spyglass/0.4/),
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

# ## Local environment
#

# Skip this step if you're ...
#
# 1. Running the tutorials on [JupyterHub](https://spyglass.hhmi.2i2c.cloud/)
# 2. A member of the Frank Lab members. Instead, ssh to a shared machine.
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
# In a terminal, ...
#
# 1. Navigate to your project directory.
# 2. Use `git` to download the Spyglass repository.
# 3. Navigate to the newly downloaded directory.
# 4. Create a `mamba` environment with either the standard `environment.yml` or
#    the `environment_position.yml`, if you intend to use the full position
#    pipeline. The latter will take longer to install.
# 5. Open this notebook with VSCode
#
# Commands for the steps above ...
#
# ```bash
# # cd /your/project/directory/ # 1
# git clone https://github.com/LorenFrankLab/spyglass/ # 2
# # cd spyglass # 3
# mamba env create -f environment.yml # 4
# code notebooks/00_Setup.ipynb # 5
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
# mamba env create -f environment_dlc.yml # 2
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
# mamba env create -f environment_moseq_cpu.yml # 2
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
# 1. Connect to an existing database.
# 2. Run your own database with [Docker](#running-your-own-database)
# 3. JupyterHub (database pre-configured, skip this step)
#
# Your choice above should result in a set of credentials, including host name,
# host port, user name, and password. Note these for the next step.
#
# <details><summary>Note for MySQL 8 users, including Frank Lab members</summary>
#
# Using a MySQL 8 server, like the server hosted by the Frank Lab, will
# require DataJoint >= 0.14.2. To keep up to data with the latest DataJoint
# features, install from GitHub
#
# ```bash
# # cd /location/for/datajoint/source/files/
# git clone https://github.com/datajoint/datajoint-python
# pip install ./datajoint-python
# ```
#
# You can then periodically fetch updates with the following commands...
#
# ```bash
# # cd /location/for/datajoint/source/files/datajoint-python
# git pull origin master
# ```
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

# ### Running your own database with Docker
#

# - First, [install Docker](https://docs.docker.com/engine/install/).
# - Add yourself to the
#   [`docker` group](https://docs.docker.com/engine/install/linux-postinstall/) so
#   that you don't have to be sudo to run docker.
# - Download the docker image for `datajoint/mysql:8.0`.
#
#   ```bash
#   docker pull datajoint/mysql:8.0
#   ```
#
# - When run, this is referred to as a 'Docker container'
# - Next start the container with a couple additional pieces of info...
#
#   - Root password. We use `tutorial`.
#   - Database name. Here, we use `spyglass-db`.
#   - Port mapping. Here, we map 3306 across the local machine and container.
#
#   ```bash
#   docker run --name spyglass-db -p 3306:3306 -e MYSQL_ROOT_PASSWORD=tutorial datajoint/mysql:8.0
#   ```
#
# - For data to persist after terminating the container,
#   [attach a volume](https://docs.docker.com/storage/volumes/) when running:
#
#   ```bash
#   docker volume create dj-vol
#   docker run --name spyglass-db -v dj-vol:/var/lib/mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=tutorial datajoint/mysql
#   ```
#
# Docker credentials are as follows:
#
# - Host: `localhost`
# - User: `root`
# - Password: `tutorial`
# - Port: `3306`
#

# ### Config
#

# Spyglass will load settings the 'custom' section of your DataJoint config file.
# The code below will generate a config
# file, but we first need to decide a 'base path'. This is generally the parent
# directory where the data will be stored, with subdirectories for `raw`,
# `analysis`, and other data folders. If they don't exist already, they will be
# created relative to the base path specified with their default names.
#
# A temporary directory is one such subfolder (default `base-dir/tmp`) to speed
# up spike sorting. Ideally, this folder should have ~500GB free.
#
# The function below will create a config file (`~/.datajoint.config` if global,
# `./dj_local_conf.json` if local).
# See also [DataJoint docs](https://datajoint.com/docs/core/datajoint-python/0.14/quick-start/#connection).
# Local is recommended for the notebooks, as
# each will start by loading this file. Custom json configs can be saved elsewhere, but will need to be loaded in startup with
# `dj.config.load('your-path')`.
#
# To point Spyglass to a folder elsewhere (e.g., an external drive for waveform
# data), simply edit the resulting json file. Note that the `raw` and `analysis` paths
# appear under both `stores` and `custom`. Spyglass will check that these match
# on startup and log a warning if not.
#

# +
import os
import datajoint as dj
from spyglass.settings import SpyglassConfig

# change to the root directory of the project
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

# connect to the database
dj.conn()

# change your password
dj.admin.set_password()

# save the configuration
SpyglassConfig().save_dj_config(
    save_method="local",  # global or local
    base_dir="/path/like/stelmo/nwb/",
    database_user="your username",
    database_password="your password",  # remove this line for shared machines
    database_host="localhost or lmf-db.cin.ucsf.edu",  # only list one
    database_port=3306,
    set_password=False,
)
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

# # Up Next
#

# Next, we'll try [introduce some concepts](./01_Concepts.ipynb)
#
