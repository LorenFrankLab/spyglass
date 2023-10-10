# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup
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
# Codespace users can skip this step.
#
# For local use, download and install ...
#
# 1. [Python 3.9](https://wiki.python.org/moin/BeginnersGuide/Download).
# 2. [mamba](https://mamba.readthedocs.io/en/latest/installation.html) as a
#    replacement for conda. Spyglass installation is significantly faster with
#    mamba.
# 3. [VS Code](https://code.visualstudio.com/docs/python/python-tutorial) with
#    relevant python extensions, including
#    [Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
#    Hold off on selecting your interpreter until after you make the environment
#    with `mamba`.
# 4. [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for
#    downloading the repository, including notebooks.
#
# In a terminal, ...
#
# 1. navigate to your project directory.
# 2. use `git` to download the Spyglass repository.
# 3. navigate to the newly downloaded directory.
# 4. create a `mamba` environment with either the standard `environment.yml` or
#    the `environment_position.yml`, if you intend to use the full position
#    pipeline. The latter will take longer to install.
# 5. open this notebook with VSCode
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
# _Note:_ Spyglass is also installable via
# [pip](<https://en.wikipedia.org/wiki/Pip_(package_manager)>)
# and [pypi](https://pypi.org/project/spyglass-neuro/) with `pip install spyglass-neuro`, but downloading from GitHub will also other files accessible.
#
# Next, within VSCode,
# [select the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
# that matches your spyglass environment created with `mamba`. To use other Python
# interfaces, be sure to activate the environment: `conda activate spyglass`
#
# See [this guide](https://datajoint.com/docs/elements/user-guide/) for additional
# details on each of these programs and the role they play in using the pipeline.
#

# ## Database Connection
#

# You have a few options for databases.
#
# 1. Connect to an existing database.
# 2. Use GitHub Codespaces (coming soon...)
# 3. Run your own database with [Docker](#running-your-own-database)
#
# Once your database is set up, be sure to configure the connection
# with your `dj_local_conf.json` file.
#

# ### Existing Database
#

# Members of the Frank Lab can run the `dj_config.py` helper script to generate
# a default `dj_local_conf.json` like the one below. Outside users should adjust
# values accordingly.
#
# ```bash
# # cd spyglass
# python config/dj_config.py <username> <base_path>
# ```
#
# Producing a json config like the following.
#
# ```json
# {
#     "database.host": "lmf-db.cin.ucsf.edu",
#     "database.user": "<username>",
#     "database.port": 3306,
#     "database.use_tls": true,
#     "enable_python_native_blobs": true,
#     "filepath_checksum_size_limit": 1 * 1024**3,
#     "stores": {
#         "raw": {
#             "protocol": "file",
#             "location": "/stelmo/nwb/raw",
#             "stage": "/stelmo/nwb/raw"
#         },
#         "analysis": {
#             "protocol": "file",
#             "location": "/stelmo/nwb/analysis",
#             "stage": "/stelmo/nwb/analysis"
#         }
#     },
#     "custom": {
#         "spyglass_dirs": {
#             "base": "/stelmo/nwb/"
#         }
#     }
# }
# ```
#

# ### Running your own database
#

# #### Setup Docker
#
# - First, [install Docker](https://docs.docker.com/engine/install/).
# - Add yourself to the
#   [`docker` group](https://docs.docker.com/engine/install/linux-postinstall/) so
#   that you don't have to be sudo to run docker.
# - Download the docker image for datajoint/mysql
#
#   ```bash
#   docker pull datajoint/mysql
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
#   docker run --name spyglass-db -p 3306:3306 -e MYSQL_ROOT_PASSWORD=tutorial datajoint/mysql
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
# #### Configure
#
# The `dj_local_conf_example.json` contains all the defaults for a Docker
# connection. Simply rename to `dj_local_conf.json` and modify the contents
# accordingly. This includes the host, password and user. For Spyglass, you'll
# want to set your base path under `custom`:
#
# ```json
# {
#   "database.host": "localhost",
#   "database.password": "tutorial",
#   "database.user": "root",
#   "custom": {
#     "database.prefix": "username_",
#     "spyglass_dirs": {
#       "base": "/your/base/path"
#     }
#   }
# }
# ```
#

# ### Loading the config
#
# We can check that the paths are correctly set up by loading the config from
# the main Spyglass directory.
#

# +
import os
import datajoint as dj

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")

from spyglass.settings import config

config
# -

# ### Connect
#
# Now, you should be able to connect to the database you set up.
#
# Let's demonstrate with an example table:
#

# +
from spyglass.common import Nwbfile

Nwbfile()
# -

# # Up Next

# Next, we'll try [inserting data](./01_Insert_Data.ipynb)
#
