# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# JupyterHub users can skip this step. Frank Lab members should first follow
# 'rec to nwb overview' steps on Google Drive to set up an ssh connection.
#
# For local use, download and install ...
#
# 1. [Python 3.9](https://wiki.python.org/moin/BeginnersGuide/Download).
# 2. [mamba](https://mamba.readthedocs.io/en/latest/installation.html) as a
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
# and [pypi](https://pypi.org/project/spyglass-neuro/) with
# `pip install spyglass-neuro`, but downloading from GitHub will also download
# other files.
#
# Next, within VSCode,
# [select the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)
# that matches your spyglass environment created with `mamba`. To use other Python
# interfaces, be sure to activate the environment: `conda activate spyglass`
#
# See [this guide](https://datajoint.com/docs/elements/user-guide/) for additional
# details on each of these programs and the role they play in using the pipeline.
#

# ## Database
#

# You have a few options for databases.
#
# 1. Connect to an existing database.
# 2. Run your own database with [Docker](#running-your-own-database)
# 3. JupyterHub (coming soon...)
#
# Your choice above should result in a set of credentials, including host name,
# host port, user name, and password. Note these for the next step.
#
# <details><summary>Note for MySQL 8 users, including Frank Lab members</summary>
#
# Using a MySQL 8 server, like the server hosted by the Frank Lab, will
# require the pre-release version of DataJoint to change one's password.
#
# ```bash
# # cd /location/for/datajoint/source/files/
# git clone https://github.com/datajoint/datajoint-python
# pip install ./datajoint-python
# ```
#
# </details>
#

# ### Existing Database
#

# Connecting to an existing database will require a user name and password.
# Please contact your database administrator for this information.
#
# Frank Lab members should contact Chris.
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
# - Host: localhost
# - Password: tutorial
# - User: root
# - Port: 3306
#

# ### Config and Connecting to the database
#

# Spyglass can load settings from either a DataJoint config file (recommended) or
# environmental variables. The code below will generate a config file, but we
# first need to decide a 'base path'. This is generally the parent directory
# where the data will be stored, with subdirectories for `raw`, `analysis`, and
# other data folders. If they don't exist already, they will be created.
#
# The function below will create a config file (`~/.datajoint.config` if global,
# `./dj_local_conf.json` if local). Local is recommended for the notebooks, as
# each will start by loading this file. Custom json configs can be saved elsewhere, but will need to be loaded in startup with
# `dj.config.load('your-path')`.
#
# To point spyglass to a folder elsewhere (e.g., an external drive for waveform
# data), simply edit the json file. Note that the `raw` and `analysis` paths
# appear under both `stores` and `custom`.
#

# +
import os
from spyglass.settings import SpyglassConfig

# change to the root directory of the project
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

SpyglassConfig().save_dj_config(
    save_method="local",  # global or local
    base_dir="/path/like/stelmo/nwb/",
    database_user="your username",
    database_password="your password",  # remove this line for shared machines
    database_host="localhost or lmf-db.cin.ucsf.edu",
    database_port=3306,
    set_password=False,
)
# -

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
# config before importing Spyglass, try setting this as an environmental variable
# before importing Spyglass.
#
# ```python
# os.environ['SPYGLASS_BASE_DIR'] = '/your/base/path'
#
# import spyglass
# from spyglass.settings import SpyglassConfig
# import datajoint as dj
# print(SpyglassConfig().config)
# dj.config.save_local() # or global
# ```
#

# # Up Next
#

# Next, we'll try [inserting data](./01_Insert_Data.ipynb)
#
