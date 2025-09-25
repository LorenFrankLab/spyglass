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

# # Sync Data
#

# ## Overview
#

# This notebook will cover ...
#
# 1. [General Kachery information](#kachery)
# 2. Setting up Kachery as a [host](#host-setup). If you'll use an existing host,
#    skip this.
# 3. Setting up Kachery in your [database](#database-setup). If you're using an
#    existing database, skip this.
# 4. Adding Kachery [data](#data-setup).
#

# ## Imports
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - To fully demonstrate syncing features, we'll need to run some basic analyses.
#   This can either be done with code in this notebook or by running another
#   notebook (e.g., [LFP](./12_LFP.ipynb))
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [these additional tutorials](https://github.com/datajoint/datajoint-tutorials)
#
# Let's start by importing the `spyglass` package and testing that your environment
# is properly configured for kachery sharing
#
# If you haven't already done so, be sure to set up your Spyglass base directory and Kachery sharing directory with [Setup](./00_Setup.ipynb)
#

# +
import os
import datajoint as dj

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection

import spyglass.common as sgc
import spyglass.sharing as sgs
from spyglass.settings import config

import warnings

warnings.filterwarnings("ignore")
# -

# For example analysis files, run the code hidden below.
#
# <details>
# <summary>Quick Analysis</summary>
#
# ```python
# from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
# import spyglass.data_import as sgi
# import spyglass.lfp as lfp
#
# nwb_file_name = "minirec20230622.nwb"
# nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
#
# sgi.insert_sessions(nwb_file_name)
# sgc.FirFilterParameters().create_standard_filters()
# lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
#     nwb_file_name=nwb_copy_file_name,
#     group_name="test",
#     electrode_list=[0],
# )
# lfp.v1.LFPSelection.insert1(
#     {
#         "nwb_file_name": nwb_copy_file_name,
#         "lfp_electrode_group_name": "test",
#         "target_interval_list_name": "01_s1",
#         "filter_name": "LFP 0-400 Hz",
#         "filter_sampling_rate": 30_000,
#     },
#     skip_duplicates=True,
# )
# lfp.v1.LFPV1().populate()
# ```
#
# </details>
#

# ## Kachery
#

# ### Cloud
#

# This notebook contains instructions for setting up data sharing/syncing through
# [_Kachery Cloud_](https://github.com/flatironinstitute/kachery-cloud), which
# makes it possible to share analysis results, stored in NWB files. When a user
# tries to access a file, Spyglass does the following:
#
# 1. Try to load from the local file system/store.
# 2. If unavailable, check if it is in the relevant sharing table (i.e.,
#    `NwbKachery` or `AnalysisNWBKachery`).
# 3. If present, attempt to download from the associated Kachery Resource to the user's spyglass analysis directory.
#
# _Note:_ large file downloads may take a long time, so downloading raw data is
# not supported. We suggest direct transfer with
# [globus](https://www.globus.org/data-transfer) or a similar service.
#

# ### Zone
#

# A [Kachery Zone](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md)
# is a cloud storage host. The Frank laboratory has three separate Kachery zones:
#
# 1. `franklab.default`: Internal file sharing, including figurls
# 2. `franklab.collaborator`: File sharing with collaborating labs.
# 3. `franklab.public`: Public file sharing (not yet active)
#
# Setting your zone can either be done as as an environment variable or an item
# in a DataJoint config. Spyglass will automatically handle setting the appropriate zone when downloading
# database files through kachery
#
# - Environment variable:
#
#   ```bash
#   export KACHERY_ZONE=franklab.default
#   export KACHERY_CLOUD_DIR=/stelmo/nwb/.kachery-cloud
#   ```
#
# - DataJoint Config:
#
#   ```json
#   "custom": {
#      "kachery_zone": "franklab.default",
#      "kachery_dirs": {
#         "cloud": "/your/base/path/.kachery-cloud"
#      }
#   }
#   ```
#

# ## Host Setup
#
# - If you are a member of a team with a pre-existing database and zone who will be sharing data, please skip to `Sharing Data`
#
# - If you are a collaborator outside your team's network and need to access files shared with you, please skip to `Accessing Shared Data`
#

# ### Zones
#

# See
# [instructions](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md)
# for setting up new Kachery Zones, including creating a cloud bucket and
# registering it with the Kachery team.
#
# _Notes:_
#
# - Bucket names cannot include periods, so we substitute a dash, as in
#   `franklab-default`.
# - You only need to create an API token for your first zone.
#

# ### Resources
#

# See [instructions](https://github.com/scratchrealm/kachery-resource/blob/main/README.md)
# for setting up zone resources. This allows for sharing files on demand. We
# suggest using the same name for the zone and resource.
#
# _Note:_ For each zone, you need to run the local daemon that listens for
# requests from that zone and uploads data to the bucket for client download when requested. An example of the bash script we use is
#
# ```bash
#     export KACHERY_ZONE=franklab.collaborators
#     export KACHERY_CLOUD_DIR=/stelmo/nwb/.kachery-cloud
#     cd /stelmo/nwb/franklab_collaborators_resource
#     npx kachery-resource@latest share
# ```
#
# For convenience, we recommend saving this code as a bash script which can be executed by the local daemon. For franklab member, these scripts can be found in the directory `/home/loren/bin/`:
#
# - run_restart_kachery_collab.sh
# - run_restart_kachery_default.sh
#

# ## Database Setup
#

# Once you have a hosted zone running, we need to add its information to the Spyglass database.
# This will allow spyglass to manage linking files from our analysis tables to kachery.
# First, we'll check existing Zones.
#

sgs.KacheryZone()

# To add a new hosted Zone, we need to prepare an entry for the `KacheryZone` table.
# Note that the `kacherycloud_dir` key should be the path for the server daemon _hosting_ the zone,
# and is not required to be present on the client machine:
#

# +
zone_name = config.get("KACHERY_ZONE")
cloud_dir = config.get("KACHERY_CLOUD_DIR")

zone_key = {
    "kachery_zone_name": zone_name,
    "description": " ".join(zone_name.split(".")) + " zone",
    "kachery_cloud_dir": cloud_dir,
    "kachery_proxy": "https://kachery-resource-proxy.herokuapp.com",
    "lab_name": sgc.Lab.fetch("lab_name", limit=1)[0],
}
# -

# Use caution when inserting into an active database, as it could interfere with
# ongoing work.
#

sgs.KacheryZone().insert1(zone_key, skip_duplicates=True)

# ## Sharing Data
#

# Once the zone exists, we can add `AnalysisNWB` files we want to share with members of the zone.
#
# The `AnalysisNwbFileKachery` table links analysis files made within other spyglass tables with a `uri`
# used by kachery. We can view files already made available through kachery here:
#

sgs.AnalysisNwbfileKachery()

# We can share additional results by populating new entries in this table.
#
# To do so we first add these entries to the `AnalysisNwbfileKacherySelection` table.
#
# _Note:_ This step depends on having previously run an analysis on the example
# file.
#

# +
nwb_copy_filename = "minirec20230622_.nwb"

analysis_file_list = (  # Grab all analysis files for this nwb file
    sgc.AnalysisNwbfile() & {"nwb_file_name": nwb_copy_filename}
).fetch("analysis_file_name")

kachery_selection_key = {"kachery_zone_name": zone_name}

for file in analysis_file_list:  # Add all analysis to shared list
    kachery_selection_key["analysis_file_name"] = file
    sgs.AnalysisNwbfileKacherySelection.insert1(
        kachery_selection_key, skip_duplicates=True
    )
# -

# With those files in the selection table, we can add them as links to the zone by
# populating the `AnalysisNwbfileKachery` table:
#

sgs.AnalysisNwbfileKachery.populate()

# Alternatively, we can share data based on its source table in the database using the helper function `share_data_to_kachery()`
#
# This will take a list of tables and add all associated analysis files for entries corresponding with a passed restriction.
# Here, we are sharing LFP and position data for the Session "minirec20230622\_.nwb"
#

# +
from spyglass.sharing import share_data_to_kachery
from spyglass.lfp.v1 import LFPV1
from spyglass.position.v1 import TrodesPosV1

tables = [LFPV1, TrodesPosV1]
restriction = {"nwb_file_name": "minirec20230622_.nwb"}
share_data_to_kachery(
    table_list=tables,
    restriction=restriction,
    zone_name=zone_name,
)
# -

# ## Managing access
#

# + [markdown] jupyter={"outputs_hidden": true}
# If all of that worked,
#
# 1. Go to https://kachery-gateway.figurl.org/admin?zone=your_zone
#    (changing your_zone to the name of your zone)
# 2. Go to the Admin/Authorization Settings tab
# 3. Add the GitHub login names and permissions for the users you want to share
#    with.
#
# If those users can connect to your database, they should now be able to use the
# `.fetch_nwb()` method to download any `AnalysisNwbfiles` that have been shared
# through Kachery.
#
# For example:
#
# ```python
# from spyglass.spikesorting import CuratedSpikeSorting
#
# test_sort = (
#     CuratedSpikeSorting & {"nwb_file_name": "minirec20230622_.nwb"}
# ).fetch()[0]
# sort = (CuratedSpikeSorting & test_sort).fetch_nwb()
# ```
#
# -

# ## Accessing Shared Data
#

# If you are a collaborator accessing datasets, you first need to be given access to the zone by a collaborator admin (see above).
#
# If you know the uri for the dataset you are accessing you can test this process below (example is for members of `franklab.collaborators`)
#

# +
import kachery_cloud as kcl

path = "/path/to/save/file/to/test"
zone_name = "franklab.collaborators"
uri = "sha1://ceac0c1995580dfdda98d6aa45b7dda72d63afe4"

os.environ["KACHERY_ZONE"] = zone_name
kcl.load_file(uri=uri, dest=path, verbose=True)
assert os.path.exists(path), f"File not downloaded to {path}"
# -

# In normal use, spyglass will manage setting the zone and uri when accessing files.
# In general, the easiest way to access data valueswill be through the `fetch1_dataframe()`
# function part of many of the spyglass tables. In brief this will check for the appropriate
# nwb analysis file in your local directory, and if not found, attempt to download it from the appropriate kachery zone.
# It will then parse the relevant information from that nwb file into a pandas dataframe.
#
# We will look at an example with data from the `LFPV1` table:
#

# +
from spyglass.lfp.v1 import LFPV1

# Here is the data we are going to access
LFPV1 & {
    "nwb_file_name": "Winnie20220713_.nwb",
    "target_interval_list_name": "pos 0 valid times",
}
# -

# We can access the data using `fetch1_dataframe()`
#

(
    LFPV1
    & {
        "nwb_file_name": "Winnie20220713_.nwb",
        "target_interval_list_name": "pos 0 valid times",
    }
).fetch1_dataframe()

# # Up Next
#

# In the [next notebook](./03_Merge_Tables.ipynb), we'll explore the details of a
# table tier unique to Spyglass, Merge Tables.
#
