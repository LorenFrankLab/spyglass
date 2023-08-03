# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# # Sync Data
#

# DEV note:
#  - set up as host, then as client
#  - test as collaborator

# ## Overview
#

# This notebook will cover ...
#
# 1. [General Kachery information](#intro)
# 2. Setting up Kachery as a [host](#host-setup). If you'll use an existing host,
#     skip this.
# 3. Setting up Kachery in your [database](#database-setup). If you're using an
#     existing database, skip this.
# 4. Adding Kachery [data](#data-setup).
#

# ## Intro
#

# This is one notebook in a multi-part series on Spyglass. Before running, be sure
# to [setup your environment](./00_Setup.ipynb) and run some analyses (e.g.
# [LFP](./12_LFP.ipynb)).
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
# 3. If present, attempt to download from the associated Kachery Resource.
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
# in a DataJoint config.
#
# - Environment variable:
#
#    ```bash
#    export KACHERY_ZONE=franklab.default
#    export KACHERY_CLOUD_DIR=/stelmo/nwb/.kachery_cloud
#    ```
#
# - DataJoint Config:
#
#    ```json
#    "custom": {
#       "kachery_zone": "franklab.default"
#    }
#    ```

# ## Host Setup

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

# ### Resources
#
# See [instructions](https://github.com/scratchrealm/kachery-resource/blob/main/README.md)
# for setting up zone resources. This allows for sharing files on demand. We
# suggest using the same name for the zone and resource.
#
# _Note:_ For each zone, you need to run the local daemon that listens for
# requests from that zone. An example of the bash script we use is
#
# ```bash
# export KACHERY_ZONE=franklab.collaborators
# export KACHERY_CLOUD_DIR=/stelmo/nwb/.kachery_cloud
# # cd /stelmo/nwb/franklab_collaborators_resource
# npx kachery-resource@latest share
# ```

# ## Database Setup
#

#
# Next we'll add zones/resources to the Spyglass database.

# +
import os
import datajoint as dj

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.sharing as sgs
from spyglass.settings import load_config

import warnings

warnings.filterwarnings("ignore")
# -

# Check existing Zones:

sgs.KacheryZone()

# Check existing file list:

sgs.AnalysisNwbfileKachery()

# Prepare an entry for the `KacheryZone` table:

zone_name = load_config().get("KACHERY_ZONE")
cloud_dir = load_config().get("KACHERY_CLOUD_DIR")
zone_key = {
    "kachery_zone_name": zone_name,
    "description": " ".join(zone_name.split(".")) + " zone",
    "kachery_cloud_dir": cloud_dir,
    "kachery_proxy": "https://kachery-resource-proxy.herokuapp.com",
    "lab_name": sgc.Lab.fetch("lab_name", limit=1)[0],
}


# Use caution when inserting into an active database, as it could interfere with
# ongoing work.

sgs.KacheryZone().insert1(zone_key)

# ## Data Setup

# Once the zone exists, we can add `AnalysisNWB` files we want to share by adding
# entries to the `AnalysisNwbfileKacherySelection` table.
#
# _Note:_ This step depends on having previously run an analysis on the example
# file.

# +
nwb_file_name = "minirec20230622_.nwb"

analysis_file_list = (  # Grab all analysis files for this nwb file
    sgc.AnalysisNwbfile() & {"nwb_file_name": nwb_file_name}
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

sgs.AnalysisNwbfileKachery.populate()

# + [markdown] jupyter={"outputs_hidden": true}
# If all of that worked,
#
# 1. go to https://kachery-gateway.figurl.org/admin?zone=your_zone
#     (changing your_zone to the name of your zone)
# 2. Go to the Admin/Authorization Settings tab
# 3. Add the GitHub login names and permissions for the users you want to share
#     with.
#
# If those users can connect to your database, they should now be able to use the
# `.fetch_nwb()` method to download any `AnalysisNwbfiles` that have been shared
# through Kachery.
#
# For example:
#
# ```python
# nwb_file_name = "wilbur20210331_.nwb"
# from spyglass.spikesorting import CuratedSpikeSorting
#
# test_sort = (CuratedSpikeSorting & {'nwb_file_name' : nwb_file_name}).fetch()[0]
# sort = (CuratedSpikeSorting & test_sort).fetch_nwb()
# ```
