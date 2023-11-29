# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# # Curation
#

# ## Overview
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [this notebook](./00_Setup.ipynb)
# - For a more detailed introduction to DataJoint with inserts, see
#   [this notebook](./01_Insert_Data.ipynb)
# - [The Spike Sorting notebook](./02_Spike_Sorting.ipynb) is a mandatory
#   prerequisite to Curation.
#

# ## Imports
#

# %env KACHERY_CLOUD_DIR="/home/cb/.kachery-cloud/"

# +
import os
import warnings
import datajoint as dj

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

from spyglass.spikesorting import SpikeSorting

# -

# ## Spikes Sorted
#
# Let's check that the sorting was successful in the previous notebook.
#

# Define the name of the file that you copied and renamed from previous tutorials
nwb_file_name = "minirec20230622.nwb"
nwb_copy_file_name = "minirec20230622_.nwb"
SpikeSorting & {"nwb_file_name": nwb_copy_file_name}

# ## `sortingview` web app
#

# As of June 2021, members of the Frank Lab can use the `sortingview` web app for
# manual curation.
#

# +
# ERROR: curation_feed_uri not a field in SpikeSorting
# -

workspace_uri = (SpikeSorting & {"nwb_file_name": nwb_copy_file_name}).fetch1(
    "curation_feed_uri"
)
print(
    f"https://sortingview.vercel.app/workspace?workspace={workspace_uri}&channel=franklab"
)

# This will take you to a workspace on the `sortingview` app. The workspace, which
# you can think of as a list of recording and associated sorting objects, was
# created at the end of spike sorting. On the workspace view, you will see a set
# of recordings that have been added to the workspace.
#
# ![Workspace view](./../notebook-images/workspace.png)
#
# Clicking on a recording then takes you to a page that gives you information
# about the recording as well as the associated sorting objects.
#
# ![Recording view](./../notebook-images/recording.png)
#
# Click on a sorting to see the curation view. Try exploring the many
# visualization widgets.
#
# ![Unit table](./../notebook-images/unittable.png)
#
# The most important is the `Units Table` and the `Curation` menu, which allows
# you to give labels to the units. The curation labels will persist even if you
# suddenly lose connection to the app; this is because the curation actions are
# appended to the workspace as soon as they are created. Note that if you are not
# logged in with your Google account, `Curation` menu may not be visible. Log in
# and refresh the page to access this feature.
#
# ![Curation](./../notebook-images/curation.png)
#

# ## Up Next
#
# Next, we'll turn our attention to [LFP data](./12_LFP.ipynb) data.
