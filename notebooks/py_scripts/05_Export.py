# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: spy
#     language: python
#     name: python3
# ---

# # Export
#

# ## Intro
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - To insert data, see [the Insert Data notebook](./01_Insert_Data.ipynb)
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [these additional tutorials](https://github.com/datajoint/datajoint-tutorials)
# - For information on what's goint on behind the scenes of an export, see
#   [documentation](https://lorenfranklab.github.io/spyglass/0.5/misc/export/)
#
# In short, Spyglass offers the ability to generate exports of one or more subsets
# of the database required for a specific analysis as long as you do the following:
#
# - Inherit `SpyglassMixin` for all custom tables.
# - Run only one export at a time.
# - Start and stop each export logging process.
#
# **NOTE:** For demonstration purposes, this notebook relies on a more populated
# database to highlight restriction merging capabilities of the export process.
# Adjust the restrictions to suit your own dataset.
#

# ## Imports
#

# Let's start by importing the `spyglass` package, along with a few others.
#

# +
import os
import datajoint as dj

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

# ignore datajoint+jupyter async warnings
from spyglass.common.common_usage import Export, ExportSelection
from spyglass.lfp.analysis.v1 import LFPBandV1
from spyglass.position.v1 import TrodesPosV1
from spyglass.spikesorting.v1.curation import CurationV1

# TODO: Add commentary, describe helpers on ExportSelection

paper_key = {"paper_id": "paper1"}
ExportSelection().start_export(**paper_key, analysis_id="test1")
a = (
    LFPBandV1 & "nwb_file_name LIKE 'med%'" & {"filter_name": "Theta 5-11 Hz"}
).fetch()
b = (
    LFPBandV1
    & {
        "nwb_file_name": "mediumnwb20230802_.nwb",
        "filter_name": "Theta 5-10 Hz",
    }
).fetch()
ExportSelection().start_export(**paper_key, analysis_id="test2")
c = (CurationV1 & "curation_id = 1").fetch_nwb()
d = (TrodesPosV1 & 'trodes_pos_params_name = "single_led"').fetch()
ExportSelection().stop_export()
Export().populate_paper(**paper_key)
# -

# ## Up Next
#

# In the [next notebook](./10_Spike_Sorting.ipynb), we'll start working with
# ephys data with spike sorting.
#
