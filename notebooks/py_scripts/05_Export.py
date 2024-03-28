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

# Let's start by connecting to the database and importing some tables that might
# be used in an analysis.
#

# +
import os
import datajoint as dj

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

from spyglass.common.common_usage import Export, ExportSelection
from spyglass.lfp.analysis.v1 import LFPBandV1
from spyglass.position.v1 import TrodesPosV1
from spyglass.spikesorting.v1.curation import CurationV1

# -

# ## Export Tables
#
# The `ExportSelection` table will populate while we conduct the analysis. For
# each file opened and each `fetch` call, an entry will be logged in one of its
# part tables.
#

ExportSelection()

ExportSelection.Table()

ExportSelection.File()

# Exports are organized around paper and analysis IDs. A single export will be
# generated for each paper, but we can delete/revise logs for each analysis before
# running the export. When we're ready, we can run the `populate_paper` method
# of the `Export` table. By default, export logs will ignore all tables in this
# `common_usage` schema.
#

# ## Logging
#
# There are a few restrictions to keep in mind when export logging:
#
# - You can only run _ONE_ export at a time.
# - All tables must inherit `SpyglassMixin`
#
# <details><summary>How to inherit <code>SpyglassMixin</code></summary>
#
# DataJoint tables all inherit from one of the built-in table types.
#
# ```python
# class MyTable(dj.Manual):
#     ...
# ```
#
# To inherit the mixin, simply add it to the `()` of the class before the
# DataJoint class. This can be done for existing tables without dropping them,
# so long as the change has been made prior to export logging.
#
# ```python
# from spyglass.utils import SpyglassMixin
# class MyTable(SpyglassMixin, dj.Manual):
#     ...
# ```
#
# </details>
#
# Let's start logging for 'paper1'.
#

# +
paper_key = {"paper_id": "paper1"}

ExportSelection().start_export(**paper_key, analysis_id="analysis1")
my_lfp_data = (
    LFPBandV1  # Logging this table
    & "nwb_file_name LIKE 'med%'"  # using a string restriction
    & {"filter_name": "Theta 5-11 Hz"}  # and a dictionary restriction
).fetch()
# -

# We can check that it was logged. The syntax of the restriction will look
# different from what we see in python, but the `preview_tables` will look
# familiar.
#

ExportSelection.Table()

# And log more under the same analysis ...
#

my_other_lfp_data = (
    LFPBandV1
    & {
        "nwb_file_name": "mediumnwb20230802_.nwb",
        "filter_name": "Theta 5-10 Hz",
    }
).fetch()

# Since these restrictions are mutually exclusive, we can check that the will
# be combined appropriately by priviewing the logged tables...
#

ExportSelection().preview_tables(**paper_key)

# Let's try adding a new analysis with a fetched nwb file. Starting a new export
# will stop the previous one.
#

ExportSelection().start_export(**paper_key, analysis_id="analysis2")
curation_nwb = (CurationV1 & "curation_id = 1").fetch_nwb()
trodes_data = (TrodesPosV1 & 'trodes_pos_params_name = "single_led"').fetch()

# We can check that the right files were logged with the following...
#

ExportSelection().list_file_paths(paper_key)

# And stop the export with ...
#

ExportSelection().stop_export()

# ## Populate
#
# The `Export` table has a `populate_paper` method that will generate an export
# bash script for the tables required by your analysis, including all the upstream
# tables you didn't directly need, like `Subject` and `Session`.
#
# **NOTE:** Populating the export for a given paper will overwrite any previous
# runs. For example, if you ran an export, and then added a third analysis for the
# same paper, generating another export will delete any existing bash script and
# `Export` table entries for the previous run.
#

Export().populate_paper(**paper_key)

# By default the export script will be located in an `export` folder within your
# `SPYGLASS_BASE_DIR`. This default can be changed by adjusting your `dj.config`.
#
# Frank Lab members will need the help of a database admin (e.g., Chris) to
# run the resulting bash script. The result will be a `.sql` file that anyone
# can use to replicate the database entries you used in your analysis.
#

# ## Up Next
#

# In the [next notebook](./10_Spike_Sorting.ipynb), we'll start working with
# ephys data with spike sorting.
#
