# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
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
# - Do not update Spyglass until the export is complete.
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
# <details><summary>Why these limitations?</summary>
#
# `SpyglassMixin` is what makes this process possible. It uses an environmental
# variable to make sure all tables are on the same page about the export ID.
# We get this feature by inheriting, but cannot set more that one value for the
# environmental variable.
#
# The export process was designed with reproducibility in mind, and will export
# your conda environment to match. We want to be sure that the analysis you run
# is replicable using the same conda environment.
#
# </details>
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
from spyglass.common.common_ephys import Electrode
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
# - _ONE_ export at a time.
# - All tables must inherit `SpyglassMixin`.
#
# Let's start logging for 'paper1'.
#

# +
paper_key = {"paper_id": "paper1"}

ExportSelection().start_export(**paper_key, analysis_id="analysis1")
my_lfp_data = (
    Electrode  # Logging this table
    & dj.AndList(
        [
            "nwb_file_name LIKE 'min%'",  # using a string restrictionshared
            {"electrode_id": 1},  # and a dictionary restriction
        ]
    )
).fetch()
# -

# **Note**: Compound resrictions (e.g., `Table & a & b`) will be logged separately
# as `Table & a` or `Table & b`. To fully restrict, use strings (e.g.,
# `Table & 'a AND b'`) or `dj.AndList([a,b])`.

# We can check that it was logged. The syntax of the restriction will look
# different from what we see in python, but the `preview_tables` will look
# familiar.
#

ExportSelection.Table()

# If there seem to be redundant entries in this part table, we can ignore them.
# They'll be merged later during `Export.populate`.
#
# Let's log more under the same analysis ...
#

my_other_lfp_data = (Electrode & "electrode_id > 125").fetch()

# Since these restrictions are mutually exclusive, we can check that the will
# be combined appropriately by previewing the logged tables...
#

ExportSelection().preview_tables(**paper_key)

# For a more comprehensive view of what would be in the export, you can look at
# `ExportSelection().show_all_tables(**paper_key)`. This may take some time to
# generate.

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
# Depending on your database's configuration, you may need an admin on your team
# to run the resulting bash script. This is true of the Frank Lab. Doing so will
# result will be a `.sql` file that anyone can use to replicate the database
# entries you used in your analysis.
#

# ## Dandi

# One benefit of the `Export` table is it provides a list of all raw data,
# intermediate analysis files, and final analysis files needed to generate a set
# of figures in a work. To aid in data-sharing standards, we have implemented
# an optional additional export step with
# tools to compile and upload this set of files as a Dandi dataset, which can then
# be used by Spyglass to directly read the data from the Dandi database if not
# available locally.
#
# We will walk through the steps to do so here:
# 1. Upload the data
# 2. Export this table alongside the previous export
# 3. Generate a sharable docker container (Coming soon!)

# <details>
#    <summary> Dandi data compliance (admins)</summary>
#
#    >__WARNING__: The following describes spyglass utilities that require database admin privileges to run. It involves altering database values to correct for metadata format errors generated prior to spyglass insert. As such it has the potential to violate data integrity and should be used with caution.
#    >
#    >The Dandi database has specific formatting standards for metadata and nwb files. If there were violations of this standard in the
#    raw nwbfile, spyglass will propagate them into all generated analysis files. In this case, running the code below will result in a list of error printouts and an error raised within the `validate_dandiset` function.
#    >
#    >To aid in correcting common formatting errors identified with changes in dandi standards, we have included the method
#    ```
#    Export().prepare_files_for_export(paper_key)
#    ```
#    >which will attempt to resolve these issues for a set of paper files. The code is __not__ guaranteed to address all errors found within the file, but can be used as a template for your specific errors
# </details>
#
#
#

# ### Dandiset Upload

# The first step you will need to do is to [create a Dandi account](https://docs.dandiarchive.org/16_account/).
# With this account you can then [register a new dandiset](https://dandiarchive.org/dandiset/create) by providing a name and basic metadata.
# Dandi's instructions for these steps are available [here](https://docs.dandiarchive.org/13_upload/).
#
# The key information you will need from your registration is the `dandiset ID` and your account `api_key`, both of which are available from your registered account.
#
# Spyglass can then use this information to compile and upload the dandiset for your paper:

# +
from spyglass.common.common_dandi import DandiPath

dandiset_id = 214304  # use the value for you registered dandiset
dandi_api_key = (
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # key connected to your Dandi account
)

DandiPath().compile_dandiset(
    paper_key,
    dandiset_id=dandiset_id,
    dandi_api_key=dandi_api_key,
    dandi_instance="dandi",
)  # use dandi_instance="dandi-staging" to use dandi's dev server
# -

# As well as uploading your dandiset, this function will populate the table `DandiPath` which will record the information needed to access a given analysis file from the Dandi server
#

DandiPath() & {"export_id": 14}

# When fetching data with Spyglass, if a file is not available locally, Syglass
# will automatically use this information to stream the file from Dandi's server
#  if available, providing an additional method for sharing data with
#  collaborators post-publication.

# ### Export Dandi Table
#
# Because we generated new entries in this process we may want to share alongside
# our export, we'll run the additional step of exporting this table as well.
#

DandiPath().write_mysqldump(paper_key)

# ## Sharing the export
#
# The steps above will generate several files in this paper's export directory.
# By default, this is relative to your Spyglass base directory:
# `{BASE_DIR}/export/{PAPER_ID}`.
#
# The `.sh` files should be run by a database administrator who is familiar with
# running `mysqldump` commands.
#
# <details><summary>Note to administrators</summary>
#
# The dump process saves the exporter's credentials as a `.my.cnf` file
# ([about these files](https://dev.mysql.com/doc/refman/8.4/en/option-files.html))
# to allow running `mysqldump` without additional flags for user, password, etc.
#
# If database permissions permit running exports from the instance that runs the
# exports, you can esure you have a similar `.my.cnf` config in place and run the
# export shell scripts as-is. Some databases, like the one used by the Frank Lab
# have protections in place that would require these script(s) to be run from the
# database instance. Resulting `.sql` files should be placed in the same export
# directory mentioned above.
#
# </details>
#
# Then, visit the dockerization repository
# [here](https://github.com/LorenFrankLab/spyglass-export-docker)
# and follow the instructions in 'Quick Start'.

# ## Up Next
#

# In the [next notebook](./10_Spike_Sorting.ipynb), we'll start working with
# ephys data with spike sorting.
#
