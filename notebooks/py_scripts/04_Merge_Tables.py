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

# # Merge Tables
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
# - For information on why we use merge tables, and how to make one, see our
#   [documentation](https://lorenfranklab.github.io/spyglass/0.4/misc/merge_tables/)
#
# In short, merge tables represent the end processing point of a given way of
# processing the data in our pipelines. Merge Tables allow us to build new
# processing pipeline, or a new version of an existing pipeline, without having to
# drop or migrate the old tables. They allow data to be processed in different
# ways, but with a unified end result that downstream pipelines can all access.
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
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.simplefilter("ignore", category=UserWarning)

import spyglass.common as sgc
import spyglass.lfp as lfp
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.utils.dj_merge_tables import delete_downstream_parts, Merge
from spyglass.common.common_ephys import LFP as CommonLFP  # Upstream 1
from spyglass.lfp.lfp_merge import LFPOutput  # Merge Table
from spyglass.lfp.v1.lfp import LFPV1  # Upstream 2

# -

# ## Example data
#

# Check to make sure the data inserted in the previour notebook is still there.
#

nwb_file_name = "minirec20230622.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
nwb_file_dict = {"nwb_file_name": nwb_copy_file_name}
sgc.Session & nwb_file_dict

# If you haven't already done so, insert data into a Merge Table.
#
# _Note_: Some existing parents of Merge Tables perform the Merge Table insert as
# part of the populate methods. This practice will be revised in the future.
#
# <!-- TODO: Add entry to another parent to cover mutual exclusivity issues. -->
#

sgc.FirFilterParameters().create_standard_filters()
lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
    nwb_file_name=nwb_copy_file_name,
    group_name="test",
    electrode_list=[0],
)
lfp_key = {
    "nwb_file_name": nwb_copy_file_name,
    "lfp_electrode_group_name": "test",
    "target_interval_list_name": "01_s1",
    "filter_name": "LFP 0-400 Hz",
    "filter_sampling_rate": 30_000,
}
lfp.v1.LFPSelection.insert1(lfp_key, skip_duplicates=True)
lfp.v1.LFPV1().populate(lfp_key)
LFPOutput.insert([lfp_key], skip_duplicates=True)

# ## Helper functions
#

# Merge Tables have multiple custom methods that begin with `merge`.
#
# `help` can show us the docstring of each
#

merge_methods = [d for d in dir(Merge) if d.startswith("merge")]
print(merge_methods)

help(getattr(Merge, merge_methods[-1]))

# ## Showing data
#

# `merge_view` shows a union of the master and all part tables.
#
# _Note_: Restrict Merge Tables with arguments, not the `&` operator.
#
# - Normally: `Table & "field='value'"`
# - Instead: `MergeTable.merge_view(restriction="field='value'"`).
#

LFPOutput.merge_view()

# UUIDs help retain unique entries across all part tables. We can fetch NWB file
# by referencing this or other features.
#

uuid_key = (LFPOutput & nwb_file_dict).fetch(limit=1, as_dict=True)[-1]
restrict = LFPOutput & uuid_key
restrict

result1 = restrict.fetch_nwb(restrict.fetch1("KEY"))
result1

nwb_key = LFPOutput.merge_restrict(nwb_file_dict).fetch(as_dict=True)[0]
nwb_key

result2 = LFPOutput().fetch_nwb(nwb_key)
result2 == result1

# ## Selecting data
#

# There are also functions for retrieving part/parent table(s) and fetching data.
#
# These `get` functions will either return the part table of the Merge table or the parent table with the source information for that part.
#

result4 = LFPOutput.merge_get_part(restriction=nwb_file_dict, join_master=True)
result4

result5 = LFPOutput.merge_get_parent(restriction='nwb_file_name LIKE "mini%"')
result5

# `fetch` will collect all relevant entries and return them as a list in
# the format specified by keyword arguments and one's DataJoint config.
#

result6 = result5.fetch("lfp_sampling_rate")  # Sample rate for all mini* files
result6

# `merge_fetch` requires a restriction as the first argument. For no restriction,
# use `True`.
#

result7 = LFPOutput.merge_fetch(True, "filter_name", "nwb_file_name")
result7

result8 = LFPOutput.merge_fetch(as_dict=True)
result8

# ## Deletion from Merge Tables
#

# When deleting from Merge Tables, we can either...
#
# 1. delete from the Merge Table itself with `merge_delete`, deleting both
#    the master and part.
#
# 2. use `merge_delete_parent` to delete from the parent sources, getting rid of
#    the entries in the source table they came from.
#
# 3. use `delete_downstream_parts` to find downstream part tables, like Merge
#    Tables, and get rid full entries, avoiding orphaned master table entries.
#
# The two latter cases can be destructive, so we include an extra layer of
# protection with `dry_run`. When true (by default), these functions return
# a list of tables with the entries that would otherwise be deleted.
#

LFPOutput.merge_delete(nwb_file_dict)  # Delete from merge table

LFPOutput.merge_delete_parent(restriction=nwb_file_dict, dry_run=True)

# `delete_downstream_parts` is available from any other table in the pipeline,
# but it does take some time to find the links downstream. If you're using this,
# you can save time by reassigning your table to a variable, which will preserve
# a copy of the previous search.
#
# Because the copy is stored, this function may not see additional merge tables
# you've imported. To refresh this copy, set `reload_cache=True`
#

# +
nwbfile = sgc.Nwbfile()

(nwbfile & nwb_file_dict).delete_downstream_parts(
    dry_run=True,
    reload_cache=False,  # if still encountering errors, try setting this to True
)
# -

# This function is run automatically whin you use `cautious_delete`, which
# checks team permissions before deleting.
#

(nwbfile & nwb_file_dict).cautious_delete()

# ## Up Next
#

# In the [next notebook](./10_Spike_Sorting.ipynb), we'll start working with
# ephys data with spike sorting.
#
