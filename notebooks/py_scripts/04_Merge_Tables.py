# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: '1335'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Merge Tables
#

# %% [markdown]
# ## Intro
#

# %% [markdown]
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

# %% [markdown]
# ## Imports
#

# %% [markdown]
# Let's start by importing the `spyglass` package, along with a few others.
#

# %%
import datajoint as dj

dj.config.load("../dj_local_conf_1335.json")
dj.conn()

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.simplefilter("ignore", category=UserWarning)

import spyglass.common as sgc
import spyglass.lfp as lfp
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.utils.dj_merge_tables import Merge
from spyglass.lfp.lfp_merge import LFPOutput  # Merge Table

# %% [markdown]
# ## Example data
#

# %% [markdown]
# Check to make sure the data inserted in the previous notebook is still there.
#

# %%
nwb_file_name = "minirec20230622.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
nwb_file_dict = {"nwb_file_name": nwb_copy_file_name}
sgc.Session & nwb_file_dict

# %% [markdown]
# If you haven't already done so, insert data into a Merge Table.
#
# _Note_: Some existing parents of Merge Tables perform the Merge Table insert as
# part of the populate methods. This practice will be revised in the future.
#
# <!-- TODO: Add entry to another parent to cover mutual exclusivity issues. -->
#

# %%
sgc.FirFilterParameters().create_standard_filters()
lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
    nwb_file_name=nwb_copy_file_name,
    group_name="test",
    electrode_list=[0],
    skip_duplicates=True,
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

# %% [markdown]
# ## Helper functions
#

# %% [markdown]
# Merge Tables expose **instance methods** that operate on a restricted table.
# The older `merge_*` class methods are deprecated (Spyglass 0.7.0) but still
# work and emit a warning on first call.
#
# <details>
# <summary><b>Familiar with the old API? Click to see the migration table.</b></summary>
#
# | Old call | New equivalent |
# |---|---|
# | `T.merge_view(r)` | `(T & r).view()` |
# | `T.merge_html(r)` | `(T & r).html()` |
# | `T.merge_restrict(r)` | `T & r` |
# | `T.merge_fetch(r, *attrs)` | `(T & r).fetch(*attrs)` |
# | `T.merge_get_part(r, ...)` | `(T & r).get_part_table(...)` |
# | `T.merge_get_parent(r, ...)` | `(T & r).get_parent_table(...)` |
# | `T.merge_delete(r)` | `(T & r).delete()` |
# | `T.merge_delete_parent(r, dry_run=True)` | `(T & r).delete_upstream(dry_run=True)` |
# | `T.merge_populate(source, keys)` | `T.populate(source, keys)` |
#
# All `merge_*` methods still work but will be removed in Spyglass 0.7.0.
#
# </details>

# %%
# New instance methods (preferred)
new_methods = [
    "view",
    "html",
    "get_part_table",
    "get_parent_table",
    "fetch",
    "fetch1",
    "super_fetch",
    "delete",
    "delete_upstream",
    "populate",
]
print("New API:", new_methods)

# Deprecated class methods (still work, emit warnings)
deprecated = [d for d in dir(Merge) if d.startswith("merge_")]
print("Deprecated:", deprecated)

# %%
help(LFPOutput.get_part_table)

# %% [markdown]
# ## Showing data
#

# %% [markdown]
# Merge tables behave like they're already joined with the parts.
#

# %%
# %load_ext autoreload
# %autoreload 2

# %%
LFPOutput()

# %% [markdown]
# This also works with `dj.Top`

# %%
LFPOutput & dj.Top()

# %% [markdown]
# To get the standard view of the table, use `T.super_view()`

# %%
LFPOutput().super_view()

# %% [markdown]
# Restrict Merge Tables with the standard `&` operator. Both dict and string
# restrictions that reference part-table fields are **resolved automatically**:
#
# ```python
# Merge & {"field": "value"}      # ✓ dict — resolves through parts
# Merge & 'field LIKE "val%"'     # ✓ string — also resolves through parts
# Merge & {"merge_id": some_uuid} # ✓ master field — unchanged
# ```

# %%
LFPOutput & nwb_file_dict

# %% [markdown]
# `fetch()` walks parts → returns part-table data

# %%
(LFPOutput & nwb_file_dict).fetch(as_dict=True)[0]

# %% [markdown]
# `fetch1()` follows DataJoint convention

# %%
restrict = LFPOutput & "lfp_electrode_group_name='test'"
restrict.fetch1()

# %%
LFPOutput.fetch("filter_name")

# %%
LFPOutput.fetch("filter_name", "lfp_sampling_rate")

# %% [markdown]
# `super_fetch()` stays on master → returns only merge_id and source

# %%
(LFPOutput & nwb_file_dict).super_fetch(as_dict=True)[0]

# %% [markdown]
# Fetch just the master primary key (bypasses part-walking)

# %%
uuid_key = (LFPOutput & nwb_file_dict).fetch("KEY", limit=1)[0]
restrict = LFPOutput & uuid_key
restrict

# %% [markdown]
# Fetch all part-table columns as a dict

# %%
nwb_key = (LFPOutput & nwb_file_dict).fetch(as_dict=True)[0]
nwb_key

# %%
LFPOutput().fetch_nwb(nwb_key)[0]

# %% [markdown]
# ## Selecting data
#

# %% [markdown]
# - `get_part_table()` returns the Merge Part table for the current restriction.
# - `get_parent_table()` returns the upstream source table (where the actual data
# lives).
#
# Both accept `join_master=True` to include `merge_id` / `source`.

# %%
result4 = (LFPOutput & nwb_file_dict).get_part_table(join_master=True)
result4

# %%
result5 = (LFPOutput & 'nwb_file_name LIKE "mini%"').get_parent_table()
result5

# %%
result5.full_table_name

# %% [markdown]
# ## Deletion from Merge Tables
#

# %% [markdown]
# When deleting from Merge Tables, we have three options:
#
# 1. **`(T & restriction).delete()`** — removes the master and part entries from
#    the Merge Table. The upstream source data (e.g., `LFPV1`) is *not* removed.
#
# 2. **`(T & restriction).delete_upstream(dry_run=True)`** — removes entries from
#    the upstream source tables. Use `dry_run=True` (default) to preview first.
#
# 3. **`table.delete_downstream_parts(dry_run=True)`** — available on any
#    upstream table; finds downstream Merge Table entries and removes them,
#    preventing orphaned master rows.
#
# The latter two are destructive. Call `delete_upstream` before `delete()` if
# you want to also remove the upstream source data in the same session.

# %%
# Preview: what would delete_upstream remove from the source tables?
(LFPOutput & nwb_file_dict).delete_upstream(dry_run=True)

# %%
# Delete from the Merge Table (master + parts)
(LFPOutput & nwb_file_dict).delete()

# %% [markdown]
# This function is run automatically when you use `cautious_delete`, which
# checks team permissions before deleting.
#

# %%
from spyglass.common import Nwbfile

(Nwbfile & nwb_file_dict).cautious_delete()

# %% [markdown]
# ## Up Next
#

# %% [markdown]
# In the [next notebook](./10_Spike_Sorting.ipynb), we'll start working with
# ephys data with spike sorting.
#
