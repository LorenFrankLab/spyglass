# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: spy
#     language: python
#     name: python3
# ---

# # Insert Data
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
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [these additional tutorials](https://github.com/datajoint/datajoint-tutorials)
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

# spyglass.common has the most frequently used tables
import spyglass.common as sgc

# spyglass.data_import has tools for inserting NWB files into the database
import spyglass.data_import as sgi

# -

# ## Visualizing the database
#

# Datajoint enables users to use Python to build and interact with a _Relational Database_. In a [Relational Data Model](https://www.smartsheet.com/relational-database-modeling), each table is an object that can reference
# information in other tables to avoid redundancy.
#
# DataJoint has built-in tools for generating/saving a _Diagram_ of the
# relationships between tables.
# [This page](https://datajoint.com/docs/core/datajoint-python/0.14/design/diagrams/) describes the notation used.
#
# Polygons are tables, colors reference
# [table type](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/tiers/):
#
# - Green rectangle: tables whose entries are entered _manually_.
# - Blue oval: tables whose entries are _imported_ from external files
#   (e.g. NWB file).
# - Red circle: tables whose entries are _computed_ from entries of other tables.
# - No shape (only text): tables whose entries are _part_ of the table upstream
#
# Lines are _dependencies_ between tables. An _upstream_ table is connected to a
# _downstream_ table via _inheritance_ of the
# [_primary key_](https://docs.datajoint.org/python/definition/07-Primary-Key.html).
# This is the set of attributes (i.e., column names) used to uniquely define an
# entry (i.e., a row)
#
# - Bold lines: the upstream primary key is the sole downstream primary key
# - Solid lines: the upstream table as part of the downstream primary key
# - Dashed lines: the primary key of upstream table as non-primary key
#

# Draw tables that are two levels below and one level above Session
dj.Diagram(sgc.Session) - 1 + 2

# By adding diagrams together, of adding and subtracting levels, we can visualize
# key parts of Spyglass.
#
# _Note:_ Notice the _Selection_ tables. This is a design pattern that selects a
# subset of upstream items for further processing. In some cases, these also pair
# the selected data with processing parameters.
#

# ## Example data
#

# After exploring the pipeline's structure, we'll now grab some example data.
# Spyglass will assume that the data is a neural recording with relevant auxiliary
# in NWB.
#
# We offer a few examples:
#
# - `minirec20230622.nwb`, .3 GB: minimal recording,
#   [Link](https://ucsf.box.com/s/k3sgql6z475oia848q1rgms4zdh4rkjn)
# - `mediumnwb20230802.nwb`, 32 GB: full-featured dataset,
#   [Link](https://ucsf.box.com/s/2qbhxghzpttfam4b7q7j8eg0qkut0opa)
# - `montague20200802.nwb`, 8 GB: full experimental recording,
#   [Link](https://ucsf.box.com/s/26je2eytjpqepyznwpm92020ztjuaomb)
# - For those in the UCSF network, these and many others on `/stelmo/nwb/raw`
#
# If you are connected to the Frank lab database, please rename any downloaded
# files (e.g., `example20200101_yourname.nwb`) to avoid naming collisions, as the
# file name acts as the primary key across key tables.
#

# +
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

# Define the name of the file that you copied and renamed
nwb_file_name = "minirec20230622.nwb"
nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
# -

# Spyglass will create a copy with this name.

nwb_copy_file_name

# ## Basic Inserts: Lab Team
#

# Let's start small by inserting personnel information.
#
# The `LabMember` table lists all lab members, with an additional part table for
# `LabMemberInfo`. This holds Google account and DataJoint username info for each
# member, for authentication purposes.
#
# We can insert lab member information using the NWB file `experimenter` field
# as follows...
#

# take a look at the lab members
sgc.LabMember.insert_from_nwbfile(nwb_file_name)

# We can [insert](https://datajoint.com/docs/core/datajoint-python/0.14/manipulation/insert/)
# into `LabMemberInfo` directly with a list of lists that reflect the order of
# the fields present in the table. See
# [this notebook](https://github.com/datajoint/datajoint-tutorials/blob/main/00-Getting_Started/01-DataJoint%20Basics%20-%20Interactive.ipynb)
# for examples of inserting with `dicts`.
#

sgc.LabMember.LabMemberInfo.insert(
    [  # Full name, Google email address, DataJoint username
        ["Firstname Lastname", "example1@gmail.com", "example1"],
        ["Firstname2 Lastname2", "example2@gmail.com", "example2"],
    ],
    skip_duplicates=True,
)
sgc.LabMember.LabMemberInfo()

# A `LabTeam` is a set of lab members who own a set of NWB files and the
# associated information in the database. This is often a subgroup that
# collaborates on the same projects. Data is associated with a given team,
# granting members analysis (e.g., curation) and deletion (coming soon)
# privileges.
#

sgc.LabTeam().create_new_team(
    team_name="My Team",  # Should be unique
    team_members=["Firstname Lastname", "Firstname2 Lastname2"],
    team_description="test",  # Optional
)

# By default, each member is part of their own team. We can see all teams and
# members by looking at the
# [part table](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/master-part/)
# `LabTeam.LabTeamMember`.
#

sgc.LabTeam.LabTeamMember()

# ## Inserting from NWB
#

#
# `spyglass.data_import.insert_sessions` helps take the many fields of data
# present in an NWB file and insert them into various tables across Spyglass. If
# the NWB file is properly composed, this includes...
#
# - the experimenter (replicating part of the process above)
# - animal behavior (e.g. video recording of position)
# - neural activity (extracellular recording of multiple brain areas)
# - etc.
#
# _Note:_ this may take time as Spyglass creates the copy. You may see a prompt
# about inserting device information.

sgi.insert_sessions(nwb_file_name)

# ## Inspecting the data
#

# To look at data, we can
# [query](https://datajoint.com/docs/core/datajoint-python/0.14/query/principles/)
# a table with `Table()` syntax.
#

sgc.Lab()

# The `Session` table has considerably more fields
#

sgc.Session.heading.names

# But a short primary key
#

sgc.Session.heading.primary_key

# The primary key is shown in bold in the html
#

sgc.Session()

# Text only interfaces designate the primary key fields with `*`
#

print(sgc.Session())

# To see a the
# [table definition](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/declare/),
# including
# [data types](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/attributes/),
# use `describe`.
#
# - `---` separates the primary key
# - `:` are used to separate field name from data type
# - `#` can be used to add comments to a field
#

# +
from pprint import pprint  # adds line breaks

pprint(sgc.Session.describe())
# -

# To look at specific entries in a table, we can use the `&`
# [operator](https://datajoint.com/docs/core/datajoint-python/0.14/query/operators/).
# Below, we _restrict_ based on a `dict`, but you can also use a
# [string](https://datajoint.com/docs/core/datajoint-python/0.14/query/restrict/#restriction-by-a-string).
#

sgc.Session & {"nwb_file_name": nwb_copy_file_name}

# `Raw` is connected to `Session` with a bold line, so it has the same primary key.
#

dj.Diagram(sgc.Session) + dj.Diagram(sgc.Raw)

sgc.Raw & {"nwb_file_name": nwb_copy_file_name}

# `IntervalList` is connected to `Session` with a solid line because it has
# additional primary key attributes. Here, you need to know both `nwb_file_name`
# and `interval_list_name` to uniquely identify an entry.
#

# join/split condenses long spaces before field comments
pprint("".join(sgc.IntervalList.describe().split("  ")))

sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}

# Raw [data types](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/attributes/) like `valid_times` are shown as `=BLOB=`. We can inspect
# these with [`fetch`](https://datajoint.com/docs/core/datajoint-python/0.14/query/fetch/)
#
# _Note:_ like `insert`/`insert1`, `fetch` can be uses as `fetch1` to raise an
# error when many (or no) entries are retrieved. To limit to one entry when there
# may be many, use `query.fetch(limit=1)[0]`
#

(
    sgc.IntervalList
    & {
        "nwb_file_name": nwb_copy_file_name,
        "interval_list_name": "pos 1 valid times",
    }
).fetch1("valid_times")

# In DataJoint [operators](https://datajoint.com/docs/core/datajoint-python/0.14/query/restrict/#restriction-by-a-string),
# `&` selects by a condition and `-` removes a condition.
#

(
    (
        (sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name})
        - {"interval_list_name": "pos 1 valid times"}
    )
    - {"interval_list_name": "pos 2 valid times"}
).fetch("interval_list_name")

# ## Deleting data
#

# Another neat feature of DataJoint is that it automatically maintains
# [data integrity](https://datajoint.com/docs/core/datajoint-python/0.14/design/integrity/)
# with _cascading deletes_. For example, if we delete our `Session` entry, all
# associated downstream entries are also deleted (e.g. `Raw`, `IntervalList`).
#
# _Note_: The deletion process can be complicated by
# [Merge Tables](https://lorenfranklab.github.io/spyglass/0.4/misc/merge_tables/)
# when the entry is referenced by a part table. To demo deletion in these cases,
# run the hidden code below.
#
# <details>
# <summary>Quick Merge Insert</summary>
#
# ```python
# import spyglass.lfp as lfp
#
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
# </details>
# <details>
# <summary>Deleting Merge Entries</summary>
#
# ```python
# from spyglass.utils.dj_merge_tables import delete_downstream_merge
#
# delete_downstream_merge(
#     sgc.Nwbfile(),
#     restriction={"nwb_file_name": nwb_copy_file_name},
#     dry_run=False, # True will show Merge Table entries that would be deleted
# )
# ```
# </details>

session_entry = sgc.Session & {"nwb_file_name": nwb_copy_file_name}
session_entry

# By default, DataJoint is cautious about deletes and will prompt before deleting.
# To delete, uncomment the cell below and respond `yes` in the prompt.
#

session_entry.delete()

# We can check that delete worked, both for `Session` and `IntervalList`
#

sgc.Session & {"nwb_file_name": nwb_copy_file_name}

sgc.IntervalList & {"nwb_file_name": nwb_copy_file_name}

# `delete` is useful for re-running something. Editing entries is possible, but
# discouraged because it can lead to
# [integrity](https://datajoint.com/docs/core/datajoint-python/0.14/design/integrity/)
# issues. Instead, re-enter and let the automation handle the rest.
#
# Spyglass falls short, however, in that deleting from `Session` doesn't also
# delete the associated entry in `Nwbfile`, which has to be removed separately
# (for now). This table offers a `cleanup` method to remove the added files (with
# the `delete_files` argument as `True`).
#
# _Note:_ this also applies to deleting files from `AnalysisNwbfile` table.
#

# +
# Uncomment to delete
# (sgc.Nwbfile & {"nwb_file_name": nwb_copy_file_name}).delete()
# -

# Note that the file (ends with `_.nwb`) has not been deleted, even if the entry
# was deleted above.
#

# !ls $SPYGLASS_BASE_DIR/raw

# We can clean these files with the `cleanup` method
#

sgc.Nwbfile().cleanup(delete_files=True)

# !ls $SPYGLASS_BASE_DIR/raw

# ## Up Next

# In the [next notebook](./02_Data_Sync.ipynb), we'll explore tools for syncing.
#
