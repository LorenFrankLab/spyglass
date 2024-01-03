# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# # Spike Sorting
#

# ## Overview
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
#   [the Insert Data notebook](./01_Insert_Data.ipynb)
#

# ### [Extract the recording](#section1)<br>
#
# 1. Specifying your [NWB](#Specifying-your-NWB-filename) file.<br>
# 2. Specifying which electrodes involved in the recording to sort data from. - [`SortGroup`](#SortGroup)<br>
# 3. Specifying the time segment of the recording we want to sort. - [`IntervalList`](#IntervalList), [`SortInterval`](#SortInterval)<br>
# 4. Specifying the parameters to use for filtering the recording. - [`SpikeSortingPreprocessingParameters`](#SpikeSortingPreprocessingParameters)<br>
# 5. Combining these parameters. - [`SpikeSortingRecordingSelection`](#SpikeSortingRecordingSelection)<br>
# 6. Extracting the recording. - [`SpikeSortingRecording`](#SpikeSortingRecording)<br>
# 7. Specifying the parameters to apply for artifact detection/removal. -[`ArtifactDetectionParameters`](#ArtifactDetectionParameters)<br>
#
# ### [Spike sorting the recording](#section2)<br>
#
# 1. Specify the spike sorter and parameters to use. - [`SpikeSorterParameters`](#SpikeSorterParameters)<br>
# 2. Combine these parameters. - [`SpikeSortingSelection`](#SpikeSortingSelection)<br>
# 3. Spike sort the extracted recording according to chose parameter set. - [`SpikeSorting`](#SpikeSorting)<br>
#
# <a href='#section1'></a>
# <a href='#section2'></a>
#

# ## Imports
#
# Let's start by importing tables from Spyglass and quieting warnings caused by
# some dependencies.
#
# _Note:_ It the imports below throw a `FileNotFoundError`, make a cell with `!env | grep X` where X is part of the problematic directory. This will show the variable causing issues. Make another cell that sets this variable elsewhere with `%env VAR="/your/path/"`
#

# +
import os
import datajoint as dj
import numpy as np

# change to the upper level folder to detect dj_local_conf.json
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
dj.config.load("dj_local_conf.json")  # load config for database connection info

import spyglass.common as sgc
import spyglass.spikesorting as sgs

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Fetch Exercise
#

# If you haven't already done so, add yourself to `LabTeam`
#

name, email, dj_user = "Firstname Lastname", "example@gmail.com", "user"
sgc.LabMember.insert_from_name(name)
sgc.LabMember.LabMemberInfo.insert1(
    [name, email, dj_user], skip_duplicates=True
)
sgc.LabTeam.LabTeamMember.insert1(
    {"team_name": "My Team", "lab_member_name": name},
    skip_duplicates=True,
)

# We can try `fetch` to confirm.
#
# _Exercise:_ Try to write a fer lines to generate a dictionary with team names as
# keys and lists of members as values. It may be helpful to add more data with the
# code above and use `fetch(as_dict=True)`.
#

my_team_members = (
    (sgc.LabTeam.LabTeamMember & {"team_name": "My Team"})
    .fetch("lab_member_name")
    .tolist()
)
if name in my_team_members:
    print("You made it in!")

# <details>
# <summary>Code hidden here</summary>
#
# ```python
# members = sgc.LabTeam.LabTeamMember.fetch(as_dict=True)
# teams_dict = {member["team_name"]: [] for member in members}
# for member in members:
#     teams_dict[member["team_name"]].append(member["lab_member_name"])
# print(teams_dict)
# ```
# </details>

# ## Adding an NWB file
#

# ### Import Data
#

# If you haven't already, load an NWB file. For more details on downloading and
# importing data, see [this notebook](./01_Insert_Data.ipynb).
#

# +
import spyglass.data_import as sdi

sdi.insert_sessions("minirec20230622.nwb")
nwb_file_name = "minirec20230622_.nwb"
# -

# ### Extracting the recording
#

# #### `SortGroup`
#

# Each NWB file will have multiple electrodes we can use for spike sorting. We
# commonly use multiple electrodes in a `SortGroup` selected by what tetrode or
# shank of a probe they were on.
#
# _Note:_ This will delete any existing entries. Answer 'yes' when prompted.
#

sgs.SortGroup().set_group_by_shank(nwb_file_name)

# Each electrode has an `electrode_id` and is associated with an
# `electrode_group_name`, which corresponds with a `sort_group_id`.
#
# For example, data recorded from a 32 tetrode (128 channel) drive results in 128
# unique `electrode_id`. This could result in 32 unique `electrode_group_name` and
# 32 unique `sort_group_id`.
#

sgs.SortGroup.SortGroupElectrode & {"nwb_file_name": nwb_file_name}

# #### `IntervalList`
#
# Next, we make a decision about the time interval for our spike sorting using
# `IntervalList`.
#

sgc.IntervalList & {"nwb_file_name": nwb_file_name}

# Let's start with the first run interval (`01_s1`) and fetch corresponding `valid_times`. For the `minirec` example, this is relatively short.
#

# +
interval_list_name = "01_s1"
interval_list = (
    sgc.IntervalList
    & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
).fetch1("valid_times")[0]


def print_interval_duration(interval_list: np.ndarray):
    duration = np.round((interval_list[1] - interval_list[0]))
    print(f"This interval list is {duration:g} seconds long")


print_interval_duration(interval_list)
# -

# #### `SortInterval`
#
# For longer recordings, Spyglass subsets this interval with `SortInterval`.
# Below, we select the first `n` seconds of this interval.
#

n = 9
sort_interval_name = interval_list_name + f"_first{n}"
sort_interval = np.array([interval_list[0], interval_list[0] + n])

# With the above, we can insert into `SortInterval`
#

sgs.SortInterval.insert1(
    {
        "nwb_file_name": nwb_file_name,
        "sort_interval_name": sort_interval_name,
        "sort_interval": sort_interval,
    },
    skip_duplicates=True,
)

# And verify the entry
#

print_interval_duration(
    (
        sgs.SortInterval
        & {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }
    ).fetch1("sort_interval")
)

# ## Preprocessing Parameters
#

# `SpikeSortingPreprocessingParameters` contains the parameters used to filter the
# recorded data in the spike band prior to sorting.
#

sgs.SpikeSortingPreprocessingParameters()

# Here, we insert the default parameters and then fetch them.
#

sgs.SpikeSortingPreprocessingParameters().insert_default()
preproc_params = (
    sgs.SpikeSortingPreprocessingParameters()
    & {"preproc_params_name": "default"}
).fetch1("preproc_params")
print(preproc_params)

# Let's adjust the `frequency_min` to 600, the preference for hippocampal data,
# and insert that into the table as a new set of parameters for hippocampal data.
#

preproc_params["frequency_min"] = 600
sgs.SpikeSortingPreprocessingParameters().insert1(
    {
        "preproc_params_name": "default_hippocampus",
        "preproc_params": preproc_params,
    },
    skip_duplicates=True,
)

# ## Processing a key
#
# _key_ is often used to describe an entry we want to move through the pipeline,
# and keys are often managed as dictionaries. Here, we'll manage the spike sort
# recording key, `ssr_key`.
#

interval_list_name

ssr_key = dict(
    nwb_file_name=nwb_file_name,
    sort_group_id=0,  # See SortGroup
    sort_interval_name=sort_interval_name,  # First N seconds above
    preproc_params_name="default_hippocampus",  # See preproc_params
    interval_list_name=interval_list_name,
    team_name="My Team",
)

# ### Recording Selection
#
# We now insert this key `SpikeSortingRecordingSelection` table to specify what
# time/tetrode/etc. of the recording we want to extract.
#

sgs.SpikeSortingRecordingSelection.insert1(ssr_key, skip_duplicates=True)
sgs.SpikeSortingRecordingSelection() & ssr_key

# ### `SpikeSortingRecording`
#
# And now we're ready to extract the recording! The
# [`populate` command](https://datajoint.com/docs/core/datajoint-python/0.14/compute/populate/)
# will automatically process data in Computed or Imported
# [table tiers](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/tiers/).
#
# If we only want to process certain entries, we can grab their primary key with
# the [`.proj()` command](https://datajoint.com/docs/core/datajoint-python/0.14/query/project/)
# and use a list of primary keys when calling `populate`.
#

ssr_pk = (sgs.SpikeSortingRecordingSelection & ssr_key).proj()
sgs.SpikeSortingRecording.populate([ssr_pk])

# Now we can see our recording in the table. _E x c i t i n g !_
#

sgs.SpikeSortingRecording() & ssr_key

# ## Artifact Detection
#
# `ArtifactDetectionParameters` establishes the parameters for removing artifacts
# from the data. We may want to target artifact signal that is within the
# frequency band of our filter (600Hz-6KHz), and thus will not get removed by
# filtering.
#
# For this demo, we'll use a parameter set to skip this step.
#

sgs.ArtifactDetectionParameters().insert_default()
artifact_key = (sgs.SpikeSortingRecording() & ssr_key).fetch1("KEY")
artifact_key["artifact_params_name"] = "none"

# We then pair artifact detection parameters in `ArtifactParameters` with a
# recording extracted through population of `SpikeSortingRecording` and insert
# into `ArtifactDetectionSelection`.
#

sgs.ArtifactDetectionSelection().insert1(artifact_key)
sgs.ArtifactDetectionSelection() & artifact_key

# Then, we can populate `ArtifactDetection`, which will find periods where there
# are artifacts, as specified by the parameters.
#

sgs.ArtifactDetection.populate(artifact_key)

# Populating `ArtifactDetection` also inserts an entry into `ArtifactRemovedIntervalList`, which stores the interval without detected artifacts.
#

sgs.ArtifactRemovedIntervalList() & artifact_key

# ## Spike sorting
#

# ### `SpikeSorterParameters`
#
# For our example, we will be using `mountainsort4`. There are already some default parameters in the `SpikeSorterParameters` table we'll `fetch`.
#

# +
sgs.SpikeSorterParameters().insert_default()

# Let's look at the default params
sorter_name = "mountainsort4"
ms4_default_params = (
    sgs.SpikeSorterParameters
    & {"sorter": sorter_name, "sorter_params_name": "default"}
).fetch1()
print(ms4_default_params)
# -

# Now we can change these default parameters to line up more closely with our preferences.
#

# +
sorter_params = {
    **ms4_default_params["sorter_params"],  # start with defaults
    "detect_sign": -1,  # downward going spikes (1 for upward, 0 for both)
    "adjacency_radius": 100,  # Sort electrodes together within 100 microns
    "filter": False,  # No filter, since we filter prior to starting sort
    "freq_min": 0,
    "freq_max": 0,
    "whiten": False,  # Turn whiten, since we whiten it prior to starting sort
    "num_workers": 4,  #  same number as number of electrodes
    "verbose": True,
    "clip_size": np.int64(
        1.33e-3  # same as # of samples for 1.33 ms based on the sampling rate
        * (sgc.Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate")
    ),
}
from pprint import pprint

pprint(sorter_params)
# -

# We can give these `sorter_params` a `sorter_params_name` and insert into `SpikeSorterParameters`.
#

sorter_params_name = "hippocampus_tutorial"
sgs.SpikeSorterParameters.insert1(
    {
        "sorter": sorter_name,
        "sorter_params_name": sorter_params_name,
        "sorter_params": sorter_params,
    },
    skip_duplicates=True,
)
(
    sgs.SpikeSorterParameters
    & {"sorter": sorter_name, "sorter_params_name": sorter_params_name}
).fetch1()

# ### `SpikeSortingSelection`
#
# **Gearing up to Spike Sort!**
#
# We now collect our various keys to insert into `SpikeSortingSelection`, which is specific to this recording and eventual sorting segment.
#
# _Note:_ the spike _sorter_ parameters defined above are specific to
# `mountainsort4` and may not work for other sorters.
#

ss_key = dict(
    **(sgs.ArtifactDetection & ssr_key).fetch1("KEY"),
    **(sgs.ArtifactRemovedIntervalList() & ssr_key).fetch1("KEY"),
    sorter=sorter_name,
    sorter_params_name=sorter_params_name,
)
ss_key.pop("artifact_params_name")
ss_key

sgs.SpikeSortingSelection.insert1(ss_key, skip_duplicates=True)
(sgs.SpikeSortingSelection & ss_key)

# ### `SpikeSorting`
#
# After adding to `SpikeSortingSelection`, we can simply populate `SpikeSorting`.
#
# _Note:_ This may take time with longer data sets. Be sure to `pip install mountainsort4` if this is your first time spike sorting.
#

# [(sgs.SpikeSortingSelection & ss_key).proj()]
sgs.SpikeSorting.populate()

# #### Check to make sure the table populated
#

sgs.SpikeSorting() & ss_key

# ## Next Steps
#
# Congratulations, you've spike sorted! See our
# [next notebook](./03_Curation.ipynb) for curation steps.
#
