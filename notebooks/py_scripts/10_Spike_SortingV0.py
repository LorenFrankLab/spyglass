# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3.10.5 64-bit
#     language: python
#     name: python3
# ---

# # Spike Sorting V0
#
# _Note_: This notebook explains the first version of the spike sorting pipeline
# and is preserved for using existing data. New users should use
# [V1](./10_Spike_SortingV1.ipynb).
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
import spyglass.spikesorting.v0 as sgs
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# -

# ## Fetch Exercise
#

# If you haven't already done so, add yourself to `LabTeam`
#

# +
# Full name, Google email address, DataJoint username, admin
name, email, dj_user, admin = (
    "Firstname_spikesv0 Lastname_spikesv0",
    "example_spikesv0@gmail.com",
    dj.config["database.user"],  # use the same username as the database
    0,
)
sgc.LabMember.insert_from_name(name)
sgc.LabMember.LabMemberInfo.insert1(
    [
        name,
        email,
        dj_user,
        admin,
    ],
    skip_duplicates=True,
)

# Make a lab team if doesn't already exist, otherwise insert yourself into team
team_name = "My Team"
if not sgc.LabTeam() & {"team_name": team_name}:
    sgc.LabTeam().create_new_team(
        team_name=team_name,  # Should be unique
        team_members=[name],
        team_description="test",  # Optional
    )
else:
    sgc.LabTeam.LabTeamMember().insert1(
        {"team_name": team_name, "lab_member_name": name}, skip_duplicates=True
    )

sgc.LabMember.LabMemberInfo() & {
    "team_name": "My Team",
    "lab_member_name": "Firstname_spikesv0 Lastname_spikesv0",
}
# -

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
# _Note:_ This will delete any existing entries. Answer 'yes' when prompted, or skip
# running this cell to leave data in place.
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

sgs.ArtifactDetectionSelection().insert1(artifact_key, skip_duplicates=True)
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
sgs.SpikeSorting.populate(ss_key)

# #### Check to make sure the table populated
#

sgs.SpikeSorting() & ss_key

# ## Automatic Curation
#
# Spikesorting algorithms can sometimes identify noise or other undesired features as spiking units.
# Spyglass provides a curation pipeline to detect and label such features to exclude them
# from downstream analysis.
#
#

# ### Initial Curation
#
# The `Curation` table keeps track of rounds of spikesorting curations in the spikesorting v0 pipeline.
# Before we begin, we first insert an initial curation entry with the spiking results.

# +
for sorting_key in (sgs.SpikeSorting() & ss_key).fetch("KEY"):
    # insert_curation will make an entry with a new curation_id regardless of whether it already exists
    # to avoid this, we check if the curation already exists
    if not (sgs.Curation() & sorting_key):
        sgs.Curation.insert_curation(sorting_key)

sgs.Curation() & ss_key
# -

# ### Waveform Extraction
#
# Some metrics used for curating units are dependent on features of the spike waveform.
# We extract these for each unit's initial curation here

# Parameters used for waveform extraction from the recording
waveform_params_name = "default_whitened"
sgs.WaveformParameters().insert_default()  # insert default parameter sets if not already in database
(
    sgs.WaveformParameters() & {"waveform_params_name": waveform_params_name}
).fetch(as_dict=True)[0]

# extract waveforms
curation_keys = [
    {**k, "waveform_params_name": waveform_params_name}
    for k in (sgs.Curation() & ss_key & {"curation_id": 0}).fetch("KEY")
]
sgs.WaveformSelection.insert(curation_keys, skip_duplicates=True)
sgs.Waveforms.populate(ss_key)

# ### Quality Metrics
#
# With these waveforms, we can calculate the metrics used to determine the quality of each unit.

# parameters which define what quality metrics are calculated and how
metric_params_name = "franklab_default3"
sgs.MetricParameters().insert_default()  # insert default parameter sets if not already in database
(sgs.MetricParameters() & {"metric_params_name": metric_params_name}).fetch(
    "metric_params"
)[0]

waveform_keys = [
    {**k, "metric_params_name": metric_params_name}
    for k in (sgs.Waveforms() & ss_key).fetch("KEY")
]
sgs.MetricSelection.insert(waveform_keys, skip_duplicates=True)
sgs.QualityMetrics().populate(ss_key)
sgs.QualityMetrics() & ss_key

# Look at the quality metrics for the first curation
(sgs.QualityMetrics() & ss_key).fetch_nwb()[0]["object_id"]

# ### Automatic Curation Labeling
#
# With these metrics, we can assign labels to the sorted units using the `AutomaticCuration` table

# We can select our criteria for unit labeling here
auto_curation_params_name = "default"
sgs.AutomaticCurationParameters().insert_default()
(
    sgs.AutomaticCurationParameters()
    & {"auto_curation_params_name": auto_curation_params_name}
).fetch1()

# We can now apply the automatic curation criteria to the quality metrics
metric_keys = [
    {**k, "auto_curation_params_name": auto_curation_params_name}
    for k in (sgs.QualityMetrics() & ss_key).fetch("KEY")
]
sgs.AutomaticCurationSelection.insert(metric_keys, skip_duplicates=True)
# populating this table will make a new entry in the curation table
sgs.AutomaticCuration().populate(ss_key)
sgs.Curation() & ss_key

# ### Insert desired curation into downstream and merge tables for future analysis
#
# Now that we've performed auto-curation, we can insert the results of our chosen curation into
# `CuratedSpikeSorting` (the final table of this pipeline), and the merge table `SpikeSortingOutput`.
# Downstream analyses such as decoding will access the spiking data from there

# +
# get the curation keys corresponding to the automatic curation
auto_curation_key_list = (sgs.AutomaticCuration() & ss_key).fetch(
    "auto_curation_key"
)

# insert into CuratedSpikeSorting
for auto_key in auto_curation_key_list:
    # get the full key information needed
    curation_auto_key = (sgs.Curation() & auto_key).fetch1("KEY")
    sgs.CuratedSpikeSortingSelection.insert1(
        curation_auto_key, skip_duplicates=True
    )
sgs.CuratedSpikeSorting.populate(ss_key)

# Add the curated spike sorting to the SpikeSortingOutput merge table
keys_for_merge_tables = (
    sgs.CuratedSpikeSorting & auto_curation_key_list
).fetch("KEY")
SpikeSortingOutput.insert(
    keys_for_merge_tables,
    skip_duplicates=True,
    part_name="CuratedSpikeSorting",
)
# Here's our result!
SpikeSortingOutput.CuratedSpikeSorting() & ss_key
# -

# ## Manual Curation with figurl

# As of June 2021, members of the Frank Lab can use the `sortingview` web app for
# manual curation. To make use of this, we need to populate the `CurationFigurl` table.
#
# We begin by selecting a starting point from the curation entries. In this case we will use
# the AutomaticCuration populated above as a starting point for manual curation, though you could also
# start from the opriginal curation entry by selecting the proper key from the `Curation` table
#
# _Note_: This step requires setting up your kachery sharing through the [sharing notebook](02_Data_Sync.ipynb)
#
#

# +
starting_curations = (sgs.AutomaticCuration() & ss_key).fetch(
    "auto_curation_key"
)  # you could also select any key from the sgs.Curation table here

username = "username"
fig_url_repo = f"gh://LorenFrankLab/sorting-curations/main/{username}/"  # settings for franklab members

sort_interval_name = interval_list_name
gh_url = (
    fig_url_repo
    + str(nwb_file_name + "_" + sort_interval_name)  # session id
    + "/{}"  # tetrode using auto_id['sort_group_id']
    + "/curation.json"
)  # url where the curation is stored

for auto_id in starting_curations:
    auto_curation_out_key = dict(
        **(sgs.Curation() & auto_id).fetch1("KEY"),
        new_curation_uri=gh_url.format(str(auto_id["sort_group_id"])),
    )
    sgs.CurationFigurlSelection.insert1(
        auto_curation_out_key, skip_duplicates=True
    )
    sgs.CurationFigurl.populate(auto_curation_out_key)
# -

# We can then access the url for the curation figurl like so:

print((sgs.CurationFigurl & ss_key).fetch("url")[0])

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
