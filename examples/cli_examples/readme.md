### Insert a lab team

Create a lab team.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-lab-team team.yaml

spyglass list-lab-teams
```

### Insert a lab member

Create a lab member.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-lab-member labmember.yaml

spyglass list-lab-members
```

### Insert a lab team member

Add a lab member to a lab team.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-lab-team-member labteammember.yaml

spyglass list-lab-team-members
```

### Insert a session

Insert a session from a raw .nwb file.

```bash
# First copy the .nwb file to the `$NWB_DATAJOINT_BASE_DIR/raw` directory.

# In the following, replace the .nwb file name as appropriate
# This will populate various tables, including:
#   Subject, DataAcquisitionDevice, CameraDevice, Probe, IntervalList, ...
spyglass insert-session RN2_20191110.nwb

spyglass list-sessions
spyglass list-interval-lists RN2_20191110_.nwb # Note the trailing underscore here
```

### Set up the sort groups

Set up the electrode sort groups.

See [set_sort_groups_by_shank.py](./set_sort_groups_by_shank.py) for an example.

### Insert a sort interval

Create a sort interval (time interval) for spike sorting.

See [insert_sort_interval.py](./insert_sort_interval.py) for an example.

### Insert spike sorting preprocessing parameters

Define spike sorting preprocessing parameters.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-spike-sorting-preprocessing-parameters parameters.yaml

spyglass list-spike-sorting-preprocessing-parameters
```

### Insert artifact detection parameters

Define artifact detection parameters.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-artifact-detection-parameters parameters.yaml

spyglass list-artifact-detection-parameters
```

### Insert spike sorting recording

Insert a spike sorting recording.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-spike-sorting-recording-selection recordingselection.yaml

spyglass list-spike-sorting-recording-selections RN2_20191110_.nwb

# Now populate the SpikeSortingRecording table
spyglass create-spike-sorting-recording recordingselection.yaml

spyglass list-spike-sorting-recordings RN2_20191110_.nwb
```

### Insert spike sorter parameters

Define spike sorter parameters.

```bash
# To see the format of the .yaml file, run without the .yaml argument
spyglass insert-spike-sorter-parameters parameters.yaml

spyglass list-spike-sorter-parameters
```