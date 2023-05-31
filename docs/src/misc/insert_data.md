# How to insert data into `spyglass`

In `spyglass`, every table corresponds to an object. An experimental session is defined as a collection of such objects. When an NWB file is ingested into `spyglass`, the information about these objects is first read and inserted into tables in the `common` module (e.g. `Institution`, `Lab`, `Electrode`, etc). However, not every NWB file has all the information required by `spyglass`. For example, many NWB files do not contain any information about the `DataAcquisitionDevice` or `Probe` because NWB does not yet have an official standard for specifying them. In addition, one might find that the information contained in the NWB file is incorrect and would like to modify it before inserting it into `spyglass` without having to go through the time-consuming process of re-generating the NWB file. For these cases, we provide an alternative approach to inserting data to `spyglass`.

This alternate approach consists of two steps. First, the user must identify entries that they would like to add to the `spyglass` database that exist independently of any particular NWB file. For example, information about a particular probe is stored in the `ProbeType` and `Probe` tables of `spyglass.common`. The user can either:

1. create these entries programmatically using DataJoint `insert` commands, for example:
    ```
    sgc.ProbeType.insert1({
    "probe_type": "128c-4s6mm6cm-15um-26um-sl",
    "probe_description": "A Livermore flexible probe with 128 channels, 4 shanks, 6 mm shank length, 6 cm ribbon length. 15 um contact diameter, 26 um center-to-center distance (pitch), single-line configuration.",
    "manufacturer": "Lawrence Livermore National Lab",
    "num_shanks": 4,
    }, skip_duplicates=True)
    ```

2. define these entries in a special YAML file called `entries.yaml` that is processed when `spyglass` is imported. One can think of `entries.yaml` as a place to define information that the database should come pre-equipped prior to ingesting any NWB files. The `entries.yaml` file should be placed in the `spyglass` base directory. An example can be found in `examples/config_yaml/entries.yaml`. It has the following structure:
    ```
    TableName:
    - TableEntry1Field1: Value
    TableEntry1Field2: Value
    - TableEntry2Field1: Value
    TableEntry2Field2: Value
    ```

    For example,
    ```
    ProbeType:
    - probe_type: 128c-4s6mm6cm-15um-26um-sl
    probe_description: A Livermore flexible probe with 128 channels, 4 shanks, 6 mm shank length, 6 cm ribbon length. 15 um contact diameter, 26 um center-to-center distance (pitch), single-line configuration.
    manufacturer: Lawrence Livermore National Lab
    num_shanks: 4
    ```

Using a YAML file over programmatically creating these entries in a notebook or script has the advantages that the YAML file maintains a record of what entries have been added that is easy to access, and the file is portable and can be shared alongside an NWB file or set of NWB files from a given experiment.

Next, the user must associate the NWB file with entries defined in the database.  This is done by creating a _configuration file_, which must:
be in the same directory as the NWB file that it configures
be in YAML format
have the following naming convention: `<name_of_nwb_file>_spyglass_config.yaml`.

Users can programmatically generate this configuration file. It is then read by spyglass when calling `insert_session` on the associated NWB file.

An example of this can be found at `examples/config_yaml/​​sub-AppleBottom_ses-AppleBottom-DY20-g3_behavior+ecephys_spyglass_config.yaml`. This file is associated with the NWB file `sub-AppleBottom_ses-AppleBottom-DY20-g3_behavior+ecephys.nwb`.

This is the general format for the config entry:
```
TableName:
- primary_key1: value1
```

For example:
```
DataAcquisitionDevice:
- data_acquisition_device_name: Neuropixels Recording Device
```

In this example, the NWB file that corresponds to this config YAML will become associated with the DataAcquisitionDevice with primary key data_acquisition_device_name: Neuropixels Recording Device. This entry must exist.
