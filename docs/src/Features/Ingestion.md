# Ingestion Process

## Step 0: NWB files

Before beginning with spyglass, data must be compiled into the standardized NWB
file format. NWB files contain everything about the experiment and form the starting
point of all analyses. Numerous [online tutorials](https://nwb.org/converting-data-to-nwb/)
exist to help get you started in this process, as well as existing packages for
lab-specific conversions ([1](https://github.com/catalystneuro),
[2](https://github.com/LorenFrankLab/trodes_to_nwb)) that can be used as a reference.

The following sections describe how data is extracted from these files and brought
into the spyglass system. For best compatibility, please use these as reference when
creating your NWB files.

## What is ingestion?

Ingestion is the process of extracting data from the raw NWB file and storing it
in Spyglass tables.

## How does it work?

### For users

For most users all you'll need to do is call `spyglass.common.insert_sessions(nwb_file_name)`
which will iterate through tables populated from the raw NWB file and create
appropriate entries.

### In the background

*Note: Migration to this format is in progress, and not yet implemented for all
ingestion tables

Tables that are populated from the raw NWB file are instances of the `SpyglassIngestion`
class. Tables of this class must define the following properties which enable finding
relevant data in the NWB file and creating corresponding table entries.

- `_source_nwb_object_type`: defines the `pynwb` object type containing data for
  this table (eg. `pynwb.misc.Units` for `ImportedSpikesorting`).
- `table_key_to_obj_attr`: A dict of dicts mapping table keys to NWB object attributes
  or callable methods that generate the value to store from the nwb object.

With these defined the table entries are populated from the following methods:

- `insert_from_nwbfile`: top-level function that extracts and inserts all entries
  for the table
- `get_nwb_objects`: returns all nwb objects from the raw file containing data for
  the table. By default, returns all instances of `_source_nwb_object_type`, but
  can be overwritten on a per-table basis for more selective restriction
- `generate_entries_from_nwb_object`: Called for each identified nwb object. Generates
  table entries using the `table_key_to_obj_attr` mapping.

## NWB to Spyglass table mappings

*In progress*: To aid in creating spyglass-compatable NWB files, we provide a
[Reference Table](../ForDevelopers/ingestion_mapping.md) which maps spyglass table
entries to the source nwb objects and attributes.

For entries not yet contained in the updated format, a complete list of this mappings
can also be found [here](../ForDevelopers/UsingNWB.md).
