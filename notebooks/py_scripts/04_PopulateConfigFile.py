# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Customizing Data Insertion into Spyglass
#
# If you would like to insert data into Spyglass that does not
# follow the naming or organizational format expected by Spyglass,
# or you would like to override what values are ingested into Spyglass from
# your NWB files, including missing values, follow this guide.
#
# ## General Approach
#
# When an NWB file is ingested into Spyglass, metadata about the session
# is first read from the NWB file and inserted into
# tables in the `common` module (e.g. `Institution`, `Lab`, `Electrode`, etc).
# However, not every NWB file has all the information required by Spyglass or
# the information in the NWB file is not in a format that Spyglass expects. For
# example, many NWB files do not contain information about the
# `DataAcquisitionDevice` or `Probe` because the NWB data standard does not yet
# have an official
# standard for specifying them. For these cases, we provide a way to customize
# how data is ingested into Spyglass.
#
# Let's say that you want to ingest an NWB file into Spyglass where the lab name
# in the file is written as "Loren Frank Lab" or it is not specified, but you
# know the data comes from the Loren Frank Lab. Let's say that in Spyglass,
# the lab name that is associated with sessions from the Loren Frank Lab is
# "Frank Lab" and you would like to use the same name in order to facilitate
# data search in Spyglass. To change the lab name when you insert your new data
# to Spyglass, you could either 1) edit the NWB file directly and then
# insert it into Spyglass, or 2) define an override value "Frank Lab" to be
# used instead of the value specified in the NWB file (or lack thereof).
#
# Note that if this is your own data and you want to make changes to
# information about how the data is interpreted, e.g., the units of measurement
# are incorrect, we recommend that you edit the NWB file directly because the
# file or derivatives of it might eventually be shared outside of Spyglass and
# they will not reflect any modifications that you have made to
# the data only in Spyglass.

# ## Define a Configuration YAML File
#
# To override values in the NWB file during insertion into Spyglass,
# you will need to create a configuration
# YAML file that lives in the same directory as your NWB file, named:
# `<name_of_nwb_file>_spyglass_config.yaml`
#
# An example configuration YAML file can be found at
# `examples/config_yaml/​​sub-AppleBottom_ses-AppleBottom-DY20-g3_behavior+ecephys_spyglass_config.yaml`.
# This file is associated with the NWB file
# `sub-AppleBottom_ses-AppleBottom-DY20-g3_behavior+ecephys.nwb`.
#
# This is the general format for entries in this configuration file:
#
# ```yaml
# TableName:
# - primary_key1: value1
# ```
#
# For example:
#
# ```yaml
# Lab:
# - lab_name: Frank Lab
# ```
#
# In this example, the NWB file that corresponds to this config YAML will become
# associated with the entry in the `Lab` table with the value `Frank Lab` for
# the primary key `lab_name`. This entry must already exist. More specifically,
# when the NWB file is ingested into Spyglass,
# a new `Session` entry will be created for the NWB file that has a foreign key to
# the `Lab` entry with `lab_name` = "Frank Lab", ignoring whatever lab value is
# in the NWB file, even if one does not exist.
#
# TODO implement this for `Lab`.
#

# ## Create Entries to Reference in the Configuration YAML
#
# As mentioned earlier, the table entry that you want to associate with your NWB
# file must already exist in the database. This entry would typically be a value
# that is independent of any particular NWB file, such as
# `DataAcquisitionDevice`, `Lab`, `Probe`, and `BrainRegion`.
#
# If the entry does not already exist, you can either:
# 1) create the entry programmatically using DataJoint `insert` commands, or
# 2) define the entry in a YAML file called `entries.yaml` that is automatically
#    processed when Spyglass is imported. You can think of `entries.yaml` as a
#    place to define information that the database should come pre-equipped prior
#    to ingesting your NWB files. The `entries.yaml` file should be placed in the
#    `spyglass` base directory (next to `README.md`). An example can be found in
#    `examples/config_yaml/entries.yaml`. This file should have the following
#    structure:
#
#     ```yaml
#     TableName:
#     - TableEntry1Field1: Value
#       TableEntry1Field2: Value
#     - TableEntry2Field1: Value
#       TableEntry2Field2: Value
#     ```
#
#     For example,
#
#     ```yaml
#     DataAcquisitionDeviceSystem:
#       data_acquisition_device_system: SpikeGLX
#     DataAcquisitionDevice:
#     - data_acquisition_device_name: Neuropixels_SpikeGLX
#       data_acquisition_device_system: SpikeGLX
#       data_acquisition_device_amplifier: Intan
#     ```
#
#     Only `dj.Manual`, `dj.Lookup`, and `dj.Part` tables can be populated
#     using this approach.
#
# Once the entry that you want to associate with your NWB file exists in the
# database, you can write the configuration YAML file and then ingest your
# NWB file. As an another example, let's say that you want to associate your NWB
# file with the `DataAcquisitionDevice` entry with `data_acquisition_device_name`
# = "Neuropixels_SpikeGLX" that was defined above. You would write the following
# configuration YAML file:
#
# ```yaml
# DataAcquisitionDevice:
# - data_acquisition_device_name: Neuropixels_SpikeGLX
# ```
#
# The example in
# `examples/config_yaml/​​sub-AppleBottom_ses-AppleBottom-DY20-g3_behavior+ecephys_spyglass_config.yaml`
# includes additional examples.

# ## Example Ingestion with Real Data
#
# For this example, you will need to download the 5 GB NWB file
# `sub-JDS-NFN-AM2_behavior+ecephys.nwb`
# from dandiset 000447 here:
# https://dandiarchive.org/dandiset/000447/0.230316.2133/files?location=sub-JDS-NFN-AM2&page=1
#
# Click the download arrow button to download the file to your computer. Add it to the folder
# containing your raw NWB data to be ingested into Spyglass.
#
# This file does not specify a data acquisition device. Let's say that the
# data was collected from a SpikeGadgets system with an Intan amplifier. This
# matches an existing entry in the `DataAcquisitionDevice` table with name
# "data_acq_device0". We will create a configuration YAML file to associate
# this entry with the NWB file.
#
# If you are connected to the Frank lab database, please rename any downloaded
# files (e.g., `example20200101_yourname.nwb`) to avoid naming collisions, as the
# file name acts as the primary key across key tables.

nwb_file_name = "sub-JDS-NFN-AM2_behavior+ecephys_rly.nwb"

# this configuration yaml file should be placed next to the downloaded NWB file
yaml_config_path = "sub-JDS-NFN-AM2_behavior+ecephys_rly_spyglass_config.yaml"
with open(yaml_config_path, "w") as config_file:
    lines = [
        "DataAcquisitionDevice",
        "- data_acquisition_device_name: data_acq_device0",
    ]
    config_file.writelines(line + "\n" for line in lines)

# Then call `insert_sessions` as usual.

# +
import spyglass.data_import as sgi

sgi.insert_sessions(nwb_file_name)
# -

# Confirm the session was inserted with the correct `DataAcquisitionDevice`

# +
import spyglass.common as sgc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)

sgc.Session.DataAcquisitionDevice & {"nwb_file_name": nwb_copy_file_name}
# -

#
