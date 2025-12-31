# Analysis File Tables

Spyglass uses NWB files to store both raw experimental data and analysis
results. This guide explains how to create and manage analysis files.

______________________________________________________________________

## What is AnalysisNwbfile?

`AnalysisNwbfile` is a DataJoint table that tracks analysis files containing
your results (intermediate computations, final outputs, spike sorting, etc.).

**Key Features**:

- Creates derivative NWB files from your experimental sessions
- Tracks analysis files in the database with checksums
- Prevents accidental file modifications
- Supports custom per-user tables for better performance

**Lifecycle**: Analysis files follow a three-step process:

```
CREATE → POPULATE → REGISTER
```

Once registered, files are **checksummed and immutable** - any modification will
break the checksum and cause errors.

______________________________________________________________________

## Table of Contents

- [How to Use (Recommended)](#how-to-use-recommended)
- [Understanding Object IDs](#understanding-object-ids)
- [Using Custom Tables](#using-custom-tables)
- [Legacy Pattern Comparison](#legacy-pattern-comparison)
- [Troubleshooting](#troubleshooting)

______________________________________________________________________

## How to Use (Recommended)

Use the `.build()` method which provides a context manager that handles the
CREATE → POPULATE → REGISTER lifecycle automatically.

### Basic Usage

```python
from spyglass.common import AnalysisNwbfile
import datajoint as dj
import pandas as pd

schema = dj.schema("my_schema")


@schema
class MyAnalysis(dj.Computed):
    definition = """
    -> SomeOtherTable
    ---
    -> AnalysisNwbfile
    results_object_id: varchar(40)  # Object ID for retrieving NWB object
    """

    def make(self, key):

        my_data = ...  # Your analysis data here

        nwb_file_name = key["nwb_file_name"]
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            # Add your data using helper methods
            # add_nwb_object returns the object_id
            object_id = builder.add_nwb_object(pd.DataFrame(my_data), "results")

            # File automatically registered on exit!
            analysis_file_name = builder.analysis_file_name

        self.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "results_object_id": object_id,
            }
        )
```

### Common Operations

**Adding multiple objects**:

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    position_object_id = builder.add_nwb_object(position_data, "position")
    velocity_object_id = builder.add_nwb_object(velocity_data, "velocity")
    metadata_object_id = builder.add_nwb_object(metadata, "analysis_params")

# Store these object_ids in your table to retrieve the objects later
```

**Adding spike sorting units**:

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_units(
        units={1: [0.1, 0.5, 1.2], 2: [0.2, 0.6]},
        units_valid_times={1: [[0, 10]], 2: [[0, 10]]},
        units_sort_interval={1: [[0, 5]], 2: [[0, 5]]},
        metrics={"snr": {1: 5.2, 2: 3.8}},
    )
```

**Direct NWB I/O** (for complex operations):

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    with builder.open_for_write() as io:
        nwbf = io.read()
        nwbf.add_unit(spike_times=[0.1, 0.5, 1.2], id=1)
        io.write(nwbf)
```

**What happens on exception**:

```python
try:
    with AnalysisNwbfile().build("session.nwb") as builder:
        builder.add_nwb_object(my_data, "results")
        raise ValueError("Something went wrong!")
except ValueError:
    # File created but NOT registered - logged for cleanup
    pass
```

______________________________________________________________________

## Understanding Object IDs

### What are Object IDs?

Object IDs are unique identifiers assigned by PyNWB to every object stored in an
NWB file. When you add data to an analysis file, PyNWB automatically assigns
each object a unique ID. Spyglass uses these IDs to efficiently retrieve
specific objects from NWB files.

### Why Use Object IDs?

**Without object IDs**, `fetch_nwb()` returns a basic dict:

```python
# Table definition without object_id
definition = """
-> SomeOtherTable
---
-> AnalysisNwbfile
"""

# fetch_nwb returns only the metadata
result = (MyAnalysis & key).fetch_nwb()[0]
# result = {
#     'analysis_file_name': 'session_ABC123.nwb',
#     'nwb_file_name': 'session.nwb',
#     ...
# }
```

**With object IDs**, `fetch_nwb()` automatically retrieves the NWB objects:

```python
# Table definition WITH object_id
definition = """
-> SomeOtherTable
---
-> AnalysisNwbfile
position_object_id: varchar(40)
velocity_object_id: varchar(40)
"""

# fetch_nwb automatically loads the NWB objects
result = (MyAnalysis & key).fetch_nwb()[0]
# result = {
#     'analysis_file_name': 'session_ABC123.nwb',
#     'position': <SpatialSeries object>,  # Automatically loaded!
#     'velocity': <SpatialSeries object>,  # Automatically loaded!
#     ...
# }
```

### How It Works

1. **When populating**, `add_nwb_object()` returns the object_id:

```python
def make(self, key):
    with AnalysisNwbfile().build(nwb_file_name) as builder:
        # add_nwb_object returns the unique object ID
        position_id = builder.add_nwb_object(position_data, "position")
        velocity_id = builder.add_nwb_object(velocity_data, "velocity")

    self.insert1(
        {
            **key,
            "analysis_file_name": builder.analysis_file_name,
            "position_object_id": position_id,
            "velocity_object_id": velocity_id,
        }
    )
```

1. **When fetching**, `fetch_nwb()` detects `*_object_id` fields:

```python
# fetch_nwb automatically:
# 1. Opens the NWB file
# 2. Retrieves objects using the stored object_ids
# 3. Strips "_object_id" suffix from field names
# 4. Returns objects with clean names

result = (MyTable & key).fetch_nwb()[0]
result["position"]  # The actual NWB object (not the ID)
result["velocity"]  # The actual NWB object (not the ID)
```

### Naming Convention

**Important**: Object ID fields must end with `_object_id`:

```python
# ✅ Correct - will auto-load as 'results'
results_object_id: varchar(40)

# ✅ Correct - will auto-load as 'lfp'
lfp_object_id: varchar(40)

# ❌ Wrong - won't be recognized
results_id: varchar(40)
object_id: varchar(40)  # Too generic
```

When fetched, the suffix is stripped:

- `position_object_id` → returned as `position`
- `lfp_object_id` → returned as `lfp`
- `spike_times_object_id` → returned as `spike_times`

### Real-World Examples

**Position tracking** (from `position_trodes_position.py:187-189`):

```python
definition = """
-> TrodesPosSelection
---
-> AnalysisNwbfile
position_object_id : varchar(80)
orientation_object_id : varchar(80)
velocity_object_id : varchar(80)
"""

# Usage:
data = (TrodesPosV1 & key).fetch_nwb()[0]
position = data["position"]  # SpatialSeries object
orientation = data["orientation"]  # SpatialSeries object
velocity = data["velocity"]  # SpatialSeries object
```

**LFP data** (from `lfp/v1/lfp.py:53`):

```python
definition = """
-> LFPSelection
---
-> AnalysisNwbfile
-> IntervalList
lfp_object_id: varchar(40)
lfp_sampling_rate: float
"""

# Usage:
data = (LFPV1 & key).fetch_nwb()[0]
lfp = data["lfp"]  # ElectricalSeries object with LFP data
sampling_rate = data["lfp_sampling_rate"]
```

### When to Use Object IDs

**Use object IDs when:**

- ✅ Storing data objects in NWB files (position, LFP, spike times, etc.)
- ✅ You need to retrieve the actual data later
- ✅ Working with PyNWB objects (SpatialSeries, TimeSeries, Units, etc.)

**Don't need object IDs when:**

- ❌ Only storing metadata (parameters, file names, etc.)
- ❌ Data is stored as blobs in DataJoint (not in NWB)
- ❌ Table only tracks analysis status or configuration

### Best Practices

1. **Always store the object_id returned by `add_nwb_object()`**:

    ```python
    object_id = builder.add_nwb_object(my_data, "results")
    # Store this ID in your table!
    ```

2. **Use descriptive prefixes**:

    ```python
    # Good - clear what each object is
    raw_lfp_object_id: varchar(40)
    filtered_lfp_object_id: varchar(40)

    # Less clear
    object_id_1: varchar(40)
    object_id_2: varchar(40)
    ```

3. **varchar(40) is standard size** for object IDs

______________________________________________________________________

## Using Custom Tables

By default, all users share the common `AnalysisNwbfile` table. When multiple
users work concurrently, this can cause database lock contention and prevent new
table declarations. To avoid this, Spyglass supports custom per-user analysis
tables for custom analysis pipelines.

### How to Use

Import from `custom_nwbfile` instead of `common_nwbfile`:

```python
import datajoint as dj

# Standard (shared table)
from spyglass.common import AnalysisNwbfile

# ---------------- OR ----------------
# Custom (your own table - better performance)
from spyglass.common.custom_nwbfile import AnalysisNwbfile

schema = dj.schema("my_schema")


# Usage is identical
@schema
class MyAnalysis(dj.Computed):
    definition = """
    -> SomeOtherTable
    ---
    -> AnalysisNwbfile
    my_object_id: varchar(40)
    """
```

**What happens**: Creates a user-specific schema `{username}_nwbfile`
automatically, providing lock isolation from other users.

**Team sharing**: Set `dj.config["custom"]["database.prefix"] = "teamname"` to
share across a team (may still have some lock contention).

______________________________________________________________________

## Legacy Pattern Comparison

**Old way** (manual lifecycle):

```python
def make(self, key):
    # CREATE
    file = AnalysisNwbfile().create("session.nwb")

    # POPULATE
    AnalysisNwbfile().add_nwb_object(file, data, "results")

    # REGISTER (easy to forget!)
    AnalysisNwbfile().add("session.nwb", file)
```

**New way** (automatic):

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_nwb_object(data, "results")
    # Auto-registered on exit
```

______________________________________________________________________

## Troubleshooting

### Error: "Cannot call add_nwb_object() in state: REGISTERED"

**Cause**: You tried to use a helper method after the file was registered.

**Solution**: Use `build()` which prevents this error:

```python
# ❌ Old way - easy to make this mistake
file = AnalysisNwbfile().create("session.nwb")
AnalysisNwbfile().add("session.nwb", file)  # Registered!
AnalysisNwbfile().add_nwb_object(file, data)  # ❌ ERROR!

# ✅ New way - impossible to make this mistake
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_nwb_object(data, "results")
    # Auto-registered on exit - can't call methods after
```

### Error: "File downloaded but did not pass checksum"

**Cause**: The file was modified after registration.

**Solutions**:

1. Delete and recreate the file
2. Discuss why the file was modified with admin to modify the checksum

### Error: "Cannot call add_nwb_object() before entering context manager"

**Cause**: You tried to use builder methods outside the `with` block.

**Solution**:

```python
# ❌ Wrong
builder = AnalysisNwbfile().build("session.nwb")
builder.add_nwb_object(data, "results")  # ❌ ERROR!

# ✅ Correct
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_nwb_object(data, "results")
```

### When should I use the builder vs. direct NWB I/O?

**Use the builder** (recommended for 90% of cases):

- Adding DataFrames or arrays
- Adding spike sorting units
- Standard analysis workflows

**Use direct I/O** (advanced, legacy code):

- Custom NWB processing modules
- Complex file modifications
- Maintaining backward compatibility

Even with direct I/O, you can still use the builder:

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    with builder.open_for_write() as io:
        # Direct PyNWB operations here
        pass
```

______________________________________________________________________

## Related Documentation

- [Database Management](../ForDevelopers/Management.md) - Cleanup, maintenance,
    and custom analysis tables
- [DataJoint External Storage](https://docs.datajoint.org/python/admin/5-blob-config.html)
- [PyNWB File I/O](https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html)
