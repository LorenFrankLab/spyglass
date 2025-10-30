# Analysis File Tables

Spyglass uses NWB files to store both raw experimental data and analysis
results. This guide explains how to create and manage analysis files.

---

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

Once registered, files are **checksummed and immutable** - any modification
will break the checksum and cause errors.

---

## Table of Contents

- [How to Use (Recommended)](#how-to-use-recommended)
- [Using Custom Tables](#using-custom-tables)
- [Legacy Pattern Comparison](#legacy-pattern-comparison)
- [Troubleshooting](#troubleshooting)

---

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
    """

    def make(self, key):

        my_data = ...  # Your analysis data here

        nwb_file_name = key["nwb_file_name"]
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            # Add your data using helper methods
            builder.add_nwb_object( pd.DataFrame(my_data))

            # File automatically registered on exit!
            analysis_file_name = builder.analysis_file_name

        self.insert1({**key, "analysis_file_name": analysis_file_name})
```

### Common Operations

**Adding multiple objects**:

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_nwb_object(position_data, "position")
    builder.add_nwb_object(velocity_data, "velocity")
    builder.add_nwb_object(metadata, "analysis_params")
```

**Adding spike sorting units**:

```python
with AnalysisNwbfile().build("session.nwb") as builder:
    builder.add_units(
        units={1: [0.1, 0.5, 1.2], 2: [0.2, 0.6]},
        units_valid_times={1: [[0, 10]], 2: [[0, 10]]},
        units_sort_interval={1: [[0, 5]], 2: [[0, 5]]},
        metrics={"snr": {1: 5.2, 2: 3.8}}
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

---

## Using Custom Tables

By default, all users share the common `AnalysisNwbfile` table. When multiple
users work concurrently, this can cause database lock contention and prevent
new table declarations. To avoid this, Spyglass supports custom per-user
analysis tables for custom analysis pipelines.

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
    """
```

**What happens**: Creates a user-specific schema `{username}_nwbfile` automatically,
providing lock isolation from other users.

**Team sharing**: Set `dj.config["custom"]["database.prefix"] = "teamname"` to
share across a team (may still have some lock contention).

---

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

---

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

---

## Related Documentation

- [Database Management](../ForDevelopers/Management.md) - Cleanup, maintenance, and custom analysis tables
- [DataJoint External Storage](https://docs.datajoint.org/python/admin/5-blob-config.html)
- [PyNWB File I/O](https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html)
