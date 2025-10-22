# Analysis File Tables

Spyglass uses NWB files to store both raw experimental data and analysis
results. This guide explains how to create and manage analysis files, and how
to use custom user-specific tables for lock isolation.

## Overview

Spyglass provides two primary tables for NWB file management:

1. **`Nwbfile`**: Tracks raw, ingested experimental data files
2. **`AnalysisNwbfile`**: Tracks analysis files containing intermediate and final results

This document focuses on `AnalysisNwbfile` and its usage patterns.

---

## Table of Contents

- [Analysis File Lifecycle](#analysis-file-lifecycle)
- [Safe Usage Patterns](#safe-usage-patterns)
- [Method Reference](#method-reference)
- [DataJoint Checksum System](#datajoint-checksum-system)
- [Common Mistakes](#common-mistakes)
- [Custom Analysis Tables](#custom-analysis-tables)

---

## Analysis File Lifecycle

Analysis files follow a strict three-step lifecycle to ensure data integrity:

```
1. CREATE   → File created on disk (not yet tracked in database)
2. POPULATE → Data written to file using PyNWB
3. REGISTER → File registered in AnalysisNwbfile (CHECKSUM LOCKED)
```

**CRITICAL**: Once a file is registered (step 3), its contents are
**checksummed by DataJoint**. Any modification after registration will **break
the checksum** and cause errors.

---

## Safe Usage Patterns

### Pattern 1: Basic Analysis File (Most Common)

Use this pattern when your analysis is based on a specific experimental session:

```python
from spyglass.common import AnalysisNwbfile
from pynwb import NWBHDF5IO

# Step 1: CREATE - File created on disk with parent metadata
analysis_file_name = AnalysisNwbfile().create(
    nwb_file_name="parent_session.nwb"
)

# Step 2: POPULATE - Write your analysis data
analysis_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)

with NWBHDF5IO(analysis_file_abs_path, mode='w') as io:
    nwbf = io.read()  # Read structure created in step 1

    # Add your analysis results
    nwbf.processing["behavior"].add(my_data_object)
    nwbf.add_scratch(my_metadata, name="analysis_params")

    io.write(nwbf)

# Step 3: REGISTER - Add to database (CHECKSUM LOCKED!)
AnalysisNwbfile().add(
    nwb_file_name="parent_session.nwb",
    analysis_file_name=analysis_file_name
)
```

**Important**:

- Parent `nwb_file_name` must exist in the `Nwbfile` table
- Use `mode='w'` for writing (safest approach)
- Never modify the file after calling `add()`

**NOTE**: The `AnalysisNwbfile` table definition includes `-> Nwbfile` as a foreign key. When using the pre-configured custom table from `spyglass.common.custom_nwbfile`, this import is handled automatically. If creating custom schemas that reference `AnalysisNwbfile`, ensure `Nwbfile` is imported in your module for foreign key resolution.

### Pattern 2: Using Helper Methods

Spyglass provides helper methods for common operations:

```python
from spyglass.common import AnalysisNwbfile
import pandas as pd

# CREATE
analysis_file_name = AnalysisNwbfile().create("parent_session.nwb")

# POPULATE - Helper handles open/write/close
object_id = AnalysisNwbfile().add_nwb_object(
    analysis_file_name,
    pd.DataFrame({"time": [1, 2, 3], "value": [0.1, 0.2, 0.3]}),
    table_name="my_results"
)

# REGISTER
AnalysisNwbfile().add("parent_session.nwb", analysis_file_name)
```

**Warning**: `add_nwb_object()` uses `mode='a'` internally. Only use it **BEFORE** calling `add()`.

### Pattern 3: Adding Spike Sorting Units

Specialized pattern for spike sorting pipelines:

```python
from spyglass.common import AnalysisNwbfile

# CREATE
analysis_file_name = AnalysisNwbfile().create("session.nwb")

# POPULATE - add_units() populates the NWB units table
units_object_id, waveforms_object_id = AnalysisNwbfile().add_units(
    analysis_file_name,
    units={1: [0.1, 0.5, 1.2], 2: [0.2, 0.6]},  # spike times per unit
    units_valid_times={1: [[0, 10]], 2: [[0, 10]]},
    units_sort_interval={1: [[0, 5]], 2: [[0, 5]]},
    metrics={"snr": {1: 5.2, 2: 3.8}},
)

# REGISTER
AnalysisNwbfile().add("session.nwb", analysis_file_name)
```

**Warning**: Like `add_nwb_object()`, `add_units()` uses `mode='a'` and is only safe **BEFORE** registration.

---

## Method Reference

### Core Methods

#### `create(nwb_file_name) -> str`

Creates an analysis NWB file derived from a parent session file.

**What it does**:

1. Reads parent NWB file
2. Creates a copy with only essential fields
3. Writes copy to disk in analysis directory
4. Returns the new filename

**Important**: Does NOT register the file. You must call `add()` after writing data.

**Returns**: `analysis_file_name` (str)

---

#### `add(nwb_file_name, analysis_file_name) -> None`

Registers a completed analysis file in the database.

**What it does**:

1. Stores the file in DataJoint's external storage system
2. Computes and stores checksums (filepath and contents)
3. Links the analysis file to its parent session

**CRITICAL**: After calling this, **NEVER edit the file again**. The checksum is locked.

---

#### `get_abs_path(analysis_file_name) -> str`

Returns the absolute filesystem path for an analysis file.

```python
path = AnalysisNwbfile.get_abs_path("session_20231015_ABC123.nwb")
# Returns: "/path/to/analysis/session_20231015_ABC123.nwb"
```

---

### Helper Methods

#### `add_nwb_object(analysis_file_name, nwb_object, table_name="pandas_table") -> str`

Adds an object to the scratch space of an analysis file.

**CRITICAL**: Only use **BEFORE** calling `add()`. Uses `mode='a'` internally.

**Parameters**:

- `analysis_file_name`: Target analysis file
- `nwb_object`: PyNWB object, DataFrame, or ndarray
- `table_name`: Name for the object in scratch

**Returns**: `nwb_object_id` (str)

---

#### `add_units(analysis_file_name, units, units_valid_times, units_sort_interval, metrics=None) -> Tuple[str, str]`

Adds spike sorting units to an analysis file.

**CRITICAL**: Only use **BEFORE** calling `add()`. Uses `mode='a'` internally.

**Returns**: `(units_object_id, waveforms_object_id)`

---

#### `copy(nwb_file_name) -> str`

Creates a copy of an existing analysis file.

**Important**: Like `create()`, does NOT register the new file. Call `add()` after modifications.

---

## DataJoint Checksum System

Understanding checksums is essential for safely working with analysis files.

### How Checksums Work

When a file is registered in `AnalysisNwbfile`, DataJoint:

1. Computes a **hash of the relative filepath** (`hash` field)
2. Computes a **hash of the file contents** (`contents_hash` field)
3. Stores both in the `~external_analysis` tracking table

**Every time you fetch a file**, DataJoint verifies the checksum. If the file has been modified, the checksum will fail and raise an error.

### Why This Matters

```python
# File is registered with checksum
AnalysisNwbfile().add("session.nwb", analysis_file_name)

# ❌ This will break the checksum!
with NWBHDF5IO(path, mode='a') as io:
    nwbf = io.read()
    nwbf.add_scratch(more_data)  # Modifies registered file
    io.write(nwbf)

# Next fetch will raise:
# DataJointError: 'file.nwb' downloaded but did not pass checksum
```

**Golden Rule**: Treat registered analysis files as **immutable (read-only)**.

---

## Custom Analysis Tables

By default, all users share the master `AnalysisNwbfile` table in the
`common_nwbfile` schema. When multiple users create analysis files
concurrently, this can cause **transaction lock contention**, leading to slow
performance or timeouts.

### When to Use Custom Tables

Consider using a custom `AnalysisNwbfile` table if you experience:

- **Lock timeouts** during file creation
- **Slow file creation** when multiple users are working concurrently
- **High-frequency analysis** where you create many files

**Note**: Custom tables are optional. If you're working solo or don't
experience performance issues, the master table works fine.

### Using a Custom Table

Spyglass provides a pre-configured custom table that automatically uses your
database username for lock isolation:

```python
# Import your user-specific table (uses database.user automatically)
from spyglass.common.custom_nwbfile import AnalysisNwbfile

schema = dj.schema("yourusername_suffix")

@schema
class MyAnalysis(dj.Computed):
    definition = """
    -> SomeParentTable
    ---
    -> AnalysisNwbfile
    """
```

**What happens**:

- Creates schema `{username}_nwbfile` automatically
- Each user has their own analysis table
- Provides maximum lock isolation

**Advanced**: Override with a team name for shared files:

```python
import datajoint as dj

# Optional: Use team name instead of username in your datajoint config
dj.config["custom"]["database.prefix"] = "franklab"

from spyglass.common.custom_nwbfile import AnalysisNwbfile
# Creates schema: "franklab_nwbfile"
```

**Note**: Team-shared tables may still experience lock contention with multiple
concurrent users.

---

## Key Principles Summary

### ✅ DO

1. **Always follow CREATE → POPULATE → REGISTER order**
2. **Use `mode='w'` when writing with NWBHDF5IO** (safest)
3. **Call `add()` AFTER all data is written**
4. **Use `add_nwb_object()` and `add_units()` BEFORE registration**
5. **Treat registered files as immutable (read-only)**

### ❌ DON'T

1. **Never use `mode='a'` on registered files**
2. **Never call helper methods after `add()`**
3. **Never edit files outside the CREATE → POPULATE → REGISTER flow**
4. **Never assume you can modify a file in the AnalysisNwbfile table**

---

## Related Documentation

- [Database Management](../ForDevelopers/Management.md) - Cleanup, maintenance, and custom analysis tables
- [DataJoint External Storage](https://docs.datajoint.org/python/admin/5-blob-config.html)
- [PyNWB File I/O](https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html)
