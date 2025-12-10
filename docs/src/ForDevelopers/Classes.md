# Spyglass Mixin Architecture

This document explains the mixin-based class architecture in Spyglass,
including the goals of the design, how mixins are organized, and how to use
them in custom pipelines.

**Related Documentation:**

- [Spyglass Mixin Features](../Features/Mixin.md) - User guide for mixin functionality
- [Custom Pipelines](./CustomPipelines.md) - Guide for creating custom schemas
- [Analysis File Tables](../Features/AnalysisTables.md) - Working with AnalysisNwbfile

---

## Table of Contents

1. [Design Goals](#design-goals)
2. [Mixin Organization](#mixin-organization)
3. [Core Composite Classes](#core-composite-classes)
4. [Usage Patterns](#usage-patterns)
5. [Class Hierarchy Reference](#class-hierarchy-reference)

---

## Design Goals

The Spyglass mixin architecture addresses several key challenges in managing
DataJoint pipelines:

### Problem: Code Duplication

Without mixins, every table would need to re-implement common operations like:

- Fetching NWB files
- Permission checks before deletion
- Logging for export tracking
- Helper functions for common queries

### Solution: Modular Mixins

Spyglass uses a **mixin pattern** where each mixin provides a focused set of
related functionality:

- **Separation of concerns**: Each mixin handles one responsibility
- **Reusability**: Tables inherit only needed functionality
- **Maintainability**: Changes to shared logic happen in one place
- **Extensibility**: New mixins can be added without modifying existing code

### Special Case: Analysis File Management

The **AnalysisMixin** provides specialized functionality for `AnalysisNwbfile`
tables. When combined with **SpyglassAnalysis**, it enables custom
user-specific analysis file tables for transaction lock isolation in multi-user
environments.

**See also:** [Custom Analysis Tables](./Management.md#custom-analysis-tables)

---

## Mixin Organization

Spyglass mixins are organized in `src/spyglass/utils/mixins/` with the following structure:

### Base Layer

**BaseMixin** (`mixins/base.py`)

- Foundation for most mixins
- Provides: `_logger`, `_test_mode`, `_spyglass_version`, `_graph_deps`
- Used by: All mixins except `FetchMixin`

### Independent Mixins

These mixins inherit from `BaseMixin` and provide focused functionality:

**CautiousDeleteMixin** (`mixins/cautious_delete.py`)

- Permission checks before deletion
- Validates user is on same team as session experimenter
- See [Mixin Features: Delete Permission](../Features/Mixin.md#delete-permission-checks)

**PopulateMixin** (`mixins/populate.py`)

- Enhanced `populate()` with non-daemon process pools
- Disable transaction protection for long-running populates
- See [Mixin Features: Populate](../Features/Mixin.md#populate-calls)

**RestrictByMixin** (`mixins/restrict_by.py`)

- Long-distance restrictions with `<<` and `>>` operators
- Navigate complex foreign key relationships
- See [Mixin Features: Long-Distance Restrictions](../Features/Mixin.md#long-distance-restrictions)

**HelperMixin** (`mixins/helpers.py`)

- Miscellaneous helper functions
- `file_like()` - restrict by filename substring
- `find_insert_fail()` - debug IntegrityErrors

**AnalysisMixin** (`mixins/analysis.py`)

- Analysis file creation and management
- Methods: `create()`, `add()`, `add_nwb_object()`, `add_units()`, `cleanup()`
- Used exclusively by `AnalysisNwbfile` tables
- See [Analysis File Tables](../Features/AnalysisTables.md)

### Fetch/Export Chain

These mixins form a dependency chain for NWB file fetching and export logging:

**FetchMixin** (`mixins/fetch.py`)

- Provides: `fetch_nwb()`, `fetch_pynapple()`, `_nwb_table_tuple`
- **Does not inherit BaseMixin** (standalone)
- Detects which NWB table to fetch from (Nwbfile or AnalysisNwbfile)

**ExportMixin** (`mixins/export.py`)

- Inherits from `FetchMixin`
- Adds: `_log_fetch()`, `_log_fetch_nwb()`
- Logs file access during exports
- Handles copy-to-common for custom AnalysisNwbfile tables
- See [Export Guide](../Features/Export.md)

***IngestionMixin* (`mixins/ingestion.py`)

- Defines a protocol for populating table entries from the raw nwb file
- Provides `insert_from_nwbfile()` which identifies relevant objects within the
  nwb file and creates table entries
- See [Ingestion Guide](../Features/Ingestion.md)

---

## Core Composite Classes

### SpyglassMixin

**Location**: `src/spyglass/utils/dj_mixin.py`

**Purpose**: The primary mixin for all Spyglass tables

**Inherits from** (in MRO order):

1. `CautiousDeleteMixin`
2. `ExportMixin` → `FetchMixin`
3. `HelperMixin`
4. `PopulateMixin`
5. `RestrictByMixin`

**Additional functionality**:

- Schema prefix validation (ensures table is in authorized schema)
- Merge table detection
- Foreign key validation (prevents multiple AnalysisNwbfile references)

**Usage**:

```python
import datajoint as dj
from spyglass.utils import SpyglassMixin

schema = dj.schema("myteam_pipeline")

@schema
class MyTable(SpyglassMixin, dj.Manual):
    definition = """
    analysis_id: int
    ---
    result: float
    """
```

**NOTE**: SpyglassMixin must be the first class inherited to ensure method
overrides work correctly.

### SpyglassMixinPart

**Location**: `src/spyglass/utils/dj_mixin.py`

**Purpose**: Specialized mixin for part tables

**Inherits from**:

- `SpyglassMixin`
- `dj.Part`

**Additional functionality**:

- Propagates delete calls from part to master table
- Handles restriction on master when deleting parts

**Usage**:

```python
@schema
class MyMaster(SpyglassMixin, dj.Manual):
    definition = """
    master_id: int
    """

    class MyPart(SpyglassMixinPart):
        definition = """
        -> master
        part_id: int
        ---
        value: float
        """
```

### SpyglassAnalysis

**Location**: `src/spyglass/utils/dj_mixin.py`

**Purpose**: Specialized mixin for custom `AnalysisNwbfile` tables

**Inherits from**:

- `SpyglassMixin` (all 5 base mixins)
- `AnalysisMixin` (analysis file operations)

**Enforces**:

- Schema naming: `{prefix}_nwbfile` (one underscore)
- Table naming: `AnalysisNwbfile` (exact match)
- Table definition: Must match common `AnalysisNwbfile`
- Auto-registration in `AnalysisRegistry`

**Usage**:

Users should import the pre-configured table rather than creating their own:

```python
# Import your user-specific table (uses database.user automatically)
from spyglass.common.custom_nwbfile import AnalysisNwbfile
```

**Why this exists**: Custom `AnalysisNwbfile` tables provide transaction lock
isolation in multi-user environments. Each user's custom table is in a separate
schema (`{username}_nwbfile`), preventing concurrent insert operations from
blocking each other.

**See also:** [Custom Analysis Tables](./Management.md#custom-analysis-tables)

### SpyglassIngestion

**Location**: `src/spyglass/utils/dj_mixin.py`

**Purpose**: Specialized mixin for generating table entries from raw nwb files

**Inherits from**:

- `SpyglassMixin` (all 5 base mixins)
- `IngestionMixin` (ingestion operations)

**Additional Functionality**:

- Defines `insert_from_nwbfile()` which identifies source data in the nwb file and
  translates into table entries. Depends on defining the following properties for
  each class:
    - `_source_nwb_object_type`: The `pynwb` type of object(s) in the file containing
        data for the given table.
    - `_source_nwb_object_name`: OPtional property which further limits ingestion
        to nwb_objects with this name attribute
    - `table_key_to_obj_attr`: A dictionary which defines a mapping from spyglass
        table column to the name of the nwb object attribute to be stored.
        Optionally, a callable function which generates the value to be stored from
        the nwb object can be used instead of the attribute name.

**Usage**:

```python
@schema
class MyIngestionTable(SpyglassIngestion, dj.Manual):
    definition = """
        -> Session
        ----
        lfp_obj_id: varchar(32)
    """
    @property
    def _source_nwb_object_type(self):
        return pynwb.ecephys.LFP

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"lfp_obj_id": "object_id"}}
```

---

## Usage Patterns

### Standard Spyglass Tables

Most Spyglass tables inherit `SpyglassMixin` + a DataJoint table type:

```python
from spyglass.utils import SpyglassMixin
import datajoint as dj

schema = dj.schema("myteam_analysis")

@schema
class MyAnalysis(SpyglassMixin, dj.Computed):
    definition = """
    -> UpstreamTable
    analysis_id: int
    ---
    result: float
    """

    def make(self, key):
        # Compute analysis
        result = process_data(key)
        self.insert1({**key, "result": result})
```

**What you get**:

- ✅ NWB file fetching (`fetch_nwb()`)
- ✅ Permission-checked deletion (`delete()`)
- ✅ Long-distance restrictions (`<<`, `>>`)
- ✅ Export logging
- ✅ Helper functions

### Custom Analysis File Tables

Use the pre-configured custom `AnalysisNwbfile` table for lock isolation:

```python
# Import from custom_nwbfile module
from spyglass.common.custom_nwbfile import AnalysisNwbfile

# Table automatically created in {username}_nwbfile schema
# Inherits SpyglassAnalysis (SpyglassMixin + AnalysisMixin)
```

**What you get**:

- ✅ All SpyglassMixin functionality
- ✅ Analysis file operations (`create()`, `add()`, `cleanup()`)
- ✅ Automatic registry tracking
- ✅ Export integration (copy-to-common)
- ✅ Lock isolation (no contention with other users)

### Referencing Custom Analysis Tables

Downstream tables can reference custom `AnalysisNwbfile` tables:

```python
from spyglass.utils import SpyglassMixin
from spyglass.common.custom_nwbfile import AnalysisNwbfile
import datajoint as dj

schema = dj.schema("myteam_results")

@schema
class MyResults(SpyglassMixin, dj.Manual):
    definition = """
    result_id: int
    ---
    -> AnalysisNwbfile  # References custom table
    metric: float
    """
```

**NOTE**: Tables should only reference ONE `AnalysisNwbfile` table (either
common or custom, not both). Spyglass validates this on table declaration.

---

## Class Hierarchy Reference

### Visual Overview

```
BaseMixin (foundation)
    ├─→ CautiousDeleteMixin
    ├─→ PopulateMixin
    ├─→ RestrictByMixin
    ├─→ HelperMixin
    └─→ AnalysisMixin

FetchMixin (standalone)
    └─→ ExportMixin

SpyglassMixin (composite)
    Combines: CautiousDelete + Export + Helper + Populate + RestrictBy

    ├─→ SpyglassMixinPart (+ dj.Part)
    └─→ SpyglassAnalysis (+ AnalysisMixin)
```

### Method Resolution Order (MRO)

Understanding Python's MRO helps debug which mixin's method is actually called:

**SpyglassMixin MRO**:

```
SpyglassMixin
→ CautiousDeleteMixin
→ ExportMixin
→ FetchMixin
→ HelperMixin
→ PopulateMixin
→ RestrictByMixin
→ BaseMixin
→ object
```

**SpyglassAnalysis MRO**:

```
SpyglassAnalysis
→ SpyglassMixin (all 6 mixins above)
→ AnalysisMixin
→ BaseMixin (appears twice in hierarchy)
→ object
```

**Key insight**: `BaseMixin` appears in most paths, but only initialized once due to Python's C3 linearization.

### File Locations

| Class | File Path |
|-------|-----------|
| BaseMixin | `src/spyglass/utils/mixins/base.py` |
| CautiousDeleteMixin | `src/spyglass/utils/mixins/cautious_delete.py` |
| PopulateMixin | `src/spyglass/utils/mixins/populate.py` |
| RestrictByMixin | `src/spyglass/utils/mixins/restrict_by.py` |
| HelperMixin | `src/spyglass/utils/mixins/helpers.py` |
| AnalysisMixin | `src/spyglass/utils/mixins/analysis.py` |
| FetchMixin | `src/spyglass/utils/mixins/fetch.py` |
| ExportMixin | `src/spyglass/utils/mixins/export.py` |
| SpyglassMixin | `src/spyglass/utils/dj_mixin.py` |
| SpyglassMixinPart | `src/spyglass/utils/dj_mixin.py` |
| SpyglassAnalysis | `src/spyglass/utils/dj_mixin.py` |

---

## Related Documentation

- **User Guides**:
    - [Spyglass Mixin Features](../Features/Mixin.md) - How to use mixin functionality
    - [Analysis File Tables](../Features/AnalysisTables.md) - Working with AnalysisNwbfile
    - [Export Guide](../Features/Export.md) - Exporting data for publication

- **Developer Guides**:
    - [Custom Pipelines](./CustomPipelines.md) - Creating custom schemas
    - [Database Management](./Management.md) - Admin tasks including custom tables
    - [Schema Design](./Schema.md) - Understanding schema structure

---

For questions about mixin architecture or extending Spyglass functionality,
please open a discussion on
[GitHub](https://github.com/LorenFrankLab/spyglass/discussions).
