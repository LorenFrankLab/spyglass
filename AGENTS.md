# AGENTS.md - Spyglass Standards for LLM Agents

This document provides essential information for Large Language Models (LLMs)
and AI agents working with the Spyglass neuroscience data analysis framework.
It outlines key standards, patterns, and conventions to ensure consistent and
correct assistance when helping developers create new Spyglass tables,
pipelines, and analyses.

## Overview

Spyglass is a neuroscience data analysis framework that:
- Uses **DataJoint** for reproducible data pipeline management
- Stores data in **NWB (Neurodata Without Borders)** format
- Provides standardized pipelines for common neuroscience analyses
- Ensures reproducible research through automated dependency tracking
- Supports collaborative data sharing and analysis

## Core Architecture

### DataJoint Foundation
- All tables inherit from DataJoint table types: `Manual`, `Lookup`,
  `Computed`, `Imported`
- Tables are organized into **schemas** (MySQL databases) by topic
- Primary keys establish relationships and dependencies between tables
- The `@schema` decorator associates tables with their schema

### Spyglass Mixin
- **All Spyglass tables MUST inherit from `SpyglassMixin`**
- Import: `from spyglass.utils import SpyglassMixin`
- Provides Spyglass-specific functionality like NWB file handling and
  permission checks
- Standard inheritance pattern: `class MyTable(SpyglassMixin, dj.TableType)`

### Schema Organization
```python
import datajoint as dj
from spyglass.utils import SpyglassMixin

schema = dj.schema("module_name")  # e.g., "common_ephys", "spikesorting"

@schema
class TableName(SpyglassMixin, dj.TableType):
    definition = """
    # Table description
    primary_key: datatype
    ---
    secondary_key: datatype
    """
```

## Table Types and Patterns

### 1. Parameters Tables
Store analysis parameters for reproducible computations.

**Conventions:**
- Name ends with `Parameters` or `Params`
- Usually `dj.Lookup` (for predefined values) or `dj.Manual`
- Primary key: `{pipeline}_params_name: varchar(32)`
- Parameters stored as: `{pipeline}_params: blob` (Python dictionary)

**Example:**
```python
@schema
class MyAnalysisParameters(SpyglassMixin, dj.Lookup):
    definition = """
    analysis_params_name: varchar(32)
    ---
    analysis_params: blob
    """
    
    contents = [
        ["default", {"threshold": 0.5, "window_size": 100}],
        ["strict", {"threshold": 0.8, "window_size": 50}],
    ]
    
    @classmethod
    def insert_default(cls):
        cls().insert(rows=cls.contents, skip_duplicates=True)
```

### 2. Selection Tables
Pair data with parameters for analysis.

**Conventions:**
- Name ends with `Selection`
- Usually `dj.Manual`
- Foreign keys to data source and parameters tables
- Sets up what will be computed

**Example:**
```python
@schema
class MyAnalysisSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DataSourceTable
    -> MyAnalysisParameters
    ---
    """
```

### 3. Computed Tables
Store analysis results.

**Conventions:**
- Usually `dj.Computed`
- Foreign key to corresponding Selection table
- Include `analysis_file_name` from `AnalysisNwbfile`
- Implement `make()` method for computation

**Example:**
```python
@schema
class MyAnalysisOutput(SpyglassMixin, dj.Computed):
    definition = """
    -> MyAnalysisSelection
    ---
    -> AnalysisNwbfile
    """
    
    def make(self, key):
        # Get parameters and data
        params = (MyAnalysisParameters & key).fetch1("analysis_params")
        
        # Perform analysis
        result = perform_analysis(data, **params)
        
        # Save to NWB file and get analysis_file_name
        analysis_file_name = create_analysis_nwb_file(result, key)
        
        # Insert result
        self.insert1({**key, "analysis_file_name": analysis_file_name})
```

### 4. NWB Ingestion Tables
Import data from NWB files.

**Conventions:**
- Usually `dj.Imported`
- Include `object_id` to track NWB objects
- Implement `make()` and `fetch_nwb()` methods

### 5. Merge Tables
Combine outputs from different pipeline versions.

**Conventions:**
- Name ends with `Output`
- Inherit from custom `Merge` class
- Enable unified downstream processing

## Development Standards

### Code Organization
- **Modules**: Group related schemas (e.g., `common`, `spikesorting`, `lfp`)
- **Schemas**: Group tables by topic within modules
- **Common module**: Shared tables across pipelines (use sparingly)
- **Custom analysis**: Create your own schema for project-specific work

### Naming Conventions
- **Classes**: PascalCase (e.g., `SpikeWaveforms`)
- **Schemas**: snake_case with module prefix (e.g., `spikesorting_recording`)
- **Variables**: snake_case
- **Primary keys**: descriptive names (e.g., `recording_id`, `sort_group_id`)

### Required Methods
- **Parameters tables**: `insert_default()` classmethod for `dj.Lookup`
- **Computed tables**: `make(self, key)` method
- **NWB tables**: `make()` and `fetch_nwb()` methods

### Documentation
- Use numpy-style docstrings
- Include clear table definition comments
- Document parameter meanings and expected formats

## Common Workflows

### Creating a New Pipeline
1. Define Parameters table with default values
2. Create Selection table linking data and parameters  
3. Implement Computed table with analysis logic
4. Add Merge table if multiple pipeline versions exist

### File Handling
- Use `AnalysisNwbfile` table for storing analysis results
- Follow NWB format standards
- Include proper metadata and provenance

### Testing
- Test parameter insertion and defaults
- Verify selection table constraints
- Test computation pipeline end-to-end
- Check NWB file creation and retrieval

## Best Practices for LLM Assistance

### When Creating Tables:
1. **Always inherit from SpyglassMixin first**:
   `class MyTable(SpyglassMixin, dj.TableType)`
2. **Use appropriate table types**: Lookup for parameters, Manual for
   selections, Computed for analysis
3. **Follow naming conventions**: Clear, descriptive names with appropriate
   suffixes
4. **Include proper foreign keys**: Establish clear data dependencies
5. **Add comprehensive docstrings**: Explain purpose and usage

### When Suggesting Code:
1. **Check existing patterns**: Look at similar tables in the same module
2. **Verify schema imports**: Ensure proper schema declaration and imports
3. **Include error handling**: Especially for file operations and computations
4. **Consider permissions**: Use SpyglassMixin methods for data access control
5. **Follow dependency order**: Parameters → Selection → Computed → Merge

### Common Mistakes to Avoid:
- Missing SpyglassMixin inheritance
- Incorrect table type selection
- Missing @schema decorator
- Improper foreign key relationships
- Not implementing required methods (make, insert_default)
- Inconsistent naming conventions

## Key Imports and Dependencies
```python
import datajoint as dj
from spyglass.utils import SpyglassMixin, logger
from spyglass.common import AnalysisNwbfile, IntervalList
```

## Resources
- [Developer Documentation](docs/src/ForDevelopers/)
- [Table Types Guide](docs/src/ForDevelopers/TableTypes.md)
- [Schema Design](docs/src/ForDevelopers/Schema.md)
- [Contributing Guide](docs/src/ForDevelopers/Contribute.md)

This document should be referenced when providing assistance with Spyglass
development to ensure consistency with established patterns and standards.