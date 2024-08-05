# Table Types

Spyglass uses DataJoint's default
[table tiers](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/tiers/).

By convention, an individual pipeline has one or more the following table types:

- Common/Multi-pipeline table
- NWB ingestion table
- Parameters table
- Selection table
- Data table
- Merge Table (see also [stand-alone doc](../Features/Merge.md))

## Common/Multi-pipeline

Tables shared across multiple pipelines for shared data types.

- Naming convention: None
- Data tier: `dj.Manual`
- Examples: `IntervalList` (time interval for any analysis), `AnalysisNwbfile`
    (analysis NWB files)

_Note_: Because these are stand-alone tables not part of the dependency
structure, developers should include enough information to link entries back to
the pipeline where the data is used.

## NWB ingestion

Automatically populated when an NWB file is ingested (i.e., `dj.Imported`) to
keep track of object hashes (i.e., `object_id`) in the NWB file. All such tables
should be included in the `make` method of `Session`.

- Naming convention: None
- Data tier: `dj.Imported`
- Primary key: foreign key from `Session`
- Non-primary key: `object_id`, the unique hash of an object in the NWB file.
- Examples: `Raw`, `Institution`, etc.
- Required methods:
    - `make`: must read information from an NWB file and insert it to the table.
    - `fetch_nwb`: retrieve the data specified by the object ID.

## Parameters

Stores the set of values that may be used in an analysis.

- Naming convention: end with `Parameters` or `Params`
- Data tier: `dj.Manual`, or `dj.Lookup`
- Primary key: `{pipeline}_params_name`, `varchar`
- Non-primary key: `{pipeline}_params`, `blob` - dict of parameters
- Examples: `RippleParameters`, `DLCModelParams`
- Possible method: if `dj.Manual`, include `insert_default`

_Notes_: Some early instances of Parameter tables (a) used non-primary keys for
each individual parameter, and (b) use the Manual rather than Lookup tier,
requiring a class method to insert defaults.

## Selection

A staging area to pair sessions with parameter sets, allowing us to be selective
in the analyses we run. It may not make sense to pair every paramset with every
session.

- Naming convention: end with `Selection`
- Data tier: `dj.Manual`
- Primary key(s): Foreign key references to
    - one or more NWB or data tables
    - optionally, one or more parameter tables
- Non-primary key: None
- Examples: `MetricSelection`, `LFPSelection`

It is possible for a Selection table to collect information from more than one
Parameter table. For example, the Selection table for spike sorting holds
information about both the interval (`SortInterval`) and the group of electrodes
(`SortGroup`) to be sorted.

## Data

The output of processing steps associated with a selection table. Has a `make`
method that carries out the computation specified in the Selection table when
`populate` is called.

- Naming convention: None
- Data tier: `dj.Computed`
- Primary key: Foreign key reference to a Selection table.
- Non-primary key: `analysis_file_name` inherited from `AnalysisNwbfile` table
    (i.e., name of the analysis NWB file that will hold the output of the
    computation).
- Required method, `make`: carries out the computation and insert a new entry;
    must also create an analysis NWB file and insert it to the `AnalysisNwbfile`
    table. Note that this method is never called directly; it is called via
    `populate`. Multiple entries can be run in parallel when called with
    `reserve_jobs=True`.
- Example: `QualityMetrics`, `LFPV1`

## Merge

Following a convention outlined in [a dedicated doc](../Features/Merge.md),
merges the output of different pipelines dedicated to the same modality as part
tables (e.g., common LFP, LFP v1, imported LFP) to permit unified downstream
processing.

- Naming convention: `{Pipeline}Output`
- Data tier: custom `_Merge` class
- Primary key: `merge_id`, `uuid`
- Non-primary key: `source`, `varchar` table name associated with that entry
- Required methods: None - see custom class methods with `merge_` prefix
- Example: `LFPOutput`, `PositionOutput`
