# Developer notes

Notes on how the repo / database is organized, intended for a new developer.

## Development workflow

New contributors should follow the
[Fork-and-Branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).
See GitHub instructions
[here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

Regular contributors may choose to follow the
[Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
for features that will involve multiple contributors.

## Code organization

- Tables are grouped into schemas by topic (e.g., `common_metrics`)
- Schemas
    - Are defined in a `py` pile.
    - Correspond to MySQL 'databases'.
    - Are organized into modules (e.g., `common`) by folders.
- The _common_ module
    - In principle, contains schema that are shared across all projects.
    - In practice, contains shared tables (e.g., Session) and the first draft of
        schemas that have since been split into their own
        modality-specific\
        modules (e.g., `lfp`)
    - Should not be added to without discussion.
- A pipeline
    - Refers to a set of tables used for processing data of a particular modality
        (e.g., LFP, spike sorting, position tracking).
    - May span multiple schema.
- For analysis that will be only useful to you, create your own schema.

## Types of tables

Spyglass uses DataJoint's default
[table tiers](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/tiers/).

By convention, an individual pipeline has one or more the following table types:

- Common/Multi-pipeline table
- NWB ingestion table
- Parameters table
- Selection table
- Data table
- Merge Table (see also [doc](./misc/merge_tables.md)

### Common/Multi-pipeline

Tables shared across multiple pipelines for shared data types.

- Naming convention: None
- Data tier: `dj.Manual`
- Examples: `IntervalList` (time interval for any analysis), `AnalysisNwbfile`
    (analysis NWB files)

_Note_: Because these are stand-alone tables not part of the dependency
structure, developers should include enough information to link entries back to
the pipeline where the data is used.

### NWB ingestion

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

### Parameters

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

### Selection

A staging area to pair sessions with parameter sets, allowing us to be selective
in the analyses we run. It may not make sense to pair every paramset with every
session.

- Naming convention: end with `Selection`
- Data tier: `dj.Manual`
- Primary key(s): Foreign key references to
    - one or more NWB or data tables
    - optionally, one or more parameter tables
- Non-primary key: None
- Examples: `MetricSelection`, `LFPSElection`

It is possible for a Selection table to collect information from more than one
Parameter table. For example, the Selection table for spike sorting holds
information about both the interval (`SortInterval`) and the group of electrodes
(`SortGroup`) to be sorted.

### Data

The output of processing steps associated with a selection table. Has a `make`
method that carries out the computation specified in the Selection table when
`populate` is called.

- Naming convention: None
- Data tier: `dj.Computed`
- Primary key: Foreign key reference to a Selection table.
- Non-primary key: `analysis_file_name` inherited from `AnalysisNwbfile` table
    (i.e., name of the analysis NWB file that will hold the output of the
    computation).
- Required methods:
    - `make`: carries out the computation and insert a new entry; must also create
        an analysis NWB file and insert it to the `AnalysisNwbfile` table. Note
        that this method is never called directly; it is called via `populate`.
        Multiple entries can be run in parallel when called with
        `reserve_jobs=True`.
    - `delete`: extension of the `delete` method that checks user privilege before
        deleting entries as a way to prevent accidental deletion of computations
        that take a long time (see below).
- Example: `QualityMetrics`, `LFPV1`

### Merge

Following a convention outlined in [the dedicated doc](./misc/merge_tables.md),
merges the output of different pipelines dedicated to the same modality as part
tables (e.g., common LFP, LFP v1, imported LFP) to permit unified downstream
processing.

- Naming convention: `{Pipeline}Output`
- Data tier: custom `_Merge` class
- Primary key: `merge_id`, `uuid`
- Non-primary key: `source`, `varchar` table name associated with that entry
- Required methods: None - see custom class methods with `merge_` prefix
- Example: `LFPOutput`, `PositionOutput`

## Integration with NWB

### NWB files

NWB files contain everything about the experiment and form the starting point of
all analyses.

- Naming: `{animal name}YYYYMMDD.nwb`
- Storage:
    - On disk, directory identified by `settings.py` as `raw_dir` (e.g.,
        `/stelmo/nwb/raw`)
    - In database, in the `Nwbfile` table
- Copies:
    - made with an underscore `{animal name}YYYYMMDD_.nwb`
    - stored in the same `raw_dir`
    - contain pointers to objects in original file
    - permit adding new parts to the NWB file without risk of corrupting the
        original data

### Analysis files

Hold the results of intermediate steps in the analysis.

- Naming: `{animal name}YYYYMMDD_{10-character random string}.nwb`
- Storage:
    - On disk, directory identified by `settings.py` as `analysis_dir` (e.g.,
        `/stelmo/nwb/analysis`). Items are further sorted into folders matching
        original NWB file name
    - In database, in the `AnalysisNwbfile` table.
- Examples: filtered recordings, spike times of putative units after sorting, or
    waveform snippets.

_Note_: Because NWB files and analysis files exist both on disk and listed in
tables, these can become out of sync. You can 'equalize' the database table
lists and the set of files on disk by running `cleanup` method, which deletes
any files not listed in the table from disk.

## Reading and writing recordings

Recordings start out as an NWB file, which is opened as a
`NwbRecordingExtractor`, a class in `spikeinterface`. When using `sortingview`
for visualizing the results of spike sorting, this recording is saved again in
HDF5 format. This duplication should be resolved in the future.

## Naming convention

The following objects should be uniquely named.

- _Recordings_: Underscore-separated concatenations of uniquely defining
    features,
    `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName`.
- _SpikeSorting_: Adds `SpikeSorter_SorterParamName` to the name of the
    recording.
- _Waveforms_: Adds `_WaveformParamName` to the name of the sorting.
- _Quality metrics_: Adds `_MetricParamName` to the name of the waveform.
- _Analysis NWB files_:
    `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName.nwb`
- Each recording and sorting is given truncated UUID strings as part of
    concatenations.

Following broader Python conventions, methods a method that will not be
explicitly called by the user should start with `_`

## Time

The `IntervalList` table stores all time intervals in the following format:
`[start_time, stop_time]`, which represents a contiguous time of valid data.
These are used to exclude any invalid timepoints, such as missing data from a
faulty connection.

- Intervals can be nested for a set of disjoint intervals.
- Some recordings have explicit
    [PTP timestamps](https://en.wikipedia.org/wiki/Precision_Time_Protocol)
    associated with each sample. Some older recordings are missing PTP times,
    and times must be inferred from the TTL pulses from the camera.

## Misc

- During development, we suggest using a Docker container. See
    [example](./notebooks/00_Setup.ipynb).
- DataJoint is unable to set delete permissions on a per-table basis. If a user
    is able to delete entries in a given table, she can delete entries in any
    table in the schema. The `SpikeSorting` table extends the built-in `delete`
    method to check if the username matches a list of allowed users when
    `delete` is called. Issues #226 and #586 track the progress of generalizing
    this feature.
- `numpy` style docstrings will be interpreted by API docs. To check for
    compliance, monitor the std out when building docs (see `docs/README.md`)
- `fetch_nwb` is currently reperated across many tables. For progress on a fix,
    follow issue #530

## Making a release

Spyglass follows [Semantic Versioning](https://semver.org/) with versioning of
the form `X.Y.Z` (e.g., `0.4.2`).

1. In `CITATION.cff`, update the `version` key.
2. Make a pull request with changes.
3. After the pull request is merged, pull this merge commit and tag it with
    `git tag {version}`
4. Publish the new release tag. Run `git push origin {version}`. This will
    rebuild docs and push updates to PyPI.
5. Make a new
    [release on GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
