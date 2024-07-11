# Using NWB

This article explains how to use the NWB format in Spyglass. It covers the
naming conventions, storage locations, and the relationships between NWB files
and other tables in the database.

## NWB files

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

## Analysis files

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

# Reading and writing recordings

Recordings start out as an NWB file, which is opened as a
`NwbRecordingExtractor`, a class in `spikeinterface`. When using `sortingview`
for visualizing the results of spike sorting, this recording is saved again in
HDF5 format. This duplication should be resolved in the future.

# Naming convention

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

# Time

The `IntervalList` table stores all time intervals in the following format:
`[start_time, stop_time]`, which represents a contiguous time of valid data.
These are used to exclude any invalid timepoints, such as missing data from a
faulty connection.

- Intervals can be nested for a set of disjoint intervals.
- Some recordings have explicit
    [PTP timestamps](https://en.wikipedia.org/wiki/Precision_Time_Protocol)
    associated with each sample. Some older recordings are missing PTP times,
    and times must be inferred from the TTL pulses from the camera.
