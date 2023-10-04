# Developer notes

Notes on how the repo / database is organized, intended for a new developer.

## Overall organization

- Tables that are about similar things are grouped into a schema. Each schema is
  defined in a `.py` file. Example: all the tables related to quality metrics
  are part of the `common_metrics` schema and are defined in `common_metrics.py`
  in `common` module.
- The `common` module only contains schema that will be useful for everyone in
  the lab. If you want to add things to `common`, first check with Loren.
- For analysis that will be only useful to you, create your own schema.

## Types of tables

### NWB-related

- Data tier: `dj.Imported`
- Primary key: foreign key from `Session`
- Non-primary key: `object_id`
- Each NWB-related table has a corresponding data object in the NWB file. This
  object can be referred by a unique hash called an _object ID_.
- These tables are automatically populated when an NWB file is first ingested
  into the database. To enable this, include the `populate` call in the `make`
  method of `Session`.
- Required methods:
  - `make`: must read information from an NWB file and insert it to the table.
  - `fetch_nwb`: retrieve the data specified by the object ID; search the repo
    for examples.
- Example: `Raw`, `Institution` etc

### Pipeline

- Each analysis pipeline defined by a schema. A typical pipeline has at least
  three tables:
  - _Parameters_ table
    - Naming convention: should end with `Parameters` (e.g. `MetricParameters`)
    - Data tier: `dj.Manual`
    - Function: holds a set of parameters for the particular analysis.
    - Primary key: `x_params_name` (str); x is the name of the pipeline (e.g.
      `metric_params_name`).
    - Non-primary key: `x_params` (dict; use `blob` in the definition); holds
      the parameters as key-value pairs.
    - Required method: `insert_default` to insert a reasonable default parameter
      into the table.
  - _Selection_ table
    - Naming convention: should end with `Selection` (e.g. `MetricSelection`)
    - Data tier: `dj.Manual`
    - Function: associates a set of parameters to the data to be applied. For
      example, in the case of computing quality metrics, one might put extracted
      waveforms and a set of metrics parameters as a single entry in this table.
    - Primary key: foreign key from a table containing data and the Parameters
      table (i.e. Selection tables are downstream of these two tables).
      - Of course, it is possible for a Selection table to collect information
        from more than one Parameter table. For example, the Selection table for
        spike sorting holds information about both the interval (`SortInterval`)
        and the group of electrodes (`SortGroup`) to be sorted.
    - Usually no other key needs to be defined
  - _Data_ table
    - Data tier: `dj.Computed`
    - carries out the computation specified in the Selection table when
      `populate` is called.
    - The primary key should be foreign key inherited from the Selection table.
      The non-primary key should be `analysis_file_name` inherited from
      `AnalysisNwbfile` table (i.e. name of the analysis NWB file that will hold
      the output of the computation).
    - Required methods:
      - `make`: carries out the computation and insert a new entry; must also
        create an analysis NWB file and insert it to the `AnalysisNwbfile`
        table. Note that this method is never called directly; it is called via
        `populate`.
      - `delete`: extension of the `delete` method that checks user privilege
        before deleting entries as a way to prevent accidental deletion of
        computations that take a long time (see below).
    - Example: `QualityMetrics`
- _Why have the Parameters table?_ Because one might want to repeat an analysis
  with different sets of parameters. This way we keep track of everything. Also
  encourages sharing of parameters.
- _Why have the Selection table instead of going directly from Parameter table
  to Data table?_ one still has to manually pass in the data and the parameters
  to use for the computation (e.g. as an argument to `populate`. Since this is
  required, defining it first in the Selection table is no less efficient. In
  addition, multiple entries in Selection table can be run in parallel when
  `populate` is called with `reserve_jobs=True` option.

### Multi-pipeline

- These are tables that are part of many pipelines.
- Examples: `IntervalList` (whose entries define time interval for any
  analysis), `AnalysisNwbfile` (whose entries define analysis NWB files created
  for any analysis), `SpikeSorting` (whose entries include many types of spike
  sorting, such as uncurated, automatically curated, manually curated etc)
- Data tier: `dj.Manual`
- Note that because these are stand-alone manual tables, they are not part of
  the dependency structure. This means one should try to include enough
  information such that they can be linked back to the pipelines.

## Integration with NWB

### NWB files

- NWB files contain everything about the experiment and form the starting point
  of all analysis
- stored in `/stelmo/nwb/raw`
- A copy of the NWB file that only contains pointers to objects in original file
  is made in the same directory; the name has an extra `_` at the end, e.g.
  `beans20190718_.nwb`; this file is made because we want to create object IDs
  to refer to parts of the NWB file, but don't want to store these object IDs in
  the original file to avoid file corruption
- Listed in the `Nwbfile` table

### Analysis NWB files

- These are NWB files that hold the results of intermediate steps in the analysis.
- Examples of data stored: filtered recordings, spike times of putative units
  after sorting, or waveform snippets.
- Stored in `/stelmo/nwb/analysis`
- Listed as an entry in the `AnalysisNwbfile` table.

Note: for both types of NWB files, the fact that a file is not listed in the
table doesn't mean the file does not exist in the directory. You can 'equalize'
the list of NWB files and the list of actual files on disk by running `cleanup`
method (i.e. it deletes any files not listed in the table from disk).

## Reading and writing recordings

- Right now the recording starts out as an NWB file. This is opened as a
  `NwbRecordingExtractor`, a class in `spikeinterface`. When using `sortingview`
  for visualizing the results of spike sorting, this recording is saved again in
  HDF5 format. This duplication should be resolved in the future.

## Naming convention

There are a few places where a name needs to be given to objects. Follow these rules:

- _Recordings_: should be given unique names. As such we have decided to simply
  concatenate everything that went into defining it separated by underscore,
  i.e. `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName`.
- _SpikeSorting_: should be unique. Simply concatenates
  `SpikeSorter_SorterParamName` to the name of the recording.
- _Waveforms_: should be unique. Concatenates `WaveformParamName` to the name of
  the sorting.
- _Quality metrics_: should be unique. concatenates `MetricParamName` to the
  name of the waveform.
- _Analysis NWB files_: same as the objects, i.e. the analysis NWB file that
  holds recording is named
  `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName.nwb`
- An alternative way to get unique names that are not as long is to generate a
  UUID for each file. Currently each recording and sorting are given such IDs.
- A method that will not be explicitly called by the user should start with `_`

## Time

- All valid intervals of any kind must be inserted into the `IntervalList` table
  prior to being used.
- Store an interval as `[start_time, stop_time]`. The list can be nested for a
  set of disjoint intervals.
- Some recordings have explicit timestamps associated with each sample. This is
  obtained by a system called PTP. In this system, time 0 is defined as 1 Jan 1970. Other (typically older) recordings do not and their times must be
  inferred from the TTL pulses from the camera (ask if this doesn't make sense).
- What is a valid interval? Because our experiments can be long, sometimes there
  are missing samples. This can be due to many reasons, such as the commutator
  connection being faulty for a few milliseconds. As a result we have 'holes' in
  our data. A valid interval is a start time and an end time between which there
  are no holes.

## Misc

- You may want to create a development/testing environment independent of the
  lab datajoint server. To do so, run your own datajoint server with Docker. See
  [example](./notebooks/docker_mysql_tutorial.ipynb).
- Datajoint is unable to set delete permissions on a per-table basis. In other
  words, if a user is able to delete entries in a given table, she can delete
  entries in any table in the schema. Some tables that hold important data
  extends the `delete` method to check if the datajoint username matches a list
  of allowed users when `delete` is called. If you think this would be useful
  for your table, see examples in `common_spikesorting.py`.
- In general, use `numpy` style docstring.
- Don't overload a single `.py` file. For each pipeline make a new `.py` file
  and define your schema / tables.
- Some of the 'rules' above may need to change or be inappropriate for some
  cases. If you want to start a discussion, talk to Loren.

## Making a release

1. In `pyproject.toml`, under `[project]`, update the `version` key to the new
   version string.
2. In `CITATION.cff`, update the `version` key to the new version string.
3. Make a pull request with these changes.
4. After merging these changes, run `git tag --sign -m "spyglass ${release}"
${release} origin/master` where `${release}` is replaced with the new version
   string.

- This step requires a
  [GPG signing key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

1. Publish the new release tag. Run `git push origin ${release}`.
2. Generate distribution packages and upload them to PyPI following [these
   instructions](https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives).
3. Make a new release on GitHub with the new release tag:
   <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>

## TODO

- Fetch nwb method is currently implemented for each table. This is unnecessary
  because (1) what matters is the query, not the table the method is attached
  to; and (2) you either look up the Nwbfile or the AnalysisNwbfile table for
  it, so really there are only two versions. It would be better to just have two
  standalone functions. Or just one that figures out which NWB file to look up.
