## Developer notes
Notes on how the repo / database is organized, intended for a new developer.

### Organization
* Tables that are about similar things are grouped into a schema. Each schema is defined in a `.py` file. Example: all the tables related to quality metrics are part of the `common_metrics` schema and are defined in `common_metrics.py` under `common` directory. 
* The `common` directory only contains schema that will be useful for everyone in the lab. If you want to add things to `common`, first check with Loren. 
* For analysis that will be only useful to you, create your own schema.
### Types of tables
__NWB-related__
  * Data tier: `dj.Imported`
  * Primary key: foreign key from `Session`
  * Non-primary key: `object_id`
  * Each table has a corresponding data object in the NWB file. This could be done via an *object ID*. 
  * Populated automatically when an NWB file is first ingested into the database. To enable this, include the `populate` call in the `make` method of `Session`.
  * Required methods:
    * `make`: must read information from NWB file and insert it to the table. 
    * `fetch_nwb`: makes it easy to retrieve the data without having to deal with `pynwb`; search the repo for examples.
  * Example: `Raw`, `Institution` etc

__Analysis__
* Each type of analysis is given a schema. For a typical analysis schema, there are usually three tables (at least):
  * _Parameters_ table
    * Data tier: `dj.Manual`
    * Function: holds a set of parameters for the particular analysis.
    * Primary key: `list_name` (str; the name of the parameter set)
    * Non-primary key: `params` (dict; use `blob` in the definition); holds the parameters as key-value pairs. 
    * Example: `MetricParameters`
    * Required method: `insert_default_params` to insert a reasonable default parameter into the table.
  * _Selection_ table
    * Data tier: `dj.Manual` 
    * Function: associates a set of parameters to the data to be applied. For example, in the case of computing quality metrics, one might put the results of a spike sorting run and a set of metrics parameters as a single entry in this table.
    * Primary key: foreign key from a table containing data and the Parameters table (i.e. downstream of these two tables). 
      * Of course, it is possible for a Selection table to collect information from more than one Parameter table. For example, during the Selection table for spike sorting holds information about the interval (`SortInterval`) and the group of electrodes (`SortGroup`) to be sorted.
    * Usually no other key needs to be defined
    * Example: `MetricSelection`
  * _Data_ table
    * Data tier: `dj.Computed` 
    *  carries out the computation specified in the Selection table when `populate` is called. 
    * The primary key should be foreign key inherited from the Selection table. The non-primary key should inherit from `AnalysisNwbfile` table (i.e. name of the analysis NWB file that will hold the output of the computation).
    * Required methods: 
      * `make`: carries out the computation and insert a new entry; must also create an analysis NWB file and insert it to the `AnalysisNwbfile` table.
      * `delete`: extension of the `delete` method that checks user privilege before deleting entries as a way to prevent accidental deletion of computations that take a long time (see below).
    * Example: `QualityMetrics`
* *Why have the Parameters table?* Because one might want to repeat an analysis with different sets of parameters. This way we keep track of everything. Also encourages sharing of parameters.
* *Why have the Selection table instead of going directly from Parameter table to Data table?* one still has to manually pass in the data and the parameter set to use for the computation (e.g. as an argument to `populate`). Since this is required, defining it first in the Selection table is no less efficient. In addition, multiple entries in Selection table can be run in parallel when `populate` is called with `reserve_jobs=True` option. 

__Multi-analysis__
* These are *analysis* tables that are part of many analyses. 
* Examples: `IntervalList` (whose entries define time interval for any analysis) and `AnalysisNwbfile` (whose entries define analysis NWB files created for any analysis).
* Data tier: `dj.Manual`

### Integration with NWB
__NWB files__ 
* NWB files contain everything about the experiment and form the starting point of all analysis
* stored in `/stelmo/nwb/raw`
* A copy of the NWB file that only contains pointers to objects in original file is made in the same directory; the name has an extra `_` at the end. E.g. `beans20190718_.nwb`; this file is made because we don't want to create and store object IDs in the original file to avoid file corruption
* Listed in the `Nwbfile` table
  
__Analysis NWB files__
* These are NWB files that hold the results of intermediate steps in the analysis. 
* Examples of data stored: filtered recordings, spike times of putative units after sorting, or waveform snippets.
* Stored in `/stelmo/nwb/analysis`
* listed as an entry in the `AnalysisNwbfile` table. 
  
Note: for both types of NWB files, that just because a file is not listed in the table, doesn't mean the file does not exist in the directory. You can  'equalize' the list of NWB files and the list of actual files on disk by running `cleanup` method (i.e. it deletes any files not listed in the table from disk).

### Reading and writing recordings
* Right now the recording starts out as an NWB file. This is opened as a `NwbRecordingExtractor`, a class in `spikeinterface`. After preprocessing the recording is saved as a `LabBoxEphysRecordingExtractor`, which uses hdf5 format different from NWB. This is done for speed but in the future we should use NWB for all the steps.

### Naming convention
There are a few places where a name needs to be given to objects. Follow these rules:
* _Recordings_: should be given unique names. As such we have decided to simply concatenate everything that went into defining it separated by underscore, i.e. `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName`. 
* _Sortings_: should be unique. Simply concatenates `SpikeSorter_SorterParamName` to the name of the recording. 
* _Waveforms_: should be unique. Concatenates `WaveformParamName` to the name of the sorting.
* _Quality metrics_: should be unique. concatenates  `MetricParamName` to the name of the waveform.
* _Analysis NWB files_: same as the objects, i.e. the analysis NWB file that holds recording is named `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName.nwb`
* An alternative way to get unique names that are not as long is to generate a UUID for each file. We may switch to this method in the future.
* If you write a method that will not be explicitly called by the user, make it start with `_`

### Time
* All intervals of any kind must be inserted into the `IntervalList` table prior to being used. Store an interval as `[start_time, stop_time]`. The list can be nested for a set of disjoint intervals. 
* Some recordings have explicit timestamps associated with each sample. This is obtained by a system called PTP. In this system, time 0 is defined as 1 Jan 1970. Other (typically older) recordings do not and their times must be inferred from the DIO signals (ask if this doesn't make sense).
* Some recordings (ephys, behavior etc) have missing samples. `valid_times` indicates non-missing (present) intervals. 

### Misc
* Currently some tables accepts both manual insertions and insertion from an NWB file (e.g. `IntervalList`). These are `dj.Manual` and have an `insert_from_nwbfile` method. This is discouraged and will be changed in the future
* You may want to create a development/testing environment independent of the lab datajoint server. To do so, run your own datajoint server with Docker. See [example](../notebook/docker_mysql_tutorial.ipynb).
* Datajoint is unable to set delete permissions on a per-table basis. In other words, if a user is able to delete entries in a given table, she can delete entries in any table in the database. Some tables that hold important data extends the `delete` method to check if the datajoint username matches a list of allowed users when `delete` is called. If you think this would be useful for your table, see examples in `common_spikesorting.py`. 
* In general, use `numpy` style docstring.
* Don't overload a single `.py` file. For each analysis make a new `.py` file and define your tables and schema. 
* Some of the 'rules' above may need to change or be inappropriate for some cases. If you want to start a discussion, talk to Loren.
