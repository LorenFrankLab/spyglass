## Developer notes
Here are a set of notes on how the repo / database is organized, intended for a new developer.
* Tables that are about similar things are grouped into a schema. Each schema is placed inside a `.py` file. For example, all the tables related to quality metrics are part of the `common_metrics` schema and are defined in `common_metrics.py` under `common` directory. 
  * The `common` directory will only hold schema that will be useful for everyone in the lab. If you want to add things to `common`, first check with Loren. 
  * For analysis that will be only useful to you, create your own schema.
* The Datajoint database is designed to mimic the structure of the NWB file. As such, there are at least two groups of tables. The first group is called *NWB-related* tables. 
  * Data tier: `dj.Imported`
  * They mimic the data structure inside an NWB file. 
  * These tables are populated automatically when an NWB file is first ingested into the database. Given that they are `dj.Imported` tables, one should define in the `make` method the steps to read information from NWB file and insert it to the table. Be sure to also include the `populate` call in the `make` method of `Session`.
  * May point to an object within an NWB file with an *object ID* (a hash that points to a particular data inside an NWB file). In that case, must have a `fetch_nwb` method defined. This method makes it easy to retrieve the data without having to deal with `pynwb`. Search the repo for examples.
* The next group of tables is called *analysis* tables. Each type of analysis is given a schmea. For a typical analysis schema, there are usually three tables (at least):
  * __Parameters__ table
    * This is a `dj.Manual` table that holds a set of parameters for the particular analysis.
    * The primary key should be `list_name` (str; the name of the parameter set). The other (non-primary) key is usually a python dictionary (use `blob` in the definition) that holds the parameters as key-value pairs. 
    * Example: `MetricParameters`
    * Must contain a method called `insert_default_params` that inserts a reasonable default parameter into the table.
  * __Selection__ table
    * This is a `dj.Manual` table that associates a set of parameters to the data to be applied. For example, in the case of computing quality metrics, one might put the results of a spike sorting run and a set of metrics parameters as a single entry in this table.
    * The primary key should be inherited from the data table and the Parameters table. In other words, it is downstream of these two tables. 
      * Of course, it is possible for a Selection table to collect information from more than one Parameter table. For example, during the Selection table for spike sorting holds information about the interval (`SortInterval`) and the group of electrodes (`SortGroup`) to be sorted.
    * Example: `MetricSelection`
  * __Data__ table
    * This is a `dj.Computed` table that is downstream of the Selection table. It carries out the computation specified in the Selection table when `populate` is called. All the code that does this should go in the `make` method. 
    * The primary key should be inherited from the Selection table. The non-primary key should inherit from `AnalysisNwbfile` table (i.e. name of the analysis NWB file that will hold the output of the computation).
      * As a result `make` should include code that creates such an analysis NWB file and inserts it to the `AnalysisNwbfile` table.
    * Must contain an extension of the `delete` method that checks user privilege before deleting entries as a way to prevent accidental deletion of computations that take a long time
    * Example: `QualityMetrics`
* *Why have the Selection table? Why not just go from Parameter table to Data table?* Well, then you still have to manually pass in which data and which parameter set to use for the computation, perhaps as an argument to `populate`. If you're going to do that, might as well store it somewhere. Another reason to have the Selection table is that you can put multiple entries in it and then run `populate` in parallel by specifying `reserve_jobs=True` option. 
* There is a third group of tables (call them *multi-analysis*). These are *analysis* tables that are part of many analyses. Examples are `IntervalList` (whose entries define time interval for any analysis) and `AnalysisNwbfile` (whose entries define analysis NWB files created for any analysis). These are `dj.Manual`.
* Analysis NWB files are NWB files that hold the results of intermediate steps in the analysis. Examples include those that store filtered recordings, spike times of putative units after sorting, or waveform snippets. These are located in `/stelmo/nwb/analysis`. Each Analysis NWB file is also listed as an entry in the `AnalysisNwbfile` table. Note that just because a file is not listed in the table, doesn't mean the file does not exist in the directory. 
  * On the other hand, the raw data is stored in `/stelmo/nwb/raw`. 
* Currently some tables accepts both manual insertions and insertion from an NWB file (e.g. `IntervalList`). These are `dj.Manual` and have an `insert_from_nwbfile` method. This is discouraged and will be changed in the future
* All intervals of any kind must be inserted into the `IntervalList` table prior to being used. Store an interval as `[start_time, stop_time]`. The list can be nested for a set of disjoint intervals. 
* Some recordings have explicit timestamps associated with each sample. This is obtained by a system called PTP. In this system, time 0 is defined as 1 Jan 1970. Other (typically older) recordings do not and their times must be inferred from the DIO signals (ask if this doesn't make sense).
* Some recordings (ephys, behavior etc) have missing samples. `valid_times` indicates non-missing (present) intervals. 
* You may want to create a development/testing environment independent of the lab datajoint server. To do so, run your own datajoint server with Docker. See [example](../notebook/docker_mysql_tutorial.ipynb).
* Datajoint is unable to set delete permissions on a per-table basis. In other words, if a user is able to delete entries in a given table, she can delete entries in any table in the database. Some tables that hold important data extends the `delete` method to check if the datajoint username matches a list of allowed users when `delete` is called. If you think this would be useful for your table, see examples in `common_spikesorting.py`. 
* In general, use `numpy` style docstring.
* Don't overload a single `.py` file. For each analysis make a new `.py` file and define your tables and schema. 
* Some of the 'rules' above may need to change or be inappropriate for some cases. If you want to start a discussion, talk to Loren.
