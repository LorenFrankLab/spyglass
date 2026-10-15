# Ingestion Process

## Step 0: NWB files

Before beginning with spyglass, data must be compiled into the standardized NWB
file format. NWB files contain everything about the experiment and form the
starting point of all analyses. Numerous
[online tutorials](https://nwb.org/converting-data-to-nwb/) exist to help get
you started in this process, as well as existing packages for lab-specific
conversions ([1](https://github.com/catalystneuro),
[2](https://github.com/LorenFrankLab/trodes_to_nwb)) that can be used as a
reference.

The following sections describe how data is extracted from these files and
brought into the Spyglass system. For best compatibility, please use these as
reference when creating your NWB files.

## What ingestion does

Ingestion extracts data from a raw NWB file and stores it across Spyglass
tables. `spyglass.data_import.insert_sessions(nwb_file_name)` is the entry
point.

It reads the whole file and works out every row it would write **before**
writing anything. Three things follow, and they are the reason to read a report
rather than a traceback:

- Every problem in the file is reported at once, not just the first.
- A blocking problem does not write to the database.
- Anything that worked is cached for the next ingestion attempt.

The [Insert Data tutorial](../notebooks/02_Insert_Data.ipynb) walks through
running it, checking a file without writing (`dry_run=True`), and the flags. The
rest of this page is what you look *up* while reading what it tells you.

## Reading the report

A clean file is one line:

```text
minirec20230622_.nwb: no_op, already ingested, nothing to do
```

Anything else leads with the same verdict, then the per-table counts, then what
it found, worst first. For example, a file missing a camera device might look
like this:

```text
docsdemo20230622_.nwb: partial_new, 190 new entries

New entries:
      4  common_behav.RawPosition.PosObject
      2  common_behav.RawPosition
      4  common_behav.PositionSource.SpatialSeries
      2  common_behav.PositionSource
      3  common_dio.DIOEvents
     32  common_ephys.ElectrodeGroup
    128  common_ephys.Electrode
      1  common_ephys.Raw
      1  common_ephys.SampleCount
      6  common_interval.IntervalList
      1  common_sensors.SensorData
      1  common_session.Session.DataAcquisitionDevice
      2  common_session.Session.Experimenter
      1  common_session.Session
      1  lfp_electrode.LFPElectrodeGroup
      1  lfp_imported.ImportedLFP

Disagrees with stored rows (1):
  common_device.CameraDevice (1):
    {'camera_name': 'test camera 1'} exists with different values for
      ['manufacturer']
      stored values: {'manufacturer': 'Allied Vision'}
  -> Edit the file to match, or update the rows if the file is right. Re-run to
     accept the stored values and ingest the rest.

Hard (1):
  [hard] common_task.TaskEpoch: parse_error: No camera device found with ID [1]
    in NWB file docsdemo20230622_.nwb. Known camera IDs: [0] (object
    0fe587d1-c276-49c3-8386-90d8a0df18a9)
      -> A table raised while parsing. Fetch `error_raw` from
         `IngestionPlanLog.Problem` for the traceback.

Blocked by the above (3): common_behav.StateScriptFile, common_behav.VideoFile,
  common_optogenetics.OptogeneticProtocol
```

Read top to bottom, the verdict says 190 entries would still be written broken
down by table. The disagreement is reported without stopping anything.

A `parse_error` and a `missing_camera` name the NWB `object_id` they were
reading, so you can find it in the file. Problems found while checking a row,
such as `missing_parent`, name the row's key instead. Each *code* gets a
one-line remedy, printed once per group rather than under every occurrence.

### Verdicts

| Verdict       | Meaning                                  |
| ------------- | ---------------------------------------- |
| `fatal`       | The file could not be read at all        |
| `no_op`       | Everything in the file is already stored |
| `all_new`     | Nothing in the file is stored yet        |
| `partial_new` | Some of it is stored; the rest is new    |

### Severities

| Severity | Blocks? | Means                                                   |
| -------- | ------- | ------------------------------------------------------- |
| `fatal`  | yes     | the file itself: unreadable, or not registered          |
| `hard`   | yes     | one table, and usually everything that depends on it    |
| `soft`   | no      | an absence that is not an error, e.g. an unplaced video |
| `info`   | no      | something was resolved; shown only with `verbose=True`  |

A divergence is the exception to `hard`: it blocks the write, but the tables
below it still plan, because the row it disagrees with exists.

### Blocked tables

If a parent table could not be ingested, child tables (with foreign key
references to that parent) are reported as blocked at the end, pointing to the
parent table's root cause.

### When the file disagrees with an existing row

An entry whose primary key already exists with different values is a
**divergence**. A re-ingested camera might give its manufacturer as
`manufacturer-1` where the stored row says `manufacturer 1`.

On a file's first plan this blocks, so nothing is written until you have seen
it. Edit the file, or update the row if the file is right. Re-running accepts
the stored values and ingests the rest, reporting the divergence as a warning.

## NWB to Spyglass table mappings

*In progress*: To aid in creating spyglass-compatable NWB files, we provide a
[Reference Table](../ForDevelopers/ingestion_mapping.md) which maps spyglass
table entries to the source nwb objects and attributes.

For entries not yet contained in the updated format, a complete list of this
mappings can also be found [here](../ForDevelopers/UsingNWB.md).

## Under the hood

How the two passes are arranged, when the database is touched, what makes a
parse reusable, and what the log stores:
[Ingestion Plans](../ForDevelopers/IngestionPlans.md).
