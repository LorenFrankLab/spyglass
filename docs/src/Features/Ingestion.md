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
      4  `common_behav`.`_raw_position__pos_object`
      2  `common_behav`.`_raw_position`
      4  `common_behav`.`position_source__spatial_series`
      2  `common_behav`.`position_source`
      3  `common_dio`.`_d_i_o_events`
     32  `common_ephys`.`_electrode_group`
    128  `common_ephys`.`_electrode`
      1  `common_ephys`.`_raw`
      1  `common_ephys`.`_sample_count`
      6  `common_interval`.`interval_list`
      1  `common_sensors`.`_sensor_data`
      1  `common_session`.`_session__data_acquisition_device`
      2  `common_session`.`_session__experimenter`
      1  `common_session`.`_session`
      1  `lfp_electrode`.`l_f_p_electrode_group`
      1  `lfp_imported`.`_imported_l_f_p`

Disagrees with stored rows (1):
  `common_device`.`camera_device` (1):
    {'camera_name': 'test camera 1'} exists with different values for ['manufacturer']
      stored values: {'manufacturer': 'Allied Vision'}
  -> The stored rows were kept. Edit the file to match, or update the rows, if the file is right.

Hard (1):
  [hard] `common_task`.`_task_epoch`: parse_error: No camera device found with ID [1] in NWB file docsdemo20230622_.nwb. Known camera IDs: [0]
      -> A table raised while parsing. The traceback names where.

Blocked by the above (3): `common_behav`.`_state_script_file`, `common_behav`.`_video_file`, `common_optogenetics`.`optogenetic_protocol`
```

One root cause, read top to bottom: the verdict says 190 entries would still be
written, the counts say which tables they belong to, the disagreement is
reported without stopping anything, and `TaskEpoch` failing accounts for the
three tables below it rather than producing three more problems.

Each problem names the NWB `object_id` where there is one, so you can find the
thing in the file. Each *code* gets a one-line remedy, printed once per group
rather than under every occurrence.

### Verdicts

| Verdict       | Meaning                                  |
| ------------- | ---------------------------------------- |
| `fatal`       | The file could not be read at all        |
| `no_op`       | Everything in the file is already stored |
| `all_new`     | Nothing in the file is stored yet        |
| `partial_new` | Some of it is stored; the rest is new    |

`fatal` is answered before the others. A file that could not be read works out
no rows, so counting them would say "nothing new". That reads as *already
ingested*, the opposite of what happened.

### Severities

| Severity | Blocks? | Means                                                   |
| -------- | ------- | ------------------------------------------------------- |
| `fatal`  | yes     | the file itself: unreadable, or not registered          |
| `hard`   | yes     | one table, and everything that depends on it            |
| `soft`   | no      | an absence that is not an error, e.g. an unplaced video |
| `info`   | no      | something was resolved; shown only with `verbose=True`  |

Soft problems are shown by default: a warning you have to ask for is not a
warning.

### Blocked tables

A table whose parent could not be planned is reported once, at the end, as
blocked, not as a failure of its own. One root cause produces one problem,
rather than one per table downstream of it.

### When the file disagrees with a stored row

An entry whose primary key already exists with different values is a
**divergence**. A re-ingested camera might give its manufacturer as
`manufacturer-1` where the stored row says `manufacturer 1`.

Nothing stops to ask. The stored value is kept, the rest of the file ingests,
and the report names the row, the columns that differ, and the revision that
would align them. It gets its own section because it does not stop anything, so
one disagreement does not retitle a file that was otherwise entirely new. The
record is kept, so you can read it after the run rather than during.

## NWB to Spyglass table mappings

*In progress*: To aid in creating spyglass-compatable NWB files, we provide a
[Reference Table](../ForDevelopers/ingestion_mapping.md) which maps spyglass
table entries to the source nwb objects and attributes.

For entries not yet contained in the updated format, a complete list of this
mappings can also be found [here](../ForDevelopers/UsingNWB.md).

## Under the hood

How the two passes are arranged, how a table joins in, what makes a parse
reusable, and what the plan log stores:
[Ingestion Plans](../ForDevelopers/IngestionPlans.md).
