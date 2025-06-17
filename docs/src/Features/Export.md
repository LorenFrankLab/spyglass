# Export Process

## Why

DataJoint does not have any built-in functionality for exporting vertical slices
of a database. A lab can maintain a shared DataJoint pipeline across multiple
projects, but conforming to NIH data sharing guidelines may require that data
from only one project be shared during publication.

## Requirements

To export data with the current implementation, you must do the following:

- All custom tables must inherit from either `SpyglassMixin` or `ExportMixin`
    (e.g., `class MyTable(SpyglassMixin, dj.ManualOrOther):`)
- Only one export can be active at a time for a given Python instance.
- Start the export process with `ExportSelection.start_export()`, run all
    functions associated with a given analysis, and end the export process with
    `ExportSelection.end_export()`.

## How

The current implementation relies on two classes in the Spyglass package
(`ExportMixin` and `RestrGraph`) and the `Export` tables.

- `ExportMixin`: See `spyglass/utils/mixins/export.py`
- `RestrGraph`: See `spyglass/utils/dj_graph.py`
- `Export`: See `spyglass/common/common_usage.py`

### Mixin

The `ExportMixin` class adds functionality to DataJoint tables. A subset of
methods are used to set an environment variable, `SPYGLASS_EXPORT_ID`, and,
while active, intercept all `fetch`, `fetch_nwb`, `restrict` and `join` calls to
tables. When these functions are called, the mixin grabs the table name and the
restriction applied to the table and stores them in the `ExportSelection` part
tables.

- `fetch_nwb` is specific to Spyglass and logs all analysis nwb files that are
    fetched.
- `fetch` is a DataJoint method that retrieves data from a table.
- `restrict` is a DataJoint method that restricts a table to a subset of data,
    typically using the `&` operator.
- `join` is a DataJoint method that joins two tables together, typically using
    the `*` operator.

This is designed to capture any way that Spyglass is accessed, including
restricting one table via a join with another table. If this process seems to be
missing a way that Spyglass is accessed in your pipeline, please let us know.

Note that logging all restrictions may log more than is necessary. For example,
`MyTable & restr1 & restr2` will log `MyTable & restr1` and `MyTable & restr2`,
despite returning the combined restriction. Logging will treat compound
restrictions as 'OR' instead of 'AND' statements. This can be avoided by
combining restrictions before using the `&` operator.

```python
MyTable & "a = b" & "c > 5"  # Will capture 'a = b' OR 'c > 5'
MyTable & "a = b AND c > 5"  # Will capture 'a = b AND c > 5'
MyTable & dj.AndList(["a = b", "c > 5"])  # Will capture 'a = b AND c > 5'
```

If this process captures too much, you can either run a process with logging
disabled, or delete these entries from `ExportSelection` after the export is
logged.

Disabling logging with the `log_export` flag:

```python
MyTable().fetch(log_export=False)
MyTable().fetch_nwb(log_export=False)
MyTable().restrict(restr, log_export=False)  # Instead of MyTable & restr
MyTable().join(Other, log_export=False)  # Instead of MyTable * Other
```

### Graph

The `RestrGraph` class uses DataJoint's networkx graph to store each of the
tables and restrictions intercepted by the `ExportMixin`'s `fetch` as 'leaves'.
The class then cascades these restrictions up from each leaf to all ancestors.
Use is modeled in the methods of `ExportSelection`.

```python
from spyglass.utils.dj_graph import RestrGraph

restr_graph = RestrGraph(seed_table=AnyTable, leaves=None, verbose=False)
restr_graph.add_leaves(
    leaves=[
        {
            "table_name": MyTable.full_table_name,
            "restriction": "any_restriction",
        },
        {
            "table_name": AnotherTable.full_table_name,
            "restriction": "another_restriction",
        },
    ]
)
restr_graph.cascade()
restricted_leaves = restr_graph.leaf_ft
all_restricted_tables = restr_graph.all_ft
```

By default, a `RestrGraph` object is created with a seed table to have access to
a DataJoint connection and graph. One or more leaves can be added at
initialization or later with the `add_leaves` method. The cascade process is
delayed until `cascade`, or another method that requires the cascade, is called.

Cascading a single leaf involves transforming the leaf's restriction into its
parent's restriction, then repeating the process until all ancestors are
reached. If two leaves share a common ancestor, the restrictions are combined.
This process also accommodates projected fields, which appear as numeric alias
nodes in the graph.

### Export Table

The `ExportSelection` is where users should interact with this process.

```python
from spyglass.common.common_usage import ExportSelection
from spyglass.common.common_usage import Export

export_key = {paper_id: "my_paper_id", analysis_id: "my_analysis_id"}
ExportSelection().start_export(**export_key)
analysis_data = (MyTable & my_restr).fetch()
analysis_nwb = (MyTable & my_restr).fetch_nwb()
ExportSelection().end_export()

# Visual inspection
touched_files = ExportSelection.list_file_paths(**export_key)
restricted_leaves = ExportSelection.preview_tables(**export_key)

# Export
Export().populate_paper(**export_key)
```

`Export`'s populate will invoke the `write_export` method to collect cascaded
restrictions and file paths in its part tables, and write out a bash script to
export the data using a series of `mysqldump` commands. The script is saved to
Spyglass's directory, `base_dir/export/paper_id/`, using credentials from
`dj_config`. To use alternative credentials, create a
[mysql config file](https://dev.mysql.com/doc/refman/8.0/en/option-files.html).

To retain the ability to delete the logging from a particular analysis, the
`export_id` is a combination of the `paper_id` and `analysis_id` in
`ExportSelection`. When populated, the `Export` table, only the maximum
`export_id` for a given `paper_id` is used, resulting in one shell script per
paper. Each shell script one `mysqldump` command per table.

## External Implementation

To implement an export for a non-Spyglass database, you will need to ...

- Create a modified version of `ExportMixin`, including ...
    - `_export_table` method to lazy load an export table like `ExportSelection`
    - `export_id` attribute, plus setter and deleter methods, to manage the status
        of the export.
    - `fetch` and other methods to intercept and log exported content.
- Create a modified version of `ExportSelection`, that adjusts fields like
    `spyglass_version` to match the new database.

Or, optionally, you can use the `RestrGraph` class to cascade hand-picked tables
and restrictions without the background logging of `ExportMixin`. The assembled
list of restricted free tables, `RestrGraph.all_ft`, can be passed to
`Export.write_export` to generate a shell script for exporting the data.

## Backwards Compatibility

Spyglass databases that were declared before varchars were reduced to
accommodate MySQL key length restrictions (see
[#630](https://github.com/LorenFrankLab/spyglass/issues/630)) will have trouble
importing exported data. Specifically, varchar mismatches will throw foreign key
errors. To fix this, you run the following bash script on the generated `sql`
files before importing them into the new database.

<details><summary>Script</summary>

```bash
#!/bin/bash

for file in ./_Pop*sql; do \
    echo $file
    sed -i 's/ DEFAULT CHARSET=[^ ]\w*//g' "$file"
    sed -i 's/ DEFAULT COLLATE [^ ]\w*//g' "$file"
    sed -i 's/ `nwb_file_name` varchar(255)/ `nwb_file_name` varchar(64)/g' "$file"
    sed -i 's/ `analysis_file_name` varchar(255)/ `analysis_file_name` varchar(64)/g' "$file"
    sed -i 's/ `interval_list_name` varchar(200)/ `interval_list_name` varchar(170)/g' "$file"
    sed -i 's/ `position_info_param_name` varchar(80)/ `position_info_param_name` varchar(32)/g' "$file"
    sed -i 's/ `mark_param_name` varchar(80)/ `mark_param_name` varchar(32)/g' "$file"
    sed -i 's/ `artifact_removed_interval_list_name` varchar(200)/ `artifact_removed_interval_list_name` varchar(128)/g' "$file"
    sed -i 's/ `metric_params_name` varchar(200)/ `metric_params_name` varchar(64)/g' "$file"
    sed -i 's/ `auto_curation_params_name` varchar(200)/ `auto_curation_params_name` varchar(36)/g' "$file"
    sed -i 's/ `sort_interval_name` varchar(200)/ `sort_interval_name` varchar(64)/g' "$file"
    sed -i 's/ `preproc_params_name` varchar(200)/ `preproc_params_name` varchar(32)/g' "$file"
    sed -i 's/ `sorter` varchar(200)/ `sorter` varchar(32)/g' "$file"
    sed -i 's/ `sorter_params_name` varchar(200)/ `sorter_params_name` varchar(64)/g' "$file"
done
```

</details>

This is essentially a series of `sed` commands that adjust varchar lengths to
their updated values. This script should be run in the directory containing the
`_Populate*.sql` files generated by the export process.
