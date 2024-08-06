# Export Process

## Why

DataJoint does not have any built-in functionality for exporting vertical slices
of a database. A lab can maintain a shared DataJoint pipeline across multiple
projects, but conforming to NIH data sharing guidelines may require that data
from only one project be shared during publication.

## Requirements

To export data with the current implementation, you must do the following:

- All custom tables must inherit from `SpyglassMixin` (e.g.,
    `class MyTable(SpyglassMixin, dj.ManualOrOther):`)
- Only one export can be active at a time.
- Start the export process with `ExportSelection.start_export()`, run all
    functions associated with a given analysis, and end the export process with
    `ExportSelection.end_export()`.

## How

The current implementation relies on two classes in the Spyglass package
(`SpyglassMixin` and `RestrGraph`) and the `Export` tables.

- `SpyglassMixin`: See `spyglass/utils/dj_mixin.py`
- `RestrGraph`: See `spyglass/utils/dj_graph.py`
- `Export`: See `spyglass/common/common_usage.py`

### Mixin

The `SpyglassMixin` class adds functionality to DataJoint tables. A subset of
methods are used to set an environment variable, `SPYGLASS_EXPORT_ID`, and,
while active, intercept all `fetch`/`fetch_nwb` calls to tables. When `fetch` is
called, the mixin grabs the table name and the restriction applied to the table
and stores them in the `ExportSelection` part tables.

- `fetch_nwb` is specific to Spyglass and logs all analysis nwb files that are
    fetched.
- `fetch` is a DataJoint method that retrieves data from a table.

### Graph

The `RestrGraph` class uses DataJoint's networkx graph to store each of the
tables and restrictions intercepted by the `SpyglassMixin`'s `fetch` as
'leaves'. The class then cascades these restrictions up from each leaf to all
ancestors. Use is modeled in the methods of `ExportSelection`.

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

- Create a modified version of `SpyglassMixin`, including ...
    - `_export_table` method to lazy load an export table like `ExportSelection`
    - `export_id` attribute, plus setter and deleter methods, to manage the status
        of the export.
    - `fetch` and other methods to intercept and log exported content.
- Create a modified version of `ExportSelection`, that adjusts fields like
    `spyglass_version` to match the new database.

Or, optionally, you can use the `RestrGraph` class to cascade hand-picked tables
and restrictions without the background logging of `SpyglassMixin`. The
assembled list of restricted free tables, `RestrGraph.all_ft`, can be passed to
`Export.write_export` to generate a shell script for exporting the data.
