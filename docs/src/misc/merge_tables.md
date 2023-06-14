# Merge Tables

## Why

A pipeline may diverge when we want to process the same data in different ways.
Merge Tables allow us to join divergent pipelines together, and unify
downstream processing steps. For a more in depth discussion, please refer to
[this notebook](https://github.com/ttngu207/db-programming-with-datajoint/blob/master/notebooks/pipelines_merging_design_master_part.ipynb)
and related discussion [here](https://github.com/datajoint/datajoint-python/issues/151).

**Note:** Deleting entries upstream of Merge Tables will throw errors related to
deleteing a part entry before the master. To circumvent this, add
`force_parts=True` to the
[`delete` function](https://datajoint.com/docs/core/datajoint-python/0.14/api/datajoint/__init__/#datajoint.table.Table.delete)
call.

## What

A Merge Table is fundametally a master table with one part for each divergent
pipeline. By convention...

1. The master table has one primary key, `merge_id`, a [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier).

2. Each part table has inherits the final table in its respective pipeline, and
   shares the same name as this table.

```python
from spyglass.utils.dj_merge_tables import Merge

@schema
class MergeTable(Merge):
    definition = """
    merge_id: uuid
    """

    class One(dj.Part):
        definition = """
        -> master
        ---
        -> One
        """

    class Two(dj.Part):
        definition = """
        -> master
        ---
        -> Two
        """
```

## How

The Merge class in Spyglass's utils is a subclass of DataJoint's
[Manual Table](https://datajoint.com/docs/core/design/tables/tiers/#data-entry-lookup-and-manual)
and adds functions to make the awkwardness of part tables more manageable. These
functions are described in the API section, under `utils.dj_merge_tables`.
