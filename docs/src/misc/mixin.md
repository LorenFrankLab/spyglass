# Spyglass Mixin

The Spyglass Mixin provides a way to centralize all Spyglass-specific
functionalities that have been added to DataJoint tables. This includes...

- Fetching NWB files
- Long-distance restrictions.
- Delete functionality, including permission checks and part/master pairs
- Export logging. See [export doc](export.md) for more information.

To add this functionality to your own tables, simply inherit from the mixin:

```python
import datajoint as dj

from spyglass.utils import SpyglassMixin

schema = dj.schema("my_schema")


@schema
class MyOldTable(dj.Manual):
    pass


@schema
class MyNewTable(SpyglassMixin, dj.Manual):
    pass
```

**NOTE**: The mixin must be the first class inherited from in order to override
default DataJoint functions.

## Fetching NWB Files

Many tables in Spyglass inheris from central tables that store records of NWB
files. Rather than adding a helper function to each table, the mixin provides a
single function to access these files from any table.

```python
from spyglass.example import AnyTable

(AnyTable & my_restriction).fetch_nwb()
```

This function will look at the table definition to determine if the raw file
should be fetched from `Nwbfile` or an analysis file should be fetched from
`AnalysisNwbfile`. If neither is foreign-key-referenced, the function will refer
to a `_nwb_table` attribute.

## Long-Distance Restrictions

In complicated pipelines like Spyglass, there are often tables that 'bury' their
foreign keys as secondary keys. This is done to avoid having to pass a long list
of foreign keys through the pipeline, potentially hitting SQL limits (see also
[Merge Tables](./merge_tables.md)). This burrying makes it difficult to restrict
a given table by familiar attributes.

Spyglass provides a function, `restrict_by`, to handle this. The function takes
your restriction and checks parents/children until the restriction can be
applied. Spyglass introduces `<<` as a shorthand for `restrict_by` an upstream
key and `>>` as a shorthand for `restrict_by` a downstream key.

```python
from spyglass.example import AnyTable

AnyTable() << 'upstream_attribute="value"'
AnyTable() >> 'downstream_attribute="value"'

# Equivalent to
AnyTable().restrict_by('downstream_attribute="value"', direction="down")
AnyTable().restrict_by('upstream_attribute="value"', direction="up")
```

Some caveats to this function:

1. 'Peripheral' tables, like `IntervalList` and `AnalysisNwbfile` make it hard
    to determine the correct parent/child relationship and have been removed
    from this search by default.
2. This function will raise an error if it attempts to check a table that has
    not been imported into the current namespace. It is best used for exploring
    and debugging, not for production code.
3. It's hard to determine the attributes in a mixed dictionary/string
    restriction. If you are having trouble, try using a pure string
    restriction.
4. The most direct path to your restriction may not be the path your data took,
    especially when using Merge Tables. When the result is empty see the
    warning about the path used. Then, ban tables from the search to force a
    different path.

```python
my_table = MyTable()  # must be instanced
my_table.ban_search_table(UnwantedTable1)
my_table.ban_search_table([UnwantedTable2, UnwantedTable3])
my_table.unban_search_table(UnwantedTable3)
my_table.see_banned_tables()

my_table << my_restriction
my_table << upstream_restriction >> downstream_restriction
```

When providing a restriction of the parent, use 'up' direction. When providing a
restriction of the child, use 'down' direction.

## Delete Functionality

The mixin overrides the default `delete` function to provide two additional
features.

### Permission Checks

By default, DataJoint is unable to set delete permissions on a per-table basis.
If a user is able to delete entries in a given table, she can delete entries in
any table in the schema.

The mixin relies on the `Session.Experimenter` and `LabTeams` tables to ...

1. Check the session and experimenter associated with the attempted deletion.
2. Check the lab teams associated with the session experimenter and the user.

If the user shares a lab team with the session experimenter, the deletion is
permitted.

This is not secure system and is not a replacement for database backups (see
[database management](./database_management.md)). A user could readily
curcumvent the default permission checks by adding themselves to the relevant
team or removing the mixin from the class declaration. However, it provides a
reasonable level of security for the average user.

### Master/Part Pairs

By default, DataJoint has protections in place to prevent deletion of a part
entry without deleting the corresponding master. This is useful for enforcing
the custom of adding/removing all parts of a master at once and avoids orphaned
masters, or null entry masters without matching data.

For [Merge tables](./merge_tables.md), this is a significant problem. If a user
wants to delete all entries associated with a given session, she must find all
part table entries, including Merge tables, and delete them in the correct
order. The mixin provides a function, `delete_downstream_parts`, to handle this,
which is run by default when calling `delete`.

`delete_downstream_parts`, also aliased as `ddp`, identifies all part tables
with foreign key references downstream of where it is called. If `dry_run=True`,
it will return a list of entries that would be deleted, otherwise it will delete
them.

Importantly, `delete_downstream_parts` cannot properly interact with tables that
have not been imported into the current namespace. If you are having trouble
with part deletion errors, import the offending table and rerun the function
with `reload_cache=True`.

```python
import datajoint as dj
from spyglass.common import Nwbfile

restricted_nwbfile = Nwbfile() & "nwb_file_name LIKE 'Name%'"

vanilla_dj_table = dj.FreeTable(dj.conn(), Nwbfile.full_table_name)
vanilla_dj_table.delete()
# DataJointError("Attempt to delete part table MyMerge.Part before ... ")

restricted_nwbfile.delete()
# [WARNING] Spyglass: No part deletes found w/ Nwbfile ...
# OR
# ValueError("Please import MyMerge and try again.")

from spyglass.example import MyMerge

restricted_nwbfile.delete_downstream_parts(reload_cache=True, dry_run=False)
```

Because each table keeps a cache of downstream merge tables, it is important to
reload the cache if the table has been imported after the cache was created.
Speed gains can also be achieved by avoiding re-instancing the table each time.

```python
# Slow
from spyglass.common import Nwbfile

(Nwbfile() & "nwb_file_name LIKE 'Name%'").ddp(dry_run=False)
(Nwbfile() & "nwb_file_name LIKE 'Other%'").ddp(dry_run=False)

# Faster
from spyglass.common import Nwbfile

nwbfile = Nwbfile()
(nwbfile & "nwb_file_name LIKE 'Name%'").ddp(dry_run=False)
(nwbfile & "nwb_file_name LIKE 'Other%'").ddp(dry_run=False)
```
