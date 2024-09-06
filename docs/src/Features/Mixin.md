# Spyglass Mixin

The Spyglass Mixin provides a way to centralize all Spyglass-specific
functionalities that have been added to DataJoint tables. This includes...

- Fetching NWB files
- Long-distance restrictions.
- Permission checks on delete
- Export logging. See [export doc](./Export.md) for more information.
- Miscellaneous helper functions

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
[Merge Tables](./Merge.md)). This burrying makes it difficult to restrict a
given table by familiar attributes.

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

## Delete Permission Checks

By default, DataJoint is unable to set delete permissions on a per-table basis.
If a user is able to delete entries in a given table, she can delete entries in
any table in the schema.

The mixin relies on the `Session.Experimenter` and `LabTeams` tables to ...

1. Check the session and experimenter associated with the attempted deletion.
2. Check the lab teams associated with the session experimenter and the user.

If the user shares a lab team with the session experimenter, the deletion is
permitted.

This is not secure system and is not a replacement for database backups (see
[database management](../ForDevelopers/Management.md)). A user could readily
curcumvent the default permission checks by adding themselves to the relevant
team or removing the mixin from the class declaration. However, it provides a
reasonable level of security for the average user.

Because parts of this process rely on caching, this process will be faster if
you assign the instanced table to a variable.

```python
# Slower
YourTable().delete()
YourTable().delete()

# Faster
nwbfile = YourTable()
nwbfile.delete()
nwbfile.delete()
```

<details><summary>Deprecated delete feature</summary>

Previous versions of Spyglass also deleted masters of parts with foreign key
references. This functionality has been migrated to DataJoint in version 0.14.2
via the `force_masters` delete argument. This argument is `True` by default in
Spyglass tables.

</details>

## Populate Calls

The mixin also overrides the default `populate` function to provide additional
functionality for non-daemon process pools and disabling transaction protection.

### Non-Daemon Process Pools

To allow the `make` function to spawn a new process pool, the mixin overrides
the default `populate` function for tables with `_parallel_make` set to `True`.
See [issue #1000](https://github.com/LorenFrankLab/spyglass/issues/1000) and
[PR #1001](https://github.com/LorenFrankLab/spyglass/pull/1001) for more
information.

### Disable Transaction Protection

By default, DataJoint wraps the `populate` function in a transaction to ensure
data integrity (see
[Transactions](https://docs.datajoint.io/python/definition/05-Transactions.html)).

This can cause issues when populating large tables if another user attempts to
declare/modify a table while the transaction is open (see
[issue #1030](https://github.com/LorenFrankLab/spyglass/issues/1030) and
[DataJoint issue #1170](https://github.com/datajoint/datajoint-python/issues/1170)).

Tables with `_use_transaction` set to `False` will not be wrapped in a
transaction when calling `populate`. Transaction protection is replaced by a
hash of upstream data to ensure no changes are made to the table during the
unprotected populate. The additional time required to hash the data is a
trade-off for already time-consuming populates, but avoids blocking other users.

## Miscellaneous Helper functions

`file_like` allows you to restrict a table using a substring of a file name.
This is equivalent to the following:

```python
MyTable().file_like("eg")
MyTable() & ('nwb_file_name LIKE "%eg%" OR analysis_file_name LIKE "%eg%"')
```

`find_insert_fail` is a helper function to find the cause of an `IntegrityError`
when inserting into a table. This checks parent tables for required keys.

```python
my_key = {"key": "value"}
MyTable().insert1(my_key)  # Raises IntegrityError
MyTable().find_insert_fail(my_key)  # Shows the parent(s) missing the key
```

## Populate Calls

The mixin also overrides the default `populate` function to provide additional
functionality for non-daemon process pools and disabling transaction protection.

### Non-Daemon Process Pools

To allow the `make` function to spawn a new process pool, the mixin overrides
the default `populate` function for tables with `_parallel_make` set to `True`.
See [issue #1000](https://github.com/LorenFrankLab/spyglass/issues/1000) and
[PR #1001](https://github.com/LorenFrankLab/spyglass/pull/1001) for more
information.

### Disable Transaction Protection

By default, DataJoint wraps the `populate` function in a transaction to ensure
data integrity (see
[Transactions](https://docs.datajoint.io/python/definition/05-Transactions.html)).

This can cause issues when populating large tables if another user attempts to
declare/modify a table while the transaction is open (see
[issue #1030](https://github.com/LorenFrankLab/spyglass/issues/1030) and
[DataJoint issue #1170](https://github.com/datajoint/datajoint-python/issues/1170)).

Tables with `_use_transaction` set to `False` will not be wrapped in a
transaction when calling `populate`. Transaction protection is replaced by a
hash of upstream data to ensure no changes are made to the table during the
unprotected populate. The additional time required to hash the data is a
trade-off for already time-consuming populates, but avoids blocking other users.
