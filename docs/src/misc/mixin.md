# Spyglass Mixin

The Spyglass Mixin provides a way to centralize all Spyglass-specific
functionalities that have been added to DataJoint tables. This includes...

- Fetching NWB files
- Delete functionality, including permission checks and part/master pairs
- Export logging. See [export doc](export.md) for more information.

To add this functionality to your own tables, simply inherit from the mixin:

```python
import datajoint as dj
from spyglass.utils import SpyglassMixin

schema = dj.schema('my_schema')

@schema
class MyOldTable(dj.Manual):
    pass

@schema
class MyNewTable(SpyglassMixin, dj.Manual):)
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
Merge entries and delete them in the correct order. The mixin provides a
function, `delete_downstream_merge`, to handle this, which is run by default
when calling `delete`.

`delete_downstream_merge`, also aliased as `ddm`, identifies all Merge tables
downsteam of where it is called. If `dry_run=True`, it will return a list of
entries that would be deleted, otherwise it will delete them.

Importantly, `delete_downstream_merge` cannot properly interact with tables that
have not been imported into the current namespace. If you are having trouble
with part deletion errors, import the offending table and rerun the function
with `reload_cache=True`.

```python
from spyglass.common import Nwbfile

restricted_nwbfile = Nwbfile() & "nwb_file_name LIKE 'Name%'"
restricted_nwbfile.delete_downstream_merge(dry_run=False)
# DataJointError("Attempt to delete part table MyMerge.Part before ...

from spyglass.example import MyMerge

restricted_nwbfile.delete_downstream_merge(reload_cache=True, dry_run=False)
```

Because each table keeps a cache of downsteam merge tables, it is important to
reload the cache if the table has been imported after the cache was created.
Speed gains can also be achieved by avoiding re-instancing the table each time.

```python
# Slow
from spyglass.common import Nwbfile

(Nwbfile() & "nwb_file_name LIKE 'Name%'").ddm(dry_run=False)
(Nwbfile() & "nwb_file_name LIKE 'Other%'").ddm(dry_run=False)

# Faster
from spyglass.common import Nwbfile

nwbfile = Nwbfile()
(nwbfile & "nwb_file_name LIKE 'Name%'").ddm(dry_run=False)
(nwbfile & "nwb_file_name LIKE 'Other%'").ddm(dry_run=False)
```
