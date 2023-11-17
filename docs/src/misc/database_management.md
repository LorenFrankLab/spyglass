# Database Management

While Spyglass can help you organize your data, there are a number of things
you'll need to do to manage users, database backups, and file cleanup.

Some these tasks should be set to run regularly.
[Cron jobs](https://www.hostinger.com/tutorials/cron-job) can help with
automation.

## MySQL Version

The Frank Lab's database is running MySQL 8.0 with a number of custom
configurations set by our system admin to reflect UCSF's IT security
requirements.

DataJoint's default docker container for MySQL is version 5.7. As the Spyglass
team has hit select compatibility issues, we've worked with the DataJoint team
to update the open source package to support MySQL 8.0.

While the Spyglass team won't be able to support earlier versions, if you run
into any issues declaring Spyglass tables with an 8.0 instance, please let us
know.

## User Management

The [DatabaseSettings](../api/utils/database_settings.md) class provides a
number of methods to help you manage users. By default, it will write out a
temporary `.sql` file and execute it on the database.

### Privileges

DataJoint schemas correspond to MySQL databases. Privileges are managed by
schema/database prefix.

- `SELECT` privileges allow users to read, write, and delete data.
- `ALL` privileges allow users to create, alter, or drop tables and schemas in
  addition to operations above.

In practice, DataJoint only permits alerations of secondary keys on existing
tables, and more derstructive operations would require using DataJoint to
execeute MySQL commands.

Shared schema prefixes are those defined in the Spyglass package (e.g.,
`common`, `lfp`, etc.). A 'user schema' is any schema with the username as
prefix. User types differ in the privileges they are granted on these prifixes.

### Users types

- `collab_user`: `ALL` on user schema, `SELECT` on all other schemas.
- `dj_guest`: `SELECT` on all schemas.
- `dj_user`: `ALL` on shared and user schema, `SELECT` on all other schemas.

### Setting Passwords

New users are generated with the password `temppass`. In order to change this,
we recommend downloading DataJoint `0.14.2` (currently pre-release).

```console
git clone https://github.com/datajoint/datajoint-python/
pip install ./datajoint-python
```

Then, you the user can reset within Python:

```python
import datajoint as dj

dj.set_password()
```

## Database Backups

Coming soon...

## File Cleanup

Spyglass is designed to hold metadata for analyses that reference NWB files on
disk. There are several tables that retain lists of files that have been
generated during analyses. If someone deletes analysis entries, files will still
be on disk.

To remove orphaned files, we run the following commands in our cron jobs:

```python
from spyglass.common import AnalysisNwbfile
from spyglass.spikesorting import SpikeSorting


def main():
    AnalysisNwbfile().nightly_cleanup()
    SpikeSorting().nightly_cleanup()
```
