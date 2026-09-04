# Database Management

While Spyglass can help you organize your data, there are a number of things
you'll need to do to manage users, database backups, and file cleanup.

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

In practice, DataJoint only permits alterations of secondary keys on existing
tables, and more derstructive operations would require using DataJoint to
execeute MySQL commands.

Shared schema prefixes are those defined in the Spyglass package (e.g.,
`common`, `lfp`, etc.). A 'user schema' is any schema with the username as
prefix. User types differ in the privileges they are granted on these prifixes.
Declaring a table with the SpyglassMixin on a schema other than a shared module
or the user's own prefix will raise a warning.

### Users roles

When a database is first initialized, the team should run `add_roles` to create
the following roles:

- `dj_guest`: `SELECT` on all schemas.
- `dj_collab`: `ALL` on user schema, `SELECT` on all other schemas.
- `dj_user`: `ALL` on shared and user schema, `SELECT` on all other schemas.
- `dj_admin`: `ALL` on all schemas.

If new shared modules are introduced, the `add_module` method should be used to
expand the privileges of the `dj_user` role.

### Setting Passwords

New users are generated with the password `temppass`. In order to change this,
we recommend downloading DataJoint `0.14.2` or later.

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

The following codeblockes are a series of files used to back up our database and
migrate the contents to another server. Some conventions to note:

- `.host`: files used in the host's context
- `.container`: files used inside the database Docker container
- `.env`: files used to set environment variables used by the scripts for
    database name, backup name, and backup credentials

This backup process uses a dedicated backup user, that an admin would need to
criate with the relevant permissions.

### mysql.env.host

<details>
<summary>MySQL host environment variables</summary>

Values may be adjusted as needed for different building images.

```bash
ROOT_PATH=/usr/local/containers/mysql # path to this container's working area

# variables for building image
SRC=ubuntu
VER=20.04
DOCKERFILE=Dockerfile.base

# variables for referencing image
IMAGE=mysql8
TAG=u20
# variables for running the container
CNAME=mysql-datajoint
MACADDR=4e:b0:3d:42:e0:70
RPORT=3306

# variables for initializing/relaunching the container
# - where the mysql data and backups will live - these values
# are examples
DB_PATH=/data/db
DB_DATA=mysql
DB_BACKUP=/data/mysql-backups

# backup info
BACK_USER=mysql-backup
BACK_PW={password}
BACK_DBNAME={database}
# mysql root password - make sure to remove this AFTER the container
# is initialized - and this file will be replicated inside the container
# on initialization, so remove it from there: /opt/bin/mysql.env
```

</details>

### backup-database.sh.host

This script runs the mysql-backup container script (exec inside the container)
that dumps the database contents for each database as well as the entire
database. Use cron to set this to run on your desired schedule.

<details>
<summary>MySQL host docker exec</summary>

```bash
#!/bin/bash

PRIOR_DIR=$(pwd)
cd /usr/local/containers/mysql || exit
. mysql.env
cd "$(dirname ${ROOT_PATH})"
#
docker exec ${CNAME} /opt/bin/mysql-backup.csh
#
cd "$(dirname ${DB_BACKUP})"
#
cd ${PRIOR_DIR}
```

</details>

### mysql-backup-xfer.csh.host

This script transfers the backup to another server 'X' and is specific for us as
it uses passwordless ssh keys to a local unprivileged user on X that has the
mysql backup area on X as that user's home.

<details>
<summary>MySQL host transfer script</summary>

```bash
#!/bin/csh
set td=`date +"%Y%m%d"`
cd /data/mysql-backups
scp -P {port} -i ~/mysql-backup -r ${database}-${td} mysql-backup@${X}:~/
/bin/rm -r lmf-db-${td}
```

</details>

### myenv.csh.container

<details>
<summary>Docker container environment variables</summary>

```bash
set db_backup=mysql-backups
set back_user=mysql-backup
set back_pw={password}
set back_dbname={database}
```

</details>

### mysql-backup.csh.container

<details>
<summary>Generate backups from within container</summary>

```bash
#!/bin/csh
source /opt/bin/myenv.csh
set td=`date +"%Y%m%d"`
cd /${db_backup}
mkdir ${back_dbname}-${td}

set list=`echo "show databases;" | mysql --user=${back_user} --password=${back_pw}`
set cnt=0

foreach db ($list)
  if ($cnt == 0) then
    echo "dumping mysql databases on $td"
  else
    echo "dumping MySQL database : $db"
    # Per-schema backups
    mysqldump $db --max_allowed_packet=512M --user=${back_user} --password=${back_pw} > /${db_backup}/${back_dbname}-${td}/mysql.${db}.sql
  endif
@ cnt = $cnt + 1
end
# Full database backup
mysqldump --all-databases --max_allowed_packet=512M --user=${back_user} --password=${back_pw} > /${db_backup}/${back_dbname}-${td}/mysql-all.sql
```

</details>

## Table/File Cleanup

Spyglass is designed to hold metadata for analyses that reference NWB files on
disk. There are several tables that retain lists of files that have been
generated during analyses. If someone deletes analysis entries, files will still
be on disk.

**NOTE**: This means that directories like analysis and recording are managed
resources. Adding files to these directories outside of Spyglass will not
automatically register them in the database, and they will be deleted.

Additionally, there are key tables such as `IntervalList` and `AnalysisNwbfile`,
which are used to store entries created by downstream tables. These entries are
not always deleted when the downstream entry is removed, creating 'orphans'.

`IntervalList` relies on a string primary key uniqueness. This could cause
issues if a user were to (a) run a `make` function on a computed table that
generates a new `IntervalList` entry, then (b) delete the computed entry but not
the `IntervalList` entry, then (c) run the `make` function again. If this `make`
function were set up to skip duplicates, it may cause the new computed entry to
attach to the old `IntervalList` entry. While all Spyglass `make`s are
idempotent (using `replace` or throwing errors on duplicates), user custom
`make` functions may not be. Spyglass takes the additional precaution of
removing all `IntervalList` orphan entries with each delete call.

Similar orphan cleanups for `Nwbfile`, `AnalysisNwbfile`, `SpikeSorting`, and
`DecodingOutput` are not as critical and can be run less frequently.

### Automated Cleanup (Admin)

For database administrators, Spyglass provides automated cleanup scripts
designed to run as cron jobs. These scripts handle all cleanup operations
including orphan detection, external file deletion, and temp directory cleanup.

**Location**: `maintenance_scripts/`

**Key Scripts**:

- `cleanup.py` - Main cleanup script that performs:
    - Table cleanups (`Nwbfile`, `AnalysisNwbfile`, `SpikeSorting`,
        `DecodingOutput`, `SpikeSortingRecording`)
    - External file deletion (unreferenced files)
    - Temp directory cleanup (files older than 7 days)
    - Version table updates (fetches latest from PyPI)
- `run_jobs.sh` - Orchestration script that:
    - Updates Spyglass repository from master branch
    - Runs database connection check
    - Executes `cleanup.py`
    - Manages logging and notifications
- `check_disk_space.sh` - Monitors disk usage and sends alerts

**Setup**:

1. Configure environment variables in `maintenance_scripts/.env`:

    ```bash
    SPYGLASS_BASE_PATH=/path/to/data
    SPYGLASS_CONDA_ENV=spyglass
    SPYGLASS_REPO_PATH=/path/to/spyglass
    SPYGLASS_LOG=/path/to/cleanup.log
    # Optional: email/slack notifications
    ```

2. Set up cron jobs (edit with `crontab -e`):

    ```bash
    # Run cleanup every Monday at 4:00 AM
    0 4 * * 1 /path/to/spyglass/maintenance_scripts/run_jobs.sh

    # Check disk space daily at 8:00 AM
    0 8 * * * /path/to/spyglass/maintenance_scripts/check_disk_space.sh
    ```

**Email/Slack Notifications**: The scripts can send notifications on errors or
disk space issues. See
[maintenance_scripts/README.md](https://github.com/LorenFrankLab/spyglass/blob/master/maintenance_scripts/README.md)
for detailed setup instructions.

### Manual Cleanup (Programmatic)

For one-off cleanup operations or testing, you can run cleanup methods directly
from Python. This is useful for debugging or when automated scripts aren't
appropriate.

```python
from spyglass.common import Nwbfile, AnalysisNwbfile
from spyglass.spikesorting.v0 import SpikeSorting, SpikeSortingRecording
from spyglass.decoding import DecodingOutput

# Cleanup operations
Nwbfile().cleanup()  # Remove unreferenced raw NWB files
AnalysisNwbfile().cleanup()  # Remove orphaned analysis files (see below)
SpikeSorting().cleanup(verbose=False)  # Remove unreferenced sorting directories
SpikeSortingRecording().cleanup(verbose=False)  # Remove untracked folders
DecodingOutput().cleanup()  # Remove unreferenced .nc and .pkl files
```

**Analysis File Cleanup**: See dedicated section below for details on
coordinated cleanup across common and custom `AnalysisNwbfile` tables.

______________________________________________________________________

## Analysis File Cleanup

Spyglass provides a cleanup system for managing analysis NWB files across both
the common `AnalysisNwbfile` table and team-specific custom tables. This system
detects and removes orphaned files that are no longer referenced by any
downstream tables.

**Note**: For automated cleanup as part of cron jobs, see "Automated Cleanup
(Admin)" section above. This section covers the programmatic API for manual or
scripted cleanup.

### Overview

The cleanup system handles:

- **Orphaned files**: Files with no downstream foreign key references
- **Uninserted files**: Files created but never added to tables
- **Multi-table coordination**: Works across common and all custom
    `AnalysisNwbfile` tables
- **Empty files**: Old, untracked NWB files with 0 bytes are removed

### Running Cleanup

Use the common `AnalysisNwbfile` table to clean up all analysis files:

```python
from spyglass.common import AnalysisNwbfile

# Preview cleanup across all tables (common + custom)
AnalysisNwbfile().cleanup(dry_run=True)

# Apply the cleanup after reviewing the dry-run output
AnalysisNwbfile().cleanup(dry_run=False)
```

The dry run reports aggregate target counts and logical candidate bytes rather
than a per-path manifest.

**Important**: Cleanup automatically coordinates across all custom
`AnalysisNwbfile` tables. A file is only deleted if it's not referenced by ANY
table (common or custom).

**Warning**: This is a destructive operation that permanently deletes files.
Ensure you have backups before running cleanup on production databases. This
operation treats the analysis directory as a managed resource.

### How It Works

Cleanup discovers common and custom analysis tables, snapshots tracked paths
once, scans the filesystem, validates the complete deletion plan, and unlinks
the candidates. It next removes orphan rows and unused DataJoint external
entries. Files made orphaned by that database phase are handled on the next run.

### Safety Features

- **Tracked and recent files are retained**: tracked paths are fetched once
    across all registered common and custom tables. Files newer than
    `min_file_age_hours` (default 24), based on target modification time, are
    deferred and reported. Pass `min_file_age_hours=0` only for intentional
    immediate cleanup. This age gate applies to the `*.nwb` filesystem sweep,
    not DataJoint's custom-table external cleanup later in the run.
- **Deletion limits catch a wrong directory or large unexpected backlog**:
    destructive cleanup refuses a plan above `max_delete_fraction` (default
    0.9) or `max_delete_to_tracked_ratio` (default 10.0). These limits apply to
    untracked or empty analysis NWB files; foreign keys continue to govern
    orphan-row deletion.
- **Leaf symlinks reclaim cross-volume analysis storage**: an old, untracked
    `*.nwb` leaf symlink authorizes deletion of both its recorded regular-file
    target and the link. Dangling links lose only the link. Directory symlinks
    are **not** traversed (`followlinks=False`): cleanup deletes files, so it
    must not follow a symlinked subdirectory out of `analysis_dir` into an
    unrelated store. Only leaf `*.nwb` symlinks are eligible, and a symlinked
    `analysis_dir` root is still scanned.
- **The analysis directory is trusted**: cleanup intentionally follows the
    normal Spyglass snapshot-and-delete model rather than defending against
    concurrent filesystem or database changes. A leaf `*.nwb` symlink below
    `analysis_dir` can authorize deletion outside that directory, including
    beneath another configured store, because there is no protected-store
    denylist. Restrict write access to the analysis tree and do not run
    cleanup concurrently with analysis writers or registration.
- **Insert blocking refuses ambiguous ownership**: cleanup checks registered
    analysis tables for existing blockers before proceeding; a destructive run
    then installs temporary `BEFORE INSERT` triggers. If a blocker already
    exists, previews and destructive runs both refuse because they cannot tell
    whether another cleanup is active or the trigger is stale. First
    confirm that no cleanup is running; only then inspect the triggers and use
    `AnalysisRegistry().unblock_new_inserts()` if every blocking trigger is
    stale. That helper removes all registered analysis blocking triggers.

**Concurrency limit**: the trigger check is not a database-wide cleanup lease,
and triggers do not carry per-run ownership. Do not overlap cleanup runs.
The tracked-path and filesystem snapshots can also become stale before unlink,
so do not concurrently register or mutate eligible analysis paths.

### Custom Tables

If you've created custom `AnalysisNwbfile` tables (see
[Custom Analysis Files](./CustomAnalysisFiles.md)), cleanup works automatically.
No special configuration needed - just run cleanup on the common table and it
handles all custom tables.
