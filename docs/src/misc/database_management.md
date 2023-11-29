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
