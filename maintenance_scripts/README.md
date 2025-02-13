# Frank Lab Maintenance Scripts

Scripts in this directory are used to maintain the database, set to run
regularly as cron jobs.

## Scripts

- `alter_table.py`
    - This script is used to alter all tables according to the latest Spyglass
        definitions.
    - If the lab follows instructions in release notes, this is unnecessary.
- `cleanup.py`
    - This script performs various cleanup operations on the database, including
        removing orphans, unreferenced files, and old temp files.
    - The function for cleaning up temp files requires that this directory be
        called either `temp` or `tmp`.
- `populate.py` - This script provides an example of how to run computations as
    part of cron jobs. This is not currently in use.
- `run_jobs.sh` - This script ...
    - Updates the spyglass repository, fetching from the master branch.
    - Runs a database connection check (relying on a valid datajoint config).
    - Runs the `cleanup.py` script.

## Setup

1. Clone the repository to the desired location.
2. Set up a config file by copying `dj_local_conf_example.json` to
    `dj_local_conf.json` and filling in the necessary information.
3. Set up a cron job to run `run_jobs.sh` at the desired interval by running
    `crontab -e` and adding the script.

In the following example, the script is set to run every Monday at 4:00 AM.

```text
0 4 * * 1 /path/to/run_jobs.sh
```
