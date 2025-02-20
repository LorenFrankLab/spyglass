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
3. Copy the `example.env` file to `.env` in the `maintenance_scripts` directory
    and fill in the necessary information, including...
    - `SPYGLASS_CONDA_ENV`: the name of the conda environment with Spyglass and
        DataJoint installed.
    - `SPYGLASS_REPO_PATH`: the path to the Spyglass repository.
    - `SPYGLASS_LOG`: the path to the log file.
    - Optional email settings. If not set, email notifications will not be sent.
        - `SPYGLASS_EMAIL_SRC`: The email address from which to send notifications.
        - `SPYGLASS_EMAIL_PASS`: the password for the email address.
        - `SPYGLASS_EMAIL_DEST`: the email address to which to send notifications.
4. Set up a cron job to run `run_jobs.sh` at the desired interval by running
    `crontab -e` and adding the script.

Note that the log file will automatically be truncated to `SPYGLASS_MAX_LOG`
lines on each run. 1000 lines should be sufficient.

### Example Cron Job

In the following example, the script is set to run every Monday at 4:00 AM.

```text
0 4 * * 1 /path/to/run_jobs.sh
```

### Email Service

The script uses `curl` to send email notifications on failure. While this can
work with
[many email services](https://everything.curl.dev/usingcurl/smtp.html), Gmail is
a common choice. To use Gmail, you will need to ...

1. Turn on [2-step verification](https://myaccount.google.com/security-checkup)
2. Turn on [less secure apps](https://myaccount.google.com/lesssecureapps)
3. Create an [app password](https://myaccount.google.com/apppasswords)

`curl` will not work with your master Gmail password, so you will need to use
the app password instead.
