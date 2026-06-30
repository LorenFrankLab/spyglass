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
        removing orphans, unreferenced files, and old temp files. It also
        maintains an inventory of all analysis files and checks for issues.
    - The function for cleaning up temp files requires that this directory be
        called either `temp` or `tmp`.
    - This script also fetches the latest version information from PyPI to update
        the `SpyglassVersions` table.
- `populate.py` - This script provides an example of how to run computations as
    part of cron jobs. This is not currently in use.
- `run_jobs.sh` - This script ...
    - Reads from the `.env` file in the same directory.
    - Updates the spyglass repository, fetching from the master branch.
    - Runs a database connection check (relying on a valid datajoint config).
    - Runs the `cleanup.py` script.
- `check_disk_space.sh` - This script ...
    - Reads from the same `.env` file as `run_jobs.sh`.
    - Checks the disk space of each drive in `SPACE_CHECK_DRIVES`.
    - If free space drops below the per-drive limit in `SPACE_DRIVE_LIMITS`, sends
        an email to each recipient in `SPACE_EMAIL_RECIPIENTS` and posts a Slack
        alert.
    - Appends a structured CSV row to `SPACE_CSV_LOG` on each run (if set).
    - Appends a data runway prediction (via `predict_runway.py`) to all reports
        when `SPACE_CSV_LOG` is set.
    - Posts a full disk space summary to Slack every Monday.
    - Accepts `--dry-run`: prints all output and alerts to stdout without writing
        to the log file, sending emails, or appending to the CSV.
- `predict_runway.py` - Reads `SPACE_CSV_LOG`, fits a linear trend to recent
    `used_bytes` for a target drive, and prints a one-line runway estimate (e.g.
    `data runway: ~17 months`). Accepts the CSV path as an argument or via
    `SPACE_CSV_LOG`; target drive and window are tunable via env vars.
- `script_utils.sh` - Shared messaging utilities sourced by both shell scripts.
    Provides `send_email_message` and `send_slack_message`.

## Setup

1. Clone the repository to the desired location.
2. Set up a config file by copying `dj_local_conf_example.json` to
    `dj_local_conf.json` and filling in the necessary information.
3. Copy the `example.env` file to `.env` in the `maintenance_scripts` directory
    and fill in the necessary information, including...
4. Set up a cron job to run `run_jobs.sh` at the desired interval by running
    - Items for running cleanup jobs:
        - `SPYGLASS_CHMOD_FILES`: if `true`, the script will set the permissions of
            all files in the data directory to 644. This is limited to files
            generated in the last week to save time. If `false`, the script will
            skip this step.[^1]
        - `SPYGLASS_BASE_PATH`: the path to the Spyglass-managed data.
        - `SPYGLASS_CONDA_PATH`: Path to conda initialization script. To find the
            root directory, run `which conda` and follow the relative path in
            `example.env`
        - `SPYGLASS_CONDA_ENV`: the name of the conda environment with Spyglass and
            DataJoint installed.[^2]
        - `SPYGLASS_REPO_PATH`: the path to the Spyglass repository.
        - `SPYGLASS_LOG`: the path to the log file.
        - Optional email settings. If not set, email notifications will not be
            sent.
            - `SPYGLASS_EMAIL_SRC`: The email address from which to send
                notifications.
            - `SPYGLASS_EMAIL_PASS`: the password for the email address.
            - `SPYGLASS_EMAIL_DEST`: the email address to which to send
                notifications.
    - Items for checking disk space:
        - `TZ`: the timezone to use for reporting local times.
        - `SPACE_CHECK_DRIVES`: space-separated list of filesystem paths to check.
        - `SPACE_DRIVE_LIMITS`: space-separated IEC thresholds (e.g. `10T 3G`), one
            per drive in the same order as `SPACE_CHECK_DRIVES`. An alert fires
            when free space falls below the corresponding threshold.
        - `SPACE_ROOT_NAME`: display name used for the root (`/`) drive.
        - `SPACE_LOG`: path to the plain-text log file.
        - `SPACE_EMAIL_SRC`/`SPACE_EMAIL_PASS`: email sender credentials.
        - `SPACE_EMAIL_ON_PASS`: if `true`, send an email summary even when all
            drives are healthy. If `false`, only email on alerts.
        - `SPACE_EMAIL_RECIPIENTS`: space-separated list of recipient addresses.
    - Items for CSV logging and runway prediction (optional):
        - `SPACE_CSV_LOG`: path for the structured CSV log. When set,
            `check_disk_space.sh` appends one row per drive per run and calls
            `predict_runway.py` to include a runway estimate in all reports.
        - `SPACE_RUNWAY_PATH`: drive path to predict runway for (default:
            `/stelmo/nwb`).
        - `SPACE_RUNWAY_DAYS`: rolling window in days for the trend fit (default:
            `90`).
        - `SPACE_RUNWAY_MIN`: shortest fallback window tried before reporting
            `stable` (default: `30`).
    - Items for posting to slack:
        - `SLACK_TOKEN`: the token for the slack app.
        - `SLACK_CHANNEL`: the channel to post to.
5. Set up a cron job to run each shell script at the desired interval by running
    `crontab -e` and adding the script.

Note that the log file will automatically be truncated to `SPYGLASS_MAX_LOG`
lines on each run. 1000 lines should be sufficient.

To enable slack notifications, you will need to create a slack app and generate
a token following the instructions
[here](https://api.slack.com/tutorials/tracks/posting-messages-with-curl). For
posting to a private channel, you will need to invite the app to the relevant
channel before attempting to post.

### Example Cron Jobs

In the following example, the cleanup script is set to run every Monday at 4:00
AM, and the disk space check is set to run every day at 8:00 AM.

```text
0 4 * * 1 /path/to/run_jobs.sh
0 8 * * * /path/to/check_disk_space.sh
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

[^1]: Depending your system, you may need to run the script as `sudo` to set the
    permissions.

[^2]: You may want to run the cronjob from a dedicated conda environment to
    avoid issues with local editable installs or other package conflicts.
