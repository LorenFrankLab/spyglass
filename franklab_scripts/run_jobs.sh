#!/bin/bash

# SETUP:
# 1. Create a conda environment with datajoint installed.
# 2. Edit the variables below to match your setup.
# 3. Set up the cron job (See README.md)
# Note that the log file will be truncated to the last 1000 lines.

SPYGLASS_CONDA_ENV=spy
SPYGLASS_REPO_PATH=/home/cb/wrk/spyglass/
SPYGLASS_LOG=/home/cb/wrk/spyglass/jobs.log

exec >> $SPYGLASS_LOG 2>&1

# print the date and time
echo "SPYGLASS CRON JOB START: $(date +"%Y-%m-%d %H:%M:%S")"

# Run from the root of the spyglass repository
cd $SPYGLASS_REPO_PATH || \
    { echo "Error: Could not change to the spyglass directory"; exit 1; }

# Update the spyglass repository
git pull https://github.com/LorenFrankLab/spyglass.git master > /dev/null || \
    { echo "Error: $PWD Could not update the spyglass repository"; exit 1; }

# Test conda environment
if ! conda env list | grep -q $SPYGLASS_CONDA_ENV; then
    echo "Error: Conda environment $SPYGLASS_CONDA_ENV not found"
    exit 1
fi

# convenience function to run a command in the spyglass conda environment
conda_run() { conda run --name $SPYGLASS_CONDA_ENV "$@"; }

# Test connection to the database
conda_run python -c "import datajoint as dj; dj.conn()" > /dev/null || \
    { echo "Error: Could not connect to the database"; exit 1; }

# Run cleanup script
conda_run python franklab_scripts/cleanup.py

echo "SPYGLASS CRON JOB END"

# truncate long log file
tail -n 1000 "$SPYGLASS_LOG" > "${SPYGLASS_LOG}.tmp" && \
  mv "${SPYGLASS_LOG}.tmp" "$SPYGLASS_LOG"
