#!/bin/bash
# AUTHOR: Chris Brozdowski
# DATE: 2025-02-20
#
# 1. Go to SPYGLASS_REPO_PATH and pull the latest changes from the master branch
# 2. Test the SPYGLASS_CONDA_ENV conda environment
# 3. Test the connection to the database
# 4. Run the cleanup script
#
# This script is intended to be run as a cron job, weekly or more frequently.
# It will store a log of its output in SPYGLASS_LOG.
# If any of the operations fail, an email will be sent to SPYGLASS_EMAIL_DEST

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" # load environment variables from this directory

if [[ -z "${SPYGLASS_BASE_PATH}"
  || -z "${SPYGLASS_CONDA_PATH}" \
  || -z "${SPYGLASS_CONDA_ENV}" \
  || -z "${SPYGLASS_REPO_PATH}" \
  || -z "${SPYGLASS_LOG}" ]]; then
  echo "Error: the followimg must be set in an .env:
        SPYGLASS_BASE_PATH, SPYGLASS_CONDA_PATH, SPYGLASS_CONDA_ENV,
        SPYGLASS_REPO_PATH, and SPYGLASS_LOG"
  exit 1
fi

source $SPYGLASS_CONDA_PATH

EMAIL_TEMPLATE=$(cat <<-EOF
From: "Spyglass" <$SPYGLASS_EMAIL_SRC>
To: $SPYGLASS_EMAIL_DEST
Subject: cron fail - $(date "+%Y-%m-%d")

%s
EOF
)

on_fail() { # $1: error message. Echo message and send as email
    echo "Error: $1"
    if [ -z "$SPYGLASS_EMAIL_SRC" ]; then
      return 1 # No email source, so don't send an email
    fi
    local error_msg="$1"
    local content
    content=$(printf "$EMAIL_TEMPLATE" "$error_msg")

    curl -sS -o /dev/null --ssl-reqd \
      --url "smtps://smtp.gmail.com:465" \
      --user "${SPYGLASS_EMAIL_SRC}:${SPYGLASS_EMAIL_PASS}" \
      --mail-from "$SPYGLASS_EMAIL_SRC" \
      --mail-rcpt "$SPYGLASS_EMAIL_DEST" \
      -T <(echo "$content")
}

exec >> $SPYGLASS_LOG 2>&1

# print the date and time
echo "SPYGLASS CRON JOB START: $(date +"%Y-%m-%d %H:%M:%S")"

# Run from the root of the spyglass repository
cd $SPYGLASS_REPO_PATH || \
    { on_fail "Could not find repo path: $SPYGLASS_REPO_PATH"; exit 1; }


# Update the spyglass repository
git pull --quiet \
  https://github.com/LorenFrankLab/spyglass.git master > /dev/null || \
    { on_fail "Could not update the spyglass repo $PWD"; exit 1; }

# Test conda environment
if ! conda env list | grep -q $SPYGLASS_CONDA_ENV; then
  on_fail "Conda environment $SPYGLASS_CONDA_ENV not found"
  exit 1
fi

# convenience function to run a command in the spyglass conda environment
conda_run() { conda run --name $SPYGLASS_CONDA_ENV "$@"; }

# Test connection to the database
CONN_TEST="import datajoint as dj; dj.logger.setLevel('ERROR'); dj.conn()"
conda_run python -c "$CONN_TEST" > /dev/null || \
  { on_fail "Could not connect to the database"; exit 1; }

# Chmod new files in past 2 days, requires sudo
if $SPYGLASS_CHMOD_FILES; then
  find $SPYGLASS_BASE_PATH -type f -mtime -2 -exec chmod 644 {} \; || \
    { on_fail "Could not chmod new files in $SPYGLASS_BASE_PATH"; exit 1; }
fi

# Run cleanup script
conda_run python maintenance_scripts/cleanup.py

echo "SPYGLASS CRON JOB END"

# truncate long log file
tail -n ${SPYGLASS_MAX_LOG:-1000} "$SPYGLASS_LOG" > "${SPYGLASS_LOG}.tmp" \
  && mv "${SPYGLASS_LOG}.tmp" "$SPYGLASS_LOG"
