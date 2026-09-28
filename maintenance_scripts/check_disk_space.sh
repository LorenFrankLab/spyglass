#!/bin/bash
# This script will...
# 1. Read variables from a .env file
# 2. Loop through $SPACE_CHECK_DRIVES
# 3. Compare available space relative to $SPACE_LIMIT
# 4. If above, send an email to $SPACE_EMAIL_RECIPIENTS

# Load env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

# Parse flags
DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# Check for required variables
if [[ -z "${SPACE_CHECK_DRIVES}" \
  || -z "${SPACE_EMAIL_SRC}" \
  || -z "${SPACE_EMAIL_PASS}" \
  || -z "${SPACE_EMAIL_RECIPIENTS}" \
  || -z "${SPACE_DRIVE_LIMITS}" \
  || -z "${SPACE_LOG}" \
  || -z "${SPACE_ROOT_NAME}" \
  || -z "${SPACE_EMAIL_ON_PASS}" ]] ; then
  echo "Error: Missing one or more variables required for check_disk_space.sh"
  exit 1
fi

# --- Initialize variables ---
IFS=' ' read -r -a DRIVE_LIST <<< "$SPACE_CHECK_DRIVES"
IFS=' ' read -r -a LIMIT_LIST <<< "$SPACE_DRIVE_LIMITS"

if [[ "${#DRIVE_LIST[@]}" -ne "${#LIMIT_LIST[@]}" ]]; then
  echo "Error: Number of drives does not match number of limits"
  exit 1
fi

LOGFILE="$SPACE_LOG"
RUNWAY_CSV="${SPACE_CSV_LOG:-}"  # preserve for runway read
RUN_TS=$(date --iso-8601=seconds)
OUTPUT=""
FOUND_ISSUE=0
MAX_DRIVE_LEN=0

# Calculate max drive name length for padding
for DRIVE in $SPACE_CHECK_DRIVES; do
    LEN=${#DRIVE}
    [[ $LEN -gt $MAX_DRIVE_LEN ]] && MAX_DRIVE_LEN=$LEN
done

[[ -n "${SPYGLASS_CONDA_PATH:-}" ]] && source "$SPYGLASS_CONDA_PATH"
PYTHON="python3"
[[ -n "${SPYGLASS_CONDA_ENV:-}" ]] && \
  PYTHON="conda run --no-capture-output -n $SPYGLASS_CONDA_ENV python3"

# Dry-run: route output to stdout, suppress CSV writes.
# RUNWAY_CSV retains the original path so the runway read still works.
if [[ "$DRY_RUN" == "true" ]]; then
    LOGFILE=/dev/stdout
    SPACE_CSV_LOG=""
fi

source "$SCRIPT_DIR/script_utils.sh"
SLACK_LOG="$LOGFILE"

# In dry-run mode, replace send functions with stdout printers
if [[ "$DRY_RUN" == "true" ]]; then
    send_email_message() {
        echo "[DRY RUN] EMAIL to: $1"
        echo "  Subject: $2"
        echo "  Body: $3"
    }
    send_slack_message() {
        echo "[DRY RUN] SLACK: $1"
    }
fi

# Initialize CSV log with header if it doesn't exist
if [[ -n "${SPACE_CSV_LOG:-}" ]] && [[ ! -f "$SPACE_CSV_LOG" ]]; then
    echo "timestamp,path,avail_bytes,total_bytes" > "$SPACE_CSV_LOG"
fi

echo "SPACE CHECK: $(date)" >> "$LOGFILE"

# Check each drive
for i in "${!DRIVE_LIST[@]}"; do
    DRIVE="${DRIVE_LIST[$i]}"
    LIMIT="${LIMIT_LIST[$i]}"

    # Convert limit to bytes
    SPACE_LIMIT_BYTES=$(numfmt --from=iec "$LIMIT")

    # Skip nonexistent drives
    if [[ ! -d "$DRIVE" ]]; then
        echo "WARNING: Drive $DRIVE not found. Skipping." >> "$LOGFILE"
        continue
    fi

    # Get free space in bytes. Multiply by 1024 to convert kilo -> bytes.
    FREE_BYTES=$(df --output=avail "$DRIVE" | awk 'NR==2 {print $1 * 1024}')
    TOTAL_BYTES=$(df --output=size "$DRIVE" | awk 'NR==2 {print $1 * 1024}')
    FREE_HUMAN=$(numfmt --to=iec "$FREE_BYTES")
    TOTAL_HUMAN=$(numfmt --to=iec "$TOTAL_BYTES")

    if [[ -n "${SPACE_CSV_LOG:-}" ]]; then
        echo "$RUN_TS,$DRIVE,$FREE_BYTES,$TOTAL_BYTES" >> "$SPACE_CSV_LOG"
    fi

    # Log with left-padded drive names
    if [[ "$DRIVE" == "/" ]]; then
        NAME="$SPACE_ROOT_NAME" # Use custom name for root
    else
        NAME="${DRIVE:1}" # Assumes first char is `/`
    fi
    line=$(\
      printf "%-*s: %s/%s\n" \
      "$MAX_DRIVE_LEN" "$NAME" "$FREE_HUMAN" "$TOTAL_HUMAN"\
    )
    OUTPUT+="$line
"

    # Do nothing if under capacity
    if [[ "$FREE_BYTES" -gt "$SPACE_LIMIT_BYTES" ]]; then
        continue
    fi

    FOUND_ISSUE=1

    BODY="Low space warning: ${NAME} has ${FREE_HUMAN}/${TOTAL_HUMAN} free"
    SUBJ="${NAME}"

    echo "$BODY" >> "$LOGFILE"

    send_slack_message "$BODY"

    for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
        send_email_message "$RECIPIENT" "ALMOST FULL: $SUBJ" "$BODY"
    done

done

# Append runway prediction if a readable CSV exists
if [[ -n "$RUNWAY_CSV" ]] && [[ -f "$RUNWAY_CSV" ]]; then
    RUNWAY=$($PYTHON "$SCRIPT_DIR/predict_runway.py" "$RUNWAY_CSV" 2>>"$LOGFILE")
    OUTPUT+="$RUNWAY
"
fi

# NOTE: `echo` may cause issues on alternative shells.
# If needed, can extend to use `printf` instead.
echo "$OUTPUT" >> "$LOGFILE"

# Send full disk space report via Slack every Monday
if [[ "$(date +%u)" == "1" ]]; then
  send_slack_message "DISK SPACE:
  $OUTPUT"
fi

if [[ "$SPACE_EMAIL_ON_PASS" == "true" ]] && [[ "$FOUND_ISSUE" == "0" ]]; then
  for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
      send_email_message "$RECIPIENT" "Disk Space OK" "$OUTPUT"
  done
fi
