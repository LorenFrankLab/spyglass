#!/bin/bash
# This script will...
# 1. Read variables from a .env file
# 2. Loop through $SPACE_CHECK_DRIVES
# 3. Compare available space relative to $SPACE_LIMIT
# 4. If above, send an email to $SPACE_EMAIL_RECIPIENTS

# Load env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" # load environment variables from this directory

# Check for required variables
if [[ -z "${SPACE_CHECK_DRIVES}" \
  || -z "${SPACE_EMAIL_SRC}" \
  || -z "${SPACE_EMAIL_PASS}" \
  || -z "${SPACE_EMAIL_RECIPIENTS}" \
  || -z "${SPACE_DRIVE_LIMITS}" \
  || -z "${SPACE_LOG}" ]] ; then
  echo "Error: Missing one or more variables required for check_disk_space.sh"
  exit 1
fi

# Inputs to arrays
IFS=' ' read -r -a DRIVE_LIST <<< "$SPACE_CHECK_DRIVES"
IFS=' ' read -r -a LIMIT_LIST <<< "$SPACE_DRIVE_LIMITS"

if [[ "${#SPACE_CHECK_DRIVES[@]}" -ne "${#SPACE_DRIVE_LIMITS[@]}" ]]; then
  echo "Error: Number of drives does not match number of limits"
  exit 1
fi

echo "SPACE CHECK: $(date)" > "$SPACE_LOG"

# Email template
EMAIL_TEMPLATE=$(cat <<-EOF
From: "Spyglass" <$SPACE_EMAIL_SRC>
To: %s
Subject: Drive almost full: %s

%s
EOF
)

# Send email alert
send_email_message() {
  local RECIPIENT="$1"
  local SUBJECT="$2"
  local BODY="$3"
  EMAIL=$(printf "$EMAIL_TEMPLATE" "$RECIPIENT" "$SUBJECT" "$BODY")
  curl -s --url "smtps://smtp.gmail.com:465" \
      --ssl-reqd \
      --user "$SPACE_EMAIL_SRC:$SPACE_EMAIL_PASS" \
      --mail-from "$SPACE_EMAIL_SRC" \
      --mail-rcpt "$RECIPIENT" \
      -T <(echo "$EMAIL")
}

# Send slack message
send_slack_message() {
  if [[ -z "$SLACK_TOKEN" || -z "$SLACK_CHANNEL" ]]; then
    return 0
  fi
  local MESSAGE="$1"
  curl -d "text=$MESSAGE" \
    -d "channel=$SLACK_CHANNEL" \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -X POST https://slack.com/api/chat.postMessage
}

# Find the longest drive name for padding
MAX_DRIVE_LEN=0
for DRIVE in $SPACE_CHECK_DRIVES; do
    LEN=${#DRIVE}
    [[ $LEN -gt $MAX_DRIVE_LEN ]] && MAX_DRIVE_LEN=$LEN
done

# Check each drive
for i in "${!DRIVE_LIST[@]}"; do
    DRIVE="${DRIVE_LIST[$i]}"
    LIMIT="${LIMIT_LIST[$i]}"

    # Convert limit to bytes
    SPACE_LIMIT_BYTES=$(numfmt --from=iec "$LIMIT")

    # Skip nonexistent drives
    if [[ ! -d "$DRIVE" ]]; then
        echo "WARNING: Drive $DRIVE not found. Skipping." >> "$SPACE_LOG"
        continue
    fi

    # Get free space in bytes
    FREE_BYTES=$(df --output=avail "$DRIVE" | awk 'NR==2 {print $1}')000
    TOTAL_BYTES=$(df --output=size "$DRIVE" | awk 'NR==2 {print $1}')000
    FREE_HUMAN=$(numfmt --to=iec "$FREE_BYTES")
    TOTAL_HUMAN=$(numfmt --to=iec "$TOTAL_BYTES")

    # Log with left-padded drive names
    NAME="${DRIVE:1}" # assumes first char is `/`
    printf "%-*s: %s/%s\n" "$MAX_DRIVE_LEN" "$NAME" \
        "$FREE_HUMAN" "$TOTAL_HUMAN" >> "$SPACE_LOG"

    # Do nothing if under capacity
    if [[ "$FREE_BYTES" -gt "$SPACE_LIMIT_BYTES" ]]; then
        continue
    fi

    # Send email alert
    BODY="Low space warning: ${NAME} has ${FREE_HUMAN}/${TOTAL_HUMAN} free"
    SUBJ="LOW SPACE: ${NAME}"

    send_slack_message "$BODY"

    for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
        send_email_message "$RECIPIENT" "$SUBJ" "$BODY"
    done
done

