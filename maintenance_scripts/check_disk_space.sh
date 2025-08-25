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
  || -z "${SPACE_LOG}" \
  || -z "${SPACE_ROOT_NAME}" \
  || -z "${SPACE_EMAIL_ON_PASS}" ]] ; then
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

echo "SPACE CHECK: $(date)" >> "$SPACE_LOG"

# Email template
EMAIL_TEMPLATE=$(cat <<-EOF
From: "Spyglass" <$SPACE_EMAIL_SRC>
To: %s
Subject: %s

%s
EOF
)

# Send email alert
send_email_message() {
  local RECIPIENT="$1"
  local SUBJECT="$2"
  local BODY="$3"
  EMAIL=$(printf "$EMAIL_TEMPLATE" "$RECIPIENT" "$SUBJECT" "$BODY")
  curl -sS --url "smtps://smtp.gmail.com:465" \
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
    -X POST https://slack.com/api/chat.postMessage \
    2>> "$SPACE_LOG"
}

# Find the longest drive name for padding
MAX_DRIVE_LEN=0
for DRIVE in $SPACE_CHECK_DRIVES; do
    LEN=${#DRIVE}
    [[ $LEN -gt $MAX_DRIVE_LEN ]] && MAX_DRIVE_LEN=$LEN
done

OUTPUT=""
FOUND_ISSUE=0

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

    # Get free space in bytes. Multiply by 1024 to convert kilo -> bytes.
    FREE_BYTES=$(df --output=avail "$DRIVE" | awk 'NR==2 {print $1 * 1024}')
    TOTAL_BYTES=$(df --output=size "$DRIVE" | awk 'NR==2 {print $1 * 1024}')
    FREE_HUMAN=$(numfmt --to=iec "$FREE_BYTES")
    TOTAL_HUMAN=$(numfmt --to=iec "$TOTAL_BYTES")

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
    OUTPUT+="$line\n"

    # Do nothing if under capacity
    if [[ "$FREE_BYTES" -gt "$SPACE_LIMIT_BYTES" ]]; then
        continue
    fi

    FOUND_ISSUE=1

    # Send email alert
    BODY="Low space warning: ${NAME} has ${FREE_HUMAN}/${TOTAL_HUMAN} free"
    SUBJ="${NAME}"

    echo $BODY >> "$SPACE_LOG"

    send_slack_message "$BODY"

    for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
        send_email_message "$RECIPIENT" "ALMOST FULL: $SUBJ" "$BODY"
    done

done

echo -e "$OUTPUT" >> "$SPACE_LOG"

if [[ "$SPACE_EMAIL_ON_PASS" == "true" ]] && [[ "$FOUND_ISSUE" == "0" ]]; then
  for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
      send_email_message "$RECIPIENT" "Disk Space OK" "$OUTPUT"
  done
fi
