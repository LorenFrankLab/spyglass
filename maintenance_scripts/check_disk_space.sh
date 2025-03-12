#!/bin/bash
# This script will...
# 1. Read variables from a .env file
# 2. Loop through $SPACE_CHECK_DRIVES
# 3. Compare percent usage against $SPACE_PERCENT_LIMIT
# 4. If above, send an email to $SPACE_EMAIL_RECIPIENTS

# Load env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" # load environment variables from this directory
timestamp=$(date +"%Y-%m-%d %H:%M:%S %Z")

if [[ -z "${SPACE_CHECK_DRIVES}" \
  || -z "${SPACE_EMAIL_SRC}" \
  || -z "${SPACE_EMAIL_PASS}" \
  || -z "${SPACE_EMAIL_RECIPIENTS}" \
  || -z "${SPACE_PERCENT_LIMIT}" \
  || -z "${SPACE_LOG}" ]] ; then
  echo "Error: Missing one or more variables required for check_disk_space.sh"
  exit 1
fi

EMAIL_TEMPLATE=$(cat <<-EOF
From: "Spyglass" <$SPACE_EMAIL_SRC>
To: %s
Subject: Drive almost full: %s

%s
EOF
)


# Check each drive
for DRIVE in $SPACE_CHECK_DRIVES; do
    NAME="${DRIVE:1}" # assumes first char is `/`

    # TODO: reduce number of slow dh calls
    USAGE_PERCENT=$(df -h "$DRIVE" | awk 'NR==2 {print $5}' | tr -d '%')
    TOTAL_TB=$(df -h --output=size "$DRIVE" | awk 'NR==2 {print $1}')
    FREE_TB=$(df -h --output=avail "$DRIVE" | awk 'NR==2 {print $1}')

    echo "Drive at ${USAGE_PERCENT}%: ${NAME}" >> "$SPYGLASS_LOG"

    # Do nothing if under capacity
    if [ "$USAGE_PERCENT" -le "$SPACE_PERCENT_LIMIT" ]; then
        continue
    fi

    # Send email alert
    BODY="${NAME} has ${FREE_TB}/${TOTAL_TB} TB free as of ${timestamp}"
    SUBJ="${NAME}, $USAGE_PERCENT%"

    for RECIPIENT in $SPACE_EMAIL_RECIPIENTS; do
      EMAIL=$(printf "$EMAIL_TEMPLATE" "$RECIPIENT" "$SUBJ" "$BODY")
      curl -s --url "smtps://smtp.gmail.com:465" \
          --ssl-reqd \
          --user "$SPACE_EMAIL_SRC:$SPACE_EMAIL_PASS" \
          --mail-from "$SPACE_EMAIL_SRC" \
          --mail-rcpt "$RECIPIENT" \
          -T <(echo "$EMAIL")
    done
done
