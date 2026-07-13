#!/bin/bash
# Shared messaging utilities. Source this file; do not execute it directly.
#
# Requires for email:
#   SPACE_EMAIL_SRC, SPACE_EMAIL_PASS
# Requires for Slack:
#   SLACK_TOKEN, SLACK_CHANNEL
# Optional:
#   SLACK_LOG  — curl output destination (default: /dev/null)

EMAIL_TEMPLATE=$(cat <<-EOF
From: "Spyglass" <$SPACE_EMAIL_SRC>
To: %s
Subject: %s

%s
EOF
)

send_email_message() {
  local RECIPIENT="$1"
  local SUBJECT="$2"
  local BODY="$3"
  local EMAIL
  EMAIL=$(printf "$EMAIL_TEMPLATE" "$RECIPIENT" "$SUBJECT" "$BODY")
  curl -sS --url "smtps://smtp.gmail.com:465" \
      --ssl-reqd \
      --user "$SPACE_EMAIL_SRC:$SPACE_EMAIL_PASS" \
      --mail-from "$SPACE_EMAIL_SRC" \
      --mail-rcpt "$RECIPIENT" \
      -T <(echo "$EMAIL")
}

send_slack_message() {
  # Note: This will not handle special characters. If needed, can extend to
  # accept a JSON payload and use `--data-binary` instead of `-d`.
  if [[ -z "$SLACK_TOKEN" || -z "$SLACK_CHANNEL" ]]; then
    return 0
  fi
  local MESSAGE="$1"
  curl --silent --show-error --fail-with-body \
    -d "text=$MESSAGE" \
    -d "channel=$SLACK_CHANNEL" \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -X POST https://slack.com/api/chat.postMessage \
    &>> "${SLACK_LOG:-/dev/null}"
}
