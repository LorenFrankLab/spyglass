#!/bin/bash
# Shared messaging utilities. Source this file; do not execute it directly.
#
# Requires for email (either naming is accepted):
#   SPACE_EMAIL_SRC / SPYGLASS_EMAIL_SRC, SPACE_EMAIL_PASS / SPYGLASS_EMAIL_PASS
# Requires for Slack:
#   SLACK_TOKEN, SLACK_CHANNEL
# Optional:
#   SLACK_LOG  — curl output destination (default: /dev/null)

send_email_message() {
  # Usage: send_email_message <to> <subject> <body> [cc]
  # <cc> is optional; when set, the message is delivered to both <to> and <cc>.
  local TO="$1"
  local SUBJECT="$2"
  local BODY="$3"
  local CC="$4"
  local SRC="${SPACE_EMAIL_SRC:-$SPYGLASS_EMAIL_SRC}"
  local PASS="${SPACE_EMAIL_PASS:-$SPYGLASS_EMAIL_PASS}"

  if [[ -z "$SRC" || -z "$PASS" ]]; then
    echo "Email credentials not set; skipping email to ${TO}." >&2
    return 1
  fi

  local EMAIL="From: \"Spyglass\" <${SRC}>
To: ${TO}"
  [[ -n "$CC" ]] && EMAIL+="
Cc: ${CC}"
  EMAIL+="
Subject: ${SUBJECT}

${BODY}"

  local RCPT_ARGS=(--mail-rcpt "$TO")
  [[ -n "$CC" ]] && RCPT_ARGS+=(--mail-rcpt "$CC")

  curl -sS --ssl-reqd \
      --url "smtps://smtp.gmail.com:465" \
      --user "${SRC}:${PASS}" \
      --mail-from "$SRC" \
      "${RCPT_ARGS[@]}" \
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
