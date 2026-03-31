#!/bin/bash
# Shared Slack utility. Source this file; do not execute it directly.
#
# Requires SLACK_TOKEN and SLACK_CHANNEL to be set in the environment.
# Logs curl output to SLACK_LOG if set, otherwise /dev/null.
# Silently skips if SLACK_TOKEN or SLACK_CHANNEL is unset.
#
# Usage:
#   source "$SCRIPT_DIR/slack_utils.sh"
#   SLACK_LOG=/path/to/logfile  # optional
#   send_slack_message "Hello from Spyglass"

send_slack_message() {
  # Note: This will not handle special characters. If needed, can extend to
  # accept a JSON payload and use `--data-binary` instead of `-d`.
  if [[ -z "$SLACK_TOKEN" || -z "$SLACK_CHANNEL" ]]; then
    return 0
  fi
  local MESSAGE="$1"
  curl -d "text=$MESSAGE" \
    -d "channel=$SLACK_CHANNEL" \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -X POST https://slack.com/api/chat.postMessage \
    2>> "${SLACK_LOG:-/dev/null}"
}
