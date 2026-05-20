#!/bin/bash
# Shared email utility for Spyglass maintenance scripts.
#
# Source this file to get send_email_message, which mirrors the approach
# used by check_disk_space.sh but adds CC support.
#
# Requires SPYGLASS_EMAIL_SRC and SPYGLASS_EMAIL_PASS to be set.
#
# Usage:
#   source email_utils.sh
#   send_email_message <to> <cc> <subject> <body>
#
# <cc> may be empty; when set, the message is delivered to both <to> and <cc>.

send_email_message() {
    local TO="$1"
    local CC="$2"
    local SUBJECT="$3"
    local BODY="$4"

    if [[ -z "$SPYGLASS_EMAIL_SRC" || -z "$SPYGLASS_EMAIL_PASS" ]]; then
        echo "SPYGLASS_EMAIL_SRC/PASS not set; skipping email to ${TO}."
        return 1
    fi

    local EMAIL
    if [[ -n "$CC" ]]; then
        EMAIL="From: \"Spyglass\" <${SPYGLASS_EMAIL_SRC}>
To: ${TO}
Cc: ${CC}
Subject: ${SUBJECT}

${BODY}"
    else
        EMAIL="From: \"Spyglass\" <${SPYGLASS_EMAIL_SRC}>
To: ${TO}
Subject: ${SUBJECT}

${BODY}"
    fi

    local RCPT_ARGS=(--mail-rcpt "$TO")
    [[ -n "$CC" ]] && RCPT_ARGS+=(--mail-rcpt "$CC")

    curl -sS --ssl-reqd \
        --url "smtps://smtp.gmail.com:465" \
        --user "${SPYGLASS_EMAIL_SRC}:${SPYGLASS_EMAIL_PASS}" \
        --mail-from "$SPYGLASS_EMAIL_SRC" \
        "${RCPT_ARGS[@]}" \
        -T <(echo "$EMAIL")
}
