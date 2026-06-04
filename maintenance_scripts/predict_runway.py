#!/usr/bin/env python3
"""Predict when a drive will run out of space based on recent usage trend.

Reads the structured CSV produced by check_disk_space.sh (or migrate_log_to_csv.py),
fits a linear trend to `used_bytes` over a rolling window, and prints a one-line
runway estimate suitable for appending to Slack / email alerts.

Usage:
    python predict_runway.py [CSV_PATH [DRIVE_PATH]]

Environment variables (override defaults):
    SPACE_CSV_LOG       path to the CSV log (required if not passed as arg)
    SPACE_RUNWAY_PATH   drive path to predict for   (default: /stelmo/nwb)
    SPACE_RUNWAY_DAYS   rolling window in days       (default: 90)
    SPACE_RUNWAY_MIN    minimum window tried before  (default: 30)
                        reporting "stable"

Output (one line, no trailing newline):
    data runway: ~11 months   — at current rate
    data runway: ~45 days     — less than 2 months
    data runway: ~2 years     — more than a year
    data runway: stable       — no net growth in any window tried
    data runway: N/A          — not enough data
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Return slope (bytes/day) from ordinary least-squares."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom


def _format_runway(days: float) -> str:
    if days < 60:
        return f"~{round(days)} days"
    if days < 730:
        return f"~{round(days / 30)} months"
    return f"~{days / 365:.1f} years"


def predict(csv_path: Path, drive: str, window_days: int, min_days: int) -> str:
    rows: list[tuple[datetime, int, int]] = []  # (ts, avail, used)
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["path"] != drive:
                continue
            ts = datetime.fromisoformat(r["timestamp"])
            avail = int(r["avail_bytes"])
            total = int(r["total_bytes"])
            rows.append((ts, avail, total - avail))

    if len(rows) < 2:
        return "N/A"

    last_ts, last_avail, _ = rows[-1]

    # Try discrete fallback windows (longest first) to avoid landing on a
    # transitional near-zero slope from an arbitrary intermediate window size.
    # Build the window set: [window_days, midpoint, min_days] deduplicated.
    mid = (window_days + min_days) // 2
    windows = sorted({window_days, mid, min_days}, reverse=True)

    for window in windows:
        cutoff = last_ts - timedelta(days=window)
        w = [(ts, avail, used) for ts, avail, used in rows if ts >= cutoff]
        if len(w) < 2:
            continue
        t0 = w[0][0]
        xs = [(ts - t0).total_seconds() / 86400 for ts, _, _ in w]
        ys = [used for _, _, used in w]
        slope = _linear_slope(xs, ys)
        if slope > 0:
            runway_days = last_avail / slope
            return _format_runway(runway_days)

    return "stable"


def main() -> None:
    args = sys.argv[1:]

    csv_path = Path(
        args[0] if args else os.environ.get("SPACE_CSV_LOG", "")
    ).expanduser()
    if not csv_path or not csv_path.exists():
        print("data runway: N/A", end="")
        print(
            f"\nError: CSV not found ({csv_path or 'unset'}). "
            "Set SPACE_CSV_LOG or pass path as argument.",
            file=sys.stderr,
        )
        sys.exit(1)

    drive = (
        args[1]
        if len(args) >= 2
        else os.environ.get("SPACE_RUNWAY_PATH", "/stelmo/nwb")
    )
    window_days = int(os.environ.get("SPACE_RUNWAY_DAYS", "90"))
    min_days = int(os.environ.get("SPACE_RUNWAY_MIN", "30"))

    result = predict(csv_path, drive, window_days, min_days)
    print(f"data runway: {result}", end="")


if __name__ == "__main__":
    main()
