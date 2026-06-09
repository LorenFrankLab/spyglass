#!/usr/bin/env python3
"""Predict when a drive will run out of space based on recent usage trend.

Reads the structured CSV produced by check_disk_space.sh, fits a linear trend
to `used_bytes` over a rolling window, and prints a one-line runway estimate
suitable for appending to Slack / email alerts.

Usage:
    python predict_runway.py [--plot] [CSV_PATH [DRIVE_PATH]]

Flags:
    --plot   Generate a multi-panel usage-history PNG alongside the runway
             estimate. Saved to <csv_dir>/spyglass_disk.png; path printed to
             stderr so it goes to the log without polluting the runway line.

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

    rows.sort(key=lambda r: r[0])
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


def plot_history(
    csv_path: Path,
    drive: str = "/stelmo/nwb",
    output_path: Path | None = None,
    window_days: int = 90,
    min_days: int = 30,
) -> Path:
    """Plot used-space history for one drive with runway projections.

    Shows data points within the longest regression window, overlaid with
    OLS fit lines (solid) extended as projections (dashed) to capacity.
    Each legend entry includes the window's predicted exhaustion date so
    the regression slope and the runway estimate are directly linked.
    Matplotlib is imported here so an ``ImportError`` never affects the
    runway-only cron path.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV log file.
    drive : str
        Drive path to plot (must match values in the ``path`` column).
    output_path : Path, optional
        Output path for the PNG. Defaults to
        ``csv_path.parent / "spyglass_disk.png"``.
    window_days : int
        Longest regression window in days (matches ``SPACE_RUNWAY_DAYS``).
    min_days : int
        Shortest regression window in days (matches ``SPACE_RUNWAY_MIN``).

    Returns
    -------
    Path
        Path where the PNG was saved.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = csv_path.parent / "spyglass_disk.png"

    rows: list[tuple[datetime, int, int]] = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["path"] != drive:
                continue
            # Strip timezone so matplotlib date handling stays simple.
            ts = datetime.fromisoformat(r["timestamp"]).replace(tzinfo=None)
            avail = int(r["avail_bytes"])
            total = int(r["total_bytes"])
            rows.append((ts, avail, total - avail))
    rows.sort(key=lambda r: r[0])

    TiB = 1024**4
    last_ts, last_avail, _ = rows[-1]
    last_total_tib = (last_avail + rows[-1][2]) / TiB

    # Same window set as predict()
    mid = (window_days + min_days) // 2
    windows = sorted({window_days, mid, min_days}, reverse=True)
    colors = ["tab:red", "tab:orange", "tab:green"]

    # Only show data points within the longest window.
    cutoff = last_ts - timedelta(days=windows[0])
    visible = [(ts, used) for ts, _, used in rows if ts >= cutoff]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.scatter(
        [r[0] for r in visible],
        [r[1] / TiB for r in visible],
        s=6,
        alpha=0.55,
        color="steelblue",
        zorder=3,
        label="actual",
    )
    for window, color in zip(windows, colors):
        w_cutoff = last_ts - timedelta(days=window)
        w = [(ts, avail, used) for ts, avail, used in rows if ts >= w_cutoff]
        if len(w) < 2:
            continue
        t0 = w[0][0]
        xs = [(ts - t0).total_seconds() / 86400 for ts, _, _ in w]
        ys = [used / TiB for _, _, used in w]
        slope = _linear_slope(xs, ys)
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        intercept = ybar - slope * xbar

        y_w0 = intercept
        y_last = intercept + slope * xs[-1]

        if slope > 0:
            x_exhaust = (last_total_tib - intercept) / slope
            days_to_exhaust = x_exhaust - xs[-1]
            if 0 < days_to_exhaust < 365 * 5:
                exhaust_label = (
                    last_ts + timedelta(days=days_to_exhaust)
                ).strftime("%b %Y")
            else:
                exhaust_label = ">5 yr"
        else:
            exhaust_label = "stable"

        ax.plot(
            [w[0][0], last_ts],
            [y_w0, y_last],
            color=color,
            linewidth=2,
            alpha=0.9,
            zorder=2,
            label=f"{window}d  →  {exhaust_label}",
        )

    ax.set_title(f"{drive} — used space & regression", fontsize=11)
    ax.set_ylabel("TiB")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    raw_args = sys.argv[1:]
    do_plot = "--plot" in raw_args
    args = [a for a in raw_args if not a.startswith("--")]

    raw = args[0] if args else os.environ.get("SPACE_CSV_LOG", "")
    if not raw:
        print("data runway: N/A", end="")
        print(
            "\nError: CSV path unset. "
            "Set SPACE_CSV_LOG or pass path as argument.",
            file=sys.stderr,
        )
        sys.exit(1)
    csv_path = Path(raw).expanduser()
    if not csv_path.is_file():
        print("data runway: N/A", end="")
        print(
            f"\nError: CSV not found ({csv_path}). "
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

    if do_plot:
        try:
            png = plot_history(
                csv_path,
                drive=drive,
                window_days=window_days,
                min_days=min_days,
            )
            print(f"\nPlot saved: {png}", file=sys.stderr)
        except Exception as exc:
            print(f"\nPlot failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
