#!/usr/bin/env python3
"""Predict when a drive will run out of space based on recent usage trend.

Reads the structured CSV produced by check_disk_space.sh and estimates a
runway suitable for appending to a weekly Slack / email report.

Model
-----
The ``used_bytes`` history of a production drive is dominated by two things a
naive linear fit handles badly: occasional large manual cleanups (a sudden drop
of tens to hundreds of TiB), and bursty day-to-day growth. Fitting an ordinary
least-squares line across a cleanup folds the drop into the slope and yields a
meaningless runway. This module instead estimates growth in three steps:

    1. Change-point detection. Split the ``used_bytes`` series at large
       single-step drops (manual cleanups) and keep only the most recent
       segment, so the fit reflects the *current* growth regime rather than a
       rate contaminated by past deletions. A drop counts as a cleanup when it
       exceeds both a fraction of capacity (SPACE_RUNWAY_DROP_FRAC) and a
       multiple of the increment MAD, so the rule adapts to drive size and
       noise level.
    2. Theil-Sen robust slope. Fit the median of all pairwise slopes over that
       segment. Unlike OLS it is insensitive to the bursty outliers, so a
       single large ingest or partial cleanup does not swing the estimate.
    3. Conservative quantile. Alongside the central (median) slope, report an
       upper-quantile slope (SPACE_RUNWAY_QUANTILE). Because a steeper assumed
       growth gives a shorter runway, this is a deliberately earlier, cautious
       estimate — the right bias for an alert, where a late warning is the
       expensive failure.

Runway = current available bytes / slope. The model anticipates fill-up from
routine accumulation; it cannot and does not forecast *future* manual cleanups
or capacity changes. Accuracy is therefore best at short (near-term) horizons,
which is also where the estimate is most actionable.

The model can be cross-validated on existing data with ``--validate`` (see
below), which reports walk-forward, out-of-sample MAE/RMSE per horizon.

Usage:
    python predict_runway.py [--plot] [--validate] [CSV_PATH [DRIVE_PATH]]

Flags:
    --plot      Generate a usage-history PNG alongside the runway estimate.
                Saved to <csv_dir>/spyglass_disk.png; path printed to stderr.
    --validate  Walk-forward cross-validation: at each historical date,
                replicate what predict() would have computed using only data
                available then, project forward 30/60/90 days, and compare
                against actuals. Prints a summary table to stdout.

Environment variables (override defaults):
    SPACE_CSV_LOG          path to the CSV log (required if not passed as arg)
    SPACE_RUNWAY_PATH      drive path to predict for     (default: /stelmo/nwb)
    SPACE_RUNWAY_DROP_FRAC change-point sensitivity: a single-step used drop
                           larger than this fraction of capacity marks a
                           cleanup                        (default: 0.03)
    SPACE_RUNWAY_QUANTILE  pairwise-slope quantile for the conservative
                           (early-warning) runway         (default: 0.75)

Output (one line, no trailing newline):
    data runway: ~11 months (conservative: ~8 months)
    data runway: ~45 days (conservative: ~30 days)
    data runway: stable       — no net growth in the current segment
    data runway: N/A          — not enough data
"""

import csv
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# A single-step used drop beyond a cleanup threshold also has to clear this
# multiple of the increment MAD, so noisy small drives do not over-segment.
_MAD_MULTIPLE = 20


def _read_rows(csv_path: Path, drive: str) -> list[tuple[datetime, int, int]]:
    """Read (timestamp, avail_bytes, used_bytes) rows for one drive, sorted.

    Timezone is stripped so day-scale arithmetic stays simple; the sub-hour
    error this introduces on DST-boundary days is negligible for slopes
    measured in bytes/day.
    """
    rows: list[tuple[datetime, int, int]] = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["path"] != drive:
                continue
            ts = datetime.fromisoformat(r["timestamp"]).replace(tzinfo=None)
            avail = int(r["avail_bytes"])
            total = int(r["total_bytes"])
            rows.append((ts, avail, total - avail))
    rows.sort(key=lambda r: r[0])
    return rows


def _format_runway(days: float) -> str:
    if days < 60:
        return f"~{round(days)} days"
    if days < 730:
        return f"~{round(days / 30)} months"
    return f"~{days / 365:.1f} years"


def _change_point_threshold(
    rows: list[tuple[datetime, int, int]], drop_frac: float
) -> float:
    """Return the bytes drop that marks a cleanup (change-point).

    Robust to scale (fraction of capacity) and to noise level (a multiple of
    the median absolute deviation of single-step increments).
    """
    incs = [rows[i][2] - rows[i - 1][2] for i in range(1, len(rows))]
    med = statistics.median(incs)
    mad = statistics.median([abs(x - med) for x in incs])
    capacity = rows[-1][1] + rows[-1][2]
    return max(drop_frac * capacity, _MAD_MULTIPLE * mad)


def _current_segment(
    rows: list[tuple[datetime, int, int]], drop_frac: float
) -> list[tuple[datetime, int, int]]:
    """Return rows from the most recent cleanup (large used drop) onward."""
    if len(rows) < 2:
        return rows
    threshold = _change_point_threshold(rows, drop_frac)
    start = 0
    for i in range(1, len(rows)):
        if rows[i - 1][2] - rows[i][2] > threshold:
            start = i
    return rows[start:]


def _pairwise_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Return sorted pairwise slopes (bytes/day) for the Theil-Sen estimator."""
    n = len(xs)
    return sorted(
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(n)
        for j in range(i + 1, n)
        if xs[j] > xs[i]
    )


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    if lo >= len(sorted_vals) - 1:
        return sorted_vals[-1]
    return sorted_vals[lo] + (k - lo) * (sorted_vals[lo + 1] - sorted_vals[lo])


def _estimate_growth(
    rows: list[tuple[datetime, int, int]], drop_frac: float, quantile: float
) -> tuple[list[tuple[datetime, int, int]], float, float] | None:
    """Estimate growth on the current segment.

    Returns ``(segment, central_slope, conservative_slope)`` in bytes/day
    (Theil-Sen median and an upper quantile of pairwise slopes), or ``None``
    if there is not enough data.
    """
    seg = _current_segment(rows, drop_frac)
    if len(seg) < 2:
        return None
    t0 = seg[0][0]
    xs = [(ts - t0).total_seconds() / 86400 for ts, _, _ in seg]
    ys = [used for _, _, used in seg]
    slopes = _pairwise_slopes(xs, ys)
    if not slopes:
        return None
    return seg, _quantile(slopes, 0.5), _quantile(slopes, quantile)


def predict(
    csv_path: Path,
    drive: str,
    drop_frac: float = 0.03,
    quantile: float = 0.75,
) -> str:
    """Return a one-line runway estimate string for ``drive``."""
    rows = _read_rows(csv_path, drive)
    if len(rows) < 2:
        return "N/A"

    est = _estimate_growth(rows, drop_frac, quantile)
    if est is None:
        return "N/A"
    _, central, conservative = est

    last_avail = rows[-1][1]
    if central <= 0:
        return "stable"

    text = _format_runway(last_avail / central)
    if conservative > 0:
        text += f" (conservative: {_format_runway(last_avail / conservative)})"
    return text


def validate(
    csv_path: Path,
    drive: str = "/stelmo/nwb",
    horizons: tuple[int, ...] = (30, 60, 90),
    drop_frac: float = 0.03,
    quantile: float = 0.75,
    step_days: int = 7,
    min_segment: int = 5,
) -> list[dict]:
    """Walk-forward cross-validation of the change-point + Theil-Sen model.

    At each prediction date (stepped weekly), re-runs the full estimate using
    only data available up to that date, projects ``used_bytes`` forward from
    the last observation at the central slope, and compares against the nearest
    actual observation. Errors are summarized as MAE and RMSE in TiB.

    Parameters
    ----------
    csv_path : Path
    drive : str
    horizons : tuple of int
        Forecast horizons in days to evaluate.
    drop_frac : float
        Change-point sensitivity (fraction of capacity).
    quantile : float
        Conservative slope quantile (unused for error scoring, kept so the
        call mirrors ``predict``).
    step_days : int
        Gap between successive prediction dates.
    min_segment : int
        Skip prediction dates whose current segment has fewer points (too soon
        after a cleanup to fit a stable slope).

    Returns
    -------
    list of dict
        One entry per horizon: ``horizon_days``, ``n``, ``mae_tib``,
        ``rmse_tib``.
    """
    rows = _read_rows(csv_path, drive)
    if len(rows) < 2:
        return []

    TiB = 1024**4
    results = []
    for horizon in horizons:
        errors: list[float] = []
        t = rows[0][0] + timedelta(days=14)
        t_end = rows[-1][0] - timedelta(days=horizon)

        while t <= t_end:
            past = [r for r in rows if r[0] <= t]
            if len(past) < 2:
                t += timedelta(days=step_days)
                continue
            t_snap = past[-1][0]

            est = _estimate_growth(past, drop_frac, quantile)
            if est is None or len(est[0]) < min_segment:
                t += timedelta(days=step_days)
                continue
            _, central, _ = est
            predicted_used = past[-1][2] + central * horizon

            target = t_snap + timedelta(days=horizon)
            future = [r for r in rows if r[0] >= target]
            if not future or (future[0][0] - target).days > 3:
                t += timedelta(days=step_days)
                continue

            errors.append(predicted_used - future[0][2])
            t += timedelta(days=step_days)

        if errors:
            abs_errs = [abs(e) for e in errors]
            mae = sum(abs_errs) / len(abs_errs) / TiB
            rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5 / TiB
            results.append(
                {
                    "horizon_days": horizon,
                    "n": len(errors),
                    "mae_tib": mae,
                    "rmse_tib": rmse,
                }
            )

    return results


def plot_history(
    csv_path: Path,
    drive: str = "/stelmo/nwb",
    output_path: Path | None = None,
    drop_frac: float = 0.03,
    quantile: float = 0.75,
) -> Path:
    """Plot the current-segment used-space history with robust runway lines.

    Shows the post-cleanup segment that ``predict()`` actually fits, overlaid
    with the Theil-Sen central slope and the conservative upper-quantile slope.
    Each legend entry carries the projected exhaustion date so the slope and
    the runway estimate are directly linked. Matplotlib is imported here so an
    ``ImportError`` never affects the runway-only cron path.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV log file.
    drive : str
        Drive path to plot (must match values in the ``path`` column).
    output_path : Path, optional
        Output path for the PNG. Defaults to
        ``csv_path.parent / "spyglass_disk.png"``.
    drop_frac : float
        Change-point sensitivity (fraction of capacity).
    quantile : float
        Conservative slope quantile.

    Returns
    -------
    Path
        Path where the PNG was saved.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = csv_path.parent / "spyglass_disk.png"

    rows = _read_rows(csv_path, drive)
    TiB = 1024**4
    seg = _current_segment(rows, drop_frac)
    capacity_tib = (seg[-1][1] + seg[-1][2]) / TiB

    t0 = seg[0][0]
    xs = [(ts - t0).total_seconds() / 86400 for ts, _, _ in seg]
    ys_tib = [used / TiB for _, _, used in seg]
    slopes = _pairwise_slopes(xs, ys_tib)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(
        [r[0] for r in seg],
        ys_tib,
        s=8,
        alpha=0.6,
        color="steelblue",
        zorder=3,
        label="actual (current segment)",
    )

    last_ts = seg[-1][0]
    lines = [
        ("median", _quantile(slopes, 0.5), "tab:green", "-"),
        (
            f"q{round(quantile * 100)}",
            _quantile(slopes, quantile),
            "tab:orange",
            "--",
        ),
    ]
    for name, slope, color, ls in lines:
        # Robust intercept: median residual anchors the line.
        intercept = statistics.median(
            [y - slope * x for x, y in zip(xs, ys_tib)]
        )
        if slope > 0:
            x_exhaust = (capacity_tib - intercept) / slope
            days_to = x_exhaust - xs[-1]
            if 0 < days_to < 365 * 5:
                label = (last_ts + timedelta(days=days_to)).strftime("%b %Y")
            else:
                label = ">5 yr"
        else:
            label = "stable"
        ax.plot(
            [seg[0][0], last_ts],
            [intercept + slope * xs[0], intercept + slope * xs[-1]],
            color=color,
            linestyle=ls,
            linewidth=2,
            alpha=0.9,
            zorder=2,
            label=f"{name} → {label}",
        )

    ax.set_title(
        f"{drive} — used space & robust runway " f"(segment from {t0.date()})",
        fontsize=11,
    )
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
    do_validate = "--validate" in raw_args
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
    drop_frac = float(os.environ.get("SPACE_RUNWAY_DROP_FRAC", "0.03"))
    quantile = float(os.environ.get("SPACE_RUNWAY_QUANTILE", "0.75"))

    result = predict(csv_path, drive, drop_frac, quantile)
    print(f"data runway: {result}", end="")

    if do_validate:
        vrows = validate(
            csv_path, drive, drop_frac=drop_frac, quantile=quantile
        )
        if vrows:
            print(
                f"\n\nWalk-forward validation — {drive} "
                "(change-point + Theil-Sen, step 7d):"
            )
            print(f"  {'horizon':>8}  {'n':>5}  {'MAE':>10}  {'RMSE':>10}")
            for r in vrows:
                print(
                    f"  {r['horizon_days']:>5}d"
                    f"     {r['n']:>5}"
                    f"  {r['mae_tib']:>7.1f} TiB"
                    f"  {r['rmse_tib']:>7.1f} TiB"
                )
        else:
            print("\n\nValidation: not enough data")

    if do_plot:
        try:
            png = plot_history(
                csv_path, drive=drive, drop_frac=drop_frac, quantile=quantile
            )
            print(f"\nPlot saved: {png}", file=sys.stderr)
        except Exception as exc:
            print(f"\nPlot failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
