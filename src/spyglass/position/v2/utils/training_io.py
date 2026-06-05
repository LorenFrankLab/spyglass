"""Training-history CSV discovery and parsing utilities."""

from pathlib import Path

import pandas as pd

_CSV_PATTERNS = [
    "**/learning_stats.csv",
    "**/log.csv",
    "**/*training*.csv",
]


def discover_training_csvs(model_dir: Path) -> "list[Path]":
    """Return unique CSV files that may contain training loss data.

    Searches *model_dir* and up to two parent directories using the
    standard DLC CSV filename patterns.  Results are deduplicated while
    preserving discovery order.

    Parameters
    ----------
    model_dir : Path
        Directory where the trained model weights live.

    Returns
    -------
    list[Path]
        Unique paths to candidate CSV files.  Empty list if none found.
    """
    search_roots = [model_dir, model_dir.parent, model_dir.parent.parent]
    seen: set = set()
    found: list = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in _CSV_PATTERNS:
            for p in root.glob(pattern):
                if p not in seen:
                    seen.add(p)
                    found.append(p)
        if found:
            break
    return found


def parse_training_csv(path: Path) -> "pd.DataFrame | None":
    """Parse a single training-history CSV into a normalised DataFrame.

    Handles both header-less (DLC ``learning_stats.csv``) and header-bearing
    formats.  Returns ``None`` when the file is empty or has fewer than 2
    columns (not parseable as training data).

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``iteration``, ``loss``, optionally ``learning_rate``,
        and ``source_file``.  ``None`` on parse failure or insufficient data.
    """
    try:
        df_raw = pd.read_csv(path, header=None)
    except Exception:
        return None

    if df_raw.empty or df_raw.shape[1] < 2:
        return None

    def _is_numeric(val):
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    first_row_numeric = all(_is_numeric(v) for v in df_raw.iloc[0])
    if first_row_numeric:
        df = df_raw
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty or df.shape[1] < 2:
            return None

    col_map: dict = {}
    cols = list(df.columns)

    if str(cols[0]) != "iteration":
        col_map[cols[0]] = "iteration"
    if str(cols[1]) != "loss":
        col_map[cols[1]] = "loss"

    lr_candidates = [
        c
        for c in cols
        if isinstance(c, str)
        and ("learning_rate" in c.lower() or c.lower() == "lr")
    ]
    if lr_candidates and str(lr_candidates[0]) != "learning_rate":
        col_map[lr_candidates[0]] = "learning_rate"
    elif (
        not lr_candidates and len(cols) >= 3 and str(cols[2]) != "learning_rate"
    ):
        col_map[cols[2]] = "learning_rate"

    if col_map:
        df = df.rename(columns=col_map)

    df["source_file"] = str(path)
    return df


def aggregate_training_stats(dfs: "list[pd.DataFrame]") -> "pd.DataFrame":
    """Combine a list of per-file training DataFrames into one sorted table.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        DataFrames produced by :func:`parse_training_csv`.

    Returns
    -------
    pd.DataFrame
        Concatenated data sorted by ``iteration`` (if present).  An empty
        DataFrame is returned when *dfs* is empty.
    """
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "iteration" in combined.columns:
        combined = combined.sort_values("iteration").reset_index(drop=True)
    return combined
