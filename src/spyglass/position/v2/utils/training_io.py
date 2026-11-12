"""Training-history CSV discovery and parsing utilities.

TODO: Fold this into tool-specific strategies after #1647 merges.
"""

from pathlib import Path

import pandas as pd

from spyglass.utils import logger

_CSV_PATTERNS = [
    "**/learning_stats.csv",
    "**/log.csv",
    "**/*training*.csv",
]


def discover_training_csvs(model_dir: Path) -> "list[Path]":
    """Return unique CSV files that may contain training loss data.

    Searches *model_dir* only, using the standard DLC CSV filename
    patterns. A project holds one CSV per model, so searching upward
    returns a sibling model's history as this model's.

    Parameters
    ----------
    model_dir : Path
        Directory where the trained model weights live.

    Returns
    -------
    list[Path]
        Unique paths to candidate CSV files.  Empty list if none found.
    """
    # A directory is searched as given; anything else, including a
    # config.yaml that does not exist, resolves to its container.
    root = model_dir if model_dir.is_dir() else model_dir.parent
    seen: set = set()
    found: list = []
    for pattern in _CSV_PATTERNS:
        for path in root.glob(pattern):
            if path not in seen:
                seen.add(path)
                found.append(path)

    if len(found) < 2:
        return found

    # A project holds one CSV per shuffle and iteration. Concatenating them
    # describes no single model, so only the newest model folder is kept.
    # Selecting by shuffle needs an identity the Model row does not carry.
    newest = max(found, key=lambda path: path.stat().st_mtime)
    kept = [path for path in found if path.parent == newest.parent]
    if len(kept) < len(found):
        logger.warning(
            f"{len(found)} training CSVs under {root}; reporting only "
            f"{newest.parent}, the most recently written."
        )
    return kept


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
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ):
        # Not parseable as training data → documented ``None`` sentinel.
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
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ):
            # Not parseable as training data → documented ``None`` sentinel.
            return None
        if df.empty or df.shape[1] < 2:
            return None

    col_map: dict = {}
    cols = list(df.columns)

    if first_row_numeric:
        # Headerless: column order is the only information available.
        col_map[cols[0]] = "iteration"
        col_map[cols[1]] = "loss"
        if len(cols) >= 3:
            col_map[cols[2]] = "learning_rate"
    else:
        # Named columns identify themselves, so position is not consulted.
        aliases = {
            "iteration": ("iteration", "epoch", "step", "global_step"),
            "loss": (
                "loss",
                "losses/train.total_loss",  # DLC 3
                "train_loss",
                "training_loss",
                "total_loss",
            ),
            "learning_rate": ("learning_rate", "lr"),
        }
        by_name = {str(c).lower(): c for c in cols}
        for target, names in aliases.items():
            for name in names:
                column = by_name.get(name)
                if column is not None and column not in col_map:
                    col_map[column] = target
                    break

        # iteration and loss are required; learning_rate is not, and is left
        # unset rather than guessed from a column such as val_loss.
        for target in ("iteration", "loss"):
            if target in col_map.values():
                continue
            for column in cols:
                if column not in col_map:
                    col_map[column] = target
                    break

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
