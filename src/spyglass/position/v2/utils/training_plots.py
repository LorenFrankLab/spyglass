"""Figure composition for DeepLabCut training histories.

Pure rendering: every function takes already-fetched data and returns or
annotates a matplotlib figure. No database access, so the DataJoint table
keeps the fetch and the file write while the drawing lives here.
"""

import pandas as pd


def build_training_history_figure(df, model_id: str, detailed: bool = False):
    """Render a model's training history.

    Parameters
    ----------
    df : pandas.DataFrame
        Training history, as returned by ``Model.get_training_history``.
        Recognised columns: ``loss``, ``iteration``, ``learning_rate``,
        ``validation_loss``, ``source_file``.
    model_id : str
        Identifier for the panel titles.
    detailed : bool, optional
        Add a smoothed loss trend and, where the data allows, a validation
        panel. Default False.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure. The caller shows or saves it.

    Raises
    ------
    ValueError
        If *df* is empty or holds no loss column.
    """
    import matplotlib.pyplot as plt

    if df is None:
        raise ValueError("No training history found for this model")

    # Check available columns for flexible plotting
    has_iteration = "iteration" in df.columns
    has_loss = "loss" in df.columns
    has_lr = "learning_rate" in df.columns
    has_source = "source_file" in df.columns

    if not has_loss:
        raise ValueError("No loss data found to plot")

    # X-axis: iteration if available, otherwise index
    x_data = df["iteration"] if has_iteration else range(len(df))
    x_label = "Training Iteration" if has_iteration else "Step"

    fig, axes = _setup_training_history_figure(
        plt=plt, detailed=detailed, has_lr=has_lr
    )

    _plot_training_loss_panel(
        ax=axes[0],
        df=df,
        x_data=x_data,
        x_label=x_label,
        model_id=model_id,
        has_iteration=has_iteration,
        detailed=detailed,
    )

    if detailed:
        _plot_validation_panel(
            ax=axes[1],
            df=df,
            x_data=x_data,
        )
        if has_lr:
            _plot_learning_rate_panel(
                ax=axes[2],
                x_data=x_data,
                learning_rate=df["learning_rate"],
                x_label=x_label,
                title_fontsize=13,
            )
        else:
            axes[1].set_xlabel(x_label, fontsize=12)
    elif has_lr:
        _plot_learning_rate_panel(
            ax=axes[1],
            x_data=x_data,
            learning_rate=df["learning_rate"],
            x_label=x_label,
            title_fontsize=14,
        )

    # Add source file information if available
    if has_source:
        _add_training_source_summary(fig=fig, df=df)

    plt.tight_layout()

    # Add some spacing if we have subtitle
    if has_source:
        plt.subplots_adjust(bottom=0.08)

    return fig


def _setup_training_history_figure(plt, detailed: bool, has_lr: bool):
    """Create figure and axes layout for training-history plots."""
    if detailed:
        n_plots = 3 if has_lr else 2
        fig, axes = plt.subplots(
            n_plots,
            1,
            figsize=(11, 10 if has_lr else 8),
            sharex=True,
        )
    else:
        n_plots = 2 if has_lr else 1
        fig, axes = plt.subplots(
            n_plots,
            1,
            figsize=(10, 8 if n_plots == 2 else 6),
        )

    if n_plots == 1:
        return fig, [axes]

    return fig, list(axes)


def _plot_training_loss_panel(
    ax,
    df: pd.DataFrame,
    x_data,
    x_label: str,
    model_id: str,
    has_iteration: bool,
    detailed: bool,
):
    """Plot primary loss panel and overlay summary diagnostics."""
    ax.plot(x_data, df["loss"], "b-", linewidth=2, alpha=0.8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        f"Training Loss Curve: {model_id}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    if detailed and len(df) > 10:
        # Show trend independently of noisy step-to-step changes.
        window = max(10, len(df) // 20)
        smoothed_loss = df["loss"].rolling(window=window).mean()
        ax.plot(
            x_data,
            smoothed_loss,
            "g-",
            linewidth=2,
            alpha=0.8,
            label=f"Smoothed Loss (window={window})",
        )
        ax.legend(loc="upper right", fontsize=9)

    if len(df) > 1:
        _annotate_training_loss_stats(
            ax=ax,
            df=df,
            has_iteration=has_iteration,
        )


def _annotate_training_loss_stats(
    ax,
    df: pd.DataFrame,
    has_iteration: bool,
):
    """Add summary statistics and final-point annotation to loss panel."""
    initial_loss = df["loss"].iloc[0]
    final_loss = df["loss"].iloc[-1]
    min_loss = df["loss"].min()
    improvement = (
        ((initial_loss - final_loss) / initial_loss) * 100
        if initial_loss > 0
        else 0
    )

    stats_text = (
        f"Initial: {initial_loss:.6f}\n"
        f"Final: {final_loss:.6f}\n"
        f"Best: {min_loss:.6f}\n"
        f"Improvement: {improvement:.1f}%"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

    if has_iteration:
        final_iter = df["iteration"].iloc[-1]
        ax.annotate(
            f"Final: {final_loss:.4f}",
            xy=(final_iter, final_loss),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )


def _plot_validation_panel(ax, df: pd.DataFrame, x_data):
    """Plot validation loss panel or fallback message."""
    if "val_loss" in df.columns:
        ax.plot(x_data, df["val_loss"], "m-", linewidth=2, alpha=0.85)
    else:
        ax.text(
            0.5,
            0.5,
            "Validation data not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Validation Loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)


def _plot_learning_rate_panel(
    ax,
    x_data,
    learning_rate,
    x_label: str,
    title_fontsize: int,
):
    """Plot learning-rate schedule panel with log-scaled y-axis."""
    ax.plot(x_data, learning_rate, "r-", linewidth=2, alpha=0.8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(
        "Learning Rate Schedule",
        fontsize=title_fontsize,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _add_training_source_summary(fig, df: pd.DataFrame):
    """Add data-source summary subtitle when multiple CSV sources exist."""
    unique_sources = df["source_file"].nunique()
    if unique_sources > 1:
        source_text = (
            f"Data from {unique_sources} files: "
            + f"{', '.join(df['source_file'].unique())}"
        )
        fig.suptitle(source_text, fontsize=10, y=0.02)
