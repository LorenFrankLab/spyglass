"""Pure helpers over fetched ``Model`` rows.

No database access: each function takes rows a caller already fetched, so the
table keeps the query and the ordering logic lives here.
"""

# Metrics that can be ordered against each other, most preferred first. A
# p-cutoff error is measured on a subset of predictions, so it cannot be
# compared with a full-test error.
COMPARABLE_METRICS = ("test_error", "test_error_p")


def rank_model_rows(rows) -> list:
    """Rank same-skeleton model rows by a comparable evaluation metric.

    Pure helper (no DB). Each row is ``{"model_id", "evaluation"}`` where
    ``evaluation`` is the tool-specific metrics dict (or None). A model is
    *scored* when its evaluation carries a real (non-bool) numeric value
    for ``test_error`` (preferred) or ``test_error_p`` (fallback); other
    keys (e.g. ``train_error``) are not comparable. Scored models sort
    first, grouped by metric and ascending by error within a group, since
    a p-cutoff error is measured on a subset of predictions and so cannot
    be ordered against a full-test error. Unscored models follow, ordered
    only by a weak, deterministic ``training_iterations`` tiebreaker (more
    iterations first). No order is ever fabricated for unscored models.

    Parameters
    ----------
    rows : list[dict]
        ``{"model_id": str, "evaluation": dict | None}`` rows.

    Returns
    -------
    list[dict]
        ``{"model_id", "metric", "value", "iterations"}`` sorted
        best-first; ``metric``/``value`` are None when no comparable
        metric is present.
    """
    ranked = []
    for row in rows:
        evaluation = row.get("evaluation")
        metric, value, iterations = None, None, None
        if isinstance(evaluation, dict):
            for key in COMPARABLE_METRICS:
                candidate = evaluation.get(key)
                if isinstance(candidate, bool):
                    continue  # a bool is not a metric value
                if isinstance(candidate, (int, float)):
                    metric, value = key, float(candidate)
                    break
            iters = evaluation.get("training_iterations")
            if isinstance(iters, (int, float)) and not isinstance(iters, bool):
                iterations = int(iters)
        ranked.append(
            {
                "model_id": row.get("model_id"),
                "metric": metric,
                "value": value,
                "iterations": iterations,
            }
        )

    # Group before comparing: test_error_p excludes predictions below a
    # confidence cutoff, so it is systematically lower than test_error
    # and the two cannot be ordered against each other.
    metric_order = {
        name: position for position, name in enumerate(COMPARABLE_METRICS)
    }

    def _sort_key(entry):
        if entry["value"] is not None:
            return (0, metric_order[entry["metric"]], entry["value"])
        iters = entry["iterations"]
        # unscored: after scored; more iterations first, missing last
        return (1, 0, -iters if iters is not None else float("inf"))

    ranked.sort(key=_sort_key)
    return ranked
