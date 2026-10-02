"""Row validation and legacy rows for ``SorterParameters``.

:func:`validate_sorter_rows` is the per-sorter validation behind
``SorterParameters.insert``: each row's ``params`` blob is validated against
its sorter's schema, the sorter name is checked against the known sorters, and
the execution backend is validated and its schema version backfilled.
:func:`legacy_si_sorter_rows` builds the ``('<sorter>', 'default')`` rows
``SorterParameters.insert_default_legacy_si_sorters`` inserts for installed,
non-curated SpikeInterface sorters.

Imports without the DB layer; SpikeInterface is imported inside the functions.
"""

from __future__ import annotations

from spyglass.spikesorting.v2._params.sorter import _get_sorter_schema
from spyglass.spikesorting.v2._recipe_catalog import _params_schema_version
from spyglass.spikesorting.v2.utils import (
    _validate_params,
    validate_lookup_rows,
)


def validate_sorter_rows(rows, heading_names, non_si_sorters) -> list:
    """Validate ``SorterParameters`` rows against their per-sorter schemas.

    Parameters
    ----------
    rows : iterable of dict or sequence
        Rows as passed to ``SorterParameters.insert``.
    heading_names : list of str
        ``SorterParameters`` column names, in order.
    non_si_sorters : frozenset of str
        Spyglass-internal sorters that run in-process
        (``SorterParameters._NON_SI_SORTERS``).

    Returns
    -------
    list of dict
        The validated rows, with ``execution_params`` and both schema-version
        columns backfilled.

    Raises
    ------
    ValueError
        If a sorter name is unknown, a row fails its schema or the installed
        wrapper's parameter vocabulary, carries reserved execution keys, asks
        an in-process sorter for a container backend, or has an
        ``execution_params_schema_version`` that disagrees with its blob.
    """
    # Validate every row (incl. ``insert_default``'s positional
    # ``_DEFAULT_CONTENTS``) before it lands, dispatching the Pydantic
    # schema per ``sorter``. Two per-row guards run after that:
    #
    # 1. Sorter-name typo guard. ``_get_sorter_schema`` falls back to
    #    the permissive ``GenericSorterParamsSchema`` for any unknown
    #    sorter (the "try any installed SI sorter" escape hatch), so a
    #    typo like ``"mountainSort4"`` would otherwise validate cleanly
    #    here and fail only much later at ``Sorting.populate`` with an
    #    opaque SI "sorter not registered" error. Reject a name that is
    #    not in ``sis.available_sorters()``, the curated v2 schema set,
    #    nor the Spyglass ``clusterless_thresholder`` path -- the check
    #    ``_get_sorter_schema``'s docstring already delegates here. The
    #    gate is deliberately ``available_sorters()`` (a pure spelling
    #    check), NOT ``insert_default``'s stricter ``installed_sorters()``
    #    availability gate: a correctly-spelled sorter whose binary is
    #    absent on THIS machine may still be staged for a compute node.
    #
    # 2. ``params_schema_version`` backfill. This Lookup is multi-sorter
    #    (MS4/MS5/KS4/clusterless each carry their own schema_version),
    #    so the column default cannot be pinned to any one sorter's
    #    version -- it defaults to the sentinel 0 ("unspecified"). The
    #    validated ``params`` blob already carries the authoritative
    #    ``schema_version``, so backfill the outer column from it when
    #    the caller left it at 0 rather than making them copy the number
    #    by hand. An explicitly-passed NON-zero value is left untouched
    #    for ``_assert_schema_version_matches`` to cross-check, so a real
    #    outer-vs-blob mismatch still raises.
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import (
        _SORTER_SCHEMAS,
        reject_internal_whiten,
        reject_reserved_execution_keys,
        validate_execution_params,
        validate_sorter_params_against_wrapper,
    )

    valid_sorters = (
        set(sis.available_sorters()) | set(_SORTER_SCHEMAS) | non_si_sorters
    )

    def _check_sorter_and_backfill_version(row, _schema_cls):
        sorter = row["sorter"]
        if sorter not in valid_sorters:
            raise ValueError(
                f"SorterParameters.insert: sorter {sorter!r} is not a "
                "known SpikeInterface sorter or the Spyglass "
                "'clusterless_thresholder' path -- check the spelling. "
                f"Curated v2 sorters: {sorted(_SORTER_SCHEMAS)}; or any "
                "sorter from spikeinterface.sorters.available_sorters()."
            )
        # Internally-whitening sorters that fall through to the permissive
        # generic schema (kilosort2_5 / kilosort3 / ironclust) must not
        # carry a truthy ``whiten`` -- it would double-whiten via the
        # runtime's external float64 whitening. (KS4 self-guards in its
        # typed schema; MS4/MS5 use whiten=True deliberately.)
        reject_internal_whiten(sorter, row["params"])
        # Permissive (``extra="allow"``) schemas pass unknown keys through;
        # check them against the installed SI wrapper's own parameter
        # vocabulary so a typo fails here, not minutes into the sort.
        validate_sorter_params_against_wrapper(sorter, row["params"])
        # Container backend / install provenance is tracked ONLY on
        # ``execution_params`` -- reject the reserved execution keys from the
        # scientific ``params`` blob (the permissive ``extra="allow"`` sorter
        # schemas would otherwise pass them straight through) AND from
        # ``job_kwargs``. The strict schemas already reject the same keys via
        # ``extra="forbid"``.
        reject_reserved_execution_keys(
            row["params"], context="SorterParameters params blob"
        )
        reject_reserved_execution_keys(
            row.get("job_kwargs"), context="SorterParameters job_kwargs"
        )
        # ``params_schema_version`` is backfilled from the validated blob by
        # ``validate_lookup_rows`` (the shared path, for every Lookup), so it
        # is NOT re-done here. The execution-params version below is
        # SorterParameters-specific and stays in this hook.
        # Validate + backfill the execution backend provenance. A row that
        # omits ``execution_params`` defaults to local execution; the outer
        # ``execution_params_schema_version`` is backfilled from the
        # validated blob when omitted and cross-checked when supplied.
        validated_execution = validate_execution_params(
            row.get("execution_params")
        )
        row["execution_params"] = validated_execution
        # In-process sorters (clusterless) run in the host process; the
        # runtime ignores any container backend, so a row that claims one
        # would make preflight falsely require a container image. Reject it
        # at insert -- these must run with backend='local' (or omit it).
        if (
            sorter in non_si_sorters
            and validated_execution["backend"] != "local"
        ):
            raise ValueError(
                f"SorterParameters.insert: sorter {sorter!r} runs "
                "in-process (clusterless) and is executed locally; it "
                "cannot use execution backend "
                f"{validated_execution['backend']!r}. Set backend='local' "
                "or omit execution_params."
            )
        inner_execution_version = int(validated_execution["schema_version"])
        if "execution_params_schema_version" not in row:
            row["execution_params_schema_version"] = inner_execution_version
        elif (
            int(row["execution_params_schema_version"])
            != inner_execution_version
        ):
            raise ValueError(
                "SorterParameters.insert: execution_params_schema_version="
                f"{row['execution_params_schema_version']} does not match "
                "the inner SorterExecutionParamsSchema schema_version="
                f"{inner_execution_version} on the validated execution_params "
                "blob. Drop the column or align it with the blob's "
                "schema_version."
            )

    validated = validate_lookup_rows(
        rows,
        heading_names,
        schema_for=lambda row: _get_sorter_schema(row["sorter"]),
        table_name="SorterParameters",
        per_row_hook=_check_sorter_and_backfill_version,
    )
    return validated


def legacy_si_sorter_rows() -> list:
    """Build ``('<sorter>', 'default')`` rows for installed non-curated sorters.

    See ``SorterParameters.insert_default_legacy_si_sorters`` for which
    sorters are included and why. Each row's ``params`` are the SpikeInterface
    wrapper's own defaults, validated through ``GenericSorterParamsSchema``
    and the wrapper vocabulary; a sorter that fails either is skipped with a
    warning.

    Returns
    -------
    list of dict
        Mapping rows ready for ``SorterParameters.insert``.
    """
    import copy

    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2._params.sorter import (
        _SORTER_SCHEMAS,
        GenericSorterParamsSchema,
        validate_sorter_params_against_wrapper,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import MATLAB_SORTERS
    from spyglass.utils import logger

    curated = set(_SORTER_SCHEMAS)  # sorters with their own schemas
    installed = set(sis.installed_sorters())
    rows = []
    skipped_not_installed = []
    skipped_matlab = []
    for sorter in sis.available_sorters():
        if sorter in curated:
            # See SorterParameters.insert_default_legacy_si_sorters: a curated
            # sorter's typed schema would fail or drop SI's default keys.
            continue
        if sorter.lower() in MATLAB_SORTERS:
            # MATLAB sorters require a container backend; a local 'default'
            # row is rejected by the dispatcher at populate time. The lab
            # inserts MATLAB rows explicitly with a container
            # execution_params, so do not seed an unrunnable local default.
            skipped_matlab.append(sorter)
            continue
        if sorter not in installed:
            # Gate on installed_sorters() to match insert_default's installed-sorters
            # gate -- a wrapper exposes its defaults even when its
            # binary is absent, so an available-but-not-installed
            # row would only fail later at populate time.
            skipped_not_installed.append(sorter)
            continue
        try:
            # Read the wrapper's own defaults rather than
            # ``get_default_sorter_params``: the latter is
            # ``_dynamic_params()`` PLUS ``get_global_job_kwargs()``
            # for a ``requires_binary_data`` sorter, and those job
            # keys (n_jobs, chunk_duration, ...) are outside the
            # wrapper vocabulary the insert guard enforces.
            sorter_class = sis.sorter_dict[sorter]
            params, _descriptions = sorter_class._dynamic_params()
            params = copy.deepcopy(params)
        except Exception as exc:  # SI may raise on metadata fetch
            logger.warning(
                "insert_default_legacy_si_sorters: skipping "
                f"{sorter!r} ({exc!r})."
            )
            continue
        # Validate through the generic schema (extra='allow') so the
        # row passes the insert gate without typo-rejection.
        try:
            validated = _validate_params(GenericSorterParamsSchema, params)
        except Exception as exc:
            logger.warning(
                f"insert_default_legacy_si_sorters: {sorter!r} did "
                "not validate against GenericSorterParamsSchema "
                f"({exc!r})."
            )
            continue
        # The insert hook applies this same guard; run it here so ONE
        # sorter whose wrapper defaults fall outside its own vocabulary
        # is skipped rather than aborting the caller's batch insert.
        try:
            validate_sorter_params_against_wrapper(sorter, validated)
        except Exception as exc:
            logger.warning(
                f"insert_default_legacy_si_sorters: {sorter!r} default "
                "params are outside the installed wrapper's parameter "
                f"vocabulary ({exc!r}); skipping."
            )
            continue
        # Append a MAPPING row (not a positional tuple) so the insert hook
        # backfills the omitted execution_params (default local execution) +
        # its schema version -- a positional row would have to enumerate all
        # SorterParameters columns and would break whenever the table gains
        # a column.
        rows.append(
            {
                "sorter": sorter,
                "sorter_params_name": "default",
                "params": validated,
                "params_schema_version": _params_schema_version(validated),
                "job_kwargs": None,
            }
        )
    if skipped_not_installed:
        logger.info(
            "insert_default_legacy_si_sorters: skipping "
            f"{sorted(skipped_not_installed)} -- available in "
            "SpikeInterface but not in installed_sorters() on this "
            "platform (a 'default' row would fail at populate time)."
        )
    return rows
