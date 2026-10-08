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
from spyglass.spikesorting.v2._core.recipe_catalog import _params_schema_version
from spyglass.spikesorting.v2._core.lookup_validation import (
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
    # Sorter-name typo guard: ``_get_sorter_schema`` falls back to the
    # permissive ``GenericSorterParamsSchema`` for any unknown sorter, so a
    # typo like ``"mountainSort4"`` would otherwise fail only at populate with
    # an opaque SI error. The gate is ``available_sorters()`` (spelling), not
    # ``installed_sorters()``: a sorter absent on this machine may still be
    # staged for a compute node. ``validate_lookup_rows`` backfills
    # ``params_schema_version`` from the validated blob.
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
        # Container provenance lives only on ``execution_params``; the
        # permissive schemas would otherwise pass these keys through.
        reject_reserved_execution_keys(
            row["params"], context="SorterParameters params blob"
        )
        reject_reserved_execution_keys(
            row.get("job_kwargs"), context="SorterParameters job_kwargs"
        )
        # A row without ``execution_params`` defaults to local execution; its
        # outer schema version is backfilled when omitted, else cross-checked.
        validated_execution = validate_execution_params(
            row.get("execution_params")
        )
        row["execution_params"] = validated_execution
        # In-process sorters ignore a container backend at run time, but
        # preflight would then falsely require its image.
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
    from spyglass.spikesorting.v2._sorting.dispatch import MATLAB_SORTERS
    from spyglass.utils import logger

    curated = set(_SORTER_SCHEMAS)  # sorters with their own schemas
    installed = set(sis.installed_sorters())
    rows = []
    skipped_not_installed = []
    for sorter in sis.available_sorters():
        if sorter in curated:
            # See SorterParameters.insert_default_legacy_si_sorters: a curated
            # sorter's typed schema would fail or drop SI's default keys.
            continue
        if sorter.lower() in MATLAB_SORTERS:
            # A local 'default' row would be rejected at populate; MATLAB
            # rows are inserted explicitly with a container backend.
            continue
        if sorter not in installed:
            # Same gate as insert_default: a wrapper exposes defaults even
            # when its binary is absent.
            skipped_not_installed.append(sorter)
            continue
        try:
            # Not ``get_default_sorter_params``, which adds SI's global job
            # kwargs for ``requires_binary_data`` sorters; those keys are
            # outside the wrapper vocabulary the insert guard enforces.
            sorter_class = sis.sorter_dict[sorter]
            params, _descriptions = sorter_class._dynamic_params()
            params = copy.deepcopy(params)
        except Exception as exc:  # SI may raise on metadata fetch
            logger.warning(
                "insert_default_legacy_si_sorters: skipping "
                f"{sorter!r} ({exc!r})."
            )
            continue
        try:
            validated = _validate_params(GenericSorterParamsSchema, params)
        except Exception as exc:
            logger.warning(
                f"insert_default_legacy_si_sorters: {sorter!r} did "
                "not validate against GenericSorterParamsSchema "
                f"({exc!r})."
            )
            continue
        # The insert hook repeats this guard; checking here skips one bad
        # sorter instead of aborting the whole batch insert.
        try:
            validate_sorter_params_against_wrapper(sorter, validated)
        except Exception as exc:
            logger.warning(
                f"insert_default_legacy_si_sorters: {sorter!r} default "
                "params are outside the installed wrapper's parameter "
                f"vocabulary ({exc!r}); skipping."
            )
            continue
        # A mapping row lets the insert hook backfill the omitted
        # execution_params (local) and its schema version.
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
