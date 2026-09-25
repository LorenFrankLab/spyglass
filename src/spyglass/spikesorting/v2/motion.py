"""Motion estimation for spike-sorting v2, independent of concatenation.

Tables:
    MotionEstimationParameters -- Named, validated SpikeInterface motion recipes.

The DB-free computation (parameter resolution, the estimation adapter, the
``Motion`` serialization) lives in ``_motion``.
"""

from __future__ import annotations

import datajoint as dj

from spyglass.spikesorting.v2._params.motion_estimation import (
    MOTION_ESTIMATION_SCHEMA_VERSION,
    MotionEstimationParamsSchema,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    motion_estimation_default_contents,
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    reject_duplicate_parameter_content,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin

schema = dj.schema("spikesorting_v2_motion")


@schema
class MotionEstimationParameters(
    ImmutableParamsLookup, SpyglassMixin, dj.Lookup
):
    """Named motion-estimation recipes: a SpikeInterface preset plus overrides.

    The ``params`` blob is validated by :class:`MotionEstimationParamsSchema`
    and every row must resolve
    (``_motion.resolve_estimation_params``): an override key that is not a
    parameter of the selected SpikeInterface method is rejected at insert.
    ``insert_default`` ships ``dredge_v1`` and ``dredge_fast_v1``; the
    ``rigid_fast`` preset is allowed but ships no row. No recipe here is
    validated for a particular probe.

    ``job_kwargs`` is the optional per-row SpikeInterface job-kwargs blob
    (``n_jobs``, ``chunk_duration``, ...) for the detect-and-localize pass. It
    is not part of the estimate's identity. The noise-estimate seed is the
    identity-bearing ``noise_levels_seed`` params field, so a ``random_seed``
    job kwarg is rejected.
    """

    definition = f"""
    motion_estimation_params_name: varchar(64)
    ---
    params: blob
    params_schema_version={MOTION_ESTIMATION_SCHEMA_VERSION}: int
    job_kwargs=null: blob  # SI job kwargs for detection and localization
    """

    _DEFAULT_CONTENTS: tuple = motion_estimation_default_contents()

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one validated motion-estimation parameter row."""
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert motion-estimation parameter rows after validation.

        Each row's blob is validated, resolved against the installed
        SpikeInterface (unknown override keys raise), and checked by the
        duplicate-content guard. ``allow_duplicate_params=True`` opts out of
        that guard; see ``reject_duplicate_parameter_content``.
        """
        from spyglass.spikesorting.v2._motion import resolve_estimation_params

        def _resolve_and_check_job_kwargs(row, _schema_cls):
            resolve_estimation_params(row["params"])
            if "random_seed" in (row.get("job_kwargs") or {}):
                raise ValueError(
                    "MotionEstimationParameters.job_kwargs must not contain "
                    "'random_seed': the noise-estimate seed is the identity-"
                    "bearing params field 'noise_levels_seed'."
                )

        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: MotionEstimationParamsSchema,
            table_name="MotionEstimationParameters",
            per_row_hook=_resolve_and_check_job_kwargs,
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="MotionEstimationParameters",
            name_attr="motion_estimation_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert the shipped motion-estimation recipes if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)
