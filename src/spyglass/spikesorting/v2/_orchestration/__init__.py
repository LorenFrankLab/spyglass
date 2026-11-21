"""Private services behind the ``pipeline`` entry points.

Read in the order a run uses them:

1. ``presets``: the named parameter bundles a run starts from.
2. ``preflight`` and its ``preflight_*`` modules: read-only checks before any
   populate (``preflight_checks`` per stage; ``preflight_geometry``,
   ``preflight_motion``, ``preflight_runtime`` and ``preflight_concat`` by
   topic; ``preflight_plan`` and ``preflight_scientific_setup`` describe the
   run; ``preflight_report`` holds the report types).
3. ``run``, ``session``, ``source``, ``stages``, ``motion``: one sort group, a
   whole session, and the stages they run.
4. ``matching``: cross-session unit matching.
5. ``epochs``, ``geometry``, ``reporting``: ``describe_*`` helpers, and
   ``initialize_v2_defaults`` (in ``reporting``).
6. ``types``: the result and plan types the entry points return.
"""
