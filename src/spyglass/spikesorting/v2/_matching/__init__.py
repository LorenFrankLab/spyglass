"""Private services behind the cross-session unit matching tables.

Read in pipeline order, after ``matcher_protocol`` (the backend contract) and
``unit_matching`` (the tables):

1. ``planning``, ``preflight``: choose each session's curation and check the
   plan before anything is written.
2. ``inputs``, ``coverage``: resolve and freeze the inputs of a
   ``UnitMatchSelection``.
3. ``fetch``, ``assets``, ``waveforms``, ``compute``, ``unitmatch_backend``:
   one ``UnitMatch`` populate (``make_fetch``, input bundles, the matcher).
4. ``graph``, ``readers``: tracked units and queries over a finished run.
5. ``provenance``: producer metadata recorded with each run.
"""
