# Integration test practices

Integration tests must verify the user's scenario against independently defined
expectations. The current implementation is not the specification.

- Declare the scenario's inputs and expected outcomes before running the public
    APIs. For spike sorting, specify unit identities, labels, event frames,
    original recording clocks, eligible populations, and expected matches.
- Derive scientific expectations from fixture ground truth, raw input artifacts,
    or hand-computed examples. Do not use a production resolver, processed
    recording, output reader, or implementation algorithm to generate its own
    expected answer. Agreement between database and NWB readers is an additional
    persistence check, not an independent correctness oracle.
- Generated run/curation IDs may come from public receipts. Use them to verify
    the declared source, generation, and relationships; derive scientific unit
    identities and values independently.
- Exercise real orchestration, storage, serialization, and downstream consumers.
    Substitute only clearly identified boundaries outside the scenario being
    verified. A scripted matcher tests the workflow around matching; native
    matcher accuracy requires separate ground-truth recovery tests.
- Assert the actual identities and values, not only success flags, row counts,
    or agreement between readers. Require nonempty outputs when the scenario
    contains events. Check meaningful failure/recovery and retry behavior.
- Use public interfaces for actions under test. Private helpers are acceptable
    for isolated setup, teardown, and explicitly substituted boundaries. Avoid
    coupling scenario tests to internal call order or incidental message
    wording.
- Isolate database/storage state and restore registrations and temporary
    changes. Required CI dependencies must fail when unavailable rather than
    turning the intended coverage into skipped tests.
- When a test fails, investigate the scenario and implementation. Do not change
    expected values, seeds, scientific thresholds, or fixtures merely to match
    observed output. Change expectations only for an intentional, documented
    change to the user-facing or scientific contract.
