# SpikeInterface 0.104.3 merge-units parity decision

Decision: retain the Spyglass committed-curation rebuild through
`_sorting_analyzer.build_analyzer`; do not adopt
`SortingAnalyzer.merge_units(..., censor_ms=0.4, merging_mode="hard")`.

The gate used a 30 kHz, one-segment recording whose timestamp vector contained
two 1,500-frame chunks separated by a ten-second wall-clock gap. Two units had:

- a same-sample cross-unit duplicate at frame 500;
- frame-adjacent spikes at 1,499 and 1,500 on opposite sides of the gap; and
- a sub-0.4 ms cross-unit duplicate at frames 2,000 and 2,001.

Both paths assigned merged unit id 2 and were reloaded from zarr before
comparison. The committed-curation path preserved frames
`[500, 1499, 1500, 2000]`; SI returned `[500, 1499, 2000]`. SI correctly
removed the same-sample and within-chunk sub-0.4 ms duplicates, but also removed
frame 1,500 even though it was about ten seconds after frame 1,499 in absolute
time. This confirms that its censor operates in sample space and is not safe for
Spyglass disjoint recordings.

The lost spike propagated after hard-merge recomputation and reload:

| Surface | Committed-curation rebuild | SI `merge_units` |
| --- | ---: | ---: |
| merged spike count | 4 | 3 |
| firing rate | 40 Hz | 30 Hz |
| correlogram total | 6 | 2 |
| templates | shape `(1, 30, 4)` | same shape, unequal values |

The resolver therefore reconstructs the exact committed curation sorting and
calls the same low-level `build_analyzer` used by curation evaluation. This is a
scientific-correctness decision, not a performance preference.
