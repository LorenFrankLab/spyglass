# Interval

Neuroscience recordings are rarely clean from start to finish. A session may
include periods where the animal was running, resting, or off-task; the hardware
may have introduced noise artifacts; or only certain epochs have been annotated
for analysis. Working with neural data means constantly answering the question:
_which samples actually belong to this analysis?_

Spyglass answers that question with **time intervals** — `[start, stop]` pairs
in seconds that mark regions of interest. The `Interval` class is the primary
tool for creating, combining, and querying these regions.

```python
from spyglass.common.common_interval import Interval
```

______________________________________________________________________

## What is an Interval?

An `Interval` is a thin wrapper around a NumPy array of shape `(N, 2)`, where
each row is one `[start, stop]` time range in seconds. It represents a set of
non-overlapping time windows — e.g., the valid portions of a recording, the
trials in an epoch, or the periods free of movement artifacts.

```python
import numpy as np

# Two valid windows: 0–5 s and 8–12 s
valid = Interval(np.array([[0.0, 5.0], [8.0, 12.0]]))
```

The underlying array is always available as `.times`:

```python
valid.times  # np.array([[0.0, 5.0], [8.0, 12.0]])
```

______________________________________________________________________

## Why use `Interval` instead of a raw array?

A plain NumPy array of shape `(N, 2)` can represent the same data, but
`Interval` adds three things that matter in practice:

**1. Input validation.** `Interval` checks that every row has `start <= stop`
and that the array is shaped `(N, 2)`. Passing a 1-D array, a transposed array,
or an interval where the stop comes before the start raises a `ValueError`
immediately rather than producing silent garbage downstream.

**2. Consistent input handling.** Methods accept any `IntervalLike` — a raw
array, a list, another `Interval`, or an `IntervalList` table key — so you never
need to think about conversion.

**3. Chainable operations.** Every method that produces a new interval set
returns a new `Interval`. Multi-step filtering pipelines can be written as a
single expression without intermediate variables.

______________________________________________________________________

## Creating an Interval

`Interval` accepts several input formats at construction time:

```python
# From a numpy array or list of [start, stop] pairs
iv = Interval([[0.0, 1.0], [2.0, 3.0]])

# From an IntervalList table key — fetches valid_times automatically
iv = Interval({"nwb_file_name": "my_file_.nwb", "interval_list_name": "01_r1"})

# From a list of frame indices — contiguous runs become intervals
# e.g. [2, 3, 4, 6, 7] -> [[2, 4], [6, 7]]
iv = Interval([2, 3, 4, 6, 7], from_inds=True)

# From an existing Interval
iv2 = Interval(iv)
```

### Overlap and duplicate handling

By default, `Interval` warns if the input contains duplicate rows or overlapping
windows. Two optional flags control this behavior:

| Flag            | Default | Effect                                              |
| --------------- | ------- | --------------------------------------------------- |
| `no_duplicates` | `True`  | Silently remove duplicate intervals                 |
| `no_overlap`    | `False` | Merge overlapping intervals into one                |
| `warn`          | `True`  | Log a warning when duplicates or overlaps are found |

```python
# Overlapping input — merged automatically
iv = Interval([[0.0, 3.0], [2.0, 5.0]], no_overlap=True)
iv.times  # [[0.0, 5.0]]
```

______________________________________________________________________

## Working with Intervals

### Combining interval sets

The most common operation is combining two sets of intervals — keeping only the
time that falls within both (intersection), within either (union), or within one
but not the other (subtraction).

```python
recording = Interval([[0.0, 100.0]])  # full recording
epoch = Interval([[10.0, 40.0], [60.0, 90.0]])  # annotated epochs
artifacts = Interval([[25.0, 27.0], [65.0, 66.0]])  # noise windows

# Keep only times that are both in the recording and in an epoch
recording.intersect(epoch).times
# [[10.0, 40.0], [60.0, 90.0]]

# Remove artifact windows from the epoch times
epoch.subtract(artifacts).times
# [[10.0, 25.0], [27.0, 40.0], [60.0, 65.0], [66.0, 90.0]]

# All times covered by either set
epoch.union(artifacts).times
# [[10.0, 40.0], [60.0, 90.0]]  (artifacts are subsumed)
```

`intersect` and `subtract` accept a `min_length` keyword to drop resulting
windows that are too short to be useful. `union` also accepts `max_length`.

### Querying timestamps

Given a dense array of timestamps (e.g., LFP samples, spike times), `Interval`
can filter them to only those falling within the valid windows:

```python
timestamps = np.linspace(0, 100, 10_000)
valid = Interval([[10.0, 40.0], [60.0, 90.0]])

# Return only the timestamps inside the intervals
valid.contains(timestamps)

# Return the *indices* of those timestamps instead
valid.contains(timestamps, as_indices=True)

# Return timestamps that fall *outside* the intervals
valid.excludes(timestamps)
```

`censor` trims the interval boundaries to the span of the timestamps — useful
when you want to ensure no interval extends beyond the available data:

```python
valid.censor(timestamps)
```

### Filtering by duration

To keep only windows long enough to be worth analyzing:

```python
iv = Interval([[0.0, 0.1], [1.0, 5.0], [6.0, 6.05]])
iv.by_length(min_length=0.5).times  # [[1.0, 5.0]]
```

### Merging overlapping windows

If you have built up a set of intervals that may overlap, `consolidate` merges
them:

```python
iv = Interval([[0.0, 3.0], [2.0, 5.0], [7.0, 9.0]])
iv.consolidate().times  # [[0.0, 5.0], [7.0, 9.0]]
```

### Chaining

Because every operation returns an `Interval`, a multi-step pipeline can be
written as a single expression. Call `.times` only at the final step when you
need the raw array:

```python
from spyglass.common.common_interval import Interval

clean = (
    Interval(raw_valid_times)
    .intersect(epoch_times)
    .subtract(artifact_times)
    .by_length(min_length=0.5)
)

# Pass `clean` to the next step, or extract the array
clean.times
```

______________________________________________________________________

## Table integration

`IntervalList` is the DataJoint table that stores valid time windows for each
session. `fetch_interval()` returns an `Interval` directly:

```python
from spyglass.common import IntervalList

iv = (
    IntervalList & {"nwb_file_name": "my_file_.nwb", "interval_list_name": "01_r1"}
).fetch_interval()

trimmed = iv.by_length(min_length=1.0)
```

To store a derived interval back in the table, use `.as_dict`:

```python
IntervalList.insert1({**my_key, **trimmed.as_dict})
```

______________________________________________________________________

## Migration from `interval_list_*` functions

Older Spyglass code used standalone functions like `interval_list_intersect`,
`interval_list_contains`, and `interval_list_complement`. These are now
deprecated wrappers around `Interval` and emit a warning on every call. They
continue to work but will be removed in a future release.

All replacements produce identical output. The only change is that intermediate
results are `Interval` objects rather than raw arrays — call `.times` at the end
if a NumPy array is required.

| Deprecated function                  | Replacement                                  |
| ------------------------------------ | -------------------------------------------- |
| `interval_list_intersect(a, b)`      | `Interval(a).intersect(b).times`             |
| `interval_list_union(a, b)`          | `Interval(a).union(b).times`                 |
| `interval_list_complement(a, b)`     | `Interval(a).subtract(b).times`              |
| `interval_set_difference_inds(a, b)` | `Interval(a).subtract(b).times`              |
| `interval_list_contains(il, ts)`     | `Interval(il).contains(ts)`                  |
| `interval_list_contains_ind(il, ts)` | `Interval(il).contains(ts, as_indices=True)` |
| `interval_list_excludes(il, ts)`     | `Interval(il).excludes(ts)`                  |
| `interval_list_excludes_ind(il, ts)` | `Interval(il).excludes(ts, as_indices=True)` |
| `interval_list_censor(il, ts)`       | `Interval(il).censor(ts).times`              |
| `interval_from_inds(frames)`         | `Interval(frames, from_inds=True).times`     |
| `intervals_by_length(il, min, max)`  | `Interval(il).by_length(min, max).times`     |
| `consolidate_intervals(il)`          | `Interval(il).consolidate().times`           |
| `union_adjacent_index(a, b)`         | `Interval(a).union_adjacent_index(b).times`  |

### Before and after

#### `interval_list_contains_ind` — get indices of timestamps inside intervals

```python
# Before
from spyglass.common.common_interval import interval_list_contains_ind

ind = interval_list_contains_ind(valid_times, timestamps)

# After
ind = Interval(valid_times).contains(timestamps, as_indices=True)
```

#### `interval_list_contains` — filter timestamps to those inside intervals

```python
# Before
from spyglass.common.common_interval import interval_list_contains

ts = interval_list_contains(valid_times, timestamps)

# After
ts = Interval(valid_times).contains(timestamps)
```

#### `intervals_by_length` — keep only intervals within a length range

```python
# Before
from spyglass.common.common_interval import intervals_by_length

result = intervals_by_length(valid_times, min_length=0.5, max_length=10.0)

# After
result = Interval(valid_times).by_length(min_length=0.5, max_length=10.0).times
```

#### `interval_from_inds` — convert frame indices to interval arrays

```python
# Before
from spyglass.common.common_interval import interval_from_inds

result = interval_from_inds([2, 3, 4, 6, 7, 8])  # → [[2, 4], [6, 8]]

# After
result = Interval([2, 3, 4, 6, 7, 8], from_inds=True).times
```

#### Multi-step pipeline

```python
# Before
from spyglass.common.common_interval import (
    interval_list_intersect,
    intervals_by_length,
    interval_list_complement,
)

step1 = interval_list_intersect(raw_times, epoch_times)
step2 = intervals_by_length(step1, min_length=0.5)
result = interval_list_complement(step2, artifact_times)

# After
result = (
    Interval(raw_times)
    .intersect(epoch_times)
    .by_length(min_length=0.5)
    .subtract(artifact_times)
    .times
)
```

#### Single operation

```python
# Before
from spyglass.common.common_interval import interval_list_intersect

result = interval_list_intersect(epoch_times, valid_times, min_length=0.5)

# After
result = Interval(epoch_times).intersect(valid_times, min_length=0.5).times
```

______________________________________________________________________

## Related migration guides

- **Fetching NWB data** — if your pipeline loads NWB files alongside interval
    filtering, see
    [Centralized Code (Mixin)](./Mixin.md#migration-from-standalone-fetch_nwb--get_nwb_table-helpers)
    for migration from `fetch_nwb` and `get_nwb_table`.
- **Long-running computations** — if your table sets `_use_transaction = False`,
    see [Populate and Long-Running Computations](./Populate.md) for the tri-part
    make migration guide.
