# Explicit decode bins and Hz model migration

This migration is part of the parameter serialization work in PR #1618, updated
from master. It targets the pending non-local-detector 0.7 API. Version 0.7 is
not published or tagged by this change; release the matching detector before
releasing this adapter. The migrated dependency range is
`non-local-detector>=0.7,<0.8`. Coordinate publication of this adapter with the
matching detector release.

Current parameter rows reconstruct their concrete detector/classifier class. The
adapter fits that reconstructed instance, preserving subclass configuration.
Legacy parameter dictionaries still construct the appropriate base detector;
legacy NonLocal dictionaries reconstructed by the serialization helper retain
their concrete class. Loading an old fitted model for inspection does not
establish new rate units or transition clocks. Recompute fitted models, decoded
results, and derived analyses from the original position and spike recording.
GLM model persistence remains an unsupported upstream Patsy serialization path;
this migration does not repair or qualify it.

PositionGroup preserves each source epoch's continuity in
`position_info.attrs['valid_position_intervals']`, clipped to the fetched
recording. Original timestamps, training masks, environment labels, and encoding
group labels remain on the tracking samples. NaN position rows remain present.
Supply explicit `valid_position_intervals` in decoding kwargs to declare
narrower measured spans when an epoch contains known tracking gaps. The adapter
does not infer continuity from an arbitrary gap threshold. Overlapping source
epochs or duplicate timestamps require an upstream ownership policy and are
rejected.

Decoder fetches retain each epoch's nearest original sample on either side of
the selected time range. These measured anchors support interpolation at
fractional camera boundaries; they are not new observations. Generic
`PositionGroup.fetch_position_info()` still returns its exact requested slice.
Use `include_bracketing=True` when preparing inputs for direct decoding or
trajectory alignment. Linearized position retains the same anchors and support.

The pipeline keeps its existing camera-sample training mask: a sample trains
only when its timestamp falls inside an encoding interval. Outside anchors have
zero training weight. This is distinct from an exact physical recording window.
For a connected window requiring exact exposure bounds, pass the native
`encoding_time_range=[start, stop]` and choose training weights on the original
samples explicitly. An encoding interval with no selected training sample raises
validation rather than silently promoting bracketing anchors. A min/max range
cannot represent disconnected acquisition windows; no acquisition-availability
feature is added here.

An exact shared tracking-boundary sample anchors both physical interpolation
endpoints. One indexed value cannot describe discontinuous positions or angles
on opposite sides; use separate measured spans or sequence-local tracking for
those epochs. Physical interpolation support and spike event ownership have
separate roles.

The decoder constructs uniform bins at its configured `sampling_frequency`,
independently of camera timestamps. Partial terminal bins are trimmed. A short
interval between camera samples is decodable when its complete bins lie inside
the measured interpolation support; it does not need a camera sample inside the
interval. A decode bin is observed only when its entire span lies between finite
measured samples of a declared tracking segment. Encoding's endpoint hold does
not extend observed decode support.

For direct native sorted-spike calls, replace `time=...` with explicit edges and
keep the original tracking arrays separate:

```python
from non_local_detector import ContFragSortedSpikesClassifier

model = ContFragSortedSpikesClassifier(sampling_frequency=500)
model.fit(
    position_time=tracking_time,
    position=tracking_position,
    spike_times=spike_times,
    valid_position_intervals=tracking_intervals,
    is_training=tracking_training_mask,
)
results = model.predict(
    position_time=tracking_time,
    position=tracking_position,
    spike_times=spike_times,
    time_edges=model.calculate_time_edges([start, stop], trim=True),
    is_missing=decode_missing_mask,
)
```

`decode_missing_mask` has one value per complete decode bin. The pipeline builds
it from measured tracking support and caller masks using `prepare_decoder_grid`;
do not pass a camera-row mask directly to the native prediction call.
Clusterless calls additionally receive `spike_waveform_features` at fit and
prediction. Saved `time_bin_start`, `time_bin_end`, and `time_bin_width`
coordinates preserve the explicit bin bounds; `time` labels their centers.

Explicit missing masks combine with measured support and retain the identity of
a requested interval. A scalar mask or an array matching the original tracking
rows uses preceding-sample ownership. A timestamped Series can use the original
tracking index or the actual decode-center index. Other arrays need one value
per requested decode bin. Use a Series on actual decode centers to remove any
ambiguity when tracking and decode row counts happen to match. A wrong timestamp
index raises an actionable error; the adapter does not drop the index.

Each prediction interval starts a separate sequence. At a truly shared closing
boundary, the earlier sequence excludes the boundary event; waveform marks use
the same event mask. EM uses one uniform grid through masked gaps, propagating
its HMM rather than resetting. Result interval labels come from actual requested
bin spans, independently of missing observations: a missing row inside requested
interval 0 still has label 0, while an EM row outside every requested interval
has label -1. Bins outside measured support are marked missing. Intervals with
no complete bin or no supported observations are skipped in prediction,
preserving the original labels of later intervals; if every interval is empty
the adapter raises.

Interval membership permits only endpoint roundoff: four ULP, capped at one
percent of a bin width. A shifted clock origin therefore retains a complete bin
without admitting a materially partial bin. Masks stream one interval at a time,
keeping workspace proportional to the number of decode rows.

Timestamped transition covariates can use the original tracking index or actual
decode centers. Numeric covariates default to continuous interpolation; Boolean
and nonnumeric columns use preceding-sample categorical ownership. For numeric
categories or angles, declare, for example:

```python
decoding_kwargs["discrete_transition_covariate_kinds"] = {
    "route_id": "categorical",
    "heading": "circular",  # radians, shortest-arc interpolation
    "speed": "continuous",
}
```

Original covariates interpolate only within finite tracking segments. Every
transition row needs finite covariates, including rows with missing
observations. For gaps, supply explicit values on the actual decode-center
index; the adapter raises instead of inventing covariates between epochs. Plain
covariate dictionaries need one value per requested decode bin. Fit transition
tensors retain their original covariate axis when saved; prediction covariates
are rebuilt for every requested interval.

Ahead/behind analyses align actual position, heading, and track segment to saved
result centers and bounds. Both 1D and 2D paths return one distance per result
row, with NaN for missing or unsupported tracking. Segment IDs use the preceding
sample; angles follow the shortest arc. A slice uses the original bracketing
tracking samples rather than truncating the interpolation basis to decode
centers.

When updating an existing analysis, create new parameter and position-group
names and point new selections at them. The tutorials use `skip_duplicates=True`
for repeatable fresh setups; it does not replace an existing parameter row or
undo previously upsampled tracking. Recreate tracking from measured timestamps
and repopulate the affected models/results when migrating.

## Validation and remaining release gates

`tests/decoding/test_time_grid_migration.py` executes actual adapter source and
real detector fitting, prediction, EM, and trajectory analysis without importing
DataJoint table schemas. Cases cover sorted and clusterless decoders at 30 and
500 Hz tracking, missing tracking, epoch support, original/scalar masks,
between-camera intervals, shared event and mark ownership, nonstationary
covariates, shifted clock origins, bounded workspace, and perfect 500-bin
trajectory decoding from 31 camera samples. It also writes and reads synthetic
NWB tracking, spikes, marks, and epochs before feeding fetched arrays to all
four actual adapter paths.

The serialization regression executes the actual DecodingParameters insert and
fetch methods using in-memory table boundaries, then fits the reconstructed
concrete instance through the actual adapter. It checks that the instance and
class survive the handoff, rates carry Hz metadata, and result rows follow the
decode clock. This complements the dedicated parameter serialization tests.

Run the hermetic tests in a supported environment with the matching detector
source checkout on PYTHONPATH:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/spyglass-migration-numba \
    MPLCONFIGDIR=/tmp/spyglass-migration-mpl \
    PYTHONPATH=/path/to/non_local_detector/src \
    uv run --no-sync python -m pytest --noconftest -o addopts= -o filterwarnings= \
    tests/decoding/test_time_grid_migration.py \
    tests/decoding/test_dj_decoder_conversion.py -q
```

No database connection, table write, release tagging, or publication is needed
for those hermetic tests. They establish API, clock, and array compatibility.
Before release, run the supported Python and conda environment matrix and
compare representative scientific outputs after recomputation. Existing skipped
suite cases and the deferred GLM path remain outside this qualification.

The master-based integration was tested against detector source commit
`139b7cae51d28d149773580400f309e780fb990d`, built locally with candidate version
metadata `0.7.0` for dependency validation. That local build is not a published
release. On Python 3.11.15, NumPy 1.26.4, SciPy 1.12.0, JAX/JAXlib 0.6.2,
SpikeInterface 0.99.1, DataJoint 0.14.10, and PyNWB 3.1.3, all **67 hermetic
tests** passed. Four independent actual-source persistence checks wrote and read
real NetCDF4 results and KDE model pickles for sorted/clusterless prediction and
EM, preserving clocks, interval labels, posteriors, concrete classes, and Hz
metadata. These checks complement the normal suite, whose save/load fixtures
mock some persistence operations.

The normal decoding suite also passed **167 tests, with 14 existing skips**, in
an isolated MySQL 8.0 container with temporary NWB test data. This exercised
actual parameter insertion/fetch for all four concrete classes, preserving
nondefault algorithm and NonLocal settings and excluding learned state. The
suite exercised native clusterless prediction/EM and sorted prediction through
table population; its existing save/load fixtures remain mocked. Test-only
Numba, pytest-env, and Mountainsort4 were installed for master's fixtures
without changing production dependency constraints. Installation from the
declared 0.7 package range still requires publication on PyPI and conda-forge.
