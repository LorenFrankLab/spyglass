# v1↔v2 parity test extensions

Implementation plan for three coverage extensions to the v1↔v2
clusterless_thresholder parity gate that currently runs only on
shank 0 of the polymer smoke fixture (`test_v2_real_data_v1_parity`):

- **A.** Other shanks (1, 2, 3) of the same polymer probe instance.
- **B.** A second polymer-probe NWB fixture (same probe type,
  different seed / templates).
- **C.** MountainSort4 parity with stochastic-tolerance bands
  (±50% unit count, ±30% median firing rate per plan-line 170).

Status of A and B is "extend an existing parametrized test." C is
new infrastructure because v1 MS4 capture has not been exercised
end-to-end through `baseline_capture.py` and the tolerance-band
comparison is a different shape than the per-spike-time check the
clusterless gate uses.

---

## Constraints / decisions already locked

1. The MEArec smoke fixture is the only synthetic fixture small
   enough for routine local capture (~30 s per shank for the v1
   side, ~30 s per shank for v2). The 60 s polymer / Neuropixels
   fixtures cost ~10 minutes per side and require the heavy
   conda env. Multi-shank coverage uses the smoke fixture.
2. Smoke-friendly sort params live in
   `tests/spikesorting/v2/_smoke_constants.py`
   (`SMOKE_CLUSTERLESS_PARAM_NAME`, `SMOKE_CLUSTERLESS_PARAMS`,
   `V1_TO_V2_*_NAMES`). All new parity-test sites consume
   these constants; no new param literals.
3. The v1 baseline-capture environment is the existing
   `spyglass-v1-parity` conda env. It must remain locally
   available for any extension work; the `/tmp/spyglass-master`
   worktree underpins it.
4. Each parity assertion uses the existing nearest-neighbor
   matcher (1.5-sample tolerance + 20%+5 extra-spike budget)
   for clusterless; MS4 uses aggregate tolerance bands.

---

## Phase A: multi-shank parity on the smoke polymer fixture

### Why first

Cheapest extension. The polymer smoke fixture is 4 shanks × 32
channels; we currently sort one. Capturing baselines for shanks
1, 2, 3 and asserting parity per-shank gives 4× more invariant
coverage at the cost of one additional capture run per shank.

### Design

`test_v2_real_data_v1_parity` becomes parametrized by `sort_group_id`:

```python
@pytest.mark.parametrize(
    "shank_baseline_dir,sort_group_id",
    [
        ("v1_baseline_shank0", 0),
        ("v1_baseline_shank1", 1),
        ("v1_baseline_shank2", 2),
        ("v1_baseline_shank3", 3),
    ],
    ids=["shank0", "shank1", "shank2", "shank3"],
)
def test_v2_real_data_v1_parity(shank_baseline_dir, sort_group_id):
    ...
```

The fixture-load step honors `meta["sort_group_id"]` (already
present in the captured JSON), and the env-var contract changes
from `SPIKESORTING_V2_BASELINE_DIR=<one dir>` to
`SPIKESORTING_V2_BASELINE_ROOT=<root>` with one subdir per
captured shank. The env-var rename is a breaking change for
existing local users; a transition window honors both:

```python
root = os.environ.get("SPIKESORTING_V2_BASELINE_ROOT")
if root:
    baseline_dir = Path(root) / shank_baseline_dir
else:
    # Backwards-compat single-shank path
    legacy = os.environ.get("SPIKESORTING_V2_BASELINE_DIR")
    if not legacy:
        pytest.skip(...)
    if shank_baseline_dir != "v1_baseline_shank0":
        pytest.skip("legacy SPIKESORTING_V2_BASELINE_DIR only "
                    "covers shank 0; set SPIKESORTING_V2_BASELINE_ROOT "
                    "for multi-shank coverage.")
    baseline_dir = Path(legacy)
```

### Capture-side changes (`baseline_capture.py`)

No code change needed -- `--sort-group-id N --output-dir <dir>`
already parameterizes the shank. Document a wrapper:

```bash
# tests/spikesorting/v2/scripts/capture_all_shanks.sh
ROOT=/tmp/spyglass-v1-baseline-multishank
for SHANK in 0 1 2 3; do
  python tests/spikesorting/v2/baseline_capture.py \
    --nwb-file tests/spikesorting/v2/fixtures/mearec_polymer_smoke.nwb \
    --team-name v2_test_team \
    --interval-list-name "raw data valid times" \
    --sort-group-id $SHANK \
    --sorter-param-name smoke_clusterless_5uv \
    --output-dir $ROOT/v1_baseline_shank$SHANK \
    --base-dir /tmp/spyglass-v1-base
done
```

### Tasks

A1. Add `SPIKESORTING_V2_BASELINE_ROOT` env-var handling to
    `test_v2_real_data_v1_parity` (with `SPIKESORTING_V2_BASELINE_DIR`
    transition shim that skips shanks 1-3).

A2. Parametrize the test over `sort_group_id ∈ {0, 1, 2, 3}` with
    per-shank baseline-dir names.

A3. Update the env-var docstring + a top-of-test docstring example
    showing the `SPIKESORTING_V2_BASELINE_ROOT` layout.

A4. Capture the four baselines locally (one-time):
    `bash scripts/capture_all_shanks.sh` under the
    `spyglass-v1-parity` conda env. ~2 min total.

A5. Run the parametrized parity test under `.venv-spikesorting-v2`
    with `SPIKESORTING_V2_BASELINE_ROOT` set. All 4 should pass
    with the same ±1.5-sample / 20%+5-extra tolerance.

A6. Add a transient diagnostic block: print per-shank
    `(v1_count, v2_count, n_matched, n_unmatched_v2)` so the
    expected per-shank doublet bias from SI 0.99→0.104
    `DetectPeakLocallyExclusive` change is visible (the polymer
    smoke fixture's planted units distribute across shanks; some
    shanks will produce more secondary-channel doublets than
    others).

### Estimated time

~1 hour code + ~10 min capture + ~10 min suite verification.

### Failure modes / open questions

- One or more shanks may have zero planted units detectable at
  `detect_threshold=5.0`. In that case the baseline-capture
  refusal-to-write-empty assertion (fix #2 from review) trips
  and the shank is documented as "below threshold on this
  fixture" rather than asserted. Decision: skip that shank's
  parametrized entry with a clear message rather than fail.
- The 20%+5 extra-spike budget was calibrated for shank 0's
  61 spikes. Shanks with fewer planted spikes have a tighter
  budget (5 extras dominates), so the assertion may be slightly
  stricter or looser per shank. Acceptable; if a shank
  consistently fails by a small margin, document the per-shank
  budget in the constants module.

---

## Phase B: second polymer-probe NWB fixture

### Why second

Tests probe-config robustness (same probe layout, different
template set / placement / seed). If Phase A passes, a second
fixture exposes any single-fixture overfitting in the parity
tolerance.

### Design

Generate a second polymer smoke fixture with a different
`MEArec` seed. Add a `--seed-suffix` flag to
`generate_mearec.py` so the existing `mearec_polymer_smoke.nwb`
isn't clobbered; the new fixture lives at
`mearec_polymer_smoke_seed1.nwb`.

The parity test gains a second parametrization axis:

```python
@pytest.mark.parametrize(
    "fixture_name",
    ["mearec_polymer_smoke", "mearec_polymer_smoke_seed1"],
)
@pytest.mark.parametrize(
    "shank_baseline_dir,sort_group_id",
    [...same as Phase A...],
)
def test_v2_real_data_v1_parity(fixture_name, shank_baseline_dir, sort_group_id):
    ...
```

`SPIKESORTING_V2_BASELINE_ROOT` gets one subdir per fixture:
`<root>/<fixture_name>/v1_baseline_shank<N>/`.

### Tasks

B1. Add a `--seed-suffix` (or `--name-suffix`) flag to
    `generate_mearec.py`'s `smoke` profile so a second
    invocation writes `mearec_polymer_smoke_<suffix>.nwb`
    with a different `MEArec` seed (the existing `seed` field
    on `FixtureSpec` is parameterizable; we just need the CLI
    surface).

B2. Generate the second smoke fixture once via
    `python tests/spikesorting/v2/fixtures/generate_mearec.py
    --smoke --seed-suffix seed1 --mearec-seed 7` (or similar).
    Commit the entry to `fixtures_manifest.json`.

B3. Parametrize the parity test over `fixture_name`. Update the
    `SPIKESORTING_V2_BASELINE_ROOT` directory layout to nest by
    fixture name first, then shank.

B4. Re-run multi-shank baseline capture for the new fixture.
    Same script as Phase A with the new `--nwb-file` path.

B5. Verify both fixtures pass the per-shank parity test under
    the same tolerance budget.

### Open questions

- The MEArec smoke profile produces 6 planted units with 2
  excitatory + 4 inhibitory across 4 shanks; a different seed
  may produce a wildly different distribution (e.g., all units
  on one shank). The 8-baseline (2 fixtures × 4 shanks) matrix
  will likely include a few "no planted units on this shank"
  entries that the Phase-A skip-when-empty path handles cleanly.
- The 4 s smoke duration is at the lower bound of usable for
  v1's MAD estimate; a different seed at the same duration
  may produce a slightly different noise floor. The
  ±1.5-sample tolerance is tight; if it trips on the second
  fixture but passes on the first, consider documenting a
  fixture-specific tolerance map in `_smoke_constants.py`
  (e.g., `PARITY_TOLERANCE_PER_FIXTURE`).

### Estimated time

~1 hour code (generate_mearec flag + parametrize) + ~5 min
generation + ~10 min multi-shank capture + ~15 min verification.

---

## Phase C: MountainSort4 stochastic parity

### Why last + hardest

MS4 is non-deterministic at the template-clustering stage and the
v1 baseline-capture script has only been exercised end-to-end on
`clusterless_thresholder`. The plan-line 170 contract gives MS4
tolerance bands (±50% unit count + ±30% median firing rate) rather
than per-spike equivalence. This is meaningful coverage but
fundamentally different from the clusterless gate.

### Design

#### Capture-side: extend `baseline_capture.py`

`run_baseline_capture` currently hard-codes
`sorter="clusterless_thresholder"`. Generalize:

```python
def run_baseline_capture(
    nwb_source: Path, *, sort_group_id, interval_list_name,
    team_name, output_dir,
    sorter: str = "clusterless_thresholder",
    sorter_param_name: str = "default_clusterless",
) -> tuple[Path, Path]:
    ...
```

The `_populate_sorting` helper already accepts `sorter_param_name`;
add a parallel `sorter` arg threaded from the CLI.

The captured spike-times pickle stays the same shape (`{unit_id:
spike_times_seconds}`); the meta JSON gains `meta["sorter"]` set
to `"mountainsort4"` so the parity test knows which tolerance
contract to apply.

Per-sorter row insertion follows the pattern of
`_ensure_smoke_sorter_param_row` -- generalize to a dispatch
helper or add a sibling `_ensure_ms4_param_row(name)` that
inserts a v1 MS4 row with calibrated thresholds suitable for the
smoke fixture.

MS4 cannot run on the 4 s smoke fixture (too short for template
clustering; needs ~30 s minimum for clean templates). Phase C
uses the 60 s polymer fixture, NOT smoke. This means each capture
run is ~5 minutes (v1) + ~5 minutes (v2 verify), so the iteration
loop is slower than Phases A/B.

#### Parity-test side: add a stochastic comparison body

A new test alongside `test_v2_real_data_v1_parity`:

```python
@pytest.mark.slow
@pytest.mark.integration
def test_v2_real_data_v1_parity_mountainsort4():
    """v1 ↔ v2 stochastic parity for mountainsort4.

    Plan-line 170 stochastic-sorter contract:
    - Unit count: |v2.n_units - v1.n_units| / v1.n_units <= 0.50
    - Median firing rate: |v2.median_fr - v1.median_fr| / v1.median_fr <= 0.30

    No per-spike or per-unit matching -- the clustering step is
    non-deterministic across sort runs (different seeds, different
    sample windows for MAD), so the contract is aggregate-only.
    """
    ...
```

The test reads the same baseline pickle structure; the
aggregate comparison body replaces the per-unit nearest-neighbor
matcher.

#### Sub-tasks

C1. Add `--sorter` to `baseline_capture.py` CLI; thread through
    `run_baseline_capture` and `_populate_sorting`.

C2. Add a `_ensure_ms4_param_row` helper to `baseline_capture.py`
    with smoke- or 60s-polymer-tuned MS4 params; or generalize
    `_ensure_smoke_sorter_param_row` to dispatch on `(sorter,
    sorter_param_name)`.

C3. Capture a v1 MS4 baseline against the 60 s polymer fixture.
    Likely needs ~5 min and a calibrated `detect_threshold` for
    MS4 (different scale than clusterless). The capture-side
    refusal-to-write-empty assertion catches misconfiguration.

C4. Add a corresponding v2 MS4 row to `SorterParameters._DEFAULT_CONTENTS`
    (or a one-off insert in the parity test) with the same
    nominal params.

C5. Write the `test_v2_real_data_v1_parity_mountainsort4` body:
    - Load baseline pickle + meta.
    - If `meta["sorter"] != "mountainsort4"`: skip with a clear
      message.
    - Run v2 MS4 on the same NWB + sort_group_id (60 s polymer).
    - Compute v1 + v2 aggregates: `n_units`, `median_firing_rate`
      (median across units of `len(spike_times)/duration_s`).
    - Assert tolerances per plan-line 170.
    - Diagnostic print of (v1_n_units, v2_n_units, v1_median_fr,
      v2_median_fr, abs/relative drift per metric).

C6. Sanity-check that MS4 actually produces multiple units on the
    60 s polymer fixture. If it produces 1 unit on both sides, the
    median-firing-rate metric is trivially comparable but the
    unit-count tolerance is degenerate; document.

C7. Decide on retries. Stochastic sorters can produce different
    unit counts on different runs from the SAME side; if v2's
    `n_units` varies across runs by ~10-30% (likely), the
    ±50% tolerance should be wide enough -- but a transient
    failure on a flaky run would surface. Two options:
    - Run the v2 sort 3x and use the median; gives stable
      aggregates but tripled cost.
    - Accept the ±50% tolerance and document run-to-run variance
      as the noise floor of the assertion.

    Recommendation: accept ±50% as-is; document.

### Open questions

- MS4 install is required (already done; verified MS4 in
  `installed_sorters()` per the existing `test_ms4_default_row_only_shipped_when_ms4_installed`
  gate). The MS4 install is fragile -- a CI re-run that loses
  the install would skip Phase C silently. Should the parity
  test FAIL when MS4 is missing rather than SKIP?
  - Recommendation: fail. The plan-line 170 contract requires
    MS4 coverage; silent skip masks regressions in MS4
    runtime availability.
- Does v1 MS4 produce comparable units to v2 MS4 on the same
  recording? The known SI 0.99→0.104 changes affect
  `detect_peaks` (clusterless slice); MS4 has its own C++
  template-clustering kernel that may or may not have changed
  semantics across versions. If v2 MS4 systematically produces
  half as many units as v1 MS4 on the same fixture, the test
  reveals that and the question becomes "is this a v2 regression
  or an SI 0.104 MS4 behavior change?" -- a real signal worth
  surfacing.
- What firing-rate metric? "Median across all units" is
  robust but discards distribution shape. An alternative is
  "median firing rate computed per matched-by-similarity
  unit pair" -- but that requires a clustering match step,
  which is a fresh source of stochasticity. Stick with the
  simpler aggregate per plan-line 170.

### Estimated time

~3-4 hours for the capture-side generalization + helper, ~2
hours for the parity-test body + tolerance calibration, ~1 hour
for one-time MS4 baseline capture + suite verification. Total
~6-7 hours.

---

## Sequencing

1. **Phase A first** (1 hour, low risk). Validates the
   parametrization shape works and surfaces per-shank doublet
   variance in the existing clusterless gate.
2. **Phase B second** (1.5 hours, low-medium risk). Adds a
   second fixture to confirm the tolerance isn't single-
   fixture-specific.
3. **Phase C last** (6-7 hours, medium-high risk). Independent
   of A/B in terms of file scope, but uses lessons learned
   (per-shank tolerance maps from A, per-fixture tolerance maps
   from B) to set sensible MS4 aggregate bounds.

Total wall-clock if done sequentially: ~10 hours. Each phase
is a separate commit (followup #14, #15, #16) so review is
focused per-phase.

---

## What this DOES NOT include

- **Other probe geometries** (e.g., Neuropixels parity). The
  Neuropixels 60s fixture exists but adding a Neuropixels
  parity capture would require a v1 baseline against it, which
  is substantial (~10 minutes capture). Defer until polymer
  parity is multi-shank + multi-fixture solid.
- **Other sorters** (Kilosort4, MountainSort5, SpykingCircus2,
  TridesClous2). Each would follow the Phase C pattern but
  with different tolerance calibration. Defer.
- **The 120 s polymer drift fixture.** Drift correction is a
  separate axis of comparison (v1 has no drift correction; v2
  may or may not). Out of scope.
- **CI integration.** All extension parity tests remain local /
  manual (the v1 conda env is too heavy for default CI). The
  existing test_optional_matching_extra_resolution job pattern
  could be adapted for CI parity gating, but the cost / value
  ratio is poor relative to local verification by a developer
  who already has the env.
