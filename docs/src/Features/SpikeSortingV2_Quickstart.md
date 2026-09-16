# Spike Sorting v2 — I have an ingested NWB, what do I run?

The supported path from an already-ingested session to **selected** spike times:
configure → sort → review → curate → select units → analyze, using the v2
pipeline. The executable version is the
[single-session notebook](../notebooks/10_Spike_SortingV2.ipynb). For presets,
curation, concatenation, and cross-session matching, see the full
[Spike Sorting v2](./SpikeSortingV2.md) reference and the
[notebooks](../notebooks/10_Spike_SortingV2.ipynb).

This assumes you have already ingested the session with `insert_sessions` (see
[Insert Data](../notebooks/02_Insert_Data.ipynb)).

## 1. Review channels and references, then create sort groups

```python
import pandas as pd

from spyglass.common import Electrode, LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_run,
    describe_sort_groups,
    preflight_v2_pipeline,
    run_v2_pipeline,
)
from spyglass.spikesorting.v2.recording import SortGroupV2

nwb_file_name = "your_session.nwb"  # already ingested

initialize_v2_defaults()  # install the default parameter rows
LabTeam.insert1(
    {"team_name": "my_team", "team_description": "..."}, skip_duplicates=True
)

electrode_config = pd.DataFrame(
    (Electrode & {"nwb_file_name": nwb_file_name}).fetch(
        "electrode_group_name",
        "electrode_id",
        "bad_channel",
        "original_reference_electrode",
        as_dict=True,
    )
)
electrode_config
```

Inspect the bad-channel flags and reference metadata **before grouping**.
Grouping excludes flagged channels; changing flags afterward does not change
existing membership. The
[first-sort notebook](../notebooks/10_Spike_SortingV2.ipynb) includes an
optional `suggest_bad_channels(..., persist=False)` inspection step followed by
explicit persistence of the reviewed suggestions. Its detector thresholds come
from Neuropixels: inspect polymer-probe suggestions and do not rely on a clean
result for small tetrode groups.

Choose the reference deliberately. `references=None` inherits acquisition
metadata, which may need changing for sorting. To override it, pass a mapping
from each included `electrode_group_name` to a reference electrode ID
(`-1`/`None`: none; `-2`: global median; nonnegative: specific electrode). For
hippocampal polymer probes, prefer a quiet anatomical reference when available
and inspect common-median choices carefully.

```python
references = None  # reviewed above; or a mapping covering every included group
# set_group_by_shank refuses to overwrite existing sort groups, so guard the
# re-run. For changes, inspect preview_existing_entries before recreating them.
if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name, references=references)

# Inspect the scientific grouping before choosing a shank. Set sort_group_id
# explicitly when the session has more than one candidate.
sort_groups = describe_sort_groups(nwb_file_name)
if sort_groups.empty:
    raise ValueError(f"No SortGroupV2 rows found for {nwb_file_name!r}.")
sort_groups
```

After reviewing the table, choose the group in a separate cell. To sort every
group or a subset, use the
[whole-session notebook](../notebooks/10_Spike_SortingV2_Presets.ipynb).

```python
available_sort_group_ids = [int(value) for value in sort_groups["sort_group_id"]]
sort_group_id = None  # replace with one reviewed ID when several are listed
if sort_group_id is None:
    if len(available_sort_group_ids) == 1:
        sort_group_id = available_sort_group_ids[0]
    else:
        raise ValueError(
            f"Choose sort_group_id from {available_sort_group_ids} after "
            "reviewing sort_groups."
        )
elif sort_group_id not in available_sort_group_ids:
    raise ValueError(
        f"sort_group_id={sort_group_id} is not one of "
        f"{available_sort_group_ids} for {nwb_file_name!r}."
    )
```

## 2. Choose a recipe, inspect the plan, then sort

The notebook default is the supported MountainSort5 alternative. For hippocampal
polymer probes, the lab production recipe is MountainSort4:

| Recipe                                    | Runtime requirement                                     |
| ----------------------------------------- | ------------------------------------------------------- |
| Local MountainSort4 production preset     | MS4 backend in a compatible `numpy<2` environment       |
| Container MountainSort4 production preset | Singularity/Apptainer and the catalog's container image |
| MountainSort5 alternative (below)         | Standard v2 environment                                 |

Use `describe_pipeline_presets()` to match the region, sampling rate, and probe
type and read each recipe's runtime notes. Selecting a preset never silently
switches its sorter or backend.

Pass `auto_curate=True` so the run doesn't stop at the uncurated root: it scores
the sort with the preset's metric and rule rows and commits an **auto-labeled**
child curation in the same call. For the preset below the rules are
`franklab_default_auto_curation_2026_06` (`nn_noise_overlap > 0.1` → `noise`,
`isi_violation > 0.02` → `reject`); call
`describe_pipeline_preset("franklab_probe_hippocampus_30khz_ms5_2026_06")` to
inspect them before running.

```python
run_kwargs = dict(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
    auto_curate=True,
)
report = preflight_v2_pipeline(**run_kwargs)
print(report.summary())
```

Inspect blockers, warnings, stages to compute or reuse, the effective sorter
configuration, workers/chunks, and scratch/cache notes. Resource notes describe
known allocations; they do not predict total memory or runtime. Then run in a
separate cell. For a new recording setup, the optional
[stage-by-stage recipe](./SpikeSortingV2.md#stage-by-stage-custom-pipeline-preset)
lets you inspect preprocessed traces and retained artifact intervals before
starting the sorter.

```python
run = run_v2_pipeline(**run_kwargs)
describe_run(run)  # stages, warnings, and the effective sorter configuration
auto_labeled = (
    run.auto_labeled_curation
)  # a CurationRef pinned to this exact generation
```

After a failure, correct the reported cause and rerun with the same inputs.
Completed stages are reused; a failed sorter restarts its stage, without
resuming internal checkpoints.

Automatic labels are suggestions written as labels, not approval, and the
auto-labeled child still holds **every** unit. `run["auto_labeled_merge_id"]`
identifies that registered output; it is not a filtered population. (A run
without `auto_curate=True` leaves it `None` and gives you only
`run.root_curation`, the uncurated root.)

## 3. Review it in the browser — and reopen the review later

`start_review` evaluates the pinned curation with a named review profile and
saves a seeded FigPack bundle (local by default). `review.open()` serves that
exact bundle from your kernel at `http://localhost:<port>/` and returns the URL.
In the browser: **Curate Figure**, select units in the unit table (its columns
are the profile's official metrics and the rule set's proposals for the curation
under review), tick labels or **Merge Selected** in the Curation pane, then
**Save Annotations** (**Finalize Curation** is only a browser flag; it commits
nothing). Requires the `spikesorting-v2-curation` extra.

```python
from spyglass.spikesorting.v2.pipeline import FigPackReview

review = run.start_review(
    "franklab_hippocampus_2026_06",  # review profile: metrics + rules + columns
    source="auto_labeled",
    upload=False,
    display_options={"max_amplitudes_per_unit": 2000},  # display budget only
)
url = review.open()  # serves the bundle; opens the browser
print(review.review_id, url)  # open_browser=False just returns the URL

# Later, in a fresh notebook: the same parent generation, profile, evaluation
# and display budget come back from the persisted identity, and open() serves
# the same bundle again -- saved edits included.
review = FigPackReview.resume(review.review_id)
changes = review.preview_import()  # reads the saved annotations.json
print(changes.summary())  # labels +/-, merges, counts, conflicts
receipt = changes.commit()  # or commit(confirm_no_changes=True)

# A merge is re-evaluated with the same profile; the merged child is NOT the
# result until you have looked at it. Open the merged units and STOP here --
# open() does not wait. A label-only review has nothing to verify.
pending_verification = None
if receipt.needs_merge_verification:
    pending_verification = receipt.continue_review()
    pending_verification.open()
    final_curation = None  # pending until the look is committed
else:
    final_curation = receipt.curation  # the committed child (labels + merges)
```

If a verification is pending: inspect the merged units (edit and **Save
Annotations** if one is wrong), then, in a later cell or session, commit that
look explicitly. Run this block again if the commit imports another merge.

```python
if pending_verification is not None:
    verification = FigPackReview.resume(pending_verification.review_id)
    changes = verification.preview_import()
    verification_receipt = changes.commit(confirm_no_changes=not changes.has_changes)
    if verification_receipt.needs_merge_verification:
        pending_verification = verification_receipt.continue_review()
        pending_verification.open()  # another merge: inspect, run again
    else:
        pending_verification = None
        final_curation = verification_receipt.curation
```

Remote kernel: forward the printed port (`ssh -L <port>:localhost:<port> host`)
and open the same `localhost` URL locally -- the frontend enables editing only
for a `localhost` origin. If you skip the browser, `auto_labeled` (or a
`save_manual_curation(...)` / `commit_merges(...)` child) is the curation you
hand to analysis next -- name it `final_curation` deliberately; never infer it
from the latest child.

## 4. Select the analysis population, then analyze

`SpikeSortingOutput().get_spike_times({"merge_id": ...})` returns every unit of
a curation, labels ignored. The supported handoff is
`select_units_for_analysis`: it applies a named `UnitSelectionParams` policy to
the curation's labels, builds the `SortedSpikesGroup` that decoding and
firing-rate consumers read, and returns a receipt naming the exact curation
generation, the policy content, and every included / excluded unit with its
reason.

| Policy                               | Include                                            | Deny                                 |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------ |
| `v2_accepted_single_units` (default) | `accept`                                           | `mua`, `noise`, `reject`, `artifact` |
| `v2_accepted_neural_units`           | `accept` or `mua`                                  | `noise`, `reject`, `artifact`        |
| `v2_unflagged_units`                 | everything not denied (MUA and unlabeled included) | `noise`, `reject`, `artifact`        |
| `all_units`                          | everything (explicit expert choice)                | —                                    |

The shipped rule sets only **flag** bad units; they never write `accept`. So
after auto-labeling alone the two `accepted` policies select nothing — accept
units in the browser review first, or choose `v2_unflagged_units` to state
explicitly that rule-passing, never-reviewed units count. Unlabeled units are
excluded by the `accepted` policies and listed on the receipt either way.

```python
from spyglass.spikesorting.v2.pipeline import select_units_for_analysis

receipt = select_units_for_analysis(final_curation, policy="v2_accepted_single_units")
print(receipt.summary())  # source, policy content, counts, why if empty
receipt.describe()  # per-unit verdict, labels, reason
spike_times, unit_ids = receipt.fetch_spike_data(return_unit_ids=True)
receipt.group_key  # the SortedSpikesGroup key downstream reads
```

For a concatenated (multi-member) sort the receipt holds one per-member group
(`receipt.groups`), each on that member's own session timeline.

## Supported workloads: what changes

- **Tetrodes**: `franklab_tetrode_hippocampus_30khz_ms5_2026_06` (the same
    parameter rows as the probe preset; `probe_type` is informational). The
    analyzer sparsity default (radius 100 µm) is effectively dense on a tetrode.
- **Polymer probes / drift**: sort same-day sessions together with the concat
    presets (motion correction) — see the
    [Cross-Session notebook](../notebooks/10_Spike_SortingV2_CrossSession.ipynb);
    `select_units_for_analysis` then returns one group per member.
- **Clusterless decoding features**: run the `clusterless_thresholder` preset
    and hand the root curation's `merge_id` to `UnitWaveformFeatures`; there is
    no unit-selection step because the thresholder yields one "unit" per channel
    group.

## That's it — where to go next

- **Inspect the run:** `describe_run(run)` renders a receipt (stages, warnings,
    the effective sorter configuration, the root vs auto-labeled merge ids) — a
    zero-unit sort can't hide in it.
- **Fail fast first:** `preflight_v2_pipeline(...)` checks prerequisites before
    any compute; `print(report.summary())` shows the execution plan and
    actionable findings. `run_v2_pipeline(..., preflight=True)` (the default)
    runs it for you.
- **Plot or export exactly what you curated:** every unit-level helper in
    `spyglass.spikesorting.v2.visualization` takes a `CurationRef`
    (`ssviz.plot_waveforms(curated, unit_ids=[...])`,
    `ssviz.export_to_phy(curated, folder)` writes `spyglass_provenance.json`
    beside the export).
- **Curate by hand, pick a different preset, concatenate, or match units across
    sessions:** the full [Spike Sorting v2](./SpikeSortingV2.md) reference and
    the how-to notebooks
    ([Curation](../notebooks/10_Spike_SortingV2_Curation.ipynb),
    [Presets](../notebooks/10_Spike_SortingV2_Presets.ipynb),
    [Cross-Session](../notebooks/10_Spike_SortingV2_CrossSession.ipynb)).

For multiple shanks/tetrodes, continue with the
[whole-session notebook](../../../notebooks/10_Spike_SortingV2_Presets.ipynb):
review each group, choose an exact final curation, and assemble one population
with a consistent label policy. It reports failed/omitted groups and checks
membership when rerun. For pair diagnostics, rasters, early/late traces, and an
additional SNR filter on returned data, use the
[curation notebook](../../../notebooks/10_Spike_SortingV2_Curation.ipynb).

Inspect the scientific setup in preflight/`describe_run`: artifact masking,
drift QC and motion correction are different operations. Concat detects and
masks each member before motion correction; Kilosort's shipped no-mask preset
does not reject artifacts merely by correcting drift.

Use `review.summary()` before opening the browser. Save Annotations saves bundle
edits; Finalize sets a browser flag; Python preview/commit creates the curation.
Pending merges still show parent metrics. Inspect the reevaluated child before
choosing it for analysis. `unavailable_qc` names missing inputs for enabled
rules; missing-policy pass means unflagged, not good. `accept` plus a deny label
such as `noise` is excluded by the accepted-unit policies.
