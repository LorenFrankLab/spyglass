# Spike Sorting v2 — I have an ingested NWB, what do I run?

The shortest safe path from an already-ingested session to **analysis-ready**
spike times, using the v2 pipeline. For presets, curation, concatenation, and
cross-session matching, see the full [Spike Sorting v2](./SpikeSortingV2.md)
reference and the [notebooks](../notebooks/10_Spike_SortingV2.ipynb).

This assumes you have already ingested the session with `insert_sessions` (see
[Insert Data](../notebooks/02_Insert_Data.ipynb)) and are on a **local / test**
database — importing `spyglass.spikesorting.v2` registers its schemas only
against a `localhost` host and otherwise raises (set
`SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1` to allow another host).

## 1. One-time setup (3 things)

```python
from spyglass.common.common_lab import LabTeam
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
from spyglass.spikesorting.v2.recording import SortGroupV2

nwb_file_name = "your_session.nwb"  # already ingested

initialize_v2_defaults()  # install the default parameter rows
LabTeam.insert1(
    {"team_name": "my_team", "team_description": "..."}, skip_duplicates=True
)
# set_group_by_shank refuses to overwrite existing sort groups, so guard the
# re-run (or delete the groups first; see the reference).
if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
```

## 2. Sort **and** auto-curate in one call

Pass `auto_curate=True` so the run doesn't stop at the uncurated root: it scores
the sort and commits an auto-labeled child in the same call, giving you an
**analysis-ready** curation to send downstream.

```python
summary = run_v2_pipeline(
    nwb_file_name=nwb_file_name,
    sort_group_id=0,  # the shank to sort
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms5_2026_06",
    auto_curate=True,
)
merge_id = summary["analysis_merge_id"]  # the handle to send downstream
```

`summary["analysis_merge_id"]` is the analysis-ready curation. (A default run
without `auto_curate=True` leaves it `None` and gives you only
`summary["root_merge_id"]`, the **uncurated** root — fine for a quick look, not
for analysis. There is deliberately no bare `merge_id` to copy by mistake.)

## 3. Use it downstream

Everything downstream keys off `merge_id`, unchanged from v0/v1:

```python
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

spike_times = SpikeSortingOutput().get_spike_times({"merge_id": merge_id})
```

## That's it — where to go next

- **Inspect the run:** `describe_run(summary)` renders a receipt (stages,
  warnings, the root vs analysis merge ids) — a zero-unit sort can't hide in it.
- **Fail fast first:** `preflight_v2_pipeline(...)` checks every prerequisite in
  ~1 s before any compute; `run_v2_pipeline(..., preflight=True)` (the default)
  runs it for you.
- **Curate by hand, browse in FigPack, pick a different preset, concatenate, or
  match units across sessions:** the full [Spike Sorting v2](./SpikeSortingV2.md)
  reference and the how-to notebooks
  ([Curation](../notebooks/10_Spike_SortingV2_Curation.ipynb),
  [Presets](../notebooks/10_Spike_SortingV2_Presets.ipynb),
  [Cross-Session](../notebooks/10_Spike_SortingV2_CrossSession.ipynb)).
