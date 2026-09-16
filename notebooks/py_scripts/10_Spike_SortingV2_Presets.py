# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (spyglass_spikesorting_v2)
#     language: python
#     name: python3
# ---

# # Spike Sorting v2 — Presets & whole-session sorting
#
# Two how-tos that build on the
# [first-sort walkthrough](./10_Spike_SortingV2.ipynb): **customizing a pipeline
# preset** (tune one knob with `clone_pipeline_preset`, or build a custom one with
# `register_pipeline_preset`) without hand-editing parameter rows, and **sorting an entire
# session** (every sort group) in one call with `run_v2_pipeline_session`.
#
# Assumes a configured DataJoint connection and an ingested session (see
# [Setup](./00_Setup.ipynb) / [Insert Data](./02_Insert_Data.ipynb)).

# +
import datajoint as dj
import pandas as pd

from spyglass.common import Electrode, LabTeam
from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.pipeline import (
    clone_pipeline_preset,
    describe_pipeline_preset,
    describe_pipeline_presets,
    describe_run,
    describe_sort_groups,
    list_pipeline_presets,
    plot_sort_group_geometry,
    preflight_v2_pipeline_session,
    register_pipeline_preset,
    run_v2_pipeline_session,
)
from spyglass.spikesorting.v2.recording import SortGroupV2

dj.config["display.limit"] = 12

# + tags=["parameters"]
nwb_file_name = "your_session.nwb"  # replace with your ingested session
team_name = "my_team"
interval_list_name = "raw data valid times"
pipeline_preset = "franklab_probe_hippocampus_30khz_ms5_2026_06"
references = None  # inherit stored references; review before creating groups
# None sorts every group; set an explicit subset such as [0, 2] if needed.
sort_group_ids = None
# -

# ## 1. One-time setup
#
# Install default parameters and the owning team, review channel quality
# and references, then create and inspect the sort groups.

initialize_v2_defaults()
LabTeam.insert1(
    {"team_name": team_name, "team_description": "spike sorting"},
    skip_duplicates=True,
)

# ### Review channels and references before creating groups
#
# Inspect the existing bad-channel flags and acquisition reference metadata.
# Finalize any bad-channel edits BEFORE creating groups: grouping omits
# flagged channels, and later flags do not change existing membership.
# If groups already exist, inspect `SortGroupV2.preview_existing_entries(
# nwb_file_name)` before explicitly recreating them; this notebook reuses them.

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

# Automated detection is optional. Its coherence/PSD thresholds are derived
# from Neuropixels; inspect suggestions for polymer probes and do not rely
# on a clean result for small tetrode groups. To propose changes:
#
# ```python
# from spyglass.spikesorting.v2.bad_channels import suggest_bad_channels
# reviewed_report = suggest_bad_channels(nwb_file_name, persist=False)
# reviewed_report
# ```
#
# After inspecting the report, persist exactly those suggestions separately:
#
# ```python
# suggest_bad_channels(
#     nwb_file_name, persist=True, reviewed_report=reviewed_report
# )
# ```
#
# Choose the sorting reference before grouping. `references=None` inherits
# each group's stored reference (`-1`/None: none, `-2`: global median,
# nonnegative: that electrode). To override it, set `references` to a mapping
# from every included `electrode_group_name` to its reference electrode ID
# or sentinel. The acquisition reference is not automatically a suitable
# sorting reference; inspect the resulting group table and geometry below.

if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
    SortGroupV2.set_group_by_shank(
        nwb_file_name=nwb_file_name, references=references
    )
sort_groups = describe_sort_groups(nwb_file_name)
if sort_groups.empty:
    raise ValueError(f"No SortGroupV2 rows found for {nwb_file_name!r}.")
plot_sort_group_geometry(nwb_file_name)
sort_groups


# ## 2. Pick a pipeline preset
#
# `describe_pipeline_presets()` returns a table of what each shipping pipeline
# preset does — the sorter, the parameter rows each stage uses, the intended
# use, and (a known footgun) the units of the detection threshold — so you can
# choose one without reading the module source.

# The default is the runnable MS5 alternative, not an automatic choice of
# the lab's preferred scientific recipe. For hippocampal polymer probes:
#
# | Choice | Recommendation | Runtime requirement |
# | --- | --- | --- |
# | MountainSort4, local | Lab production recipe | MS4 backend in a compatible `numpy<2` environment |
# | MountainSort4, container | Same production recipe on a modern host | Singularity/Apptainer and the catalog's container image |
# | MountainSort5 | Alternative; notebook default | Standard v2 environment |
#
# Match the catalog's region, sampling rate and probe metadata to your
# recording. `production` means lab-recommended, `alternative` is a supported
# substitute, and `experimental` needs scientific validation for your use.
# Inspect `notes` for the exact execution requirements; selecting a preset
# never silently switches its sorter or backend.
#

describe_pipeline_presets()

# ### Customize a preset
#
# The shipping presets cover the common Frank Lab recipes, but you can adapt them
# without hand-editing parameter rows. `describe_pipeline_preset(name)` expands
# one preset into the exact parameter row each stage uses:

describe_pipeline_preset(pipeline_preset)

# `clone_pipeline_preset` derives a new preset from an existing one by tuning a single
# knob: pass the parameter you want to change as a keyword and it builds only the
# new parameter rows that differ, reusing the base preset's rows for every
# untouched stage. Here we lower the MountainSort5 detection threshold (a common
# tweak for low-amplitude units). The clone is then selectable by name like any
# shipping preset — set `pipeline_preset` to it to use it below. (Registering a
# name that already exists raises, so each cell is guarded to be safe to re-run.)

if "my_lab_ms5_lower_threshold" not in list_pipeline_presets():
    clone_pipeline_preset(
        pipeline_preset,
        "my_lab_ms5_lower_threshold",
        detect_threshold=5.0,
    )
describe_pipeline_preset("my_lab_ms5_lower_threshold")

# For a fully custom pipeline — a different sorter, or a stage combination no
# shipping preset covers — `register_pipeline_preset(name, {...})` adds a preset from the
# parameter-row names you choose (each must already exist; `initialize_v2_defaults`
# seeded the rows below). The registration lives for this session; to ship a
# durable recipe, add it to the preset catalog. The mapping is the stages a run
# touches:

if "my_lab_custom" not in list_pipeline_presets():
    register_pipeline_preset(
        "my_lab_custom",
        {
            "preprocessing_params_name": "franklab_hippocampus_2026_06",
            "artifact_detection_params_name": "default",
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "franklab_default_auto_curation_2026_06",
        },
    )

# ## 3. Sort the whole session at once
#
# A real session has one sort group per shank. `run_v2_pipeline_session` loops
# the single-group runner over all of them (run `preflight_v2_pipeline_session`
# first for a read-only whole-session check), returning one entry per group with
# an `outcome` of `"ok"` or `"failed"`. With `continue_on_error=True` a failed
# group is recorded (with `error_type`, `error`, and `partial_run_summary`)
# instead of stopping the batch. Pick `pipeline_preset` explicitly from
# `describe_pipeline_presets()` — the session runner infers no default.
# `describe_run(session_results)` renders the whole batch as one receipt: a
# summary row with the ok / failed / zero-unit / with-warnings counts, then a
# row per group (and per warning), so failed and zero-unit groups don't hide in
# a long list.

# Read-only whole-session check first (run_v2_pipeline_session also preflights
# internally); inspect report.ok / report.errors before committing compute.
session_report = preflight_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    sort_group_ids=sort_group_ids,
)
# Freeze the validated target list so execution uses exactly these groups.
target_sort_group_ids = [
    row["sort_group_id"] for row in session_report.group_reports
]
print(session_report.summary())
sort_groups[sort_groups["sort_group_id"].isin(target_sort_group_ids)]

# Inspect the plan above before running this cell. Groups run sequentially;
# completed stages are reused on a retry, while a failed stage starts again.
session_results = run_v2_pipeline_session(
    nwb_file_name=nwb_file_name,
    interval_list_name=interval_list_name,
    team_name=team_name,
    pipeline_preset=pipeline_preset,
    sort_group_ids=target_sort_group_ids,
    continue_on_error=True,
)
describe_run(session_results)
