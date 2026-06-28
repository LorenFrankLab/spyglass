# FigPack spike-sorting curation environment — resolver evidence

Durable record of the isolated environment used to verify the FigPack
spike-sorting curation path against the modern (SortingAnalyzer) stack, and the
verified API contract the DataJoint FigPack curation tables will wrap. This is a
feasibility/verification artifact, not shipped API docs.

**Verdict: feasible.** A curation view builds from a `SortingAnalyzer`, displays
fully offline (local server) or via cloud upload, and an edited curation state
(labels dict + merge groups) round-trips back into Python in a retrievable form
(`annotations.json`). No stop/escalate condition was hit. The round trip was
verified **offline only** (no cloud upload), per the owner's scope decision.

## Environment

- Python **3.11.15**.
- Dedicated `uv` virtualenv, separate from the SI-0.99 base environment, the
  SI-0.104 dev environment, and the matching environment. The curation extra is
  **not** installed into any shared/base environment.
- Install command (basis for the `spikesorting-v2-curation` optional-dependency
  group; `numba`/`scikit-learn`/`scipy`/`pandas` are pulled by the SI analyzer
  extensions the curation view needs, not by FigPack itself):
  ```bash
  uv venv .venv-figpack-curation --python 3.11
  uv pip install "spikeinterface>=0.104,<0.105" figpack figpack-spike-sorting \
                 numba scikit-learn scipy pandas zarr
  ```

## Resolved versions (key packages)

| Package | Version |
| --- | --- |
| figpack | 0.3.20 |
| figpack_spike_sorting | 0.1.14 |
| spikeinterface | 0.104.7 |
| numpy | 2.4.6 |
| zarr | 2.18.7 |
| numba | 0.65.1 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |
| pandas | 3.0.4 |
| probeinterface | 0.3.2 |
| pydantic | 2.13.4 |
| requests | 2.34.2 |

`uv pip check` → **all installed packages are compatible**.

## Packages and import paths (verified)

- Curation UI is **two packages**: core `figpack` plus the spike-sorting
  extension `figpack-spike-sorting` (PyPI name; **imported as
  `figpack_spike_sorting`**). `figpack` alone has no spike-sorting views.
- SpikeInterface 0.104 ships a **figpack widgets backend** (`backend="figpack"`)
  that builds the spike-sorting sub-views directly from a `SortingAnalyzer`
  (`spikeinterface/widgets/utils_figpack.py`,
  `spikeinterface/widgets/sorting_summary.py`). This is the cleanest adapter and
  is preferred over hand-porting v1's `SpikeSortingView` layout.

```python
import figpack.views as vv                  # layout: Box, Splitter, LayoutItem, ...
import figpack_spike_sorting.views as ssv    # SortingCuration, UnitsTable, ...
import spikeinterface.widgets as sw          # plot_* (backend="figpack")
from spikeinterface.widgets.utils_figpack import generate_unit_table_view
```

## View construction (verified working path)

The released `figpack_spike_sorting==0.1.14` `SortingCuration` signature is:

```python
ssv.SortingCuration(*, default_label_options=["mua", "good", "noise"], curation={})
```

Build the curation view with the **minimal-attach** approach: let SpikeInterface
build the whole summary view via `plot_sorting_summary(curation=False)` and attach
**only** the `SortingCuration` control as a sibling. This keeps SpikeInterface the
owner of the layout/sub-view composition (we do not reimplement it) and isolates
our hand-written surface to a single widget plus a one-line wrapper — important
because of the upstream kwarg mismatch documented below.

Required analyzer extensions (checked by SI's summary widget): `correlograms`,
`spike_amplitudes`, `unit_locations`, `template_similarity` — plus
`templates`/`waveforms`/`noise_levels`/`random_spikes`, and
`quality_metrics`/`template_metrics` for the displayed unit columns.

**v2 analyzer source.** `Sorting().get_analyzer(key)`
([sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py)) returns a
SortingAnalyzer whose **base** extension set is only
`random_spikes`/`noise_levels`/`templates`/`waveforms`
([_sorting_analyzer.py](../../../../src/spyglass/spikesorting/v2/_sorting_analyzer.py)).
The four curation-view extras above (`spike_amplitudes`, `correlograms`,
`unit_locations`, `template_similarity`) are **not** in that base set, so the
curation-view builder must compute them on top before calling
`plot_sorting_summary`. v2 already has the machinery for this:
`Sorting.add_extensions` / `ensure_extensions` and `_visualization`'s
`DISPLAY_WIDGET_EXTENSIONS` (used by the existing v2 viz path,
[visualization.py](../../../../src/spyglass/spikesorting/v2/visualization.py)).
So a real v2 analyzer differs from the standalone one only in provenance, not in
the inputs the figpack views consume.

```python
# Ensure the curation-view extensions on top of v2's base analyzer set (in v2 this
# is Sorting.add_extensions / ensure_extensions), then let SpikeInterface build the
# entire summary view and attach ONLY the curation control as a sibling.
analyzer.compute(
    ["spike_amplitudes", "correlograms", "unit_locations", "template_similarity"]
)
summary = sw.plot_sorting_summary(
    analyzer, curation=False, backend="figpack",
    generate_url=False, display=False,
).view                                       # SI owns this composition
control = ssv.SortingCuration(default_label_options=["accept", "mua", "noise"])
view = vv.Box(
    direction="vertical",
    items=[
        vv.LayoutItem(view=summary, title="Sorting summary", stretch=1),
        vv.LayoutItem(view=control, title="Curation", max_size=260),
    ],
)
```

Verified: this composed view saves, serves, and round-trips edited curation
(`labelsByUnit` / `mergeGroups`) identically to a fully hand-built layout.

### Released-version drift — why we attach instead of one-call `curation=True`

`sw.plot_sorting_summary(analyzer, curation=True, label_choices=[...],
backend="figpack")` **raises** with this released combo:

```
TypeError: SortingCuration.__init__() got an unexpected keyword argument 'label_choices'
```

SpikeInterface 0.104.7's figpack backend calls `SortingCuration(label_choices=...)`,
but `figpack_spike_sorting==0.1.14` (the latest on PyPI; upstream `main` is the
same) expects `default_label_options=...`. The mismatch is unresolved upstream on
both released and `main`, and it breaks **only** the curation-control
instantiation. Hence the minimal-attach path above:

- Use `plot_sorting_summary(curation=False)` (works) for the whole summary view and
  attach `ssv.SortingCuration(default_label_options=...)` ourselves — we do NOT
  reimplement SI's layout, and we do NOT depend on SI's `curation=True` path.
- Add a small **compatibility probe** in the table module that detects when the
  one-call `curation=True` path becomes usable on a future release, so the attach
  shim can be dropped and SI can own the curation control too. Worth filing the
  SI↔extension kwarg mismatch upstream.
- `figpack_spike_sorting` is **0.1.14, beta** — pin exact versions and re-check on
  any bump.

## Display / publish / save (verified)

`figpack/core/figpack_view.py :: FigpackView.show / .save`:

```python
url = view.show(*, title, upload=False, open_in_browser=False, inline=False,
                wait_for_input=False, port=None, ephemeral=False, ...)  # -> str
view.save(output_path, *, title, description="", script="")            # folder or .tar.gz
```

- **Offline (verified):** `upload=False` starts a local server
  (`ThreadingHTTPServer`, daemon thread, started with `enable_file_upload=True`)
  and returns `http://localhost:<port>/figure_<id>`. `wait_for_input=False` makes
  it non-blocking. This is the path used for the round trip below.
- **Static bundle (verified):** `view.save(folder)` writes
  `index.html`, `data.zarr`, `extension-figpack-spike-sorting.js`,
  `extension_manifest.json`, `assets/`. No annotation server (read-only artifact).
- **Cloud (NOT exercised — offline-only scope):** `upload=True` uploads to
  figpack.org and returns a hosted URL (`https://figures.figpack.org/figures/.../`).
  Requires env `FIGPACK_API_KEY` unless `ephemeral=True`
  (`figpack/core/_show_view.py`). Config: `FIGPACK_API_BASE_URL`
  (default `https://figpack-api.figpack.org`), bucket `figpack-figures`
  (`figpack/core/config.py`). The annotations contract below is identical for a
  hosted figure; only the write path is authenticated.

> **Durability caveat (affects the persisted table's default).** A
> `FigPackCuration` Computed row stores a `figpack_uri`. For that URI to be both
> **durable** and **editable** (round-trippable), it must be the hosted/cloud
> path (or a self-hosted figpack figure service): the offline local-server URL is
> process-ephemeral, and `view.save()` produces a durable but **read-only** bundle
> (no annotation server, so no edit round trip). 5a verified only the offline
> mechanism. Therefore 5b must verify the cloud round trip (`FIGPACK_API_KEY` +
> a cloud smoke that PUT/GETs `annotations.json` on a hosted figure) before
> relying on `upload=True` as the table's path. Do not assume the cloud annotation
> write path works just because the offline one does — only the read (GET) path is
> proven identical.

## Curation-state round trip (the gating deliverable — verified offline)

Edited curation is **not** written back to the zarr `curation` dataset. It is
stored as a figpack **figure annotation** and persisted as a plain file at
`<figure_url>/annotations.json`
(`figpack_spike_sorting/.../FPSortingCuration.tsx`,
`figpack-figure/.../useSavedFigureAnnotations.ts`):

```json
{"annotations": {"/": {"sorting_curation": "<json-string>"}}}
```

where `<json-string>` decodes to:

```json
{"labelsByUnit": {"<unit_id>": ["label", ...]},
 "mergeGroups": [[unit_id, unit_id, ...], ...],
 "labelChoices": ["accept", "mua", "noise"],
 "isClosed": true}
```

- **Save** (browser "Save"): HTTP `PUT <figure_url>/annotations.json`. The local
  server writes it to disk (`figpack/core/_file_handler.py`); a hosted figure
  writes via the authenticated figpack API.
- **Retrieve** (Python): HTTP `GET <figure_url>/annotations.json` (append a
  cache-buster `?cb=<n>`), then read `["annotations"]["/"]["sorting_curation"]`.

`labelsByUnit` / `mergeGroups` is the **same shape** as v1 FigURL's kachery JSON
(`v1/figurl_curation.py :: get_labels / get_merge_groups`), so it maps directly
onto `CurationV2.insert_curation(labels=..., merge_groups=...)`.

Verified end to end against the real local server: `PUT -> 201`, `GET -> 200`,
parsed `labels == {0: ["noise"], 1: ["accept"], 2: ["mua"]}` and
`merge_groups == [[3, 4]]` (exact match to the written state, confirmed persisted
on disk by a second independent GET).

### macOS local-server quirk (harmless)

On macOS the local server's PUT **success response body** can be malformed: it
computes `file_path.relative_to(self.directory)`, and the temp dir under
`/var` ⇄ `/private/var` symlink makes `relative_to` raise, so a `500` line is
appended into the body after the `201` status line. The file is still written
correctly. The retrieval path must rely on the **GET** (which is what
`fetch_curation_from_uri` does anyway), not on parsing the PUT response body.

### Seeding an initial curation into an editable view

The Python-side `ssv.SortingCuration(curation=...)` init dict is written to a
zarr `curation` dataset, but the **editable** frontend reads/writes the
`annotations` path (`sorting_curation`), not that dataset. To seed prior labels
into an editable view, pre-write `annotations.json` with the `sorting_curation`
key (same shape as above) rather than relying on the `curation=` kwarg.

## Helper shapes the DataJoint tables will wrap

- `build_curation_view(curation_key, label_options=None, metrics=None,
  upload=..., ephemeral=False) -> str`: load the curation's `SortingAnalyzer`
  via `Sorting().get_analyzer(...)`, **ensure the curation-view extensions**
  (`spike_amplitudes`/`correlograms`/`unit_locations`/`template_similarity`) on
  top of the base set via `Sorting.add_extensions`/`ensure_extensions`, build the
  summary with `sw.plot_sorting_summary(analyzer, curation=False, backend="figpack",
  generate_url=False, display=False).view` and **attach** only
  `ssv.SortingCuration(default_label_options=label_options or <CurationLabel
  order>)` as a sibling layout item (minimal-attach; do not reimplement SI's
  layout), then `view.show(upload=..., ephemeral=...,
  wait_for_input=False, open_in_browser=False)` (or `view.save(...)` for a local
  artifact) and return the figure URL. Default `label_options` should be the v2
  `CurationLabel` order `["accept", "mua", "noise"]`, not FigURL-era `"good"`.
  The `upload` default is intentionally left open here: only the offline path is
  verified, and a persisted+editable `figpack_uri` needs the cloud/hosted path
  (see the durability caveat above), so 5b must gate `upload=True` on
  `FIGPACK_API_KEY` + a cloud smoke before adopting it as the table default.
- `fetch_curation_from_uri(uri) -> tuple[dict, list]`:
  `GET <uri>/annotations.json`, parse `["annotations"]["/"]["sorting_curation"]`,
  and return `({int(unit_id): labels}, [list(group) for group in mergeGroups])` —
  ready for `CurationV2.insert_curation(labels=..., merge_groups=...)`. Empty /
  missing annotations → `({}, [])`.

## Reproduction and the analyzer-fixture decision

Standalone, no DataJoint: a deterministic SortingAnalyzer is built with
`spikeinterface.generate_ground_truth_recording(seed=0)` +
`create_sorting_analyzer` + the extensions above; the view is built, saved, served
locally, and the annotations round trip is exercised over the real local-server
HTTP PUT/GET.

**Fixture decision (accepted).** This feasibility check used a *standalone*
SpikeInterface analyzer rather than a populated v2 DataJoint analyzer — an
explicit owner decision, made because the isolated curation environment has no
DataJoint/MySQL stack and the figpack views consume a `SortingAnalyzer` + the
extensions above regardless of how the analyzer was produced. The standalone
analyzer is representative: it carries v2's base extensions plus the same
curation-view extras computed on top (see "v2 analyzer source" above).

**Residual risk for 5b (must smoke).** 5b binds to the concrete v2 surface —
`Sorting().get_analyzer(key)` and v2's `add_extensions`/`ensure_extensions` — which
this offline check did not exercise. The first 5b step should run one documented
smoke of `build_curation_view` against a **real populated v2 analyzer** (needs the
Docker/MySQL v2 environment) to confirm `get_analyzer()` returns a usable analyzer
and the curation-view extensions ensure cleanly end to end.
