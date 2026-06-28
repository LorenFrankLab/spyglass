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

Build the curation view by composing SpikeInterface figpack sub-views (each
widget returns a figpack view via `.view`) with a `SortingCuration` control and
figpack layout views. Required analyzer extensions (checked by SI's summary
widget): `correlograms`, `spike_amplitudes`, `unit_locations`,
`template_similarity` — plus `templates`/`waveforms`/`noise_levels`/
`random_spikes`, and `quality_metrics`/`template_metrics` for the displayed unit
columns.

```python
def _subview(plot_fn, analyzer, **kw):
    return plot_fn(analyzer, backend="figpack",
                   generate_url=False, display=False, **kw).view

units    = generate_unit_table_view(analyzer, ["firing_rate", "snr"])
templates = _subview(sw.plot_unit_templates, analyzer, hide_unit_selector=True)
amps      = _subview(sw.plot_amplitudes,     analyzer, hide_unit_selector=True)
corr      = _subview(sw.plot_crosscorrelograms, analyzer, hide_unit_selector=True)
curation  = ssv.SortingCuration(default_label_options=["accept", "mua", "noise"])

view = vv.Splitter(
    direction="horizontal",
    item1=vv.LayoutItem(view=vv.Box(direction="vertical", items=[
        vv.LayoutItem(view=units, title="Units"),
        vv.LayoutItem(view=curation, title="Curation"),
    ])),
    item2=vv.LayoutItem(view=vv.Splitter(
        direction="vertical",
        item1=vv.LayoutItem(view=templates, title="Templates"),
        item2=vv.LayoutItem(view=vv.Splitter(
            direction="vertical",
            item1=vv.LayoutItem(view=amps, title="Amplitudes"),
            item2=vv.LayoutItem(view=corr, title="Cross-correlograms"),
        )),
    )),
)
```

### Released-version drift (must work around in the table)

`sw.plot_sorting_summary(analyzer, curation=True, label_choices=[...],
backend="figpack")` **raises** with this released combo:

```
TypeError: SortingCuration.__init__() got an unexpected keyword argument 'label_choices'
```

SpikeInterface 0.104.7's figpack backend calls
`SortingCuration(label_choices=...)`, but `figpack_spike_sorting==0.1.14` expects
`default_label_options=...`. So the curation-table builder must construct the
`SortingCuration` control itself (as above), **not** rely on SI's
`plot_sorting_summary(curation=True)`. `plot_sorting_summary(curation=False)`
works and returns a `figpack.views.Splitter` (usable for the read-only sub-views).
`figpack_spike_sorting` is **0.1.14, beta** — pin exact versions and re-check this
on any bump.

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
  upload=True, ephemeral=False) -> str`: load the curation's `SortingAnalyzer`,
  ensure the required extensions, build the units table + sub-views +
  `ssv.SortingCuration(default_label_options=label_options or <CurationLabel
  order>)`, compose with figpack layout, then `view.show(upload=..., ephemeral=...,
  wait_for_input=False, open_in_browser=False)` (or `view.save(...)` for a local
  artifact) and return the figure URL. Default `label_options` should be the v2
  `CurationLabel` order `["accept", "mua", "noise"]`, not FigURL-era `"good"`.
- `fetch_curation_from_uri(uri) -> tuple[dict, list]`:
  `GET <uri>/annotations.json`, parse `["annotations"]["/"]["sorting_curation"]`,
  and return `({int(unit_id): labels}, [list(group) for group in mergeGroups])` —
  ready for `CurationV2.insert_curation(labels=..., merge_groups=...)`. Empty /
  missing annotations → `({}, [])`.

## Reproduction

Standalone, no DataJoint: a deterministic SortingAnalyzer is built with
`spikeinterface.generate_ground_truth_recording(seed=0)` +
`create_sorting_analyzer` + the extensions above; the view is built, saved, served
locally, and the annotations round trip is exercised over the real local-server
HTTP PUT/GET. A real DataJoint-produced v2 analyzer carries the same extensions,
so the contract transfers unchanged.
