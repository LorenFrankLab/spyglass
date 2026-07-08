# Ingestion Process

## Step 0: NWB files

Before beginning with spyglass, data must be compiled into the standardized NWB
file format. NWB files contain everything about the experiment and form the
starting point of all analyses. Numerous
[online tutorials](https://nwb.org/converting-data-to-nwb/) exist to help get
you started in this process, as well as existing packages for lab-specific
conversions ([1](https://github.com/catalystneuro),
[2](https://github.com/LorenFrankLab/trodes_to_nwb)) that can be used as a
reference.

The following sections describe how data is extracted from these files and
brought into the spyglass system. For best compatibility, please use these as
reference when creating your NWB files.

## What is ingestion?

Ingestion is the process of extracting data from the raw NWB file and storing it
in Spyglass tables.

## How does it work?

### For users

For most users all you'll need to do is call
`spyglass.common.insert_sessions(nwb_file_name)` which will iterate through
tables populated from the raw NWB file and create appropriate entries.

### In the background

\*Note: Migration to this format is in progress, and not yet implemented for all
ingestion tables

Tables that are populated from the raw NWB file are instances of the
`SpyglassIngestion` class. Tables of this class must define the following
properties which enable finding relevant data in the NWB file and creating
corresponding table entries.

- `_source_nwb_object_type`: defines the `pynwb` object type containing data for
    this table (eg. `pynwb.misc.Units` for `ImportedSpikesorting`).
- `table_key_to_obj_attr`: A dict of dicts mapping table keys to NWB object values.
  There are several options for identifying object values:
    - `str`: Spyglass will take the object attribute of that name
    - `Tuple[str, any]`: Spyglass will take the object attribute matching the first
    term, defaulting to the value of the second if unavailable
    - `Callable`: A function that takes the nwb object and returns the value to
    insert

With these defined the table entries are populated from the following methods:

- `insert_from_nwbfile`: top-level function that extracts and inserts all
    entries for the table
- `get_nwb_objects`: returns all nwb objects from the raw file containing data
    for the table. By default, returns all instances of
    `_source_nwb_object_type`, but can be overwritten on a per-table basis for
    more selective restriction
- `generate_entries_from_nwb_object`: Called for each identified nwb object.
    Generates table entries using the `table_key_to_obj_attr` mapping.

## NWB to Spyglass table mappings

*In progress*: To aid in creating spyglass-compatable NWB files, we provide a
[Reference Table](../ForDevelopers/ingestion_mapping.md) which maps spyglass
table entries to the source nwb objects and attributes.

For entries not yet contained in the updated format, a complete list of this
mappings can also be found [here](../ForDevelopers/UsingNWB.md).

## Fiber photometry

Spyglass ingests fiber-photometry setups recorded with the
[`ndx-fiber-photometry`](https://github.com/catalystneuro/ndx-fiber-photometry)
NWB extension into the `common_photometry` schema. `insert_sessions` populates:

- Six reusable device/reagent tables — `Indicator`, `ExcitationSource`,
    `Photodetector`, `DichroicMirror`, `OpticalFilter`, and `OpticalFiber` — a
    shared catalog keyed by device name and storing only reusable model specs.
    Each is reference-scoped: it ingests only the objects a `FiberPhotometryTable`
    references, so a file without a `FiberPhotometry` container is a clean no-op.
- `FiberPhotometryConfig` — one row per `FiberPhotometryTable` row, with foreign
    keys to the device tables plus the fiber's session-specific insertion
    metadata and per-channel excitation/emission wavelengths. When the fiber's
    indicator carries a viral injection, the row also links (nullably) to the
    shared, session-scoped `common_optogenetics.VirusInjection` — the same table
    optogenetic effectors use — rather than duplicating injection modeling.
- `FiberPhotometryResponseSeries` — one row per recorded
    `FiberPhotometryResponseSeries`, storing the NWB object id (not the array) so
    the trace is retrievable via `fetch_nwb()`. Its `.Fiber` part maps each data
    column to the `FiberPhotometryConfig` row it records, and it references an
    `IntervalList` of the series' valid (recorded) times so the trace can be
    time-restricted against the rest of Spyglass (as `Raw` does).

The recorded fluorescence traces are **not** copied into the database — the table
stores only the NWB object id and the trace stays in the file. A file typically
holds several response series, so restrict to a single one (e.g. by `name`) and
retrieve it as a time-indexed `pandas.DataFrame` (per-fiber columns, time axis
from the series' `rate`/`starting_time` or explicit `timestamps`):

```python
from spyglass.common import FiberPhotometryResponseSeries

file_key = {"nwb_file_name": "my_session_.nwb"}
# inspect the available series for this file
(FiberPhotometryResponseSeries & file_key).fetch("name")

# retrieve one series' trace
series = FiberPhotometryResponseSeries & file_key & {"name": "green_DLS"}
df = series.fetch1_dataframe()
```

Viral-injection provenance (construct, titer, site) for a fiber's indicator is
retrieved with `FiberPhotometryConfig.fetch_injection()`, which resolves the link
through the shared `VirusInjection`/`Virus` tables. Use it rather than a bare join:
`FiberPhotometryConfig` and `VirusInjection` share several column names (fiber
*insertion* vs injection *site*), so a natural join would silently match on them
and drop rows.

```python
from spyglass.common import FiberPhotometryConfig

FiberPhotometryConfig().fetch_injection(as_dict=True)  # construct/titer/site per fiber
```

An unmodeled `FiberPhotometryTable` column or device attribute is ignored with a
warning naming it, rather than dropped silently. Analysis (dF/F, isosbestic
correction, downsampling) is left to a follow-up; this layer stops at retrieving
the raw trace.

Data-production constraint: files must embed NWB **`core` 2.9.0** — write them
with pynwb 3.1.x. This lets the feature ship on the current dependency floor with
no `pynwb`/`hdmf` bump; the `ndx-fiber-photometry` package is only needed to
*build* test fixtures and is never imported at ingest time (NWB types are matched
by class name and gated on the file-embedded namespace version).

Ingesting a photometry file also touches `common_optogenetics`, whose optical
fiber tables (`OpticalFiberDevice`/`OpticalFiberImplant`) share the underlying
`ndx-ophys-devices` types. Those tables now skip photometry-referenced fibers via
a `get_nwb_objects()` gate — a behavioral change only, with **no schema change**;
a non-photometry file is unaffected.
