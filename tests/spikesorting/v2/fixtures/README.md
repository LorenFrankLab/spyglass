# Modern spike-sorting validation fixtures

Simulated ground-truth recordings used to validate the modern (`v2`) spike
sorting pipeline. MEArec plants known spikes from biophysically simulated
neurons, so a sort can be scored against a true answer.

The recording/NWB fixtures are **not** committed to git — they are large and
deterministic, so they are regenerated locally or in CI from
[`generate_mearec.py`](generate_mearec.py).

## Fixture strategy

| Fixture | Source | Used by | Real spikes? |
|---|---|---|---|
| `minirec20230622.nwb` | Existing v0/v1 fixture | Plumbing tests (module import, schema validation, populate-doesn't-crash). Not used for sort correctness. | Likely none — too short |
| `mearec_polymer_128ch_60s.nwb` | MEArec → NWB | Ground-truth precision/recall; brain-region tracing. **Primary fixture** — 128-channel LLNL polymer probe, the current Frank-lab implant configuration. | Yes (planted) |
| `mearec_neuropixels_60s.nwb` | MEArec → NWB | Dense-probe sorter smoke/correctness coverage | Yes (planted) |
| `mearec_polymer_128ch_drift_120s.nwb` | MEArec → NWB | Motion-correction validation | Yes (planted, with drift) |
| Real lab dataset | User-provided via `SPIKESORTING_V2_REAL_NWB_PATH` | v1-parity smoke test, memory/runtime budget, end-to-end "works on real data" smoke. Tests skip if the variable is unset. | Yes |

The polymer probe geometry (4 shanks × 32 contacts, 26 µm pitch) mirrors the
Frank-lab reference probe metadata `128c-4s6mm6cm-15um-26um-sl`. It is defined
once in [`mearec_to_nwb.py`](../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py)
(`polymer_probe_layout`) and shared by fixture generation and NWB conversion so
the MEArec channel order always matches the NWB electrode order.

## Generating the fixtures

Fixture generation needs the validation extra **and** NEURON + LFPy (MEArec
simulates templates biophysically):

```bash
conda activate spyglass_spikesorting_v2
# The env (environments/environment_spikesorting_v2.yml) already installs
# .[test,spikesorting-v2,spikesorting-v2-validation] + SpikeInterface 0.104.
# Fixture GENERATION additionally needs NEURON + LFPy:
pip install "neuron<9" "LFPy<2.3.7"
```

The version caps are load-bearing: NEURON 9 dropped the legacy random API
(`scop_random` / `nrn_random_arg`) that MEArec's bundled BBP `.mod`
mechanisms still use, so `nrnivmodl` cannot compile them under 9.x. LFPy
2.3.7 requires `neuron>=9.0.1`, so the last working pair is LFPy 2.3.6
(`neuron>=7.7.2`) with NEURON 8.2.x. The same pins are applied in the
`pytest-v2` CI job (`.github/workflows/test-conda.yml`).

`neuroconv` is overridden to a modern release because the project's pinned
`spikeinterface` would otherwise drag it down to a version with no `mearec`
extra.

Generation is a long-running biophysical simulation — run it inside `tmux`:

```bash
# Fast end-to-end pipeline check first (tiny templates, short recording):
python tests/spikesorting/v2/fixtures/generate_mearec.py --smoke

# Full fixture set:
python tests/spikesorting/v2/fixtures/generate_mearec.py \
    --base-dir tests/_data/spikesorting_v2 --database-prefix pytests
```

The script registers each probe with MEAutility, simulates a cached template
library, generates each recording, converts it to a Spyglass-ingestible NWB
file, and round-trips the NWB through `insert_sessions` to confirm it ingests.
Template libraries and recordings are cached under
`<base_dir>/mearec_work/`; rerunning skips finished steps. Provenance — package
versions, seeds, and SHA-256 hashes — is written to `fixtures_manifest.json`.

`generate_mearec.py` always runs against the isolated Docker test database and
a temporary `SPYGLASS_BASE_DIR`; it never connects to production. Pass
`--skip-ingestion` to produce fixtures without a database connection.

## Downloading the fixtures (canonical artifacts)

The MEArec recording is the output of a NEURON biophysical simulation and is
**not byte-reproducible across OS/arch** — macOS arm64, Linux x86, and a
committed value all produce different `recording_h5_sha256` (only `n_units` and
`probe_json_sha256` are cross-platform stable). So regenerating on every CI run
is slow *and* cannot be hash-gated reliably. Instead the canonical NWBs are
generated **once**, uploaded, and **downloaded** on demand by
[`_fetch.py`](_fetch.py). Because the downloaded bytes are fixed, `nwb_sha256`
from `fixtures_manifest.json` becomes a real integrity gate again.

**Configure once (after uploading to Box):** fill in `FIXTURE_URLS` in
[`_fetch.py`](_fetch.py) with each file's Box **direct-download** URL — *not*
the `/s/<token>` web share link (that returns an HTML preview page). Share the
file "people with the link" + download allowed; from its share link
`https://ucsf.box.com/s/<token>` the direct URL is:

```python
"mearec_polymer_smoke": (
    "https://ucsf.box.com/index.php?rm=box_download_shared_file"
    "&shared_name=<token>&file_id=f_<file-id>"
),
```

`<file-id>` is the numeric Box file id (file info panel, or the share-page
HTML); `urllib` follows Box's redirect to the file. Leave an entry `None` to
keep generate-or-skip behaviour for that fixture.

**How it's consumed:**

- **Locally** — the v2 `conftest.py` `pytest_sessionstart` hook calls
  `ensure_fixture("mearec_polymer_smoke")` before any test, so a configured URL
  auto-downloads + verifies the per-PR fixture. Set `SPYGLASS_V2_FETCH_FULL=1`
  to also pull the larger nightly fixtures. No-op when the file already exists.
- **In CI** — the `pytest-v2` job downloads via
  `python tests/spikesorting/v2/fixtures/_fetch.py <name>` (smoke per-PR; the 60s
  polymer on nightly/manual), behind an `actions/cache` step so unchanged
  fixtures aren't re-downloaded. There is **no** generate fallback: `_fetch`
  retries transient network errors but a hash mismatch or a bad/rotated Box URL
  **fails the job loudly** (a silent regenerate would hide a corrupt artifact
  and re-introduce the NEURON/LFPy toolchain). Regeneration is the manual
  operation below, not a CI step — so the per-PR job carries no biophysical
  toolchain.

### Regenerate → upload → re-commit runbook

When the probe layout or a fixture spec changes (or you intentionally refresh):

1. Regenerate the affected fixture(s): `generate_mearec.py` (full) or
   `--only <name>` (one fixture; preserves the other manifest entries).
2. Upload the new `<name>.nwb` to Box and update `FIXTURE_URLS` in
   [`_fetch.py`](_fetch.py) with the new Box download URL (resolve the `/s/`
   share link to the `box_download_shared_file` form shown above).
3. Commit the regenerated `fixtures_manifest.json` (its `nwb_sha256` now matches
   the uploaded bytes — this is the gate `_fetch.py` enforces on download).
   The CI cache auto-busts: its key hashes `fixtures_manifest.json` + `_fetch.py`,
   so a fixture refresh or URL change invalidates it with no manual bump.

The fixtures themselves are gitignored (large); only `fixtures_manifest.json`
and the `FIXTURE_URLS` in `_fetch.py` are committed.

## Baselines

`baselines/` holds the v1 spike-sorting baseline that Phase 1 parity tests
compare against. Unlike the fixtures, the small `.pkl` / `.json` baseline
artifacts **are** committed; see [`../baseline_capture.py`](../baseline_capture.py).
