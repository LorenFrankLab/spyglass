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
source .venv-spikesorting-v2/bin/activate
uv pip install -e ".[test,spikesorting-v2-validation]"
uv pip install "spikeinterface>=0.104,<0.105" "neuroconv[mearec]"
uv pip install neuron LFPy
```

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

## Baselines

`baselines/` holds the v1 spike-sorting baseline that Phase 1 parity tests
compare against. Unlike the fixtures, the small `.pkl` / `.json` baseline
artifacts **are** committed; see [`../baseline_capture.py`](../baseline_capture.py).
