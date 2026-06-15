# spikesorting-v2 operations runbook

Operational notes distilled from session traps. Read this **before** touching
fixtures, baselines, the v1 capture pipeline, or the Docker/MySQL setup —
each section catalogs a concrete mistake that cost real time in earlier
sessions and how to avoid repeating it.

---

## 1. Fixture regeneration lifecycle

### When `nwb_sha256` changes, decide patch vs recapture

Spike times are deterministic from the tuple
`(recording_h5, sorter, sorter_params, seed)`. If only the NWB sidecar
writer changes (and the upstream MEArec h5 is unchanged), the spike times
are unchanged too — even though `nwb_sha256` drifts because the sidecar
bytes differ.

**Decision tree before kicking off a full v1 baseline recapture:**

1. Did the MEArec `.h5` file change? (Templates regenerated? Recording
   regenerated?) → **Full recapture required.**
2. Did sorter params (preprocessing, sorter, sorter_params, artifact_params,
   seed) change? → **Full recapture required.**
3. Only the NWB-writer code changed and added/edited metadata columns? →
   **Patch `nwb_sha256` in each `baseline_v1_recording_meta.json`.**

The patch path is seconds; recapture is ~15 min per script × multiple
scripts × every infrastructure hiccup. Don't conflate "fingerprint
changed" with "underlying data changed."

**Equivalence verification when patching:** run the v1 sorter on one
shank fresh and `set(samples_old) ^ set(samples_fresh)` against the
captured pickle. Zero diff = baseline still valid; just refresh the
hash.

### `templates_overrides` is required for narrow probes

`mearec_to_spyglass_nwb` loads `templates` (and `load_waveforms=False` for
memory), which pulls `template_locations` so the GT sidecar's
`position_x/y/z` columns are populated. Without `load=['templates']`,
positions silently write as NaN. The polymer fixture works with MEArec's
default placement volume; the **tetrode** fixture rejects all candidate
cells unless `FixtureSpec.templates_overrides` opens `ylim`/`zlim` and
drops `min_amp` (see `generate_mearec.py:_profiles`).

---

## 2. v1 capture script invariants

### `--sorter-param-name` must match the original capture exactly

The most common mistake when targeted-recapturing a subset of shanks:
using `--sorter-param-name default` instead of `smoke_clusterless_5uv`.
`default` carries `noise_levels=[1.0]` in the v1 row, which then appears
in the parity fingerprint as `right-only=['noise_levels']` and breaks
`canonical_sorter_params` equality.

**Always copy the arg verbatim from the production capture script:**

```bash
grep "sorter-param-name" tests/spikesorting/v2/scripts/capture_polymer_*.sh
```

Currently:
- Clusterless: `--sorter-param-name smoke_clusterless_5uv`
- MS4: `--sorter-param-name ms4_60s_polymer`

### v1 schemas don't honor `database.prefix`

`spikesorting_v1_*` schemas are hardcoded — the `--database-prefix` flag
only namespaces v2 tables. Between consecutive captures, leftover v1 rows
cause `IntegrityError: foreign key constraint fails` on
`__spike_sorting_recording`.

**Drop v1 schemas between capture sessions:**

```python
import pymysql
c = pymysql.connect(host='127.0.0.1', port=<container_port>,
                    user='root', password='tutorial')
cur = c.cursor()
cur.execute('SET FOREIGN_KEY_CHECKS = 0')
cur.execute('SHOW DATABASES')
for (d,) in cur.fetchall():
    if d.startswith('spikesorting_v1') or d.startswith('common_'):
        cur.execute(f'DROP DATABASE IF EXISTS `{d}`')
cur.execute('SET FOREIGN_KEY_CHECKS = 1')
c.commit()
```

Note: `FOREIGN_KEY_CHECKS = 0` is required — `common_interval.interval_list`
is referenced by `common_ephys._raw` via FK, plain DROP fails.

### Capture scripts must NOT run while pytest is in flight

Both `DockerMySQLManager` (pytest) and the capture script's
`_docker_test_server_credentials` target the same container name
`spyglass-pytest-{branch}`. Pytest tears the container down at end-of-run,
killing any captures mid-write. Sequence:

1. Captures complete (verify all pickles refreshed) → 2. Run pytest.

Never overlap.

---

## 3. Docker / MySQL coordination

### Anonymous volumes survive container removal

`datajoint/mysql:8.0` defines `/var/lib/mysql` as a VOLUME. Removing the
container leaves the anonymous volume behind. The next container with the
same name attaches the orphaned volume, finds files, and the MySQL
entrypoint aborts with `--initialize specified but the data directory has
files in it`.

**Always use `-v` when removing a broken container:**

```bash
docker rm -fv spyglass-pytest-spikesorting-v2
docker volume prune -f  # cleans orphans from past sessions too
```

A past session left ~125 GB of orphan MySQL volumes on the host. Run
`docker volume prune -f` periodically.

### `DROP DATABASE` destabilizes the running container

After dropping all user schemas, the container's healthcheck may flip
unhealthy, Docker auto-restarts it, the restart triggers the
`--initialize` codepath, the data dir already has files → container dies.

Two safer alternatives to mass-DROP:

1. **Selective drop** of just the schemas you want to clear (don't touch
   `mysql`, `performance_schema`, `sys`, `information_schema`).
2. **Container rebuild**: `docker rm -fv` + `docker volume prune` + start
   fresh.

### Port mapping is branch-derived, not 3306

`DockerMySQLManager._container_name_from_branch` deterministically maps
branch name → port via SHA256 hash mod port range. The
`spikesorting-v2` branch maps to port 32719. Connect via:

```bash
docker ps --filter "name=spyglass-pytest" --format "{{.Ports}}"
# e.g. "0.0.0.0:32719->3306/tcp"
```

Don't assume 3306.

---

## 4. MEArec / NWB coordinate conventions

### Multi-shank probes: MEArec puts shanks along Z

For the polymer 128-channel probe:

| NWB electrode column | MEArec channel axis | Meaning            |
|----------------------|---------------------|--------------------|
| `rel_x`              | `z`                 | Shank-spanning     |
| `rel_y`              | `y`                 | Within-shank depth |
| `rel_z`              | `x`                 | Depth into tissue  |

Confirmed by inspecting `recgen.channel_positions` — 4 unique z values
at 0/350/774/1124 µm.

### GT sidecar `position_*` columns are MEArec frame

The writer copies `template_locations[unit] = (x, y, z)` directly into
`(position_x, position_y, position_z)`. No axis transformation. So
`position_z` is the shank-spanning axis for polymer, while the NWB
electrode shank-spanning axis is `rel_x`. If you compare them, they share
SCALE but are different SYMBOLS.

### Tetrode: probe must be in MEArec's XY plane

Tetrodes built with `rel_z != 0` raise
`TypeError: 'NoneType' object is not subscriptable` deep inside
MEAutility because `electrode.normal` is never computed. See
`mearec_to_nwb.tetrode_probe_layout` for the axis-swap that puts the
4 contacts at `rel_x = ±6.25`, `rel_y = ±6.25`, `rel_z = 0`.

---

## 5. GT-gate design: soma position ≠ best detection shank

### Why clusterless GT uses max-over-shanks (not nearest-by-position)

For per-shank matching, the obvious approach — "assign each planted unit
to its nearest shank by `position_z`, match its spikes only against that
shank's v2 peaks" — empirically mis-routes 18/24 polymer 60s units. The
GT cell `position_z` spans `[-570, +583]` µm but the polymer shanks are
at z = 0/350/774/1124, so cells off the probe edge in z get routed to
the geometrically-nearest shank even though they fire most strongly on a
DIFFERENT shank (the one whose electrodes actually pick them up).

**Use max-over-shanks instead:** per planted unit, compute time-recall
against each shank's stream independently and take the max. The
constraint is "v2 peaks from a SINGLE shank must explain the planted
spikes" — cross-shank peaks no longer satisfy a planted spike. Lets
electrical reality determine the assignment rather than Euclidean
distance. See `test_clusterless_thresholder_ground_truth` for the
implementation.

### Time tolerance: ±0.4 ms (NOT ±N samples)

Planted intracellular fire moment is offset from detected extracellular
trough by filter group delay + axonal propagation — empirically ~1-3
samples at 32 kHz. Sub-sample tolerances (e.g. ±1.5 samples) reject
valid detections and produce false-low recall. Use SI's standard
`delta_time=0.4 ms` (which is what `compare_sorter_to_ground_truth` uses
internally and what our sibling MS4/MS5 GT gates use).

---

## 6. Cheap-check-first reflexes

Before writing any logic that depends on a precondition, **verify the
precondition with the shortest possible Python expression first**. Each
of these would have saved 15+ minutes in an earlier session:

```python
# Before writing position-routing logic:
python -c "import pynwb, numpy as np; \
  from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import get_ground_truth_units_table; \
  io = pynwb.NWBHDF5IO('<fixture.nwb>', 'r', load_namespaces=True); \
  gt = get_ground_truth_units_table(io.read()); \
  print('any NaN:', np.isnan(gt['position_x'][:]).any())"

# Before claiming "v1 = v2 by parity":
python -c "import pickle; \
  a = pickle.load(open('<old.pkl>', 'rb')); \
  b = pickle.load(open('<new.pkl>', 'rb')); \
  print(all(set(a[k].tolist()) == set(b[k].tolist()) for k in a))"

# Before scheduling a multi-hour recapture:
# Ask: does the data the sorter actually CONSUMES change,
# or just metadata that the parity test happens to fingerprint?
```

Don't search the whole filesystem for paths whose shape you know:

```bash
# BAD (~minutes, scans /):
find / -name baseline_v1_spike_times.pkl

# GOOD (instant):
find /cumulus/edeno/spikesorting-v2-baselines -name baseline_v1_spike_times.pkl
```
