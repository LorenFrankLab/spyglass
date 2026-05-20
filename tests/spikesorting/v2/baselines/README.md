# Modern-spike-sorting baselines

Small artifacts the v2 validation suite compares against.

## `legacy_schemas.json`

Byte-level snapshot of every v0/v1 DataJoint ``definition`` string consumed
by `test_no_legacy_schema_changes` in
[`../test_legacy_runtime_boundary.py`](../test_legacy_runtime_boundary.py).
The legacy-runtime boundary is meant to be quiescent: any drift here would
mean a v0/v1 schema migration sneaked in alongside an SI bump.

Regenerate (when a v0/v1 schema legitimately needs to change):

```bash
source .venv-spikesorting-v2-si0104/bin/activate
python tests/spikesorting/v2/baselines/regen_legacy_schemas.py
```

Commit the regenerated `legacy_schemas.json` in the same change that edits
the source ``definition`` so the drift is explicit and reviewable.

## `baseline_v1_spike_times.pkl` / `baseline_v1_recording_meta.json`

Written by [`../baseline_capture.py`](../baseline_capture.py) when run
against a real lab NWB. Manually invoked; not produced in CI. See the v1
baseline-capture docstring for invocation.
