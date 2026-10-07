# Host the two-session UnitMatch fixtures

**Status:** Not started. Needs the owner to upload two files.

**Why:** `test_v2_unitmatch_polymer_mearec_ground_truth`
(`tests/spikesorting/v2/test_unitmatch.py`) checks cross-session UnitMatch
recovery on two simulated sessions of the same 128-channel polymer probe. Its
two fixtures are not hosted, so the test skips everywhere, and the CI step
"Two-session matcher ground-truth gate (nightly / manual dispatch)" in
`.github/workflows/test-conda.yml` fails by design until they are.

**Files:** `mearec_polymer_128ch_2sessions_s1.nwb` and
`mearec_polymer_128ch_2sessions_s2.nwb`. Generate them locally (generation is
too heavy for CI: 120 s of 128-channel MEArec per session):

```bash
python tests/spikesorting/v2/fixtures/generate_mearec.py \
    --only mearec_polymer_128ch_2sessions_s1 \
    --only mearec_polymer_128ch_2sessions_s2
```

## Steps

1. Check the local files against `nwb_sha256` in
   `tests/spikesorting/v2/fixtures/fixtures_manifest.json` (already recorded
   for both). The fetcher verifies downloads against it, and MEArec generation
   is not byte-reproducible, so upload the files that match; if you regenerate,
   update the manifest checksums in the same commit.
2. Owner: upload both NWB files to the same Box location as the other v2
   fixtures and create shared download links. Then set their entries in
   `FIXTURE_URLS` in `tests/spikesorting/v2/fixtures/_fetch.py` (currently
   `None`).
3. Update `tests/spikesorting/v2/fixtures/README.md` ("Two-session polymer pair
   (not hosted)") and the comment above the CI step to say they are hosted.
4. Run the CI step once with `workflow_dispatch` and confirm the test runs
   (not skips) and passes.

**Alternative if they should not be hosted:** remove the CI step, and document
the test as local-only in the fixtures README.

Delete this file once done.
