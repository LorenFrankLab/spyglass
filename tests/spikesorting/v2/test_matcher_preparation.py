"""Custom matching preparation is independent of UnitMatchPy and bundle format."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "custom_layout,bad_identity", [(False, False), (True, False), (True, True)]
)
def test_custom_matcher_uses_its_preparer_without_unitmatch(
    tmp_path, custom_layout, bad_identity
):
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        import uuid
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import spikeinterface as si

        class BlockUnitMatch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('UnitMatchPy'):
                    raise AssertionError('Custom matching imported UnitMatchPy')

        sys.meta_path.insert(0, BlockUnitMatch())
        from spyglass import settings
        from spyglass.spikesorting.v2 import _unit_match_compute as compute
        from spyglass.spikesorting.v2 import _sorting_analyzer, _units_nwb
        from spyglass.spikesorting.v2.matcher_protocol import (
            MatchPair, PreparedMatcherInput, SessionMatcherInput,
            register_matcher,
        )
        settings.temp_dir = sys.argv[1]
        custom_layout = sys.argv[2] == 'True'
        bad_identity = sys.argv[3] == 'True'
        recording = si.NumpyRecording(np.ones((500, 2), dtype=np.float32), 1000)
        recording.set_channel_locations(np.array([[0, 0], [0, 20]]))
        recording.set_channel_gains(1.0)
        recording.set_channel_offsets(0.0)
        sorting = si.NumpySorting.from_unit_dict(
            {17: np.array([100, 200, 300]), 29: np.array([120, 220, 320])}, 1000
        )
        plans = [
            dict(input_index=i, sorting_id=uuid.uuid4(), curation_id=1,
                 recordings=[{'nwb_file_name': f'{i}.nwb'}],
                 sorting_input=object(), units=object(), matchable_unit_ids=[17, 29],
                 input_start_time=f'2026-01-0{i+1}T00:00:00Z',
                 statistics_spans=[(0, 500)])
            for i in range(2)
        ]
        prepared_dirs = []

        class Preparer:
            def prepare(self, source, directory, params, jobs):
                assert source.statistics_spans == [(0, 500)]
                assert jobs['n_jobs'] == 1
                directory.mkdir()
                np.save(directory / 'channel_positions.npy', [[0, 0], [0, 20]])
                (directory / 'spikes.json').write_text(json.dumps({
                    '17': source.sorting.get_unit_spike_train(17).tolist()
                }))
                prepared_dirs.append(directory)
                key = dict(source.curation_key)
                if bad_identity:
                    key['curation_id'] = 99
                return PreparedMatcherInput(
                    SessionMatcherInput(key, directory, directory / 'channel_positions.npy',
                                        source.recording_date), (29,), 'were intentionally excluded'
                )

        class Backend:
            name = 'independent'
            def match(self, inputs, params):
                assert [s.recording_date for s in inputs] == [p['input_start_time'] for p in plans]
                for item in inputs:
                    prepared_dirs.append(item.waveform_dir)
                    if custom_layout:
                        assert not (item.waveform_dir / 'RawWaveforms').exists()
                        assert json.loads((item.waveform_dir / 'spikes.json').read_text()) == {
                            '17': [100, 200, 300]
                        }
                    else:
                        wave = np.load(item.waveform_dir / 'RawWaveforms/Unit17_RawSpikes.npy')
                        np.testing.assert_array_equal(wave, np.ones((2, 2, 2)))
                # Return reversed sides to exercise canonical orientation too.
                a, b = inputs[1].curation_key, inputs[0].curation_key
                return [MatchPair(str(a['sorting_id']), a['curation_id'], 17,
                                  str(b['sorting_id']), b['curation_id'], 17, 0.9)]

        kwargs = {'input_preparer': Preparer()} if custom_layout else {}
        register_matcher(Backend(), type('Schema', (), {}), **kwargs)
        with patch.object(_sorting_analyzer, 'read_canonical_recording', return_value=recording), \
             patch.object(_units_nwb, 'read_stored_units', return_value=sorting):
            if custom_layout and bad_identity:
                try:
                    compute.extract_and_match(plans[::-1], 'independent', {}, {'n_jobs': 1})
                except ValueError as exc:
                    assert 'preserve the frozen' in str(exc)
                else:
                    raise AssertionError('Mismatched preparation identity was accepted')
            else:
                pairs, runtime = compute.extract_and_match(
                    plans[::-1], 'independent', {}, {'n_jobs': 1}
                )
                assert len(pairs) == 1 and runtime >= 0
                assert str(pairs[0]['session_a_sorting_id']) == str(plans[0]['sorting_id'])
                assert pairs[0]['unit_a_id'] == 17
        assert prepared_dirs and all(not p.exists() for p in prepared_dirs)
        assert not any(name.startswith('UnitMatchPy') for name in sys.modules)
        """)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(custom_layout),
            str(bad_identity),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
