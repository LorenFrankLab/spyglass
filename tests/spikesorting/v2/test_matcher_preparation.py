"""Custom matching preparation is independent of UnitMatchPy and bundle format."""

import subprocess
import sys
import textwrap
import uuid
from types import SimpleNamespace

import pytest


@pytest.fixture
def preparation_contract(tmp_path, monkeypatch):
    from spyglass import settings
    from spyglass.spikesorting.v2 import _sorting_analyzer, _units_nwb
    from spyglass.spikesorting.v2 import matcher_protocol as protocol
    from spyglass.spikesorting.v2._unit_match_compute import extract_and_match

    monkeypatch.setattr(settings, "temp_dir", str(tmp_path))
    monkeypatch.setattr(
        _sorting_analyzer, "read_canonical_recording", lambda _: object()
    )
    monkeypatch.setattr(
        _units_nwb,
        "read_stored_units",
        lambda _: SimpleNamespace(select_units=lambda ids: object()),
    )
    plans = [
        {
            "input_index": i,
            "sorting_id": uuid.UUID(int=i + 1),
            "curation_id": 1,
            "recordings": [{"nwb_file_name": f"{i}.nwb"}],
            "sorting_input": object(),
            "units": object(),
            "matchable_unit_ids": [1, 17, 29],
            "input_start_time": f"2026-01-0{i + 1}",
            "statistics_spans": [(0, 100)],
        }
        for i in range(2)
    ]
    directories = []
    emitted = [17, 17]
    excluded = [29]

    class Preparer:
        def prepare(self, source, directory, params, job_kwargs):
            directory.mkdir()
            directories.append(directory)
            return protocol.PreparedMatcherInput(
                protocol.SessionMatcherInput(
                    dict(source.curation_key),
                    directory,
                    directory / "positions.npy",
                    source.recording_date,
                ),
                tuple(excluded),
            )

    def match(inputs, params):
        # Reverse orientation so each side's prepared set follows identity.
        a, b = inputs[1].curation_key, inputs[0].curation_key
        return [
            protocol.MatchPair(
                str(a["sorting_id"]),
                a["curation_id"],
                emitted[1],
                str(b["sorting_id"]),
                b["curation_id"],
                emitted[0],
                0.9,
            )
        ]

    monkeypatch.setattr(
        protocol, "get_matcher", lambda _: SimpleNamespace(match=match)
    )
    monkeypatch.setattr(protocol, "get_input_preparer", lambda _: Preparer())
    yield lambda: extract_and_match(
        plans, "fixture", {}, {"n_jobs": 1}
    ), emitted, excluded
    assert directories and all(
        not directory.exists() for directory in directories
    )


@pytest.mark.parametrize("emitted", [(29, 17), (17, 29), (999, 17), (17, 999)])
def test_matching_rejects_excluded_or_unknown_units(
    preparation_contract, emitted
):
    run, pair_units, _ = preparation_contract
    pair_units[:] = emitted
    with pytest.raises(ValueError, match="prepared matching input"):
        run()


def test_matching_retains_prepared_units(preparation_contract):
    run, _, _ = preparation_contract
    (pair,), _ = run()
    assert pair["unit_a_id"] == pair["unit_b_id"] == 17
    assert pair["session_a_sorting_id"] == str(uuid.UUID(int=1))


@pytest.mark.parametrize("unit", [29.0, True])
def test_matching_rejects_noninteger_exclusions(preparation_contract, unit):
    run, _, excluded = preparation_contract
    excluded[:] = [unit]
    with pytest.raises(ValueError, match="excluded_unit_id must be an integer"):
        run()


def test_matching_input_error_keeps_reason_and_original_cause(
    preparation_contract, monkeypatch
):
    from spyglass.spikesorting.v2 import matcher_protocol as protocol
    from spyglass.spikesorting.v2._waveform_bundles import NoMatchableUnitsError

    run, _, _ = preparation_contract
    original = protocol.get_input_preparer("fixture")
    reason = "every unit had fewer than two sampled spikes with full waveform support"

    def fail(*args):
        original.prepare(*args)
        raise NoMatchableUnitsError(
            f"No bundle can be built at {args[1]}", reason=reason
        )

    monkeypatch.setattr(
        protocol, "get_input_preparer", lambda _: SimpleNamespace(prepare=fail)
    )
    with pytest.raises(NoMatchableUnitsError) as raised:
        run()
    message = str(raised.value)
    assert "input_index 0" in message and "0.nwb" in message
    assert reason in message
    assert "unitmatch_" not in message
    assert isinstance(raised.value.__cause__, NoMatchableUnitsError)
    assert "unitmatch_" in str(raised.value.__cause__)


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
                (directory / 'spikes.json').write_text(json.dumps({
                    '17': source.sorting.get_unit_spike_train(17).tolist()
                }))
                prepared_dirs.append(directory)
                key = dict(source.curation_key)
                if bad_identity:
                    key['curation_id'] = 99
                return PreparedMatcherInput(
                    SessionMatcherInput(key, bundle_dir=directory,
                                        layout='spike_times_json', layout_version=1,
                                        recording_date=source.recording_date),
                    (29,), 'were intentionally excluded'
                )

        class Backend:
            name = 'independent'
            def match(self, inputs, params):
                assert [s.recording_date for s in inputs] == [p['input_start_time'] for p in plans]
                for item in inputs:
                    prepared_dirs.append(item.bundle_dir)
                    if custom_layout:
                        assert item.layout == 'spike_times_json' and item.layout_version == 1
                        assert item.geometry_path is None
                        assert not (item.bundle_dir / 'RawWaveforms').exists()
                        assert json.loads((item.bundle_dir / 'spikes.json').read_text()) == {
                            '17': [100, 200, 300]
                        }
                    else:
                        assert item.layout == 'split_half_waveforms' and item.layout_version == 1
                        assert item.geometry_path.is_file()
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
