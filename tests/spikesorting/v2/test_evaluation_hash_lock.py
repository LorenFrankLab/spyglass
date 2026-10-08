"""Evaluation and stale checks hold their cache lock while hashing data."""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from filelock import Timeout


@pytest.mark.parametrize("operation", ["evaluation", "stale"])
@pytest.mark.parametrize("wants_pc", [False, True])
@pytest.mark.parametrize("fail_hash", [False, True])
def test_hash_reads_share_cache_generation(
    tmp_path, monkeypatch, operation, wants_pc, fail_hash
):
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage import analyzer_cache as cache
    from spyglass.spikesorting.v2._curation import (
        evaluation_analyzers as evaluation,
    )
    from spyglass.spikesorting.v2._curation import metric_fetch as fetch
    from spyglass.spikesorting.v2._storage import recompute as hashing
    from spyglass.spikesorting.v2._sorting import analyzer as builders

    sorting_id = uuid.uuid4()
    monkeypatch.setattr(cache, "analyzer_cache_root", lambda: tmp_path)
    reads = []
    failure = RuntimeError("hash read failed")

    def contend():
        try:
            with cache.analyzer_cache_lock(sorting_id).acquire(timeout=0):
                return True
        except Timeout:
            return False

    class Analyzer:
        def __init__(self, role):
            self.role = role

        def has_extension(self, name):
            return name in hashing.ANALYZER_RECOMPUTE_EXTENSIONS

        def get_extension(self, name):
            def read():
                with ThreadPoolExecutor(max_workers=1) as pool:
                    assert not pool.submit(contend).result(
                        timeout=5
                    ), "cache deletion/rebuild could enter during hashing"
                reads.append((self.role, name))
                if fail_hash:
                    raise failure
                return np.array([1.0, 2.0], dtype=np.float32)

            return SimpleNamespace(get_data=read)

    display, metric = Analyzer("display"), Analyzer("metric")
    if operation == "evaluation":
        monkeypatch.setattr(evaluation, "read_stored_units", lambda _: object())
        monkeypatch.setattr(
            builders,
            "load_or_rebuild_analyzer_from_resolved",
            lambda **kwargs: (
                metric
                if kwargs["analyzer_folder"].name == "metric"
                else display
            ),
        )
        monkeypatch.setattr(
            evaluation._metric_curation,
            "evaluate_analyzers",
            lambda *args, **kwargs: ("metrics", {}, []),
        )

        def work():
            return evaluation.evaluate_cached_analyzers(
                object(),
                object(),
                sorting_inputs=SimpleNamespace(
                    sorting_id=sorting_id,
                    raw_units=object(),
                    raw_n_units=1,
                    expected_unit_ids=[1],
                ),
                analyzer_inputs=SimpleNamespace(
                    display_analyzer_folder=str(tmp_path / "display"),
                    metric_analyzer_folder=str(tmp_path / "metric"),
                    display_waveform_params={},
                    metric_waveform_params={},
                    sorter_row={},
                    analyzer_job_kwargs={},
                ),
                metric_inputs=object(),
                wants_pc=wants_pc,
                observation_metrics=object(),
                statistics_spans=[],
            )

    else:

        class Relation:
            def __init__(self, row):
                self.row = row

            def __and__(self, key):
                return self

            def fetch1(self):
                return self.row

        selection_module = ModuleType(
            "spyglass.spikesorting.v2.metric_curation"
        )
        selection_module.CurationEvaluationSelection = Relation(
            {"sorting_id": sorting_id, "metric_waveform_params_name": "metric"}
        )
        sorting_module = ModuleType("spyglass.spikesorting.v2.sorting")
        sorting_module.Sorting = lambda: SimpleNamespace(
            get_analyzer=lambda key, **kwargs: (
                metric
                if kwargs["waveform_params_name"] == "metric"
                else display
            )
        )
        monkeypatch.setitem(
            sys.modules, selection_module.__name__, selection_module
        )
        monkeypatch.setitem(
            sys.modules, sorting_module.__name__, sorting_module
        )
        roles = {"display": "array-v2:old"}
        if wants_pc:
            roles["metric"] = "array-v2:old"
        table = Relation(
            {
                "spikeinterface_version": si.__version__,
                "source_analyzer_hashes": roles,
            }
        )

        def work():
            return fetch.detect_stale_source(table, {})

    if fail_hash:
        with pytest.raises(RuntimeError) as exc:
            work()
        assert exc.value is failure
    else:
        result = work()
        expected_roles = {"display", "metric"} if wants_pc else {"display"}
        assert {role for role, _ in reads} == expected_roles
        if operation == "evaluation":
            assert set(result[3]) == expected_roles
        else:
            assert (
                set(result["source_analyzer_hashes"]["current"])
                == expected_roles
            )
    # Success and failure both release the lock for another worker.
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(contend).result(timeout=5)
