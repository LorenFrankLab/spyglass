"""Named CI lanes: the tests each CI job runs.

``pytest --lane NAME`` collects only that lane, so a workflow job names its
lane instead of listing paths. Entries are paths relative to the repository
root; ``path::test`` keeps one test of that file, with all its parameters.
Tests run in entry order.

This module imports only pytest, so a job that skips the suite's conftest
(``--noconftest``) can still load it with ``-p tests._ci_lanes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

import pytest

V2 = "tests/spikesorting/v2"

# Full-sort modules, routed to the v2 ``integration`` shard to balance runtime.
# Add a new full-sort test file here; every other v2 file runs in ``v2-unit``.
INTEGRATION_MODULES = (
    "test_unitmatch.py",
    "test_unit_match_generation.py",
    "test_direct_curation_unitmatch.py",
    "test_epoch_unitmatch.py",
    "test_user_curation_workflow.py",
    "test_unitmatch_concat.py",
    "test_session_group_concat.py",
    "test_recompute.py",
    "test_curation_evaluation.py",
    "test_analyzer_lifecycle.py",
    "test_pipeline_session.py",
    "test_downstream_consumers.py",
    "test_clusterless_waveform_features.py",
    "test_fixture_ingestion.py",
    "test_brain_region_attribution.py",
    "test_recording_nwb.py",
    "test_motion_correction.py",
    "test_motion_consumers.py",
)
_INTEGRATION = tuple(f"{V2}/{name}" for name in INTEGRATION_MODULES)


@dataclass(frozen=True)
class Lane:
    """Collection paths for one CI lane.

    Parameters
    ----------
    include : tuple of str
        Paths or ``path::test`` entries to run; empty means all of ``tests/``.
    exclude : tuple of str
        Paths never collected.
    """

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


LANES = {
    # The default env: SpikeInterface 0.104 without the v2 extras.
    "core": Lane(
        exclude=("tests/spikesorting/v0", "tests/spikesorting/v1", V2)
    ),
    # SpikeInterface 0.99. The v2 file holds the legacy-environment guard
    # test; other v2 modules need SpikeInterface 0.104 to import.
    "legacy": Lane(
        include=(
            "tests/spikesorting/v0",
            "tests/spikesorting/v1",
            f"{V2}/test_clusterless_waveform_features.py",
        )
    ),
    # v2 workload shards, balanced by runtime; together they hold every v2
    # test file once (test_ci_shards.py).
    "v2-single-session": Lane(include=(f"{V2}/single_session",)),
    "v2-integration": Lane(include=_INTEGRATION),
    "v2-unit": Lane(
        include=(V2,), exclude=(f"{V2}/single_session", *_INTEGRATION)
    ),
    # The only env with UnitMatchPy (spikesorting-v2-matching extra).
    "v2-matching": Lane(
        include=(
            f"{V2}/test_v1_parity.py::test_optional_matching_extra_resolution",
            f"{V2}/test_unitmatch.py",
            f"{V2}/test_unitmatch_concat.py",
            f"{V2}/test_unitmatch_backend.py",
            f"{V2}/test_unitmatch_daily_concat.py",
            f"{V2}/test_direct_curation_unitmatch.py",
            f"{V2}/test_motion_consumers.py::test_unitmatch_bundle_of_a_corrected_sort",
            f"{V2}/test_motion_consumers.py::test_corrected_concat_bundle_keeps_windows_in_spans",
            f"{V2}/test_notebook_execution.py::test_cross_session_notebook_runs",
        )
    ),
    # The only env with FigPack and Playwright (spikesorting-v2-curation).
    "v2-curation": Lane(
        include=(
            f"{V2}/test_figpack_curation_integration.py",
            f"{V2}/test_review_api_integration.py",
            f"{V2}/test_review_delivery.py",
            f"{V2}/test_review_unit_properties.py",
            f"{V2}/test_review_inspection.py",
            f"{V2}/test_review_browser.py",
            f"{V2}/test_review_browser_journey.py",
            f"{V2}/test_user_curation_workflow.py",
            f"{V2}/test_epoch_unitmatch.py",
            f"{V2}/test_notebook_execution.py::test_curation_notebook_runs",
        )
    ),
    # Python 3.10 and 3.12 with only the spikesorting-v2 extra: contracts
    # that need neither the DB harness nor its dependencies (--noconftest).
    "v2-supported-python": Lane(
        include=(
            f"{V2}/test_dependency_contract.py",
            f"{V2}/test_pipeline_facade.py",
            f"{V2}/test_service_import_contracts.py::test_pipeline_type_contracts_are_reexported_from_facade",
            f"{V2}/test_params_validation.py::test_preprocessing_non_default_blob_round_trips",
            f"{V2}/test_preprocessing_order.py",
            f"{V2}/test_signal_math.py",
            f"{V2}/test_mountainsort4_runtime.py",
        )
    ),
}


def pytest_addoption(parser):
    parser.addoption(
        "--lane",
        choices=sorted(LANES),
        default=None,
        help="Collect only one CI lane's tests (see tests/_ci_lanes.py).",
    )


def _path(entry: str) -> PurePosixPath:
    return PurePosixPath(entry.split("::", 1)[0])


def _within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    return path == parent or parent in path.parents


def missing_paths(lane: Lane, root) -> list[str]:
    """Lane paths absent under ``root``, so a renamed file cannot drop out."""
    entries = lane.include + lane.exclude
    return sorted(
        {str(_path(e)) for e in entries if not (root / _path(e)).exists()}
    )


def collects(lane: Lane, rel: PurePosixPath) -> bool:
    """Whether ``lane`` collects the repository-relative path ``rel``.

    Lane paths, what they contain and the directories leading to them are
    collected, minus the excluded paths.
    """
    if any(_within(rel, _path(e)) for e in lane.exclude):
        return False
    return not lane.include or any(
        _within(rel, _path(e)) or _within(_path(e), rel) for e in lane.include
    )


def pytest_ignore_collect(collection_path, config):
    name = config.getoption("lane", default=None)
    if name is None:
        return None
    try:
        rel = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    # None, not False, so --ignore and collect_ignore still apply.
    return None if collects(LANES[name], PurePosixPath(rel)) else True


def _entry_index(nodeid: str, lane: Lane) -> int | None:
    """Index of the lane entry holding test ``nodeid``; None if outside it."""
    base = nodeid.split("[", 1)[0]
    file = PurePosixPath(nodeid.split("::", 1)[0])
    if any(_within(file, _path(e)) for e in lane.exclude):
        return None
    if not lane.include:
        return 0
    for index, entry in enumerate(lane.include):
        if "::" in entry:
            if base == entry or base.startswith(entry + "::"):
                return index
        elif _within(file, PurePosixPath(entry)):
            return index
    return None


def pytest_collection_modifyitems(config, items):
    name = config.getoption("lane", default=None)
    if name is None:
        return
    lane = LANES[name]
    if missing := missing_paths(lane, config.rootpath):
        raise pytest.UsageError(f"Lane {name!r} names missing paths: {missing}")
    # Also filters test paths given on the command line, which pytest never
    # passes through pytest_ignore_collect.
    index = {item: _entry_index(item.nodeid, lane) for item in items}
    deselected = [item for item in items if index[item] is None]
    # Stable sort: entry order, then collection order within an entry.
    items[:] = sorted(
        (item for item in items if index[item] is not None), key=index.get
    )
    if deselected:
        config.hook.pytest_deselected(items=deselected)
