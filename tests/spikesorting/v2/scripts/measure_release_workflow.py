"""Release-workload measurement: the supported v2 workflow end to end.

Runs prepare -> sort -> auto-label -> review bundle (+ reopen from its id) ->
merge -> reevaluate -> detailed inspection (waveform plot + Phy export) ->
unit selection -> warm rerun on ONE NWB file, against a PRIVATE MySQL
container (``tests/container.py``) and a private base dir, while sampling
the WHOLE process tree's RSS (SpikeInterface workers included) and the
scratch-disk footprint. Writes ``<label>.json`` into ``--out-dir`` with stage
timings, per-stage completed / skipped / failed results, peak memory, peak
scratch, the monitored roots, bundle bytes, the preflight effective
configuration and resource notes, the ``describe_run`` receipt and hardware
details. A stage failure still writes the JSON (``outcome: failed``) before
the error propagates, so partial measurements are never lost.

This is the repeatable release run for the long-recording gate: run it on a
lab Linux machine with one representative 1-3 h tetrode recording and one
>= 1 h high-channel-count probe recording, then set the supported machine
budgets from the JSON. Numbers from short MEArec fixtures are NOT evidence of
long-recording capacity. Report the channels of the sorted GROUP, not the
file: a high-channel-count NWB sorted one shank at a time is a per-shank
benchmark.

Usage (from the repo root, in the spikesorting-v2 environment)::

    python tests/spikesorting/v2/scripts/measure_release_workflow.py \\
        --nwb /path/to/session.nwb \\
        --preset franklab_probe_hippocampus_30khz_ms5_2026_06 \\
        --label probe_1h --n-jobs 8 [--sort-group-id N] [--port 3313] \\
        [--out-dir DIR] [--base-dir DIR]

Docker is reached through ``docker.from_env()`` exactly as the caller has it
configured (``DOCKER_HOST`` / current context). Colima users export
``DOCKER_HOST=unix://$HOME/.colima/default/docker.sock`` before running.

Measurement semantics: ``peak_process_tree_rss_gb`` is the peak SUM of RSS
over the runner and its descendants (SpikeInterface workers) -- shared pages
count once per process, so it is an upper bound on physical RAM, not a
deduplicated measurement. ``peak_scratch_disk_gb`` is the peak total size
under ``scratch_roots`` (nested roots are deduplicated); Phy exports written
to ``--out-dir`` are not counted. The run is CPU-only (CUDA disabled).

Run ONE measurement at a time: never alongside a pytest session on the same
machine (they would share scratch and CPU). On macOS SpikeInterface spawns
fresh worker processes per computation (slow imports), so only Linux timings
are representative of lab machines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # tests/spikesorting/v2/scripts -> repo root
sys.path.insert(0, str(REPO))


def distinct_roots(paths) -> list[Path]:
    """Resolve ``paths`` and drop any root nested under another one.

    Summing recursive sizes of both a directory and its subdirectory would
    count the subdirectory twice (the default analyzer cache root lives under
    the temp dir); a separately configured analyzer root stays separate.
    """
    resolved = sorted({Path(p).expanduser().resolve() for p in paths})
    kept: list[Path] = []
    for path in resolved:
        if any(path == root or path.is_relative_to(root) for root in kept):
            continue
        kept.append(path)
    return kept


class TreeMonitor:
    """Sample RSS of this process + all descendants, and scratch-root sizes."""

    def __init__(self, scratch_dirs, interval=0.25):
        self._proc = psutil.Process()
        self._scratch = distinct_roots(scratch_dirs)
        self._interval = interval
        self.peak_rss = 0
        self.peak_scratch = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def roots(self) -> list[Path]:
        return list(self._scratch)

    @staticmethod
    def dir_bytes(path) -> int:
        total = 0
        if not Path(path).exists():
            return 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    def _run(self):
        tick = 0
        while not self._stop.is_set():
            rss = 0
            try:
                procs = [self._proc, *self._proc.children(recursive=True)]
            except psutil.Error:
                procs = [self._proc]
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            if tick % 8 == 0:
                self.peak_scratch = max(
                    self.peak_scratch,
                    sum(self.dir_bytes(d) for d in self._scratch),
                )
            tick += 1
            self.samples += 1
            time.sleep(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--port", type=int, default=3313)
    ap.add_argument("--container-name", default="spyglass-measure")
    ap.add_argument(
        "--base-dir",
        default=str(REPO / "tests" / "_data_release_measure"),
        help="Private Spyglass base dir (must sit under a 'tests' directory).",
    )
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--profile", default="franklab_hippocampus_2026_09")
    ap.add_argument("--sort-group-id", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    os.environ["SPYGLASS_BASE_DIR"] = str(base)
    # CPU-only measurement; GPU sorters are out of scope for this runner.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import datajoint as dj

    from tests.container import DockerMySQLManager

    server = DockerMySQLManager(
        container_name=args.container_name, port=args.port
    )
    server.wait()
    dj.config.update(server.credentials)
    dj.config["database.prefix"] = "measure"
    dj.config["loglevel"] = "ERROR"
    dj.config.setdefault("custom", {})["spyglass_dirs"] = {"base": str(base)}
    dj.config["custom"]["test_mode"] = True
    dj.config["custom"]["spikesorting_v2_job_kwargs"] = {
        "n_jobs": args.n_jobs,
        "chunk_duration": "1s",
        "progress_bar": False,
    }
    dj.conn().ping()

    from spyglass.settings import temp_dir
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_cache_root

    monitor = TreeMonitor(
        [
            Path(temp_dir),
            analyzer_cache_root(),
            base / "analysis",
            base / "recording",
        ]
    )
    timings: dict[str, float] = {}
    result: dict = {
        "label": args.label,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        ),
        "nwb": args.nwb,
        "preset": args.preset,
        "n_jobs": args.n_jobs,
        "hardware": {
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "platform": platform.platform(),
            "cuda": "disabled (CPU-only run)",
            "filesystems": [
                {"mountpoint": p.mountpoint, "type": p.fstype}
                for p in psutil.disk_partitions(all=True)
                if base.resolve().is_relative_to(Path(p.mountpoint))
            ],
        },
        "timings_s": timings,
        "stage_results": {},
        "scratch_roots": [str(root) for root in monitor.roots],
        "measurement_notes": [
            "peak_process_tree_rss_gb sums RSS over the runner and its "
            "descendants; shared pages count once per process (upper bound "
            "on physical RAM).",
            "peak_scratch_disk_gb is the peak total under scratch_roots "
            "(nested roots deduplicated); Phy exports under --out-dir are "
            "not counted.",
        ],
    }
    out = Path(args.out_dir) / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def finish(outcome: str) -> None:
        result["outcome"] = outcome
        result["peak_process_tree_rss_gb"] = round(
            monitor.peak_rss / 1024**3, 3
        )
        result["peak_scratch_disk_gb"] = round(
            monitor.peak_scratch / 1024**3, 3
        )
        result["monitor_samples"] = monitor.samples
        result["analyzer_cache_bytes"] = TreeMonitor.dir_bytes(
            analyzer_cache_root()
        )
        out.write_text(json.dumps(result, indent=2, default=str))

    with monitor:
        try:
            _run_workflow(args, result, timings)
        except BaseException:
            finish("failed")
            raise
    finish("completed")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("describe_run", "preflight")
            },
            indent=2,
            default=str,
        )
    )


def _run_workflow(args, result: dict, timings: dict) -> None:
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

    from spyglass.common import IntervalList, LabTeam, Raw
    from spyglass.spikesorting.analysis.v1 import group as analysis_group
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2 import visualization as ssviz
    from spyglass.spikesorting.v2._pipeline_presets import _PIPELINE_PRESETS
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import (
        FigPackReview,
        describe_run,
        preflight_v2_pipeline,
        run_v2_pipeline,
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.recording import SortGroupV2

    stage_results = result["stage_results"]

    class stage:
        """Time a stage; record completed / failed so a partial run is honest."""

        def __init__(self, name):
            self.name = name

        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, exc_type, *exc):
            timings[self.name] = round(time.perf_counter() - self.t0, 2)
            stage_results[self.name] = (
                "completed"
                if exc_type is None
                else f"failed: {exc_type.__name__}"
            )

    def skipped(name: str, why: str) -> None:
        stage_results[name] = f"skipped: {why}"

    with stage("ingest"):
        nwb_file_name = copy_and_insert_nwb(args.nwb)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "measure", "team_description": ""},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & {"nwb_file_name": nwb_file_name}):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    groups = sorted(
        (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch("sort_group_id")
    )
    sort_group_id = (
        args.sort_group_id if args.sort_group_id is not None else int(groups[0])
    )
    n_channels = len(
        SortGroupV2.SortGroupElectrode
        & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
    )
    raw = (Raw & {"nwb_file_name": nwb_file_name}).fetch1()
    valid = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    result["recording"] = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "n_sort_groups": len(groups),
        "n_channels_in_group": n_channels,
        "sampling_rate_hz": float(raw["sampling_rate"]),
        "duration_s": float(sum(b - a for a, b in valid)),
    }
    common = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="measure",
        pipeline_preset=args.preset,
    )
    report = preflight_v2_pipeline(**common, auto_curate=True)
    assert report.ok, report.errors
    result["preflight"] = {
        "effective_config": report.effective_config,
        "resource_notes": report.resource_notes,
        "scientific_config": report.scientific_config,
    }

    with stage("prepare_sort_root"):
        run = run_v2_pipeline(**common)
    result["stage_seconds_run1"] = dict(run["stage_seconds"])
    with stage("auto_label"):
        auto = run_v2_pipeline(**common, auto_curate=True)
    result["stage_seconds_auto"] = dict(auto["stage_seconds"])
    result["n_units"] = int(auto["n_units"])
    result["describe_run"] = describe_run(auto).to_dict(orient="records")
    auto_ref = auto.auto_labeled_curation

    figpack_ok = importlib.util.find_spec("figpack") is not None
    if not figpack_ok:
        skipped("first_review_bundle", "figpack not installed")
        skipped("reopen_review", "figpack not installed")
    elif result["n_units"] == 0:
        skipped("first_review_bundle", "zero-unit sort")
        skipped("reopen_review", "zero-unit sort")
    else:
        with stage("first_review_bundle"):
            review = auto.start_review(
                args.profile, source="auto_labeled", upload=False
            )
        result["review"] = {
            "review_id": str(review.review_id),
            "bundle_bytes": TreeMonitor.dir_bytes(Path(review.uri)),
            "display_options": review.display_options.as_dict(),
            "stages": {s.name: s.status for s in review.stages},
        }
        with stage("reopen_review"):
            reopened = FigPackReview.resume(review.review_id)
            _ = reopened.preview_import()

    bundle_preset = _PIPELINE_PRESETS[args.preset]
    with stage("evaluate_auto_labeled_child"):
        evaluation = auto_ref.evaluate(
            metric_params_name=bundle_preset.metric_params_name,
            auto_curation_rules_name=bundle_preset.auto_curation_rules_name,
        )
    unit_ids = sorted(
        int(u) for u in (CurationV2.Unit & auto_ref.as_key()).fetch("unit_id")
    )
    if len(unit_ids) >= 2:
        with stage("merge_and_reevaluate"):
            receipt = evaluation.merge_and_evaluate([unit_ids[:2]])
        merged_ref = receipt.child
        result["merge"] = {
            "groups": [unit_ids[:2]],
            "child_curation_id": merged_ref.curation_id,
        }
        if figpack_ok:
            with stage("review_bundle_after_merge"):
                review2 = merged_ref.start_review(args.profile, upload=False)
            result["review_after_merge_bundle_bytes"] = TreeMonitor.dir_bytes(
                Path(review2.uri)
            )
        else:
            skipped("review_bundle_after_merge", "figpack not installed")
    else:
        why = f"{len(unit_ids)} unit(s); a merge needs >= 2"
        skipped("merge_and_reevaluate", why)
        skipped("review_bundle_after_merge", why)
        merged_ref = auto_ref

    if not unit_ids:
        skipped("detailed_inspection_waveforms", "zero-unit sort")
        skipped("export_phy", "zero-unit sort")
    else:
        with stage("detailed_inspection_waveforms"):
            import matplotlib

            matplotlib.use("Agg")
            merged_units = sorted(
                int(u)
                for u in (CurationV2.Unit & merged_ref.as_key()).fetch(
                    "unit_id"
                )
            )
            ssviz.plot_waveforms(merged_ref, unit_ids=merged_units[:3])
        with stage("export_phy"):
            ssviz.export_to_phy(
                merged_ref, Path(args.out_dir) / f"phy_{args.label}"
            )
    # The private database uses test mode for setup. Measure production label
    # filtering: the shared analysis test fixture otherwise bypasses it.
    analysis_group.test_mode = False
    with stage("select_units"):
        sel = select_units_for_analysis(merged_ref, policy="v2_unflagged_units")
        times, selected_ids = sel.fetch_spike_data(return_unit_ids=True)
        assert {unit["unit_id"] for unit in selected_ids} == set(
            sel.included_unit_ids
        )
    result["selection"] = {
        "policy": sel.policy_name,
        "n_included": len(sel.included_unit_ids),
        "n_excluded": len(sel.excluded_units),
        "n_unlabeled": len(sel.unlabeled_unit_ids),
        "n_spike_trains": len(times),
    }
    with stage("rerun_warm"):
        run_v2_pipeline(**common, auto_curate=True)


if __name__ == "__main__":
    main()
