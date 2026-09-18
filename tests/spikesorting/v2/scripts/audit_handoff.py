"""Opt-in backup, account-handoff, and population-quality rehearsals."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import datajoint as dj
import numpy as np

from tests.spikesorting.v2.scripts.audit_lifecycle import (
    workflow as workflow,  # noqa: PLC0414
)


def test_database_and_file_restore(workflow, server, base_dir, monkeypatch):
    from spyglass.common import AnalysisNwbfile
    from spyglass.spikesorting.analysis.v1 import group as gm
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.utils.nwb_helper_fn import close_nwb_files

    monkeypatch.setattr(gm, "test_mode", False)
    ref = workflow["child"]
    curation_key = ref.as_key()
    receipt = select_units_for_analysis(ref)
    before, ids = receipt.fetch_spike_data(return_unit_ids=True)
    row = (CurationV2 & ref.as_key()).fetch1()
    path = Path(AnalysisNwbfile.get_abs_path(row["analysis_file_name"]))
    backup = workflow["out"] / "file-backup"
    base = Path(base_dir)
    assert path.is_relative_to(base)
    close_nwb_files()
    shutil.copytree(base, backup)
    databases = [
        r[0]
        for r in dj.conn().query("SHOW DATABASES").fetchall()
        if r[0]
        not in {"mysql", "sys", "information_schema", "performance_schema"}
    ]
    container = server.client.containers.get(server.container_name)
    dumped = container.exec_run(
        [
            "mysqldump",
            "-uroot",
            "--single-transaction",
            "--set-gtid-purged=OFF",
            "--databases",
            *databases,
        ],
        environment={"MYSQL_PWD": server.password},
    )
    assert dumped.exit_code == 0, dumped.output.decode()
    # Demonstrate recovery from actual missing DB rows and a missing NWB file.
    (CurationV2 & ref.as_key()).delete(safemode=False)
    path.unlink(missing_ok=True)
    assert not (CurationV2 & curation_key) and not path.exists()
    restored = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "-e",
            f"MYSQL_PWD={server.password}",
            server.container_name,
            "mysql",
            "-uroot",
        ],
        input=dumped.output,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert restored.returncode == 0, restored.stderr.decode()
    shutil.copy2(backup / path.relative_to(base), path)
    script = """
import json, sys, uuid
import datajoint as dj
dj.config.load(sys.argv[1])
from spyglass.spikesorting.analysis.v1 import group as gm
gm.test_mode = False
from spyglass.spikesorting.v2.curation_api import CurationRef
from spyglass.spikesorting.v2.analysis_selection import select_units_for_analysis
ref = CurationRef.from_key({'sorting_id': uuid.UUID(sys.argv[2]), 'curation_id': int(sys.argv[3])})
receipt = select_units_for_analysis(ref)
spikes, ids = receipt.fetch_spike_data(return_unit_ids=True)
json.dump({'uuid': str(ref.curation_uuid), 'ids': [r['unit_id'] for r in ids],
    'spikes': [s.tolist() for s in spikes], 'observation': receipt.observation.intervals.tolist()},
    open(sys.argv[4], 'w'))
"""
    output = workflow["out"] / "restored.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(workflow["config"]),
            str(ref.sorting_id),
            str(ref.curation_id),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(output.read_text())
    assert after["uuid"] == str(ref.curation_uuid)
    assert after["ids"] == [r["unit_id"] for r in ids]
    for expected, actual in zip(before, after["spikes"], strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        after["observation"], receipt.observation.intervals
    )
    print(
        "RESTORE",
        json.dumps(
            {
                "databases": len(databases),
                "dump_bytes": len(dumped.output),
                "selected_units": after["ids"],
                "curation_uuid": after["uuid"],
            }
        ),
    )


def test_separate_accounts_commit_separate_branches(
    planted_two_unit_sort, curation_evaluation_defaults, tmp_path
):
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile

    root = CurationRef.from_key(
        CurationV2.insert_curation(planted_two_unit_sort, reuse_existing=True)
    )
    CurationReviewProfile.insert1(
        {
            "review_profile_name": "account_review",
            "metric_params_name": "minimal",
            "auto_curation_rules_name": "none",
            "displayed_unit_properties": ["isi_violation"],
            "label_options": ["accept", "noise"],
            "label_import_mode": "replace",
        },
        skip_duplicates=True,
    )
    config = tmp_path / "db.json"
    dj.config.save(str(config))
    config.chmod(0o600)
    script = """
import json, sys, uuid
from pathlib import Path
import datajoint as dj
dj.config.load(sys.argv[1])
dj.config['database.user'] = sys.argv[2]
dj.config['database.password'] = 'audit-only'
from spyglass.spikesorting.v2.curation_api import CurationRef
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2._figpack_curation import labels_and_merges_to_annotations
root = CurationRef.from_key({'sorting_id': uuid.UUID(sys.argv[3]), 'curation_id': int(sys.argv[4])})
review = root.start_review('account_review')
labels = {int(sys.argv[5]): ['accept']}
annotations = labels_and_merges_to_annotations(labels, [], label_options=['accept', 'noise'])
(Path(review.uri) / 'annotations.json').write_text(json.dumps(annotations))
child = review.preview_import().commit().curation
from spyglass.spikesorting.analysis.v1 import group as gm
gm.test_mode = False
from spyglass.spikesorting.v2.analysis_selection import select_units_for_analysis
selection = select_units_for_analysis(child)
_, ids = selection.fetch_spike_data(return_unit_ids=True)
row = (CurationV2 & child.as_key()).fetch1()
json.dump({'curation_id': child.curation_id, 'curation_uuid': str(child.curation_uuid),
    'parent': row['parent_curation_id'], 'created_by': row['created_by'],
    'review_id': str(review.review_id), 'selected': [i['unit_id'] for i in ids]}, open(sys.argv[6], 'w'))
"""
    results = []
    users = ("audit_alice", "audit_bob")
    try:
        for unit, user in enumerate(users):
            dj.conn().query(
                f"CREATE USER '{user}'@'%%' IDENTIFIED BY 'audit-only'"
            )
            # These credentials exist only in the disposable fixture container.
            dj.conn().query(f"GRANT ALL PRIVILEGES ON *.* TO '{user}'@'%%'")
            output = tmp_path / f"{user}.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(config),
                    user,
                    str(root.sorting_id),
                    str(root.curation_id),
                    str(unit),
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=240,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            results.append(json.loads(output.read_text()))
        assert results[0]["curation_uuid"] != results[1]["curation_uuid"]
        assert [r["created_by"] for r in results] == list(users)
        assert [r["selected"] for r in results] == [[0], [1]]
        assert all(r["parent"] == root.curation_id for r in results)
        assert not (CurationV2.UnitLabel & root.as_key())
        print("ACCOUNTS", json.dumps(results))
    finally:
        config.unlink(missing_ok=True)
        for user in users:
            dj.conn().query(f"DROP USER IF EXISTS '{user}'@'%%'")


def test_ground_truth_population_after_auto_labels(workflow, monkeypatch):
    import pynwb
    import spikeinterface as si
    from spikeinterface.comparison import compare_sorter_to_ground_truth

    from spyglass.spikesorting.analysis.v1 import group as gm
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )
    from spyglass.spikesorting.v2._recipe_catalog import FRANKLAB_CURATION_RULES
    from spyglass.spikesorting.v2.analysis_selection import (
        select_units_for_analysis,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.setattr(gm, "test_mode", False)
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2

    name = workflow["name"]
    group = int(
        min((SortGroupV2 & {"nwb_file_name": name}).fetch("sort_group_id"))
    )
    root = run_v2_pipeline(
        nwb_file_name=name,
        sort_group_id=group,
        interval_list_name="raw data valid times",
        team_name="lifecycle_audit",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    ).root_curation
    evaluation = root.evaluate(
        metric_params_name="franklab_default",
        auto_curation_rules_name=FRANKLAB_CURATION_RULES,
    )
    child = evaluation.accept_labels()
    receipt = select_units_for_analysis(child, policy="v2_unflagged_units")
    raw = Sorting().get_sorting({"sorting_id": root.sorting_id})
    selected = raw.select_units(list(receipt.included_unit_ids))
    observed = receipt.observation
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/mearec_tetrode_60s.nwb"
    )
    with pynwb.NWBHDF5IO(str(fixture), "r") as io:
        units = get_ground_truth_units_table(io.read())
        trains = {
            int(u): np.rint(
                np.asarray(units["spike_times"][i])[
                    observed.contains(np.asarray(units["spike_times"][i]))
                ]
                * raw.sampling_frequency
            ).astype(np.int64)
            for i, u in enumerate(units.id[:])
        }
    truth = si.NumpySorting.from_unit_dict(
        trains, sampling_frequency=raw.sampling_frequency
    )
    measurements = {}
    for name, sorting in (("raw", raw), ("selected", selected)):
        comparison = compare_sorter_to_ground_truth(
            truth, sorting, delta_time=0.4, match_score=0.5, exhaustive_gt=True
        )
        performance = comparison.get_performance(
            method="by_unit", output="pandas"
        )
        measurements[name] = {
            "units": sorting.get_num_units(),
            "truth_accuracy": performance.accuracy.tolist(),
            "mean_precision": float(performance.precision.mean()),
            "mean_recall": float(performance.recall.mean()),
        }
    print(
        "POPULATION",
        json.dumps(
            {
                "measurement": measurements,
                "included": list(receipt.included_unit_ids),
                "excluded": dict(receipt.excluded_units),
            },
            default=str,
        ),
    )
    assert (
        selected.get_num_units() > 0
    ), "Default auto-labeling excluded every sorted unit"
