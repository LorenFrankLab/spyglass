"""Rehearse the published upgrade against retained data in the test database.

The missing-policy column represents an already completed migration step: old
rules have the documented 'error' backfill while new dated rules may coexist.
Other changed columns are removed and restored through the actual documented
procedure. Existing test rows are restored afterward so their generation IDs
and recipe values remain valid for other fixtures.
"""

import re
import subprocess
import sys
import uuid
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest


def _build_documented_upgrade_script(tmp_path: Path) -> tuple[Path, Path]:
    """Extract the doc's upgrade code block into a runnable script.

    Shared by every rehearsal test in this module. Returns
    ``(script_path, config_path)``; the caller runs the script (possibly more
    than once, to check resumability) and removes ``config_path`` afterward.
    """
    config_file = tmp_path / "test-db.json"
    dj.config.save(str(config_file))
    config_file.chmod(0o600)
    repo = Path(__file__).resolve().parents[3]
    document = (
        repo / "docs/src/Features/SpikeSortingV2_Migration.md"
    ).read_text()
    section = document.split("### Upgrading a preproduction v2 database", 1)[1]
    code = re.search(r"```python\n(.*?)\n```", section, re.DOTALL).group(1)
    code = code.replace(".alter(context=", ".alter(prompt=False, context=")
    script = tmp_path / "upgrade.py"
    script.write_text(
        f"import datajoint as dj\ndj.config.load({str(config_file)!r})\n" + code
    )
    return script, config_file


def _column_data_type(table, column: str) -> str:
    """The column's MySQL ``DATA_TYPE`` (e.g. ``'double'``, ``'float'``)."""
    return table.connection.query(
        "SELECT DATA_TYPE FROM information_schema.COLUMNS WHERE "
        "TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME=%s",
        args=(table.database, table.table_name, column),
    ).fetchone()[0]


def _column_ddl(table, column: str) -> str:
    """The column's full definition from ``SHOW CREATE TABLE``.

    Includes its type, nullability, default and comment, so
    ``ALTER TABLE ... MODIFY COLUMN <ddl>`` restores it exactly.
    """
    create = table.connection.query(
        f"SHOW CREATE TABLE {table.full_table_name}"
    ).fetchone()[1]
    (ddl,) = [
        line.strip().rstrip(",")
        for line in create.splitlines()
        if line.strip().startswith(f"`{column}` ")
    ]
    return ddl


def _run_script(script: Path) -> subprocess.CompletedProcess:
    """Run ``script`` from a fresh interpreter so no cached heading hides
    the old DDL the script is about to alter."""
    repo = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def test_documented_upgrade_widens_single_precision_threshold_columns(
    dj_conn, tmp_path
):
    """Columns downgraded to ``float`` (simulating a database predating the
    double-precision change) come back as ``double`` after the documented
    alter loop, and a value stored while the column was ``float`` keeps its
    single-precision value once widened -- exactly the case the
    ``math.isclose`` tolerance in ``rules_payloads_match`` exists for: the
    loop's call to ``initialize_v2_defaults()`` re-runs ``insert_rules``
    against the widened shipped row, which would raise without it.
    """
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )

    initialize_v2_defaults()  # ensure the shipped rows read below exist
    rule_key = {
        "auto_curation_rules_name": "v1_default_nn_noise_2026_09",
        "rule_index": 0,
    }
    assert (AutoCurationRules.Rule & rule_key).fetch1("threshold") == 0.1

    altered = {
        AutoCurationRules.Rule: "threshold",
        QualityMetricParameters: "observed_presence_bin_duration_s",
    }
    snapshots = {table: table.fetch(as_dict=True) for table in altered}
    declared = {
        table: _column_ddl(table, column) for table, column in altered.items()
    }
    script, config_file = _build_documented_upgrade_script(tmp_path)
    try:
        for table, column in altered.items():
            table.connection.query(
                f"ALTER TABLE {table.full_table_name} "
                f"MODIFY COLUMN `{column}` FLOAT"
            )
        result = _run_script(script)
        assert result.returncode == 0, result.stdout + result.stderr

        for table, column in altered.items():
            assert _column_data_type(table, column) == "double"

        widened = float(np.float32(0.1))
        assert widened != 0.1  # the widening is real, not a no-op
        assert (AutoCurationRules.Rule & rule_key).fetch1(
            "threshold"
        ) == widened
    finally:
        config_file.unlink(missing_ok=True)
        # A failed upgrade must not leave the persistent test database with
        # single-precision columns that would silently change later tests.
        for table, column in altered.items():
            if _column_data_type(table, column) != "double":
                table.connection.query(
                    f"ALTER TABLE {table.full_table_name} "
                    f"MODIFY COLUMN {declared[table]}"
                )
        for table, rows in snapshots.items():
            restored = dj.FreeTable(dj.conn(), table.full_table_name)
            for row in rows:
                restored.update1(row)


@pytest.mark.parametrize("interrupted_uuid_add", [False, True])
def test_documented_upgrade_preserves_retained_data(
    planted_two_unit_sort, tmp_path, interrupted_uuid_add
):
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactSelection,
        SharedGroupArtifactSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        CurationEvaluationSelection,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.review_profile import CurationReviewProfile
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # Seed the two historic rule names exactly as the missing-policy migration
    # leaves them. Initialization must neither overwrite them nor collide with
    # them when it installs current defaults.
    historical_names = {
        "v1_default_nn_noise_2026_09": "v1_default_nn_noise",
        "franklab_default_auto_curation_2026_09": (
            "franklab_default_auto_curation_2026_06"
        ),
    }
    for master, rules in AutoCurationRules._default_payloads():
        if master["auto_curation_rules_name"] not in historical_names:
            continue
        AutoCurationRules.insert_rules(
            {
                **master,
                "auto_curation_rules_name": historical_names[
                    master["auto_curation_rules_name"]
                ],
            },
            [{**rule, "missing_policy": "error"} for rule in rules],
            skip_duplicates=True,
        )
    old_rules = AutoCurationRules.Rule & [
        {"auto_curation_rules_name": name} for name in historical_names.values()
    ]
    rules_before = old_rules.fetch(as_dict=True, order_by="KEY")
    root = CurationV2.insert_curation(
        planted_two_unit_sort, reuse_existing=True
    )
    CurationV2.insert_curation(
        planted_two_unit_sort,
        parent_curation_id=root["curation_id"],
        labels={0: ["accept"]},
        description="retained migration child",
        reuse_existing=True,
    )
    labels_before = CurationV2.UnitLabel.fetch(as_dict=True, order_by="KEY")
    artifacts_before = SortingSelection.ArtifactDetectionSource.fetch(
        as_dict=True, order_by="KEY"
    )
    profiles_before = CurationReviewProfile.fetch(as_dict=True, order_by="KEY")
    columns = {
        CurationV2: ["curation_uuid", "created_at", "created_by"],
        QualityMetricParameters: ["observed_presence_bin_duration_s"],
        CurationEvaluationSelection: ["observation_version"],
        RecordingArtifactSelection: ["manual_excluded_times"],
        SharedGroupArtifactSelection: ["manual_excluded_times"],
    }
    snapshots = {
        table: table.proj(*names).fetch(as_dict=True)
        for table, names in columns.items()
    }
    script, config_file = _build_documented_upgrade_script(tmp_path)
    try:
        for table, names in columns.items():
            table.connection.query(
                f"ALTER TABLE {table.full_table_name} "
                + ", ".join(f"DROP COLUMN `{name}`" for name in names)
            )
        prior_uuid = uuid.uuid4()
        if interrupted_uuid_add:
            dj.conn().query(
                f"ALTER TABLE {CurationV2.full_table_name} "
                "ADD COLUMN `curation_uuid` BINARY(16) NULL"
            )
            dj.conn().query(
                f"UPDATE {CurationV2.full_table_name} SET curation_uuid=%s "
                "WHERE sorting_id=%s AND curation_id=%s",
                args=(
                    prior_uuid.bytes,
                    root["sorting_id"].bytes,
                    root["curation_id"],
                ),
            )
        # Repeat the complete published procedure to prove it is resumable.
        for _ in range(2):
            result = _run_script(script)
            assert result.returncode == 0, result.stdout + result.stderr
        current = dj.FreeTable(dj.conn(), CurationV2.full_table_name)
        actual_uuids = current.fetch("curation_uuid")
        assert len(actual_uuids) == len(snapshots[CurationV2])
        assert len(set(actual_uuids)) == len(actual_uuids)
        assert all(value is not None for value in actual_uuids)
        if interrupted_uuid_add:
            assert (current & root).fetch1("curation_uuid") == prior_uuid
        assert current.heading.indexes[("curation_uuid",)]["unique"]
        assert old_rules.fetch(as_dict=True, order_by="KEY") == rules_before
        assert (
            CurationV2.UnitLabel.fetch(as_dict=True, order_by="KEY")
            == labels_before
        )
        assert (
            SortingSelection.ArtifactDetectionSource.fetch(
                as_dict=True, order_by="KEY"
            )
            == artifacts_before
        )
        assert (
            CurationReviewProfile.fetch(as_dict=True, order_by="KEY")
            == profiles_before
        )
    finally:
        config_file.unlink(missing_ok=True)
        for table, rows in snapshots.items():
            restored = dj.FreeTable(dj.conn(), table.full_table_name)
            for row in rows:
                restored.update1(row)
