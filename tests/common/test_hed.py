"""Test ingestion of an NWB file's HED annotations (ndx-hed).

A small file holds a HedLabMetaData with one definition, and two HedTags
columns: one on a task table, one on the trials table. It verifies that:

- the schema version and definitions are ingested in HedMetadata;
- each HedTags column gets one HedAnnotation entry, with its table, column and
  one Row per table row;
- fetch_annotations_dataframe returns the HED strings with their table.
"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pytest
from hdmf.common import DynamicTable, VectorData
from packaging.version import Version
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

ndx_hed = pytest.importorskip("ndx_hed")
# ndx-hed < 0.3 (the releases that install next to numpy<2 and pynwb 3) cannot
# read back a HedTags column it wrote; 0.3 needs pynwb >= 4.2 (#1694).
if Version(version("ndx-hed")) < Version("0.3.0"):
    pytest.skip("HED tests need ndx-hed >= 0.3.0", allow_module_level=True)

SCHEMA_VERSION = "8.5.0"
# In the form ndx-hed stores it (it normalizes definitions on write), so the
# ingested value can be compared with it directly.
DEFINITIONS = "(Definition/test-condition,(Indoors,Task))"
TASK_HED = ["Def/Test-condition"]
TRIALS_HED = ["Sensory-event, Visual-presentation", "Agent-action, Press"]


@pytest.fixture(scope="module")
def hed_nwb_key(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwbfile = NWBFile(
        session_description="test HED ingestion",
        identifier="test_hed",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    nwbfile.subject = Subject(
        subject_id="test_hed_subject", species="Rattus norvegicus"
    )
    nwbfile.add_lab_meta_data(
        ndx_hed.HedLabMetaData(
            hed_schema_version=SCHEMA_VERSION, definitions=DEFINITIONS
        )
    )

    task_table = DynamicTable(
        name="task_table",
        description="task of the session",
        columns=[
            VectorData(
                name="task_name", description="task name", data=["test"]
            ),
            ndx_hed.HedTags(name="HED", description="task HED", data=TASK_HED),
        ],
    )
    nwbfile.create_processing_module("tasks", "tasks").add(task_table)

    nwbfile.add_trial_column(
        name="HED", description="trial HED", col_cls=ndx_hed.HedTags
    )
    for i, hed in enumerate(TRIALS_HED):
        nwbfile.add_trial(start_time=float(i), stop_time=i + 0.5, HED=hed)

    raw_file_name = "test_hed.nwb"
    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    yield key

    with verbose_context:
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)


def test_hed_metadata(common, hed_nwb_key):
    entry = (common.HedMetadata & hed_nwb_key).fetch1()
    assert entry["hed_schema_version"] == SCHEMA_VERSION
    assert entry["definitions"] == DEFINITIONS


def test_hed_annotations_per_column(common, hed_nwb_key):
    columns = (common.HedAnnotation & hed_nwb_key).fetch(
        "table_name",
        "column_name",
        "n_rows",
        as_dict=True,
        order_by="table_name",
    )
    assert columns == [
        dict(table_name="task_table", column_name="HED", n_rows=1),
        dict(table_name="trials", column_name="HED", n_rows=2),
    ]

    trials = common.HedAnnotation & hed_nwb_key & {"table_name": "trials"}
    rows = (common.HedAnnotation.Row & trials).fetch(
        "hed_string", order_by="row_index"
    )
    assert list(rows) == TRIALS_HED


def test_fetch_annotations_dataframe(common, hed_nwb_key):
    df = (common.HedAnnotation & hed_nwb_key).fetch_annotations_dataframe()
    task = df[df["table_name"] == "task_table"]
    assert task["hed_string"].tolist() == TASK_HED
    assert len(df) == len(TASK_HED) + len(TRIALS_HED)
