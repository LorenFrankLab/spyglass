"""Parameter-table content columns follow the Spyglass naming convention.

Each v2 parameter table keyed by ``{stage}_params_name`` stores its content
blob in ``{stage}_params`` (``docs/src/ForDevelopers/TableTypes.md``). The
fingerprint and id payloads name the blob themselves, so the column name is
in none of them; the pins below were computed at 914f984a2, when every one of
these columns was named ``params``.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import uuid
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).resolve().parents[3] / "src/spyglass/spikesorting/v2"

# table -> (module, name attribute, content column)
CONTENT_COLUMNS = {
    "PreprocessingParameters": (
        "recording",
        "preprocessing_params_name",
        "preprocessing_params",
    ),
    "ArtifactDetectionParameters": (
        "artifact",
        "artifact_detection_params_name",
        "artifact_detection_params",
    ),
    "SorterParameters": ("sorting", "sorter_params_name", "sorter_params"),
    "AnalyzerWaveformParameters": (
        "sorting",
        "waveform_params_name",
        "waveform_params",
    ),
    "MotionEstimationParameters": (
        "motion",
        "motion_estimation_params_name",
        "motion_estimation_params",
    ),
    "MotionInterpolationParameters": (
        "motion",
        "motion_interpolation_params_name",
        "motion_interpolation_params",
    ),
    "MatcherParameters": (
        "unit_matching",
        "matcher_params_name",
        "matcher_params",
    ),
}
# Keyed by a ``*_params_name`` but with named columns instead of one blob.
NO_CONTENT_BLOB = {"QualityMetricParameters", "MotionCorrectionParameters"}

_ATTRIBUTE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(?:=[^:]*)?:\s*(\w+)")


def _declared_definitions():
    """Yield ``(class name, primary attrs, secondary attrs)`` per v2 table.

    Attributes are ``{name: type}``; foreign keys and comments are skipped.
    Parsed from source, so no database is needed.
    """
    for path in sorted(V2_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not (
                    isinstance(item, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "definition"
                        for t in item.targets
                    )
                    and isinstance(item.value, ast.Constant)
                ):
                    continue
                primary, secondary = {}, {}
                current = primary
                for line in item.value.value.splitlines():
                    if line.strip().startswith("---"):
                        current = secondary
                        continue
                    match = _ATTRIBUTE.match(line.split("#", 1)[0])
                    if match and "->" not in line:
                        current[match.group(1)] = match.group(2)
                yield node.name, primary, secondary


def test_parameter_content_column_is_the_name_key_without_name():
    """A ``*_params_name``-keyed table stores its blob in ``{key}`` minus
    ``_name``, and no v2 table has a bare ``params`` column."""
    seen = {}
    for name, primary, secondary in _declared_definitions():
        assert "params" not in primary and "params" not in secondary, name
        name_keys = [k for k in primary if k.endswith("_params_name")]
        if not name_keys:
            continue
        (name_key,) = name_keys
        column = name_key.removesuffix("_name")
        if name in NO_CONTENT_BLOB:
            assert column not in secondary, name
            continue
        assert secondary.get(column) in {
            "blob",
            "longblob",
        }, f"{name}: content blob must be {column!r}, declared {secondary}"
        seen[name] = (name_key, column)
    assert seen == {
        table: (key, column)
        for table, (_mod, key, column) in CONTENT_COLUMNS.items()
    }


def test_params_column_derives_the_column_from_the_name_attribute():
    from spyglass.spikesorting.v2._core.lookup_validation import (
        params_column,
    )

    assert params_column("sorter_params_name") == "sorter_params"
    assert (
        params_column(["sorter", "sorter_params_name", "sorter_params"])
        == "sorter_params"
    )
    for bad in ("sorter", ["a_params_name", "b_params_name"]):
        with pytest.raises(ValueError, match="_params_name"):
            params_column(bad)


def _table(name):
    module, _key, _column = CONTENT_COLUMNS[name]
    return getattr(
        importlib.import_module(f"spyglass.spikesorting.v2.{module}"), name
    )


@pytest.mark.database
@pytest.mark.parametrize("table_name", sorted(CONTENT_COLUMNS))
def test_bare_params_key_is_refused_with_the_column_name(dj_conn, table_name):
    """An insert row that names its blob ``params`` says the column name."""
    table = _table(table_name)
    table.insert_default()
    _module, name_key, column = CONTENT_COLUMNS[table_name]
    row = table.fetch(as_dict=True, limit=1)[0]
    row[name_key] = "_pytest_bare_params"
    row["params"] = row.pop(column)
    with pytest.raises(ValueError, match=f"column is '{column}', not 'params'"):
        table.insert1(row, allow_duplicate_params=True)
    assert not (table & {name_key: "_pytest_bare_params"})


# Computed at 914f984a2 through the same calls. The kilosort4 default row is
# not installed without Kilosort4, so it is not pinned here.
PINNED_FINGERPRINTS = {
    "AnalyzerWaveformParameters": {
        "franklab_cortex_actual_waveforms": "6276baa9475535cf3e5f25e2724d3d24a8fede382b3b9187b721d6dc63ac782f",
        "franklab_cortex_metric_waveforms": "a92fd940d0209d0d2c788857659acd7d8df809968e4fa0c3859a201805e37284",
        "franklab_hippocampus_actual_waveforms": "ecc6f6b314dd7831781203f42ee7a221b54d06597cb5913748d0b52aa2c26814",
        "franklab_hippocampus_metric_waveforms": "0845e6464309df7e7c89ffd15b1c2e9435eeb2134a104c55739f18b820bfe1ed",
    },
    "ArtifactDetectionParameters": {
        "default": "c9c3f821754935d71fc5c3e2025635dc975f4d8d150708263b6359b8da2519c8",
        "franklab_100uv_p07_2026_06": "6e6ddbc7e5b7b2659d4bd681a7af9c8dfb5db07289abca4a56b8cd7cf7835087",
        "franklab_50uv_p07_2026_06": "a4a9158675e813176d8690e37e0f57317178099960f89b4e5875fd68d3fa9804",
        "none": "adce7dc59da281a91645d713468ba3b11c20a0b02bb2bb6a7da8f7de44f57639",
    },
    "MatcherParameters": {
        "unitmatch_default": "403e1eea0e43701d0cebc08df93745af79a1bb726d509cfea7d5c3f959af3b8a",
    },
    "MotionEstimationParameters": {
        "dredge_fast_v1": "015276a2390c25593e63ab88fcb592614016bd6ebd6c0c09cf410b30f4a47aab",
        "dredge_v1": "dd54934d9c1614b034aa629afb7c83d734c1163c00a944fe5e065e4a53ebe336",
    },
    "MotionInterpolationParameters": {
        "kriging_force_extrapolate_v1": "b6259b4e20b37188e0e995e46b3f283ea25e67ecb88ee28e2480f951a2ea8223",
        "kriging_remove_channels_v1": "42e59f25e8bab5a4dcb61ce15d031ef151fd2775e4b83ee805c3629fcdd26700",
    },
    "PreprocessingParameters": {
        "default": "cf3eb959ae3cec0a46d3dffc6198db656ce95e7352c9e5eb8065624a8cec27f1",
        "default_neuropixels": "fa3649076b324c9f375f75a489db82d2f6ac36cbbeef48825c6f2868a9b5f3ca",
        "franklab_cortex_2026_06": "2872d9dcb49760ed18e372246dd340a71858dc3ae3344d08bf0192b00fa832b5",
        "franklab_hippocampus_2026_06": "a23acc3c06a9eb001ba0126dda55b41a58630a086960da61e38a5256f039415a",
        "no_filter": "607134848039c754a84903b990af2d0207abf33735d62c2b4e6fb47ade0daf68",
    },
    "SorterParameters": {
        "clusterless_thresholder/default": "ee2a831a3ac152088d0c0d5f833a31a36e61bb39257bac658fc75b84d1d13e42",
        "clusterless_thresholder/franklab_clusterless_60uv_2026_11": "df07727a289382561c64fea9720c10fd9de77ddba91c769b2fbc9f20ccd331cb",
        "mountainsort4/franklab_20khz_ms4_2026_06": "ae4c949b5e50eb020cead2c7800dee4f37d3561886d0df2d8985d9b6c87475d4",
        "mountainsort4/franklab_30khz_ms4_2026_06": "c151d63638a0afdc8fee5b3d9688d4de715c198360c81d32d193ab00f5e62244",
        "mountainsort4/franklab_probe_hippocampus_30khz_ms4_singularity_2026_06": "a48d8698387da8cd1c464b684dc50dff21c6d337441a6aee8e8a04ce2eb013da",
        "mountainsort5/franklab_30khz_ms5_2026_06": "78e6f83eab996f89bf5c6420e96e863e8f04982a8345a366ca454ef10eaf018b",
        "spykingcircus2/default": "8164b856d41945d1554c69859becf6a9851afe12e44f0251e2c9b6cb3fcfeafb",
        "spykingcircus2/v2_preprocessed": "1a2ad98b9d743da2b8c1b933c201261a3ba4ae42ff1aaf86d7e2d66b74b53be1",
        "tridesclous2/default": "a408625695654cc7c9a708d21a3eeb4e4df7e8b6926096f0848743d20bae22e6",
        "tridesclous2/v2_preprocessed": "8f780db74141a4ed4f8f59fd0bc4429813aac3e74e75220dcf91dfb2c4efc4dc",
    },
}
# describe_parameter_rows fingerprints these tables over their stored columns,
# not through parameter_row_fingerprint, so they are pinned separately.
PINNED_DESCRIBE_FINGERPRINTS = {
    "AnalyzerWaveformParameters": {
        "franklab_cortex_actual_waveforms": "d6e5ee7a93c2",
        "franklab_cortex_metric_waveforms": "039714532f6e",
        "franklab_hippocampus_actual_waveforms": "b8b8dc7176bb",
        "franklab_hippocampus_metric_waveforms": "39bf0c4b9297",
    },
    "MatcherParameters": {"unitmatch_default": "b3e92f13b903"},
    "MotionEstimationParameters": {
        "dredge_fast_v1": "65272361b401",
        "dredge_v1": "c892599f3c16",
    },
    "MotionInterpolationParameters": {
        "kriging_force_extrapolate_v1": "f7034f2a8b47",
        "kriging_remove_channels_v1": "f39dfec57891",
    },
}
_PRESET = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
PINNED_PRESET = {
    "describe": "877caaf215c6fad03e23c6e6968efecadf5edb3305ac19588c131e510ad592f9",
    "clone_describe": "81d82f14628e68d83f4b385341a8d6622708cf4454a56b82d08d3f645dc25ab2",
    "clone_sorter_fingerprint": "552f82bf0f635aa500a5ccc93b76d3966fe0c93b7e1623a0c3e1ab56e47b7cf1",
}
# recipe -> (motion_estimate_id, motion_corrected_recording_id)
PINNED_MOTION_IDS = {
    "dredge_fast_v1": (
        "d92ab158-f54a-5375-86c8-b524187ac15e",
        "013206af-4284-5092-a14e-bf4f383413e3",
    ),
    "dredge_v1": (
        "872a54d3-b665-5b8c-8847-1a6f37ba08ef",
        "9b06fc57-5df6-5018-b8c2-87af5a7f362a",
    ),
}


def _stored_row(table_name, label):
    table = _table(table_name)
    _module, name_key, _column = CONTENT_COLUMNS[table_name]
    if table_name == "SorterParameters":
        sorter, name = label.split("/")
        return (table & {"sorter": sorter, name_key: name}).fetch1()
    return (table & {name_key: label}).fetch1()


def _describe_sha(name):
    from spyglass.spikesorting.v2 import describe_pipeline_preset
    from spyglass.utils.identity import sha256_json

    detail = describe_pipeline_preset(name)
    return sha256_json(json.loads(detail.to_json(orient="records")))


@pytest.mark.database
def test_parameter_fingerprints_and_preset_identity_are_pinned(
    dj_conn, monkeypatch
):
    """Shipped rows, a cloned preset and motion ids keep their base values."""
    import spikeinterface

    import spyglass.spikesorting.v2._orchestration.presets as presets_mod
    from spyglass.spikesorting.v2 import (
        clone_pipeline_preset,
        describe_parameter_rows,
        initialize_v2_defaults,
    )
    from spyglass.spikesorting.v2._core.lookup_validation import (
        parameter_row_fingerprint,
    )
    from spyglass.spikesorting.v2._motion.estimation import (
        motion_corrected_selection_identity,
        motion_estimate_selection_identity,
    )
    from spyglass.spikesorting.v2._orchestration.preflight_motion import (
        resolve_motion_recipe,
    )
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        waveform_recipe_hash,
    )
    from spyglass.spikesorting.v2.pipeline_preset import PipelinePreset
    from spyglass.spikesorting.v2.sorting import SorterParameters

    initialize_v2_defaults()
    report = describe_parameter_rows()
    for table_name, pinned in PINNED_FINGERPRINTS.items():
        rows = {label: _stored_row(table_name, label) for label in pinned}
        actual = {
            label: parameter_row_fingerprint(
                table_name,
                row,
                sorter_keyed=table_name == "SorterParameters",
                matcher_keyed=table_name == "MatcherParameters",
            )
            for label, row in rows.items()
        }
        assert actual == pinned, table_name
        if table_name == "AnalyzerWaveformParameters":
            # The analyzer manifests store this hash.
            assert {
                label: waveform_recipe_hash(row) for label, row in rows.items()
            } == pinned
        sub = report[report["table"] == table_name]
        described = {
            (f"{s}/{p}" if s else p): f
            for s, p, f in zip(
                sub["sorter"], sub["parameter_name"], sub["fingerprint"]
            )
        }
        expected = PINNED_DESCRIBE_FINGERPRINTS.get(
            table_name, {label: fp[:12] for label, fp in pinned.items()}
        )
        assert {label: described[label] for label in expected} == expected

    assert _describe_sha(_PRESET) == PINNED_PRESET["describe"]
    monkeypatch.setattr(
        presets_mod, "_PIPELINE_PRESETS", dict(presets_mod._PIPELINE_PRESETS)
    )
    new_name = "pin_clone_ms5_2026_11"
    try:
        clone_pipeline_preset(_PRESET, new_name, detect_threshold=4.0)
        row = (
            SorterParameters
            & {"sorter": "mountainsort5", "sorter_params_name": new_name}
        ).fetch1()
        assert (
            parameter_row_fingerprint(
                "SorterParameters", row, sorter_keyed=True
            )
            == PINNED_PRESET["clone_sorter_fingerprint"]
        )
        assert _describe_sha(new_name) == PINNED_PRESET["clone_describe"]
    finally:
        (PipelinePreset & {"pipeline_preset": new_name}).delete_quick()
        (SorterParameters & {"sorter_params_name": new_name}).delete_quick()

    # The recipes are read through preflight; the source ids are fixed.
    monkeypatch.setattr(spikeinterface, "__version__", "0.104.3")
    for recipe_name, pinned in PINNED_MOTION_IDS.items():
        recipe = resolve_motion_recipe(recipe_name)
        estimate_id, _ = motion_estimate_selection_identity(
            source_kind="recording",
            source_id=uuid.UUID(int=1),
            source_content_hash="c" * 64,
            artifact_detection_id=uuid.UUID(int=3),
            motion_estimation_params_name=recipe.recipe[
                "motion_estimation_params_name"
            ],
            estimation_params=recipe.estimation_params,
        )
        corrected_id, _ = motion_corrected_selection_identity(
            motion_estimate_id=estimate_id,
            motion_interpolation_params_name=recipe.recipe[
                "motion_interpolation_params_name"
            ],
            interpolation_params=recipe.interpolation_params,
        )
        assert (str(estimate_id), str(corrected_id)) == pinned, recipe_name


@pytest.mark.database
def test_selection_ids_are_pinned(populated_sorting):
    """The smoke recording, artifact detection, sort and a match keep their
    ids; the recording id reads the preprocessing row's content."""
    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
    )
    from spyglass.spikesorting.v2._matching.graph import input_set_hash
    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection
    from tests.spikesorting.v2.test_matcher_graph_inputs import _parts

    row = (SortingSelection & populated_sorting).fetch1()
    artifact_detection_id = (
        RecordingArtifactSelection
        & {
            "recording_id": row["recording_id"],
            "artifact_detection_params_name": "none",
        }
    ).fetch1("artifact_detection_id")
    unitmatch_id = deterministic_id(
        "unitmatch",
        {
            "matcher_params_name": "unitmatch_default",
            "input_set_hash": input_set_hash(*_parts()),
        },
    )
    assert {
        "recording_id": str(row["recording_id"]),
        "artifact_detection_id": str(artifact_detection_id),
        "sorting_id": str(row["sorting_id"]),
        "unitmatch_id": str(unitmatch_id),
    } == {
        "recording_id": "91781355-f84d-536a-8af7-63a5aac15fc8",
        "artifact_detection_id": "17d7ef83-2d69-527f-b05f-aaf70ce93955",
        "sorting_id": "430bbbc6-dfb8-5f09-9024-a77c0f3f64d0",
        "unitmatch_id": "a8a7f837-d673-54b7-be27-b370c1be8cf1",
    }
