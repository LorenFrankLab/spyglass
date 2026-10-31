"""Tier classification fails closed and respects pytest fixture overrides."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.spikesorting.v2._test_tiers import (
    ModuleTiers,
    TierClassificationError,
    TierManifest,
    classify_test,
    database_fixture_names,
    fixture_minimums,
    load_manifest,
)

pytestmark = pytest.mark.unit


def test_unknown_module_requires_an_explicit_tier():
    manifest = TierManifest({}, {})
    with pytest.raises(TierClassificationError, match="Unclassified"):
        classify_test(manifest, "test_new.py", "test_example")
    assert (
        classify_test(
            manifest, "test_new.py", "test_example", declared_tier="stage"
        )
        == "stage"
    )


@pytest.mark.parametrize(
    "dependencies",
    [
        {"needs_database": True},
        {"needs_external_data": True},
        {"fixture_tiers": {"nested_population": "pipeline"}},
    ],
)
def test_a_new_dependency_cannot_silently_enter_the_unit_tier(dependencies):
    manifest = TierManifest({"test_pure.py": ModuleTiers("unit")}, {})
    with pytest.raises(TierClassificationError, match="classified unit"):
        classify_test(manifest, "test_pure.py", "test_new", **dependencies)


def test_function_override_keeps_a_mixed_module_out_of_unit_selection():
    manifest = TierManifest(
        {
            "test_mixed.py": ModuleTiers(
                "unit", {"TestDatabase::test_rows": "db_unit"}
            )
        },
        {},
    )
    assert (
        classify_test(
            manifest,
            "test_mixed.py",
            "TestDatabase::test_rows",
            declared_tier="unit",
            needs_database=True,
        )
        == "db_unit"
    )
    assert classify_test(manifest, "test_mixed.py", "test_helper") == "unit"


def test_pipeline_fixture_cannot_be_assigned_to_a_lower_database_tier():
    manifest = TierManifest({"test_db.py": ModuleTiers("db_unit")}, {})
    with pytest.raises(TierClassificationError, match="requires pipeline"):
        classify_test(
            manifest,
            "test_db.py",
            "test_rows",
            fixture_tiers={"populated_sorting": "pipeline"},
        )


def test_explicit_heavy_marker_can_raise_a_module_default():
    manifest = TierManifest({"test_pure.py": ModuleTiers("unit")}, {})
    assert (
        classify_test(
            manifest,
            "test_pure.py",
            "test_memory",
            declared_tier="regression_gate",
        )
        == "regression_gate"
    )


def test_database_fixture_graph_follows_only_the_resolved_override():
    root = SimpleNamespace(argnames=("dj_conn",))
    isolated_override = SimpleNamespace(argnames=())
    wrapper = SimpleNamespace(argnames=("mini_insert",))
    definitions = {
        "mini_insert": (root, isolated_override),
        "wrapper": (wrapper,),
    }
    assert database_fixture_names(definitions) == set()
    definitions["mini_insert"] = (root,)
    assert database_fixture_names(definitions) == {
        "dj_conn",
        "mini_insert",
        "wrapper",
    }


def test_database_dependency_is_found_behind_nested_fixtures():
    definitions = {
        "sorting": (SimpleNamespace(argnames=("recording",)),),
        "recording": (SimpleNamespace(argnames=("dj_conn",)),),
    }
    assert database_fixture_names(definitions) == {
        "dj_conn",
        "recording",
        "sorting",
    }


def test_new_database_fixture_requires_reviewed_registration():
    definition = SimpleNamespace(
        baseid="tests/spikesorting/v2/test_new.py", argnames=("dj_conn",)
    )
    with pytest.raises(
        TierClassificationError, match="Unclassified database fixture"
    ):
        fixture_minimums(TierManifest({}, {}), {"new_sort": (definition,)})
    manifest = TierManifest(
        {}, {"tests/spikesorting/v2/test_new.py:new_sort": "pipeline"}
    )
    assert fixture_minimums(manifest, {"new_sort": (definition,)}) == {
        "tests/spikesorting/v2/test_new.py:new_sort": "pipeline"
    }


def test_unit_runtime_guard_rejects_queries_and_connections(monkeypatch):
    from tests import conftest

    class Connection:
        def connect(self):
            return "connection unexpectedly opened"

        def query(self, query):
            return "query unexpectedly ran"

    monkeypatch.setattr(conftest, "_UNIT_DATABASE_METHODS", {})
    monkeypatch.setattr(conftest, "dj", SimpleNamespace(Connection=Connection))
    conftest._block_unit_database_access()
    connection = Connection()
    for method, args in (
        (connection.connect, ()),
        (connection.query, ("SELECT 1",)),
    ):
        with pytest.raises(AssertionError, match="v2 unit tier"):
            method(*args)


def test_manifest_overrides_reference_real_test_definitions():
    import ast

    manifest = load_manifest()
    suite = Path(__file__).parent
    for module, assignments in manifest.modules.items():
        path = suite / module
        assert path.is_file(), f"Stale module tier: {module}"
        tree = ast.parse(path.read_text())
        names = set()

        def visit(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}::")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        names.add(f"{prefix}{node.name}")

        visit(tree.body)
        stale = set(assignments.overrides) - names
        assert not stale, f"Stale tier overrides in {module}: {sorted(stale)}"
