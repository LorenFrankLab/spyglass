"""Explicit v2 test tiers, checked against pytest's resolved fixture graph.

The manifest records reviewed module defaults and function overrides. A new
module must declare a tier or be added to the manifest; a new database fixture
must also declare its minimum tier here. No unknown module defaults to unit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

TIERS = ("unit", "db_unit", "stage", "pipeline", "regression_gate")
TIER_RANK = {tier: index for index, tier in enumerate(TIERS)}
DB_FIXTURES = frozenset({"dj_conn", "dj_config", "server"})
EXTERNAL_DATA_FIXTURES = frozenset(
    {"smoke_nwb", "mini_path", "mini_content", "mini_open", "mini_closed"}
)
MANIFEST_PATH = Path(__file__).with_name("test_tiers.json")
SUITE_PATH = MANIFEST_PATH.parent


class TierClassificationError(ValueError):
    """A test or fixture has no valid, isolated tier assignment."""


@dataclass(frozen=True)
class ModuleTiers:
    """Reviewed default and named overrides for a test module."""

    tier: str
    overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TierManifest:
    """Assignments independent of test collection and runtime execution."""

    modules: Mapping[str, ModuleTiers]
    fixtures: Mapping[str, str]


def _valid_tier(value: object, context: str) -> str:
    if not isinstance(value, str) or value not in TIER_RANK:
        raise TierClassificationError(
            f"{context}: expected one of {TIERS}, got {value!r}"
        )
    return value


def load_manifest(path: Path = MANIFEST_PATH) -> TierManifest:
    """Read validated assignments; malformed entries never select unit."""
    raw = json.loads(path.read_text())
    if raw.get("version") != 1:
        raise TierClassificationError(
            "Unsupported v2 test-tier manifest version"
        )
    modules = {}
    for module, entry in raw["modules"].items():
        modules[module] = ModuleTiers(
            _valid_tier(entry["tier"], module),
            {
                name: _valid_tier(tier, f"{module}::{name}")
                for name, tier in entry.get("overrides", {}).items()
            },
        )
    fixtures = {
        name: _valid_tier(tier, name) for name, tier in raw["fixtures"].items()
    }
    return TierManifest(modules, fixtures)


def classify_test(
    manifest: TierManifest,
    module: str,
    name: str,
    *,
    declared_tier: str | None = None,
    fixture_tiers: Mapping[str, str] | None = None,
    needs_database: bool = False,
    needs_external_data: bool = False,
) -> str:
    """Resolve an assignment and reject fixture dependencies below its tier.

    Named manifest overrides define each module's reviewed exceptions. An
    explicit marker can raise a module default (e.g. a new regression gate),
    while lowering a reviewed default requires a named manifest override.
    """
    entry = manifest.modules.get(module)
    if entry is None:
        if declared_tier is None:
            raise TierClassificationError(
                f"Unclassified v2 test {module}::{name}; declare a tier or "
                "add the module to tests/spikesorting/v2/test_tiers.json"
            )
        tier = _valid_tier(declared_tier, f"{module}::{name}")
    elif name in entry.overrides:
        tier = entry.overrides[name]
    else:
        choices = [entry.tier]
        if declared_tier is not None:
            choices.append(_valid_tier(declared_tier, f"{module}::{name}"))
        tier = max(choices, key=TIER_RANK.__getitem__)

    minimums = dict(fixture_tiers or {})
    if needs_database:
        minimums["database dependency"] = "db_unit"
    if needs_external_data:
        minimums["external test data"] = "db_unit"
    violations = [
        f"{fixture} requires {minimum}"
        for fixture, minimum in minimums.items()
        if TIER_RANK[tier] < TIER_RANK[minimum]
    ]
    if violations:
        raise TierClassificationError(
            f"{module}::{name} is classified {tier}, but "
            + "; ".join(violations)
            + ". Update the tier; unit tests must use isolated fixtures."
        )
    return tier


def fixture_key(baseid: str, name: str) -> str:
    """Distinguish overridden fixtures by their defining pytest node."""
    return f"{baseid}:{name}"


def database_fixture_names(fixture_defs: Mapping) -> set[str]:
    """Find database-dependent fixtures using the selected override only."""
    found: set[str] = set()

    def needs_db(name: str, visiting: frozenset[str] = frozenset()) -> bool:
        if name in DB_FIXTURES or name in found:
            found.add(name)
            return True
        if name in visiting:
            return False
        definitions = fixture_defs.get(name)
        if not definitions:
            return False
        selected = definitions[-1]
        if any(
            needs_db(dependency, visiting | {name})
            for dependency in selected.argnames
        ):
            found.add(name)
            return True
        return False

    for name in fixture_defs:
        needs_db(name)
    return found


def fixture_minimums(
    manifest: TierManifest, fixture_defs: Mapping
) -> dict[str, str]:
    """Require an explicit minimum for every resolved database fixture.

    Only the last definition is active in pytest's fixture closure; an
    overridden parent must neither inflate a pure tier nor hide a new DB
    dependency in the selected override.
    """
    database_names = database_fixture_names(fixture_defs)
    minimums = {}
    for name, alternatives in fixture_defs.items():
        if not alternatives:
            continue
        selected = alternatives[-1]
        key = fixture_key(selected.baseid, name)
        minimum = manifest.fixtures.get(key)
        if minimum is not None:
            minimums[key] = minimum
        elif name in database_names:
            raise TierClassificationError(
                f"Unclassified database fixture {key}; add its minimum tier "
                "to tests/spikesorting/v2/test_tiers.json"
            )
    return minimums
