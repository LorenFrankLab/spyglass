"""The ingestion plan cache: what it reuses, and what invalidates it."""

import json

import pytest


@pytest.fixture
def cache(common):
    """The plan cache module, emptied before and after each test."""
    from spyglass.data_import import plan_cache

    plan_cache.clear_cache()
    yield plan_cache
    plan_cache.clear_cache()


def test_config_fingerprint_ignores_key_order(cache):
    """The same config written two ways is the same config."""
    a = {"Institution": [{"institution_name": "UCSF"}], "Lab": []}
    b = {"Lab": [], "Institution": [{"institution_name": "UCSF"}]}

    assert cache.config_fingerprint(a) == cache.config_fingerprint(
        b
    ), "Key order must not change the digest"
    assert cache.config_fingerprint({}) == cache.config_fingerprint(
        None
    ), "An absent config and an empty one are the same input"
    assert cache.config_fingerprint(a) != cache.config_fingerprint(
        {"Institution": [{"institution_name": "Other"}]}
    ), "A different config must not reuse a plan"


def test_file_fingerprint_tracks_the_file(cache, tmp_path):
    """A rewritten file gets a new fingerprint; a missing one gets none."""
    path = tmp_path / "f.nwb"
    path.write_bytes(b"one")
    first = cache.file_fingerprint(path)

    assert first, "A readable file should fingerprint"
    assert cache.file_fingerprint(path) == first, "Unchanged file, same key"

    path.write_bytes(b"a longer body")
    assert (
        cache.file_fingerprint(path) != first
    ), "A changed file must not reuse the old plan"

    assert (
        cache.file_fingerprint(tmp_path / "missing.nwb") is None
    ), "An unreadable file yields no key, so nothing is cached under one"


def test_plan_round_trips_through_the_cache(cache, mini_copy_name):
    """A saved plan is returned intact for the same provenance."""
    from spyglass.data_import.planner import plan_nwbfile

    first = plan_nwbfile(mini_copy_name, use_cache=True)

    assert first.nwb_hash, "A whole-file plan should carry its provenance"
    assert first.config_hash and first.spyglass_version

    second = plan_nwbfile(mini_copy_name, use_cache=True)

    assert (
        second.plan_hash == first.plan_hash
    ), "The cached plan should be the plan that was cached"
    assert second.nwb_file_name == first.nwb_file_name
    assert [t.table_name for t in second.table_plans] == [
        t.table_name for t in first.table_plans
    ], "Table plans should survive the round trip, in order"


def test_a_different_config_is_a_different_plan(cache, mini_copy_name):
    """Config is part of the key, not something the cache ignores."""
    from spyglass.data_import.planner import plan_nwbfile

    plan_nwbfile(mini_copy_name, use_cache=True)
    other = plan_nwbfile(
        mini_copy_name,
        config={"Institution": [{"institution_name": "Xyz"}]},
        use_cache=True,
    )

    assert other.config_hash != cache.config_fingerprint(
        {}
    ), "A configured run must not be served the unconfigured plan"


def test_a_table_subset_is_never_cached(cache, common, mini_copy_name):
    """A partial plan must not later stand in for the whole file."""
    from spyglass.data_import.planner import plan_nwbfile

    partial = plan_nwbfile(
        mini_copy_name, tables=[common.Institution], use_cache=True
    )

    assert (
        partial.nwb_hash is None
    ), "A subset plan carries no provenance, so it cannot be keyed"
    assert not list(
        cache.cache_dir().glob("*.json")
    ), "A subset plan must not be written to the cache"


def test_use_cache_false_reparses(cache, mini_copy_name):
    """The cache is escapable without clearing it."""
    from spyglass.data_import.planner import plan_nwbfile

    plan_nwbfile(mini_copy_name, use_cache=True)
    cached_files = list(cache.cache_dir().glob("*.json"))
    assert cached_files, "The first whole-file plan should be cached"

    plan = plan_nwbfile(mini_copy_name, use_cache=False)
    assert plan.nwb_hash is None, "An uncached run does not key itself"


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(cache, mini_copy_name):
    """A bad cache costs a re-parse, never an ingestion."""
    from spyglass.data_import.planner import plan_nwbfile

    first = plan_nwbfile(mini_copy_name, use_cache=True)
    (path,) = list(cache.cache_dir().glob("*.json"))
    path.write_text("{not json")

    second = plan_nwbfile(mini_copy_name, use_cache=True)

    assert (
        second.plan_hash == first.plan_hash
    ), "A corrupt entry should be re-planned, not raised on"


def test_provenance_mismatch_is_rejected(cache, mini_copy_name):
    """A file whose contents disagree with its key is not served."""
    from spyglass.data_import.planner import plan_nwbfile

    plan_nwbfile(mini_copy_name, use_cache=True)
    (path,) = list(cache.cache_dir().glob("*.json"))

    tampered = json.loads(path.read_text())
    tampered["config_hash"] = "not-the-config-this-was-built-with"
    path.write_text(json.dumps(tampered))

    plan = plan_nwbfile(mini_copy_name, use_cache=True)

    assert (
        plan.config_hash != "not-the-config-this-was-built-with"
    ), "A plan whose stored provenance disagrees must be discarded"
