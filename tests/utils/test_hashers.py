import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile


def make_mvp_hash_files(base_dir: Path):
    """
    Create minimal files:
      - sample.json (valid JSON)
      - array.npy   (valid NumPy array)
      - whatever.xyz (arbitrary contents)
    under base_dir / tmp / test_hasher
    """
    target = Path(base_dir) / "tmp" / "test_hasher"
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "sample.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"foo": 1, "bar": [1, 2, 3]}, f)

    npy_path = target / "array.npy"
    np.save(npy_path, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    xyz_path = target / "whatever.xyz"
    xyz_path.write_text("this can be anything\n", encoding="utf-8")

    nwb_path = target / "dummy.nwb"
    nwbfile = mock_NWBFile()  # creates a small, valid NWBFile for testing
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    return target


@pytest.fixture
def dir_hasher(base_dir):
    from spyglass.utils.nwb_hash import DirectoryHasher

    mvp_hash_dir = make_mvp_hash_files(base_dir)
    yield DirectoryHasher(mvp_hash_dir, keep_obj_hash=True)


def test_dir_hasher(dir_hasher):
    hash = dir_hasher.hash
    assert hash is not None
    assert isinstance(hash, str)
    assert len(hash) == 32

    cache = dir_hasher.cache
    assert "sample.json" in cache
    assert "array.npy" in cache
    assert "whatever.xyz" in cache
    assert "dummy.nwb" in cache


@pytest.fixture
def nwb_hasher(mini_path):
    from spyglass.utils.nwb_hash import NwbfileHasher

    yield NwbfileHasher(mini_path, precision_lookup=5, keep_obj_hash=True)


@pytest.mark.slow
def test_nwb_hasher(nwb_hasher):

    hash = nwb_hasher.hash
    assert hash is not None
    assert isinstance(hash, str)
    assert len(hash) == 32
    assert hash.startswith("1d393"), "Unexpected NWB file hash"

    # Check that individual object hashes are present
    cache = nwb_hasher.objs
    assert "acquisition" in cache
    assert "processing" in cache

    precision = nwb_hasher.precision.get("ProcessedElectricalSeries")
    assert precision == 5

    roundable = nwb_hasher.is_roundable(5)
    not_roundable = nwb_hasher.is_roundable(None)
    assert roundable is True and not_roundable is False

    skipped_obj = SimpleNamespace(name="version")
    assert nwb_hasher.hash_dataset(skipped_obj) is None


@pytest.fixture
def nwb_legacy_hasher(mini_path):
    from spyglass.utils.nwb_hash import NwbfileHasher

    yield NwbfileHasher(
        mini_path, precision_lookup=5, keep_obj_hash=True, legacy_mode=True
    )


@pytest.mark.slow
def test_nwb_legacy_hasher(nwb_legacy_hasher):
    hash = nwb_legacy_hasher.hash
    assert hash.startswith("4e0c"), "Unexpected NWB file hash"


def test_object_id_index_does_not_change_the_file_hash(mini_path):
    """The index is additive: stored hashes must not shift because of it.

    Production hashes in `SpikeSortingRecording` depend on this value, so the
    new flag has to be provably inert with respect to it.
    """
    from spyglass.utils.nwb_hash import NwbfileHasher

    plain = NwbfileHasher(mini_path)
    indexed = NwbfileHasher(mini_path, object_ids=True)

    assert (
        plain.hash == indexed.hash
    ), "object_ids=True must not alter the file hash"
    assert indexed.obj_ids, "The index should hold the file's objects"
    assert not plain.obj_ids, "The index should be off by default"


def test_object_id_index_addresses_nested_objects(mini_path):
    """Group children are keyed by name, so nested objects are reachable.

    `add_to_cache` was called with a literal "k" rather than the loop
    variable, collapsing every child of a group onto one overwritten key.
    """
    from spyglass.utils.nwb_hash import NwbfileHasher

    hasher = NwbfileHasher(mini_path, object_ids=True)
    paths = [path for path, _ in hasher.obj_ids.values()]

    assert not [
        p for p in paths if p.endswith("/k")
    ], "No object should be indexed under a literal 'k'"
    assert any(
        p.count("/") >= 2 for p in paths
    ), "Nested objects should be addressable by their own path"


def test_read_set_digest_is_order_free_and_absence_sensitive(mini_path):
    """The digest answers 'did what this table read change?'"""
    from spyglass.utils.nwb_hash import NwbfileHasher

    hasher = NwbfileHasher(mini_path, object_ids=True)
    read = list(hasher.obj_ids)[:4]

    assert hasher.read_set_digest(read) == hasher.read_set_digest(
        list(reversed(read))
    ), "The order a table read objects in is not part of its identity"
    assert hasher.read_set_digest(read) != hasher.read_set_digest(
        read[:-1]
    ), "Reading fewer objects is a different read-set"
    assert hasher.read_set_digest(read) != hasher.read_set_digest(
        read + ["not-in-this-file"]
    ), "An object that has vanished must not hash as unchanged"
    assert (
        hasher.digest_for("not-in-this-file") is None
    ), "An unknown object id has no digest"


def test_object_indexed_where_it_lives_not_where_it_is_linked(mini_path):
    """A soft-linked object is indexed at its real home.

    NWB soft-links a device into the series that uses it, and h5py hands back
    the dereferenced object at both paths. Rolling the subtree digest up at
    the link would cover the object's own attributes and nothing beneath it,
    so a probe would hash the same however its shanks changed.
    """
    import h5py

    from spyglass.utils.nwb_hash import NwbfileHasher

    hasher = NwbfileHasher(mini_path, object_ids=True)

    with h5py.File(mini_path, "r") as file:
        linked = [
            path
            for path, _ in hasher.obj_ids.values()
            if isinstance(file.get(path, getlink=True), h5py.SoftLink)
        ]

    assert not linked, f"Objects indexed at a soft link: {linked}"


def test_a_child_change_moves_its_container_digest(mini_path, tmp_path):
    """A container's digest covers what is under it, not just its own attrs."""
    import shutil

    import h5py

    from spyglass.utils.nwb_hash import NwbfileHasher

    copy = tmp_path / "copy_.nwb"
    shutil.copy2(mini_path, copy)

    before = NwbfileHasher(copy, object_ids=True)
    container = next(
        (oid, path)
        for oid, (path, _) in before.obj_ids.items()
        if any(p.startswith(f"{path}/") for p, _ in before.obj_ids.values())
    )
    object_id, path = container

    with h5py.File(copy, "r+") as file:
        child = next(iter(file[path]))
        file[f"{path}/{child}"].attrs["_probe_of_change"] = "edited"

    after = NwbfileHasher(copy, object_ids=True)

    assert before.digest_for(object_id) != after.digest_for(
        object_id
    ), "A change beneath a container must change the container's digest"
