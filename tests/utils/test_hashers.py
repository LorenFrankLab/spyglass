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


def test_nwb_hasher(nwb_hasher):

    hash = nwb_hasher.hash
    assert hash is not None
    assert isinstance(hash, str)
    assert len(hash) == 32
    assert hash.startswith("4e0c"), "Unexpected NWB file hash"

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
