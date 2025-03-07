"""This schema is used to track recompute capabilities for existing files.

Tables
------
RecordingRecomputeSelection:
    - Manual table to track recompute attempts.
    - Records the pip dependencies used to attempt recompute.
    - Records whether the attempt was logged at creation, skipping recomputes.
RecordingRecompute:
    - Computed table to track recompute success.
    - Runs `SpikeSortingRecording()._make_file` to recompute files, saving the
        resulting folder to `temp_dir/spikesort_v0_recompute/{attempt_id}`.
"""

from functools import cached_property
from pathlib import Path
from typing import Tuple

import datajoint as dj
from numpy import __version__ as np_version
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version

from spyglass.settings import recording_dir, temp_dir
from spyglass.spikesorting.utils import DEFAULT_ATTEMPT_ID
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)  # noqa F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.h5_helper_fn import H5pyComparator
from spyglass.utils.nwb_hash import DirectoryHasher

schema = dj.schema("cbroz_temp_v0")


@schema
class RecordingRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingRecording
    attempt_id: varchar(32) # name for environment used to attempt recompute
    ---
    logged_at_creation=0: bool
    pip_deps: blob # dict of pip dependencies
    """

    @cached_property
    def pip_deps(self):
        return dict(
            spikeinterface=si_version,
            probeinterface=pi_version,
            numpy=np_version,
        )

    def insert(self, rows, logged_at_creation=False, **kwargs):
        """Custom insert to ensure dependencies are added to each row."""
        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        inserts = []
        for row in rows:
            key_pk = self.dict_to_pk(row)
            inserts.append(
                dict(
                    **key_pk,
                    attempt_id=row.get("attempt_id", DEFAULT_ATTEMPT_ID),
                    pip_deps=self.pip_deps,
                    logged_at_creation=logged_at_creation,
                )
            )
        super().insert(inserts, **kwargs)

    def attempt_all(
        self, restr: dict = True, attempt_id: str = None, **kwargs
    ) -> None:
        """Insert all files into the recompute table.

        Parameters
        ----------
        restr : dict, optional
            Key or restriction of SpikeSortingRecording. If None, all files
            are inserted.
        attempt_id : str, optional
            Name for the environment used to attempt recompute. If None, the
            current user and conda environment are used.
        """
        source = SpikeSortingRecording() & restr
        kwargs["skip_duplicates"] = True

        inserts = [
            {
                **key,
                "attempt_id": attempt_id or DEFAULT_ATTEMPT_ID,
                "logged_at_creation": False,
            }
            for key in source.fetch("KEY", as_dict=True)
        ]
        self.insert(inserts, **kwargs)

    # --- Gatekeep recompute attempts ---

    @property  # key_source must not be a cached_property
    def this_env(self):
        """Restricted table matching pynwb env and pip env.

        Serves as key_source for RecordingRecompute. Ensures that recompute
        attempts are only made when the pynwb and pip environments match the
        records. Also skips files whose environment was logged on creation.
        """

        restr = []
        for key in self:
            if key["pip_deps"] != self.pip_deps:
                continue
            restr.append(self.dict_to_pk(key))
        return self & restr

    def _has_matching_pip(self, key: dict, show_err=True) -> bool:
        """Check current env for matching pip versions."""
        query = self.this_env & key

        if not len(query) == 1:
            raise ValueError(f"Query returned {len(query)} entries: {query}")

        need = query.fetch1("pip_deps")
        ret = need == self.pip_deps

        if not ret and show_err:
            logger.error(
                f"Pip version mismatch. Skipping key: {self.dict_to_pk(key)}"
                + f"\n\tHave: {self.pip_deps}"
                + f"\n\tNeed: {need}"
            )

        return ret


@schema
class RecordingRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingRecomputeSelection
    ---
    matched:bool
    """

    _hasher_cache = dict()

    class Name(dj.Part):
        definition = """ # File names missing from old or new versions
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(dj.Part):
        definition = """ # File hashes that differ between old and new versions
        -> master
        name : varchar(255)
        """

        def get_objs(self, key, name: str = None):
            old, new = RecordingRecompute()._get_paths(key, as_str=True)

            old_hasher = RecordingRecompute()._hash_one(old)
            new_hasher = RecordingRecompute()._hash_one(new)

            name = name or key["name"]
            old_obj = old_hasher.cache.get(name)
            new_obj = new_hasher.cache.get(name)

            return old_obj, new_obj

        def compare(self, key, name: str = None):
            return H5pyComparator(*self.get_objs(key, name))

    def _parent_key(self, key):
        ret = SpikeSortingRecording * RecordingRecomputeSelection & key
        if len(ret) != 1:
            raise ValueError(f"Query returned {len(ret)} entries: {ret}")
        return ret.fetch(as_dict=True)[0]

    def _hash_one(self, path):
        str_path = str(path)
        if str_path in self._hasher_cache:
            return self._hasher_cache[str_path]
        hasher = DirectoryHasher(directory_path=path, keep_file_hash=True)
        self._hasher_cache[str_path] = hasher
        return hasher

    def _get_temp_dir(self, key):
        return Path(temp_dir) / "spikesort_v0_recompute" / key["attempt_id"]

    def _get_paths(self, key, as_str=False) -> Tuple[Path, Path]:
        """Return the old and new file paths."""
        rec_name = SpikeSortingRecording()._get_recording_name(key)

        old = Path(recording_dir) / rec_name
        new = self._get_temp_dir(key) / rec_name

        return (str(old), str(new)) if as_str else (old, new)

    def make(self, key):
        # Skip recompute for files logged at creation
        parent = self._parent_key(key)
        if parent.get("logged_at_creation", True):
            logger.info(f"Skipping logged_at_creation {key}")
            self.insert1({**key, "matched": True})
            return

        # Ensure dependencies unchanged since selection insert
        if not RecordingRecomputeSelection()._has_matching_pip(key):
            logger.info(f"Skipping due to pip mismatch: {key}")
            return

        old, new = self._get_paths(key)

        new_hasher = (
            self._hash_one(new)
            if new.exists()
            else SpikeSortingRecording()._make_file(
                key, base_dir=self._get_temp_dir(key), return_hasher=True
            )["hash"]
        )

        # hack pending table alter
        old_hash = parent.get("hash") or self._hash_one(old).hash

        if old_hash == new_hasher.hash:
            logger.info(f"Matched {key}")
            self.insert1({**key, "matched": True})
            return

        old_hasher = self._hash_one(old)
        names, hashes = [], []
        for file in set(new_hasher.cache) | set(old_hasher.cache):
            if file not in old_hasher.cache:
                names.append(dict(key, name=file, missing_from="old"))
                continue
            if file not in new_hasher.cache:
                names.append(dict(key, name=file, missing_from="new"))
                continue
            if old_hasher.cache[file] != new_hasher.cache[file]:
                hashes.append(dict(key, name=file))

        logger.info(f"Failed to match {key}")
        self.insert1(dict(key, matched=False))
        self.Name().insert(names)
        self.Hash().insert(hashes)

    def delete_files(self, restriction=True, dry_run=True):
        """If successfully recomputed, delete files for a given restriction."""
        query = self & "matched=1" & restriction
        file_names = query.fetch("analysis_file_name")
        prefix = "DRY RUN: " if dry_run else ""
        msg = f"{prefix}Delete {len(file_names)} files?\n\t" + "\n\t".join(
            file_names
        )

        if dry_run:
            logger.info(msg)
            return

        if dj.utils.user_choice(msg).lower() not in ["yes", "y"]:
            return

        for key in query.proj():
            old, new = self._get_paths(key)
            new.unlink(missing_ok=True)
            old.unlink(missing_ok=True)
