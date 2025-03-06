"""This schema is used to track recompute capabilities for existing files."""

from functools import cached_property
from os import environ as os_environ
from pathlib import Path
from typing import Tuple

import datajoint as dj
from numpy import __version__ as np_version
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version

from spyglass.settings import recording_dir, temp_dir
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)  # noqa F401
from spyglass.utils import logger
from spyglass.utils.nwb_hash import DirectoryHasher

schema = dj.schema("cbroz_temp_v0")


@schema
class RecordingRecomputeSelection(dj.Manual):
    definition = """
    -> SpikeSortingRecording
    attempt_id: varchar(32) # name for environment used to attempt recompute
    ---
    logged_at_creation=0: bool
    pip_deps: blob # dict of pip dependencies
    """

    @cached_property
    def default_attempt_id(self):
        user = dj.config["database.user"]
        conda = os_environ.get("CONDA_DEFAULT_ENV", "base")
        return f"{user}_{conda}"

    @cached_property
    def pip_deps(self):
        return dict(
            spikeinterface=si_version,
            probeinterface=pi_version,
            numpy=np_version,
        )

    def key_pk(self, key):
        return {k: key[k] for k in self.primary_key}

    def insert(self, rows, logged_at_creation=False, **kwargs):
        """Custom insert to ensure dependencies are added to each row."""
        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        inserts = []
        for row in rows:
            key_pk = self.key_pk(row)
            inserts.append(
                dict(
                    **key_pk,
                    attempt_id=row.get("attempt_id", self.default_attempt_id),
                    dependencies=self.pip_deps,
                    logged_at_creation=logged_at_creation,
                )
            )
        super().insert(inserts, **kwargs)

    # --- Gatekeep recompute attempts ---

    @cached_property
    def this_env(self):
        """Restricted table matching pynwb env and pip env.

        Serves as key_source for RecordingRecompute. Ensures that recompute
        attempts are only made when the pynwb and pip environments match the
        records. Also skips files whose environment was logged on creation.
        """

        restr = []
        for key in self:
            if key["dependencies"] != self.pip_deps:
                continue
            restr.append(self.key_pk(key))
        return self & restr

    def _has_matching_pip(self, key, show_err=True) -> bool:
        """Check current env for matching pip versions."""
        query = self.this_env & key

        if not len(query) == 1:
            raise ValueError(f"Query returned {len(query)} entries: {query}")

        need = query.fetch1("dependencies")
        ret = need == self.pip_deps

        if not ret and show_err:
            logger.error(
                f"Pip version mismatch. Skipping key: {self.key_pk(key)}"
                + f"\n\tHave: {self.pip_deps}"
                + f"\n\tNeed: {need}"
            )

        return ret


@schema
class RecordingRecompute(dj.Computed):
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

    def _parent_key(self, key):
        ret = SpikeSortingRecording * RecordingRecomputeSelection & key
        if len(ret) != 1:
            raise ValueError(f"Query returned {len(ret)} entries: {ret}")
        return ret.fetch(as_dict=True)[0]

    def _hash_one(self, path):
        str_path = str(path)
        if str_path in self._hasher_cache:
            return self._hasher_cache[str_path]
        hasher = DirectoryHasher(path=path, keep_file_hash=True)
        self._hasher_cache[str_path] = hasher
        return hasher

    def _get_temp_dir(self, key):
        return Path(temp_dir) / "spikesort_v0_recompute" / key["attempt_id"]

    def _get_paths(self, key, as_str=False) -> Tuple[Path, Path]:
        """Return the old and new file paths."""
        rec_name = SpikeSortingRecording().__get_recording_name(key)

        old = Path(recording_dir) / rec_name
        new = self._get_temp_dir(key) / rec_name

        return (str(old), str(new)) if as_str else (old, new)

    def make(self, key):
        # Skip recompute for files logged at creation
        parent = (SpikeSortingRecording & key).fetch1()
        if parent.get("logged_at_creation", True):
            self.insert1({**key, "matched": True})

        # Ensure dependencies unchanged since selection insert
        if not RecordingRecomputeSelection()._has_matching_pip(key):
            return

        old, new = self._get_paths(key)

        new_hasher = (
            self._hash_one(new)
            if new.exists()
            else SpikeSortingRecording()._make_file(
                key, base_dir=self._get_temp_dir(key), return_hasher=True
            )["hash"]
        )

        if parent["hash"] == new_hasher.hash:
            self.insert1({**key, "matched": True})
            return

        old_hasher = self._hash_one(old)
        names, hashes = [], []
        for file in set(new_hasher.cache) | set(old_hasher.cache):
            if file not in old_hasher.cache:
                names.append((key, file, "old"))
                continue
            if file not in new_hasher.cache:
                names.append((key, file, "new"))
                continue
            if old_hasher[file] != new_hasher[file]:
                hashes.append((key, file))

        self.insert1(dict(key, matched=False))
        self.Name().insert(names)
        self.Hash().insert(hashes)

    def delete_file(self, key, dry_run=True):
        """If successfully recomputed, delete files for a given restriction."""
        query = self & "matched=1" & key
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
