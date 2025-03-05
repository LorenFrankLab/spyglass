"""This schema is used to track recompute capabilities for existing files."""

from functools import cached_property
from os import environ as os_environ

import datajoint as dj
from numpy import __version__ as np_version
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version

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

    def insert(self, rows, at_creation=False, **kwargs):
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
                    logged_at_creation=at_creation,
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
        for key in self & "logged_at_creation=0":
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

    def _hash_one(self, key):
        key_hash = dj.hash.key_hash(key)
        if key_hash in self._hasher_cache:
            return self._hasher_cache[key_hash]
        hasher = DirectoryHasher(
            path=self._parent_key(key)["recording_path"],
            keep_file_hash=True,
        )
        self._hasher_cache[key_hash] = hasher
        return hasher

    def make(self, key):
        pass

    def delete_file(self, key):
        pass  # TODO: Add means of deleting repliacted files
