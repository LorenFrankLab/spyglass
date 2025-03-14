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

from pprint import pprint  # TODO: Remove before merge
from shutil import rmtree as shutil_rmtree

from functools import cached_property
from pathlib import Path
from typing import Tuple

import datajoint as dj
from numpy import __version__ as np_version
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version

from spyglass.settings import recording_dir, temp_dir
import json

from spyglass.common.common_user import UserEnvironment
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)  # noqa F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.h5_helper_fn import H5pyComparator
from spyglass.utils.nwb_hash import DirectoryHasher
import re
from subprocess import run as sub_run
from packaging.version import parse as version_parse

schema = dj.schema("cbroz_recomp_v0")  # TODO: spikesorting_recompute_v0


@schema
class RecordingRecomputeVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    spikeinterface: varchar(16)
    probeinterface: varchar(16)
    """

    _version_cache = dict()

    # --- Gatekeep recompute attempts ---

    def _fetch_versions(self, package):
        if cached := self._version_cache.get(package):
            return cached
        run_kwargs = dict(capture_output=True, text=True)
        result = sub_run(["pip", "index", "versions", package], **run_kwargs)
        if result.returncode != 0:
            raise ValueError(f"Failed to fetch index: {result.stderr}")
        ret = result.stdout.splitlines()[1].split(": ")[1].split(", ")
        self._version_cache["package"] = sorted(ret, key=version_parse)
        return self._version_cache["package"]

    @property  # @cached_property
    def this_env(self):
        """Return restricted version of self for the current environment."""
        return (
            self & self._package_restr("spikeinterface", si_version)
        ) & self._package_restr("probeinterface", pi_version)

    def _package_restr(self, package, version):
        """Return a restriction string for a package and version.

        If the version is an official release, allow any version that starts
        with the same base. If the version is a dev release, allow any version
        that starts with the same base or the next available version.
        """
        base = re.match(r"(\d+\.\d+\.\d+)", version)
        if "dev" not in version or not base:
            return f"{package} LIKE '{version}%'"
        base = base.group(1)
        avail = self._fetch_versions(package)
        if base not in avail:
            raise ValueError(f"Version {version} not found in {package}")
        next_index = avail.index(base) + 1
        next_avail = avail[next_index] if next_index < len(avail) else base
        return f"{package} like '{base}%' OR {package}='{next_avail}'"

    # --- Inventory versions ---

    def _extract_version(self, obj):
        if isinstance(obj, (list, set, tuple)):
            obj_sets = set()
            for o in obj:
                obj_sets.update(self._extract_version(o))
            return obj_sets
        if (
            isinstance(obj, (str, Path))
            and Path(obj).exists()
            and Path(obj).suffix == ".json"
        ):
            with open(obj, "r") as file:
                data = json.load(file)
            return self._extract_version(data)
        elif not isinstance(obj, dict):
            return {None}

        versions = set([obj.get("version")])
        for val in obj.values():
            versions.update(self._extract_version(val))

        return {v for v in versions if v is not None}

    def make(self, key):
        """Inventory the namespaces present in an analysis file."""
        query = SpikeSortingRecording() & key
        rec_path = Path(query.fetch1("recording_path"))

        si_version = self._extract_version(
            [
                rec_path / f"{fname}.json"
                for fname in ["binary", "si_folder", "provenance"]
            ]
        )
        if len(si_version) != 1:
            raise ValueError(f"Multiple si versions found: {si_version}")

        with open(rec_path / "probe.json", "r") as file:
            pi_version = json.load(file).get("version", None)

        self.insert1(
            dict(
                key,
                probeinterface=pi_version,
                spikeinterface=next(iter(si_version)),
            )
        )


# TODO: try to insert with the new upstream table


@schema
class RecordingRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingRecomputeVersions
    -> UserEnvironment
    ---
    logged_at_creation=0: bool
    """

    @cached_property
    def pip_deps(self):
        return dict(
            spikeinterface=si_version,
            probeinterface=pi_version,
            numpy=np_version,
        )

    def insert(self, rows, logged_at_creation=False, limit=None, **kwargs):
        """Custom insert to ensure dependencies are added to each row."""
        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        this_env = UserEnvironment().insert_current_env()

        inserts = []
        for i, row in enumerate(rows):
            if limit and i >= limit:
                break
            key_pk = self.dict_to_pk(row)
            inserts.append(
                dict(
                    **key_pk,
                    **this_env,
                    logged_at_creation=logged_at_creation,
                )
            )
        super().insert(inserts, **kwargs)

    def attempt_all(
        self, restr: dict = True, attempt_id: str = None, limit=None, **kwargs
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

        if limit:
            source &= dj.condition.Top(limit=limit, order_by="RAND()")

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

    def _has_matching_pip(self, key: dict, show_err=False) -> bool:
        """Check current env for matching pip versions."""
        ret, need = False, "Unknown"
        query = self.this_env & key

        if len(query) == 1:
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

        def compare(self, key=None, name: str = None):
            if key is None:
                key = self.restriction or True
            if isinstance(key, list):
                for row in key:
                    self.compare(row)
                return
            if this_name := name or key.get("name"):
                old, new = RecordingRecompute()._get_paths(key)
                print(f"Comparing {this_name}\n\t{old}\n\t{new}")
                return H5pyComparator(*self.get_objs(key, name))
            query = self & key
            if len(query) == 0 or len(query) == len(self):
                return
            for row in query:
                self.compare(row)

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
        log_key = key.get("nwb_file_name", key)
        if parent.get("logged_at_creation", True):
            logger.info(f"Skipping logged_at_creation {log_key}")
            self.insert1({**key, "matched": True})
            return

        # Ensure dependencies unchanged since selection insert
        if not RecordingRecomputeSelection()._has_matching_pip(key):
            logger.info(
                f"Skipping due to pip mismatch, {key['attempt_id']}: {log_key}"
            )
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
            logger.info(f"Matched {log_key}")
            self.insert1({**key, "matched": True})
            return

        # only show file name if available
        logger.info(f"Failed to match {log_key}")

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

    def delete(self, *args, **kwargs):

        attempt_dirs = []
        for key in self:
            _, new = self._get_paths(key)
            attempt_dirs.append(new)

        msg = "Delete attempt files?\n\t" + "\n\t".join(attempt_dirs)
        if dj.utils.user_choice(msg).lower() == "yes":
            for dir in attempt_dirs:
                shutil_rmtree(dir, ignore_errors=True)

        super().delete(*args, **kwargs)
