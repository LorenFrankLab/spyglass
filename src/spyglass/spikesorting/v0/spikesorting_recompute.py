"""This schema is used to track recompute capabilities for existing files.

Tables
------
UserEnvironment (upstream): Table to track the current environment.
RecordingRecomputeVersions: Computed table to track versions of spikeinterface
    and probeinterface used to generate existing files.
RecordingRecomputeSelection: Plan a recompute attempt for a given key.
RecordingRecompute: Computed table to track recompute success.
    Runs `SpikeSortingRecording()._make_file` to recompute files, saving the
    resulting folder to `temp_dir/{this database}/{env_id}`. If the new
    directory matches the old, the new directory is deleted.
"""

import json
import re
from functools import cached_property
from pathlib import Path
from pprint import pprint  # TODO: Remove before merge
from shutil import rmtree as shutil_rmtree
from subprocess import run as sub_run
from typing import Tuple

import datajoint as dj
from numpy import __version__ as np_version
from packaging.version import parse as version_parse
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version

from spyglass.common.common_user import UserEnvironment
from spyglass.settings import recording_dir, temp_dir
from spyglass.spikesorting import USER_TBL
from spyglass.spikesorting.v0.spikesorting_recording import (
    SpikeSortingRecording,
)  # noqa F401
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.h5_helper_fn import H5pyComparator, sort_dict
from spyglass.utils.nwb_hash import DirectoryHasher

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

    @cached_property
    def this_env(self) -> dj.expression.QueryExpression:
        """Return restricted version of self for the current environment."""
        return (
            self & self._package_restr("spikeinterface", si_version)
        ) & self._package_restr("probeinterface", pi_version)

    def _has_matching_env(self, key: dict, show_err=True) -> bool:
        """Check current env for matching pynwb versions."""
        key_pk = self.dict_to_pk(key)
        if not self & key:
            self.make(key)
        ret = bool(self.this_env & key_pk)
        if not ret and show_err:
            need = sort_dict(
                (self & key).fetch1(
                    "spikeinterface", "probeinterface", as_dict=True
                )
            )
            have = sort_dict(
                dict(spikeinterface=si_version, probeinterface=pi_version)
            )
            logger.error(f"Versions mismatch: {need} != {have}")

        return ret

    def _package_restr(self, package, version):
        """Return a restriction string for a package and version.

        Restricts to versions matching the most recent official release
        (e.g., 0.1.0dev1 and 0.1.0 -> 0.1.0). For dev releases, also includes
        the next official release (e.g., 0.1.0dev1 -> 0.1.1). For official
        releases, also includes the previous release (e.g., 0.1.0 -> 0.9.3).
        """
        base = re.match(r"(\d+\.\d+\.\d+)", version)
        if not base:
            return f"{package} LIKE '{version}%'"
        base = base.group(1)
        avail = self._fetch_versions(package)
        if base not in avail:
            raise ValueError(f"Version {version} not found in {package}")

        if "dev" in version:  # if dev release, look at next
            next_index = min(avail.index(base) + 1, len(avail) - 1)
        else:  # if official release, look at prev dev release
            next_index = max(avail.index(base) - 1, 0)

        next_avail = avail[next_index]

        return f"{package} like '{base}%' OR {package} like '{next_avail}%'"

    # --- Inventory versions ---

    def _extract_version(self, obj):
        """Extract version numbers from a nested dictionary.

        Developed to be recursive to handle provenance.json files, which could
        theoretically contain different versions for different processing steps.
        """
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

        return versions

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
        si_version.discard(None)
        n_versions = len(si_version)
        if n_versions != 1:
            raise ValueError(f"{n_versions} si versions found: {si_version}")

        with open(rec_path / "probe.json", "r") as file:
            pi_version = json.load(file).get("version", None)

        self.insert1(
            dict(
                key,
                probeinterface=pi_version,
                spikeinterface=next(iter(si_version)),
            ),
            allow_direct_insert=True,
        )


# Initializing table to prevent recompute of cached properties
REC_VER_TBL = RecordingRecomputeVersions()


@schema
class RecordingRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingRecomputeVersions
    -> UserEnvironment
    ---
    logged_at_creation=0: bool
    """

    def insert(self, rows, at_creation=False, **kwargs):
        """Custom insert to ensure dependencies are added to each row."""
        if not rows:
            return
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        inserts = []
        for row in rows:
            key_pk = self.dict_to_pk(row)
            if not REC_VER_TBL._has_matching_env(key_pk):
                continue
            key_pk.update(USER_TBL.this_env)
            key_pk.setdefault("logged_at_creation", at_creation)
            inserts.append(key_pk)
        super().insert(inserts, **kwargs)

    def attempt_all(
        self, restr: dict = True, limit: int = None, **kwargs
    ) -> None:
        """Insert all files into the recompute table.

        Parameters
        ----------
        restr : dict, optional
            Key or restriction of SpikeSortingRecording. If None, all files
            are inserted.
        limit : int, optional
            Maximum number of rows to insert, randomly selected. For
            retrospective recompute attempts, randomly selecting potential
            recompute attaempts can be useful for trying a diverse set of
            files.
        """
        source = REC_VER_TBL.this_env & restr
        kwargs["skip_duplicates"] = True

        if limit:
            source &= dj.condition.Top(limit=limit, order_by="RAND()")

        inserts = [
            {
                **key,
                "logged_at_creation": False,
            }
            for key in source.fetch("KEY", as_dict=True)
        ]
        self.insert(inserts, at_creation=False, **kwargs)

    # --- Gatekeep recompute attempts ---

    @cached_property
    def this_env(self) -> dj.expression.QueryExpression:
        """Restricted table matching pynwb env and pip env."""
        return self & USER_TBL.this_env


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

        def compare(self, key=None):
            """Compare embedded objects as available.

            If key is a list or tuple, compare all entries. If no key, compare
            all entries in self (likely restricted). If key specifies a name,
            use key to find paths to files and compare the objects.
            """
            if isinstance(key, (list, tuple)) or not key:
                comps = []
                for row in key or self:
                    comps.append(self.compare(row))
                return comps

            if not key.get("name"):
                self.compare(self & key)
                return

            old, new = RecordingRecompute()._get_paths(key)
            this_file = key.get("name")
            msg = f" {this_file}\n\t{old}\n\t{new}"
            if str(old).endswith("raw"):
                print("Cannot compare raw files" + msg)
                return
            print("Comparing" + msg)
            return H5pyComparator(old / this_file, new / this_file)

    def _parent_key(self, key):
        ret = (
            SpikeSortingRecording
            * RecordingRecomputeVersions
            * RecordingRecomputeSelection
            & key
        )
        if len(ret) != 1:
            raise ValueError(f"Query returned {len(ret)} entries: {ret}")
        return ret.fetch(as_dict=True)[0]

    def _hash_one(self, path):
        str_path = str(path)
        if str_path in self._hasher_cache:
            return self._hasher_cache[str_path]
        hasher = DirectoryHasher(directory_path=path, keep_obj_hash=True)
        self._hasher_cache[str_path] = hasher
        return hasher

    def _get_temp_dir(self, key):
        return Path(temp_dir) / self.database / key["env_id"]

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
        if not REC_VER_TBL._has_matching_env(key):
            logger.info(
                f"Skipping due to pip mismatch, {key['env_id']}: {log_key}"
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
        """Delete recompute attempts when deleting rows."""
        attempt_dirs = []
        for key in self:
            _, new = self._get_paths(key)
            attempt_dirs.append(new)

        msg = "Delete attempt files?\n\t" + "\n\t".join(attempt_dirs)
        if dj.utils.user_choice(msg).lower() == "yes":
            for dir in attempt_dirs:
                shutil_rmtree(dir, ignore_errors=True)

        super().delete(*args, **kwargs)
