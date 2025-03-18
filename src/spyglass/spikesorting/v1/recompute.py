"""This schema is used to track recompute capabilities for existing files.

Tables
------
RecordingRecomputeVersions: What versions are present in an existing analysis
    file? Allows restrict of recompute attempts to pynwb environments that are
    compatible with a pre-existing file.
RecordingRecomputeSelection: Plan a recompute attempt. Capture a list of
    pip dependencies under an attempt label, 'env_id', and set the desired
    level of precision for the recompute (i.e., rounding for ElectricalSeries
    data).
RecordingRecompute: Attempt to recompute an analysis file, saving a new file
    to a temporary directory. If the new file matches the old, the new file is
    deleted. If the new file does not match, the differences are logged in
    the Hash table.
"""

import atexit
from functools import cached_property
from pathlib import Path
from typing import Tuple, Union

import datajoint as dj
import pynwb
from datajoint.hash import key_hash
from h5py import File as h5py_File
from hdmf import __version__ as hdmf_version
from hdmf.build import TypeMap
from numpy import __version__ as np_version
from spikeinterface import __version__ as si_version

from spyglass.common import AnalysisNwbfile
from spyglass.common.common_user import UserEnvironment
from spyglass.settings import analysis_dir, temp_dir
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.h5_helper_fn import H5pyComparator, sort_dict
from spyglass.utils.nwb_hash import NwbfileHasher, get_file_namespaces

schema = dj.schema("cbroz_recomp_v1")  # TODO: spikesorting_v1_recompute

USER_TBL = UserEnvironment()


@schema
class RecordingRecomputeVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    nwb_deps=null:blob
    """

    # expect
    #  - core
    #  - hdmf_common
    #  - hdmf_experimental
    #  - ndx_franklab_novela
    #  - ndx_optogenetics
    #  - spyglass

    @cached_property
    def nwb_deps(self):
        """Return a restriction of self for the current environment."""
        return self.namespace_dict(pynwb.get_manager().type_map)

    @cached_property
    def this_env(self):
        """Return restricted version of self for the current environment."""
        logger.info("Checking current nwb environment.")
        restr = []
        for key in self:
            key_deps = key["nwb_deps"]
            _ = key_deps.pop("spyglass", None)
            if key_deps != self.nwb_deps:
                continue
            restr.append(self.dict_to_pk(key))
        return self & restr

    def _has_matching_env(self, key: dict, show_err=True) -> bool:
        """Check current env for matching pynwb versions."""
        if not self & key:
            self.make(key)
        ret = self.this_env & key

        if not ret and show_err:
            need = sort_dict(self.key_env(key))
            have = sort_dict(self.nwb_deps)
            key_pk = self.dict_to_pk(key)
            logger.warning(
                f"PyNWB version mismatch. Skipping key: {self.dict_to_pk(key)}"
                + f"\n\tHave: {have}"
                + f"\n\tNeed: {need}"
            )
        return bool(ret)

    def key_env(self, key):
        """Return the pynwb environment for a given key."""
        if not self & key:
            self.make(key)
        query = self & key
        if len(query) != 1:
            raise ValueError(f"Key matches {len(query)} entries: {query}")
        this_env = query.fetch("nwb_deps", as_dict=True)[0]["nwb_deps"]
        _ = this_env.pop("spyglass", None)  # ignore spyglass version
        return this_env

    def namespace_dict(self, type_map: TypeMap):
        """Remap namespace names to hyphenated field names for DJ compatibility."""
        name_cat = type_map.namespace_catalog
        return {
            field: name_cat.get_namespace(field).get("version", None)
            for field in name_cat.namespaces
        }

    def make(self, key):
        """Inventory the namespaces present in an analysis file."""
        query = SpikeSortingRecording() & key
        if not len(query) == 1:
            raise ValueError(
                f"SpikeSortingRecording & {key} has {len(query)} "
                + f"matching entries: {query}"
            )

        parent = query.fetch1()
        path = AnalysisNwbfile().get_abs_path(parent["analysis_file_name"])

        nwb_deps = get_file_namespaces(path).copy()

        with h5py_File(path, "r") as f:
            script = f.get("general/source_script")
            if script is not None:  # after `=`, remove quotes
                script = str(script[()]).split("=")[1].strip().replace("'", "")
            nwb_deps["spyglass"] = script

        self.insert1(dict(key, nwb_deps=nwb_deps), allow_direct_insert=True)


# Initializing table to prevent recompute of cached properties
REC_VER_TBL = RecordingRecomputeVersions()


@schema
class RecordingRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingRecomputeVersions
    -> UserEnvironment
    rounding=4: int # rounding for float ElectricalSeries
    ---
    logged_at_creation=0: bool # whether the attempt was logged at creation
    """

    # --- Insert helpers ---
    @cached_property
    def default_rounding(self) -> int:
        """Return the default rounding for ElectricalSeries data."""
        return int(self.heading.attributes["rounding"].default)

    @cached_property
    def pip_deps(self) -> dict:
        """Return the pip dependencies for the current environment."""
        return dict(
            pynwb=pynwb.__version__,
            hdmf=hdmf_version,
            spikeinterface=si_version,
            numpy=np_version,
        )

    @cached_property
    def env_dict(self):
        logger.info("Inserting current environment.")
        return USER_TBL.insert_current_env()

    def insert(self, rows, limit=None, at_creation=False, **kwargs) -> None:
        """Custom insert to ensure dependencies are added to each row."""
        if not rows:
            logger.info("No rows to insert.")
            return
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        logger.info(f"Inserting {len(rows)} recompute attempts.")

        inserts = []
        for row in rows:
            row_pk = self.dict_to_pk(row)
            if not REC_VER_TBL._has_matching_env(row_pk):
                continue
            full_key = self.dict_to_full_key(row)
            full_key.update(dict(self.env_dict, logged_at_creation=at_creation))
            inserts.append(full_key)

        logger.info(f"Inserting {len(inserts)} valid recompute attempts.")

        super().insert(inserts, **kwargs)

    def attempt_all(
        self,
        restr: dict = True,
        rounding: int = None,
        limit: int = None,
        **kwargs,
    ) -> None:
        """Insert recompute attempts for all existing files.

        Parameters
        ----------
        restr : dict
            Key or restriction for RecordingRecomputeVersions. Default all
            available files.
        rounding : int, optional
            Rounding for float ElectricalSeries data. Default is the table's
            default_rounding, 4.
        limit : int, optional
            Maximum number of rows to insert. Randomly selected from available.
        """
        source = REC_VER_TBL.this_env & restr
        kwargs["skip_duplicates"] = True

        if limit:
            source &= dj.condition.Top(limit=limit, order_by="RAND()")

        logger.info(f"Inserting recompute attempts for {len(source)} files.")

        inserts = [
            {
                **key,
                **self.env_dict,
                "rounding": rounding or self.default_rounding,
            }
            for key in source.fetch("KEY", as_dict=True)
        ]
        self.insert(inserts, at_creation=False, **kwargs)

    # --- Gatekeep recompute attempts ---

    @cached_property
    def this_env(self) -> dj.expression.QueryExpression:
        """Restricted table matching pynwb env and pip env."""
        return self & self.env_dict

    def _has_matching_env(self, key) -> bool:
        """Check current env for matching pynwb and pip versions."""
        return REC_VER_TBL._has_matching_env(key) and bool(self.this_env & key)


@schema
class RecordingRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingRecomputeSelection
    ---
    matched: bool
    err_msg=null: varchar(255)
    """

    class Name(dj.Part):
        definition = """ # Object names missing from old or new versions
        -> master
        name : varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(dj.Part):
        definition = """ # Object hashes that differ between old and new
        -> master
        name : varchar(255)
        """

        def get_objs(self, key, name=None):
            old, new = (self & key).fetch1("old", "new")
            if old is not None and new is not None:
                return old, new
            old, new = RecordingRecompute()._open_files(key)
            this_obj = name or key["name"]
            return old.get(this_obj, None), new.get(this_obj, None)

        def compare(self, key=None, name=None):
            if isinstance(key, (list, tuple)) or not key:
                comps = []
                for row in key or self:
                    comps.append(self.compare(row))
                return comps

            if not key.get("name"):
                self.compare(self & key)
                return

            return H5pyComparator(*self.get_objs(key, name=name))

    _key_cache = dict()
    _hasher_cache = dict()
    _files_cache = dict()
    _cleanup_registered = False

    @property
    def with_names(self) -> dj.expression.QueryExpression:
        """Return tables joined with analysis file names."""
        return self * SpikeSortingRecording.proj("analysis_file_name")

    # --- Cache management ---

    def _cleanup(self) -> None:
        """Close all open files."""
        for file in self._file_cache.values():
            file.close()
        self._file_cache = {}
        for hasher in self._hasher_cache.values():
            hasher.cleanup()
        if self._cleanup_registered:
            atexit.unregister(self._cleanup)
            self._cleanup_registered = False

    def _open_files(self, key) -> Tuple[h5py_File, h5py_File]:
        """Open old and new files for comparison."""
        if not self._cleanup_registered:
            atexit.register(self._cleanup)
            self._cleanup_registered = True

        old, new = self._get_paths(key, as_str=True)
        if old not in self._file_cache:
            self._file_cache[old] = h5py_File(old, "r")
        if new not in self._file_cache:
            self._file_cache[new] = h5py_File(new, "r")

        return self._file_cache[old], self._file_cache[new]

    def _hash_one(self, path, precision) -> NwbfileHasher:
        """Return the hasher for a given path. Store in cache."""
        cache_val = f"{path}_{precision}"
        if cache_val in self._hasher_cache:
            return self._hasher_cache[cache_val]
        hasher = NwbfileHasher(
            path,
            verbose=False,
            keep_obj_hash=True,
            keep_file_open=True,
            precision_lookup=precision,
        )
        self._hasher_cache[cache_val] = hasher
        return hasher

    # --- Path management ---

    def _get_paths(self, key, as_str=False) -> Tuple[Path, Path]:
        """Return the old and new file paths."""
        key = self.get_parent_key(key)

        def get_subdir(key) -> Path:
            """Return the analysis file's subdirectory."""
            file = key["analysis_file_name"] if isinstance(key, dict) else key
            parts = file.split("_")
            subdir = "_".join(parts[:-1])
            return subdir + "/" + file

        old = Path(analysis_dir) / get_subdir(key)
        new = (
            Path(temp_dir)
            / self.database
            / key.get("env_id", "")
            / get_subdir(key)
        )

        return (str(old), str(new)) if as_str else (old, new)

    # --- Database checks ---

    def get_parent_key(self, key) -> dict:
        """Return the parent key for a given recompute key."""
        key = self.dict_to_pk(key)
        hashed = key_hash(key)
        if hashed in self._key_cache:
            return self._key_cache[hashed]
        parent = (
            SpikeSortingRecording
            * RecordingRecomputeVersions
            * RecordingRecomputeSelection
            & key
        ).fetch1()
        self._key_cache[hashed] = parent
        return parent

    def _other_roundings(
        self, key, operator="<"
    ) -> dj.expression.QueryExpression:
        """Return other planned precision recompute attempts.

        Parameters
        ----------
        key : dict
            Key for the current recompute attempt.
        operator : str, optional
            Comparator for rounding field.
            Default 'less than', return attempts with lower precision than key.
            Also accepts '!=' or '>'.
        """
        return (
            RecordingRecomputeSelection()
            & {k: v for k, v in key.items() if k != "rounding"}
            & f'rounding {operator} "{key["rounding"]}"'
        ).proj() - self

    def _is_lower_rounding(self, key) -> bool:
        """Check for lesser precision recompute attempts after match."""
        this_key = {k: v for k, v in key.items() if k != "rounding"}
        has_match = bool(self & this_key & "matched=1")
        return (
            False
            if not has_match  # Only if match, report True of lower precision
            else bool(self._other_roundings(key) & key)
        )

    # --- Recompute ---

    def _recompute(self, key) -> Union[None, dict]:
        """Attempt to recompute the analysis file. Catch common errors."""

        _, new = self._get_paths(key)
        parent = self.get_parent_key(key)
        default_rounding = RecordingRecomputeSelection().default_rounding

        try:
            new_vals = SpikeSortingRecording()._make_file(
                parent,
                recompute_file_name=parent["analysis_file_name"],
                save_to=new.parent.parent,
                rounding=key.get("rounding", default_rounding),
            )
        except RuntimeError as e:  # fail bc error in recompute, will retry
            logger.warning(f"{e}: {new.name}")
        except ValueError as e:
            e_info = e.args[0]
            if "probe info" in e_info:  # make failed bc missing probe info
                self.insert1(dict(key, matched=False, err_msg=e_info))
            else:  # unexpected ValueError, will retry
                logger.warning(f"ValueError: {e}: {new.name}")
        except KeyError as err:
            e_info = err.args[0]
            if "H5 object missing" in e_info:  # failed bc missing parent obj
                e = e_info.split(", ")[1].split(":")[0].strip()
                self.insert1(dict(key, matched=False, err_msg=e_info))
                self.Name().insert1(
                    dict(key, name=f"Parent missing {e}", missing_from="old")
                )
            else:
                logger.warning(f"KeyError: {err}: {new.name}")
        else:
            return new_vals

    def make(self, key, force_check=False) -> None:
        """Attempt to recompute an analysis file and compare to the original."""
        parent = self.get_parent_key(key)
        rounding = key.get("rounding")

        # Skip recompute for files logged at creation
        if parent["logged_at_creation"]:
            self.insert1(dict(key, matched=True))

        # Ensure dependencies unchanged since selection insert
        if not RecordingRecomputeSelection()._has_matching_env(key):
            logger.info(f"Skipping due to env mismatch: {key}")
            return

        # Ensure not duplicate work for lesser precision
        if self._is_lower_rounding(key) and not force_check:
            logger.warning(
                f"Match at higher precision. Assuming match for {key}\n\t"
                + "Run with force_check=True to recompute."
            )

        old, new = self._get_paths(parent)

        new_hasher = (
            self._hash_one(new, rounding)
            if new.exists()
            else self._recompute(key)["hash"]
        )

        if new_hasher is None:  # Error occurred during recompute
            return

        old_hasher = self._hash_one(old, rounding)

        if new_hasher.hash == old_hasher.hash:
            logger.info(f"Matched {new.name}")
            self.insert1(dict(key, matched=True))
            if not self._other_roundings(key, operator="!="):
                # if no other recompute attempts
                new.unlink(missing_ok=True)
            return

        logger.info(f"Comparing mismatched {new.name}")

        names, hashes = [], []
        for obj in set({**old_hasher.objs, **new_hasher.objs}):
            old_obj, old_hash = old_hasher.objs.get(obj, (None, None))
            new_obj, new_hash = new_hasher.objs.get(obj, (None, None))

            if old_hash is None:
                names.append(dict(key, name=obj, missing_from="old"))
            if new_hash is None:
                names.append(dict(key, name=obj, missing_from="new"))
            if old_hash != new_hash:
                hashes.append(dict(key, name=obj))

        self.insert1(dict(key, matched=False))
        self.Name().insert(names)
        self.Hash().insert(hashes)

    def delete_files(self, restriction=True, dry_run=True) -> None:
        """If successfully recomputed, delete files for a given restriction."""
        query = self.with_names & "matched=1" & restriction
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

    def delete(self, *args, **kwargs) -> None:
        """Delete recompute attempts when deleting rows."""
        attempt_paths = []
        for key in self:
            _, new = self._get_paths(key)
            attempt_paths.append(new)

        msg = "Delete attempt files?\n\t" + "\n\t".join(attempt_dirs)
        if dj.utils.user_choice(msg).lower() == "yes":
            for path in attempt_paths:
                path.unlink(missing_ok=True)

        super().delete(*args, **kwargs)
