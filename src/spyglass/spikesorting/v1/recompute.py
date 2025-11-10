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
from typing import Optional, Tuple, Union

import datajoint as dj
import pynwb
from datajoint.hash import key_hash
from h5py import File as h5py_File
from hdmf.build import TypeMap
from tqdm import tqdm

from spyglass.common import AnalysisNwbfile
from spyglass.common.common_user import UserEnvironment  # noqa: F401
from spyglass.settings import analysis_dir, temp_dir
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import bytes_to_human_readable
from spyglass.utils.h5_helper_fn import H5pyComparator, sort_dict
from spyglass.utils.nwb_hash import NwbfileHasher, get_file_namespaces

schema = dj.schema("spikesorting_v1_recompute")


@schema
class RecordingRecomputeVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    nwb_deps=null:blob
    """

    # expected nwb_deps: core, hdmf_common, hdmf_experimental, spyglass
    #                    ndx_franklab_novela, ndx_optogenetics, ndx_pose

    @cached_property
    def nwb_deps(self):
        """Return a restriction of self for the current environment."""
        return sort_dict(self.namespace_dict(pynwb.get_manager().type_map))

    @cached_property
    def this_env(self) -> dj.expression.QueryExpression:
        """Return restricted version of self for the current environment.

        Ignores the spyglass version.
        """
        restr = []
        for key in self:
            key_deps = key["nwb_deps"]
            _ = key_deps.pop("spyglass", None)
            if sort_dict(key_deps) != self.nwb_deps:  # comment out to debug
                continue
            restr.append(self.dict_to_pk(key))
        return self & restr

    def _has_key(self, key: dict) -> bool:
        """Attempt make, return status"""
        if not SpikeSortingRecording & key:
            logger.warning(
                f"Attempt to populate Recompute before Recording: {key}"
            )
        if not self & key:
            self.make(key)
        return bool(self & key)

    def _has_matching_env(self, key: dict, show_err=False) -> bool:
        """Check current env for matching pynwb versions."""
        if not self._has_key(key):
            return False  # # pragma: no cover

        need = sort_dict(self.key_env(key))
        ret = self.nwb_deps == need

        if not ret and show_err:
            logger.warning(  # pragma: no cover
                f"PyNWB version mismatch. Skipping key: {self.dict_to_pk(key)}"
                + f"\n\tHave: {self.nwb_deps}"
                + f"\n\tNeed: {need}"
            )
        return bool(ret)

    def key_env(self, key):
        """Return the pynwb environment for a given key."""

        if not self & key:
            self.make(key)
        query = self & key
        if len(query) == 0:
            return None
        if len(query) != 1:
            raise ValueError(f"Key matches {len(query)} entries: {query}")
        this_env = query.fetch("nwb_deps", as_dict=True)[0]["nwb_deps"]
        _ = this_env.pop("spyglass", None)  # ignore spyglass version
        return this_env

    def namespace_dict(self, type_map: TypeMap):
        """Return a dictionary of namespaces and their versions."""
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
        try:
            path = AnalysisNwbfile().get_abs_path(parent["analysis_file_name"])
        except (FileNotFoundError, dj.DataJointError) as e:
            logger.warning(  # pragma: no cover
                f"Issue w/{parent['analysis_file_name']}. Skipping.\n{e}"
            )
            return  # pragma: no cover

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
    def env_dict(self):
        logger.info("Initializing UserEnvironment")
        return UserEnvironment().insert_current_env()

    def insert(
        self, rows, limit=None, at_creation=False, force_attempt=False, **kwargs
    ) -> None:
        """Custom insert to ensure dependencies are added to each row.

        Parameters
        ----------
        rows : list or dict
            Row(s) to insert.
        limit : int, optional
            Maximum number of rows to insert. Default is None, inserts all.
        at_creation : bool, optional
            Whether the rows are logged at creation. Default is False.
        force_attempt : bool, optional
            Whether to force an attempt to insert rows even if the environment
            does not match. Default is False.
        """

        if not self.env_dict.get("env_id"):  # likely not using conda
            logger.warning("Cannot log for recompute without UserEnvironment.")
            return

        if not rows:
            logger.info("No rows to insert.")
            return
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        editable_env = (UserEnvironment & self.env_dict).fetch1("has_editable")
        if at_creation and editable_env:  # assume fail if editable env
            at_creation = False  # # pragma: no cover

        inserts = []
        for row in rows:
            key_pk = self.dict_to_pk(row)
            if not force_attempt and not REC_VER_TBL._has_matching_env(key_pk):
                continue
            full_key = self.dict_to_full_key(row)
            full_key.update(dict(self.env_dict, logged_at_creation=at_creation))
            inserts.append(full_key)

        if not len(inserts):
            return

        super().insert(inserts, **kwargs)

    def attempt_all(
        self,
        restr: Optional[dict] = True,
        rounding: Optional[int] = None,
        limit: Optional[int] = None,
        force_attempt: bool = False,
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
            Maximum number of rows to insert, randomly selected. For
            retrospective recompute attempts, randomly selecting potential
            recompute attaempts can be useful for trying a diverse set of
            files.
        force_attempt : bool, optional
            Whether to force an attempt to insert rows even if the environment
            does not match. Default is False.
        """
        upstream = REC_VER_TBL if force_attempt else REC_VER_TBL.this_env
        source = upstream & restr
        kwargs["skip_duplicates"] = True

        if limit:
            source &= dj.condition.Top(limit=limit, order_by="RAND()")

        inserts = [
            {
                **key,
                **self.env_dict,
                "rounding": rounding or self.default_rounding,
            }
            for key in source.fetch("KEY", as_dict=True)
            if len(RecordingRecompute & key) == 0
        ]
        if not inserts:
            logger.info(f"No rows to insert from:\n\t{source}")
            return

        logger.info(f"Inserting recompute attempts for {len(inserts)} files.")

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
            old, new = RecordingRecompute()._get_paths(key)
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
        if not new.exists():
            logger.warning(f"New file does not exist: {new.name}")
            return None, None  # pragma: no cover

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
        allow_ins = dict(allow_direct_insert=True)

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
                self.insert1(  # pragma: no cover
                    dict(key, matched=False, err_msg=e_info), **allow_ins
                )
            else:  # unexpected ValueError
                raise  # pragma: no cover
        except KeyError as err:
            e_info = err.args[0]  # pragma: no cover
            if "H5 object missing" in e_info:  # failed bc missing parent obj
                e = e_info.split(", ")[1].split(":")[0].strip()
                self.insert1(
                    dict(key, matched=False, err_msg=e_info),
                    **allow_ins,
                )
                self.Name().insert1(
                    dict(key, name=f"Parent missing {e}", missing_from="old"),
                    **allow_ins,
                )
            elif "ndx-" in e_info:  # failed bc missing pynwb extension
                raise ModuleNotFoundError(  # pragma: no cover
                    "Please install the missing pynwb extension:\n\t" + e_info
                )
            else:
                raise
        except TypeError as err:
            e_info = err.args[0]  # pragma: no cover
            if "unexpected keyword" in e_info:
                self.insert1(  # pragma: no cover
                    dict(key, matched=False, err_msg=e_info), **allow_ins
                )
            else:
                logger.warning(f"TypeError: {err}: {new.name}")
        else:
            return new_vals
        return dict(hash=None)  # pragma: no cover

    def _hash_both(self, key) -> Tuple[NwbfileHasher, NwbfileHasher]:
        """Compare old and new files for a given key."""
        old, new = self._get_paths(key)
        new_hasher = (
            self._hash_one(new, key.get("rounding"))
            if new.exists()
            else self._recompute(key)["hash"]
        )
        if new_hasher is None:  # Error occurred during recompute_file_name
            return None, None
        old_hasher = self._hash_one(old, key.get("rounding"))
        if new_hasher.hash == old_hasher.hash and not self._other_roundings(
            key, operator="!="
        ):
            new.unlink(missing_ok=True)
        return old_hasher, new_hasher

    def recheck(self, key) -> None:
        """Recheck a previous recompute attempt."""
        old_hasher, new_hasher = self._hash_both(key)

        new_path = (
            new_hasher.path.name if new_hasher else self._get_paths(key)[1]
        )
        if new_hasher is None:  # Error occurred during recompute_file_name
            logger.error(f"V1 Recheck failed: {new_path}")
            return None

        if new_hasher.hash == old_hasher.hash:
            return True

        logger.error(f"V1 Recheck mismatch: {new_path}")
        return False

    def make(self, key, force_check=False) -> None:
        """Attempt to recompute an analysis file and compare to the original."""
        rec_dict = dict(recording_id=key["recording_id"])
        if self & rec_dict & "matched=1":
            return

        parent = self.get_parent_key(key)

        # Skip recompute for files logged at creation
        if parent["logged_at_creation"]:
            self.insert1(dict(key, matched=True))

        # Ensure not duplicate work for lesser precision
        if self._is_lower_rounding(key) and not force_check:
            logger.warning(
                f"Match at higher precision. Assuming match for {key}\n\t"
                + "Run with force_check=True to recompute."
            )

        old_hasher, new_hasher = self._hash_both(key)

        if new_hasher is None:  # Error occurred during recompute
            return

        if new_hasher.hash == old_hasher.hash:
            self.insert1(dict(key, matched=True))
            return

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

    def get_disk_space(self, which="new", restr: dict = None) -> Path:
        """Return the file(s) disk space for a given key or restriction.

        Parameters
        ----------
        which : str
            Which file to check disk space for, 'old' or 'new'. Default 'new'.
        restr : dict, optional
            Restriction for RecordingRecompute. Default is "matched=0".
        """
        restr = restr or "matched=0"
        total_size = 0
        for key in tqdm(self & restr, desc="Calculating disk space"):
            old, new = self._get_paths(key)
            this = old if which == "old" else new
            if this.exists():
                total_size += this.stat().st_size
        return f"Total: {bytes_to_human_readable(total_size)}"

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

        msg = "Delete attempt files?\n\t" + "\n\t".join(
            str(p) for p in attempt_paths
        )
        if dj.utils.user_choice(msg).lower() == "yes":
            for path in attempt_paths:
                path.unlink(missing_ok=True)
            kwargs["safemode"] = False  # pragma: no cover
            super().delete(*args, **kwargs)
