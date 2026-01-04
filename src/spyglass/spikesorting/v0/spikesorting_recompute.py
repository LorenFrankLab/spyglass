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

XFAIL Patterns
--------------
Use check_xfail() to test entries against known expected failure patterns.
"""

import json
import re
from datetime import datetime
from functools import cached_property
from pathlib import Path
from shutil import rmtree as shutil_rmtree
from subprocess import run as sub_run
from typing import Any, List, Optional, Tuple, Union

import datajoint as dj
from packaging.version import parse as version_parse
from probeinterface import __version__ as pi_version
from spikeinterface import __version__ as si_version
from tqdm import tqdm

from spyglass.common.common_user import UserEnvironment  # noqa F401
from spyglass.settings import recording_dir, temp_dir
from spyglass.spikesorting.v0.spikesorting_recording import (  # noqa F401
    SpikeSortingPreprocessingParameters,
    SpikeSortingRecording,
)
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import bytes_to_human_readable
from spyglass.utils.h5_helper_fn import H5pyComparator, sort_dict
from spyglass.utils.nwb_hash import DirectoryHasher

VERBOSE = True
if logger.level > 20:
    VERBOSE = False


schema = dj.schema("spikesorting_recompute_v0")


def check_xfail(*args, **kwargs) -> Tuple[bool, Optional[str]]:
    return RecordingRecomputeSelection()._check_xfail(*args, **kwargs)


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

    def _fetch_versions(self, package: str) -> list:
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

    def _has_matching_env(
        self, key: dict, show_err: Optional[bool] = False
    ) -> bool:
        """Check current env for matching pynwb dependency versions."""
        key_pk = self.dict_to_pk(key)
        if not self & key:
            self.make(key)
        ret = bool(self.this_env & key_pk)
        if not ret and show_err:
            this_entry = (self & key_pk).fetch(  # pragma: no cover
                "spikeinterface", "probeinterface", as_dict=True
            )[0]
            need = sort_dict(this_entry)  # pragma: no cover
            have = sort_dict(  # pragma: no cover
                dict(spikeinterface=si_version, probeinterface=pi_version)
            )
            logger.error(f"Versions mismatch:\n\tNeed: {need}\n\tHave: {have}")

        return ret

    def _package_restr(self, package: str, version: str) -> str:
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

    def _extract_version(self, obj: Any) -> set:
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

    def make(self, key: dict) -> None:
        """Inventory the namespaces present in an analysis file."""
        query = SpikeSortingRecording() & key
        rec_path = Path(query.fetch1("recording_path"))

        missing_dir = not rec_path.exists() or not rec_path.is_dir()
        empty_dir = rec_path.is_dir() and not any(rec_path.iterdir())

        if missing_dir or empty_dir:
            logger.warning(f"Problem with recording path: {rec_path}")
            return  # pragma: no cover

        si_version = self._extract_version(
            [
                rec_path / f"{fname}.json"
                for fname in ["binary", "si_folder", "provenance"]
            ]
        )
        si_version.discard(None)
        n_versions = len(si_version)
        if n_versions != 1:
            logger.warning(f"{n_versions} si versions found: {si_version}")
            return  # pragma: no cover

        probe_path = rec_path / "probe.json"
        if not probe_path.exists():
            logger.warning(f"Probe file not found for {key['nwb_file_name']}")
            return  # pragma: no cover

        with open(rec_path / "probe.json", "r") as file:
            pi_version = json.load(file).get("version", None)

        self.insert1(
            dict(
                self.dict_to_pk(key),
                probeinterface=pi_version,
                spikeinterface=next(iter(si_version)),
            ),
            allow_direct_insert=True,
            skip_duplicates=True,
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
    xfail_reason=NULL   : varchar(127)
    """

    @cached_property
    def env_dict(self) -> dict:
        logger.info("Initializing UserEnvironment")
        return UserEnvironment().insert_current_env()

    def _check_xfail(
        self,
        key: dict,
        rec_path: Optional[Path] = None,
        skip_padlen: bool = True,
        skip_si094: bool = True,
        skip_low_sr: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """Check if entry matches known xfail (expected failure) patterns.

        Parameters
        ----------
        key : dict
            Recording key with nwb_file_name, sort_group_id, etc.
        recording_path : Path, optional
            Path to recording directory. If None, computed from key.
        skip_padlen : bool, optional
            Check for padlen errors (recordings <35 samples). Default True.
        skip_si094 : bool, optional
            Check for SI 0.94.x incompatibility. Default True.
        skip_low_sr : bool, optional
            Check for low sampling rate causing Wn[0] < Wn[1] errors. Default
            True.

        Returns
        -------
        is_xfail : bool
            True if entry matches any enabled xfail pattern
        reason : str or None
            Description of xfail pattern matched, or None
        """
        rec_tbl = SpikeSortingRecording()

        if rec_path is None:
            rec_name = rec_tbl._get_recording_name(key)
            rec_path = Path(recording_dir) / rec_name

        if not rec_path.exists():
            return True, "Recording path does not exist"

        required_files = [
            "si_folder.json",
            "binary.json",
            "provenance.json",
            "probe.json",
        ]
        for req_file in required_files:
            if not (rec_path / req_file).exists():
                err_msg = f"missing_required_file ({req_file})"
                return True, err_msg

        # Pattern 1: Padlen error (short recordings or short segments)
        if skip_padlen:
            # Check for short segments (early exit on first segment < 35)
            min_samples, seg_index = rec_tbl._get_min_segment_length(
                key, min_threshold=35
            )
            if min_samples is not None and min_samples < 35:
                err_msg = (
                    f"padlen_short_segment (segment {seg_index}: "
                    f"{min_samples} samples < 35)"
                )
                return True, err_msg

            # Fallback: also check total n_samples from si_folder.json
            n_samples = rec_tbl._get_n_samples(key=key, rec_path=rec_path)
            if n_samples is not None and n_samples < 35:
                err_msg = f"padlen_short_recording ({n_samples} samples)"
                return True, err_msg

        # Pattern 2: SI 0.94.x incompatibility
        if skip_si094:
            version = (REC_VER_TBL & key).fetch1("spikeinterface")
            if version.startswith("0.94."):
                err_msg = f"si094_incompatibility (SI {version})"
                return True, err_msg

        # Pattern 3: Low sampling rate (Wn[0] < Wn[1] error)
        if skip_low_sr:
            # Pass key to enable NWB fallback if si_folder.json unavailable
            samp_rate = rec_tbl._get_sampling_rate(key=key, rec_path=rec_path)

            # If sampling rate is None/invalid even after NWB fallback, mark xfail
            if samp_rate is None:
                err_msg = "invalid_sampling_rate (None or NaN from source)"
                return True, err_msg

            nyquist = samp_rate / 2
            freq_max = (
                SpikeSortingPreprocessingParameters().fetch_params(key)
            ).get("frequency_max", 0)

            if freq_max > 0 and freq_max >= nyquist:
                err_msg = (
                    f"low_sampling_rate (freq_max={freq_max} Hz >= "
                    f"Nyquist={nyquist:.1f} Hz)"
                )
                return True, err_msg

        # Add more xfail patterns here as discovered
        # Each should have a corresponding skip_* parameter

        return False, None

    def insert(
        self,
        rows: List[dict],
        at_creation: Optional[bool] = False,
        force_attempt: bool = False,
        skip_xfail: bool = True,
        skip_padlen: bool = True,
        skip_si094: bool = True,
        skip_low_sr: bool = True,
        **kwargs,
    ) -> None:
        """Custom insert to ensure dependencies are added to each row.

        Parameters
        ----------
        rows : list of dict
            Recording keys to insert
        at_creation : bool, optional
            Mark entries as logged at creation time. Default False.
        force_attempt : bool, optional
            Force insertion even if version mismatch. Default False.
        skip_xfail : bool, optional
            Skip entries matching known xfail patterns. Default True.
        skip_padlen : bool, optional
            Skip short recordings (<35 samples). Default True.
        skip_si094 : bool, optional
            Skip SI 0.94.x incompatible recordings. Default True.
        skip_low_sr : bool, optional
            Skip recordings with freq_max >= Nyquist frequency. Default True.
        **kwargs
            Additional arguments passed to DataJoint insert
        """
        if not rows:
            return
        if not isinstance(rows, (list, tuple)):
            rows = [rows]
        if not isinstance(rows[0], dict):
            raise ValueError("Rows must be a list of dicts")

        for row in rows:
            if not RecordingRecomputeVersions & row:
                RecordingRecomputeVersions().make(row)
        if not self.env_dict.get("env_id"):  # likely not using conda
            logger.warning("Cannot log for recompute without UserEnvironment.")
            return

        editable_env = (UserEnvironment & self.env_dict).fetch1("has_editable")
        if at_creation and editable_env:  # no assume pass if gen w/editable env
            at_creation = False  # pragma: no cover

        inserts = []
        xfail_kwargs = dict(
            skip_padlen=skip_padlen,
            skip_si094=skip_si094,
            skip_low_sr=skip_low_sr,
        )
        for row in tqdm(rows, desc="Preparing inserts"):
            no_env = {k: v for k, v in row.items() if k != "env_id"}
            if bool(RecordingRecompute & no_env & "matched=1"):
                logger.info(
                    f"Skipping already matched: {row.get('nwb_file_name', row)}"
                )
                continue

            key_pk = self.dict_to_pk(row)
            if not force_attempt and not REC_VER_TBL._has_matching_env(key_pk):
                continue

            # Check xfail patterns if enabled
            xfail_reason = None
            if skip_xfail:
                _, xfail_reason = check_xfail(key_pk, **xfail_kwargs)

            key_pk.update(dict(self.env_dict, xfail_reason=xfail_reason))
            key_pk.setdefault("logged_at_creation", at_creation)
            inserts.append(key_pk)

        super().insert(inserts, **kwargs)

        if not inserts:
            logger.warning("No rows inserted.")  # pragma: no cover

    def attempt_all(
        self,
        restr: Optional[dict] = True,
        limit: Optional[int] = None,
        **kwargs,
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
            if len(RecordingRecompute & key) == 0
        ]
        self.insert(inserts, at_creation=False, **kwargs)

    # --- Gatekeep recompute attempts ---

    @cached_property
    def this_env(self) -> dj.expression.QueryExpression:
        """Restricted table matching pynwb env and pip env."""
        return self & self.env_dict

    def remove_matched(
        self,
        restriction: Optional[Union[str, dict]] = True,
        dry_run: bool = True,
    ) -> int:
        """Remove selection entries for files already successfully matched.

        This method cleans up redundant entries in RecordingRecomputeSelection
        for files that have already been successfully matched in
        RecordingRecompute (potentially in a different environment).

        Parameters
        ----------
        restriction : bool, str, dict, optional
            Additional restriction to apply. Default True (all entries).
        dry_run : bool, optional
            If True, only show what would be deleted without deleting. Default True.

        Returns
        -------
        int
            Number of entries that were (or would be) deleted.

        Example
        -------
        >>> # Remove all redundant selection entries
        >>> RecordingRecomputeSelection().remove_matched(dry_run=False)
        """
        # Get all successfully matched entries
        matched_entries = RecordingRecompute & "matched=1"

        # Get primary keys excluding env_id
        pk_fields = [
            k for k in SpikeSortingRecording.primary_key if k != "env_id"
        ]

        # Get unique matched file keys
        matched_keys = (dj.U(*pk_fields) & matched_entries).fetch(
            "KEY", as_dict=True
        )

        # Find selection entries that match these files
        redundant = (self & restriction & matched_keys) - matched_entries.proj()
        count = len(redundant)

        if count == 0:
            logger.debug("No redundant matched entries")
            return 0

        prefix = "DRY RUN: " if dry_run else ""
        logger.info(
            f"{prefix}Found {count} selection entries for already-matched files"
        )

        if dry_run:
            # Show sample of what would be deleted
            sample = redundant.fetch("KEY", as_dict=True, limit=10)
            logger.info(f"{prefix}Sample entries (up to 10):")
            for i, key in enumerate(sample, 1):
                nwb_file = key.get("nwb_file_name", "unknown")
                env_id = key.get("env_id", "unknown")
                logger.info(f"  {i}. {nwb_file} (env: {env_id})")
            if count > 10:
                logger.info(f"  ... and {count - 10} more")
            return redundant

        # Actually delete the redundant entries
        redundant.delete_quick()
        logger.info(f"Deleted {count} redundant entries")

        return count


@schema
class RecordingRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingRecomputeSelection
    ---
    matched:bool
    err_msg=null: varchar(255)
    created_at=null: datetime # Timestamp when the original file was created
    deleted=0: bool # whether the old file has been deleted after a match
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

        def get_objs(
            self, key: dict, name: Optional[str] = None
        ) -> Tuple[str, str]:
            old, new = RecordingRecompute()._get_paths(key, as_str=True)

            old_hasher = RecordingRecompute()._hash_one(old)
            new_hasher = RecordingRecompute()._hash_one(new)

            name = name or key["name"]
            old_obj = old_hasher.cache.get(name)
            new_obj = new_hasher.cache.get(name)

            return old_obj, new_obj

        def compare(
            self, key: Optional[Union[dict, list, tuple]] = None
        ) -> Optional[H5pyComparator]:
            """Compare embedded objects as available.

            If key is a list or tuple, compare all entries. If no key, compare
            all entries in self (likely restricted). If key specifies a name,
            use key to find paths to files and compare the objects.
            """
            if isinstance(key, (list, tuple)) or not key:
                rows = key or self
                return [self.compare(row) for row in rows]

            if not key.get("name"):
                self.compare(key=(self & key).fetch(as_dict=True))
                return

            old, new = RecordingRecompute()._get_paths(key)

            this_file = key.get("name")

            if this_file.endswith("raw"):
                print(f"Cannot compare raw files: {this_file}")
                return
            print(f"Comparing {this_file}")
            return H5pyComparator(old / this_file, new / this_file)

    def _parent_key(self, key: dict) -> dict:
        ret = (
            SpikeSortingRecording
            * RecordingRecomputeVersions
            * RecordingRecomputeSelection
            & key
        )
        if len(ret) != 1:
            raise ValueError(f"Query returned {len(ret)} entries: {ret}")
        return ret.fetch(as_dict=True)[0]

    def _hash_one(self, path: Union[str, Path]) -> DirectoryHasher:
        str_path = str(path)
        if str_path in self._hasher_cache:
            return self._hasher_cache[str_path]
        hasher = DirectoryHasher(directory_path=path, keep_obj_hash=True)
        self._hasher_cache[str_path] = hasher
        return hasher

    def _get_temp_dir(self, key: dict) -> Path:
        return Path(temp_dir) / self.database / key["env_id"]

    def _get_paths(
        self, key: dict, as_str: Optional[bool] = False
    ) -> Tuple[Path, Path]:
        """Return the old and new file paths."""
        rec_name = SpikeSortingRecording()._get_recording_name(key)

        old = Path(recording_dir) / rec_name
        new = self._get_temp_dir(key) / rec_name

        return (str(old), str(new)) if as_str else (old, new)

    def _get_file_created_at(self, key: dict):
        """Get directory creation timestamp from filesystem.

        Defaults to current time if file not found.

        Parameters
        ----------
        key : dict
            Primary key for the recording.
        """

        old, _ = self._get_paths(key)

        if not old.exists():
            return datetime.now()
        return datetime.fromtimestamp(old.stat().st_mtime)

    def _hash_both(
        self, key: dict, strict: bool = False
    ) -> Union[Tuple[DirectoryHasher, DirectoryHasher], Tuple[None, str]]:
        """Return the old and new file hashers."""
        old, new = self._get_paths(key)

        try:
            new_hasher = (
                self._hash_one(new)
                if new.exists()
                else SpikeSortingRecording()._make_file(
                    key, base_dir=self._get_temp_dir(key), return_hasher=True
                )["hash"]
            )
        except ValueError as err:  # pragma: no cover
            # Some spikeinterface version can't handle small batches
            e_info = err.args[0]  # pragma: no cover
            if not strict and "greater than padlen" in e_info:
                logger.error(f"Failed to recompute {new.name}: {e_info}")
                return None, e_info[:255]  # pragma: no cover
            raise  # pragma: no cover

        if not new_hasher.hash:
            return None, None

        old_hasher = self._hash_one(old)

        if (
            old_hasher.hash == new_hasher.hash
            and "tmp" in str(new)
            and new.exists()
        ):
            shutil_rmtree(new, ignore_errors=True)

        return old_hasher, new_hasher

    def recheck(self, key: dict) -> bool:
        """Recheck if a given key matches without recomputing."""
        old_hasher, new_hasher = self._hash_both(key, strict=True)

        new_path = self._get_paths(key, as_str=True)[1]

        if not new_hasher.hash:
            logger.error(f"V0 Recheck failed: {new_path}")
            return None

        if old_hasher.hash == new_hasher.hash:
            new_path = new_hasher.dir_path
            if "tmp" in str(new_path) and new_path.exists():
                shutil_rmtree(new_path, ignore_errors=True)
            return True

        logger.error(f"V0 Recheck mismat: {new_path}")
        return False

    def make(self, key: dict, force_check=False) -> None:
        rec_key = {k: v for k, v in key.items() if k != "env_id"}
        if not force_check and (self & rec_key & "matched=1"):
            logger.info(f"Already matched {rec_key['nwb_file_name']}")
            RecordingRecomputeSelection().remove_matched(rec_key, dry_run=False)
            return  # pragma: no cover

        # Skip recompute for files logged at creation
        parent = self._parent_key(key)
        log_key = key.get("nwb_file_name", key)
        created_key = dict(created_at=self._get_file_created_at(key))
        if parent.get("logged_at_creation", True):
            logger.info(f"Skipping logged_at_creation {log_key}")
            self.insert1({**key, "matched": True, **created_key})
            return

        # Skip recompute for known xfail patterns
        if parent.get("xfail_reason", None):
            logger.info(f"Skipping xfail {log_key}: {parent['xfail_reason']}")
            self.insert1(
                {
                    **key,
                    "matched": False,
                    "err_msg": f"xfail: {parent['xfail_reason']}",
                    **created_key,
                }
            )
            return

        old_hasher, new_hasher = self._hash_both(key, strict=True)

        if isinstance(new_hasher, str):  # pragma: no cover
            logger.error(f"Recompute error for {log_key}: {new_hasher}")
            self.insert1(  # if `_hash_both` returned an error message
                {**key, "matched": False, "err_msg": new_hasher, **created_key}
            )
            return  # pragma: no cover

        if old_hasher.hash == new_hasher.hash:
            logger.info(f"V0 Matched {log_key}")
            self.insert1({**key, "matched": True, **created_key})
            return

        # only show file name if available
        logger.info(f"V0: Failed to match {log_key}")

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

        self.insert1(dict(key, matched=False, **created_key))
        self.Name().insert(names)
        self.Hash().insert(hashes)

    def get_disk_space(self, which="new", restr: dict = None) -> Path:
        """Return the new file(s) disk space for a given key or restriction.

        Parameters
        ----------
        which : str
            Which file to check disk space for, 'old' or 'new'. Default 'new'.
        restr : dict, optional
            Restriction for RecordingRecompute. Default is "matched=0".
        """
        restr = restr or "matched=0"
        total_size = 0
        for key in tqdm(
            self & restr, desc="Calculating disk space", disable=not VERBOSE
        ):
            old, new = self._get_paths(key)
            this = old if which == "old" else new
            if this.exists():
                total_size += sum(
                    f.stat().st_size for f in this.glob("**/*") if f.is_file()
                )
        return f"Total: {bytes_to_human_readable(total_size)}"

    def delete_files(
        self,
        restriction: Optional[Union[str, dict]] = True,
        dry_run: Optional[bool] = True,
        days_since_creation: int = 7,
    ) -> None:
        """Delete old files for successfully recomputed entries.

        Parameters
        ----------
        restriction : bool, str, dict, optional
            Restriction to apply to matched entries. Default True (all matched).
        dry_run : bool, optional
            If True, only show what would be deleted. Default True.
        days_since_creation : int, optional
            Skip files created within this many days. Default 7.
        """
        # Apply base restrictions
        query = self & "matched=1 AND deleted=0" & restriction

        # Skip recently created files
        if days_since_creation > 0:
            date_temp = "created_at < DATE_SUB(CURDATE(), INTERVAL {} DAY)"
            query = query & date_temp.format(days_since_creation)
            logger.info(
                f"Excluding files created within {days_since_creation} days"
            )

        file_names = query.fetch("nwb_file_name")
        prefix = "DRY RUN: " if dry_run else ""
        msg = f"{prefix}Delete {len(file_names)} files?\n\t" + "\n\t".join(
            file_names
        )

        if dry_run:
            total_size = 0
            for key in tqdm(query, total=len(query), desc="Calculating size"):
                old, _ = self._get_paths(key)
                if not old.exists():
                    continue
                total_size += sum(
                    f.stat().st_size for f in old.glob("**/*") if f.is_file()
                )

            total_human = bytes_to_human_readable(total_size)
            msg += f"\nTotal size: {total_human}"
            logger.info(msg)
            return total_human

        if dj.utils.user_choice(msg).lower() not in ["yes", "y"]:
            return

        for key in tqdm(query.proj(), total=len(query), desc="Deleting files"):
            old, new = self._get_paths(key)
            logger.info(f"Deleting old: {old}, new: {new}")
            shutil_rmtree(old, ignore_errors=True)
            shutil_rmtree(new, ignore_errors=True)

    def delete(self, *args, **kwargs) -> None:
        """Delete recompute attempts when deleting rows."""
        attempt_dirs = []
        for key in self:
            _, new = self._get_paths(key)
            attempt_dirs.append(new)

        if attempt_dirs:
            msg = "Delete attempt files?\n\t" + "\n\t".join(attempt_dirs)
            if dj.utils.user_choice(msg).lower() in ["yes", "y"]:
                for dir in attempt_dirs:
                    shutil_rmtree(dir, ignore_errors=True)

        super().delete(*args, **kwargs)

    def update_secondary(
        self, restriction: Optional[Union[str, dict]] = True
    ) -> None:
        """Update secondary keys for entries matching restriction.

        Parameters
        ----------
        restriction : bool, str, dict, optional
            Restriction to apply to entries. Default True (all entries).
        """

        query = self & restriction & "created_at IS NULL"
        total = len(query)

        if total == 0:
            logger.warning("No entries found matching restriction")
            return

        logger.info(
            f"Updating created_at for {total} entries from filesystem..."
        )

        for key in tqdm(query, total=total):
            created_at = self._get_file_created_at(key)
            old, _new = self._get_paths(key)
            self.update1(
                dict(key, created_at=created_at, deleted=not old.exists())
            )
