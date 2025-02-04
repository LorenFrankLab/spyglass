"""This schema is used to transition manage files for recompute.

Tables
------
RecordingVersions: What versions are present in an existing analysis file?
    Allows restrict of recompute attempts to pynwb environments that are
    compatible with a pre-existing file. For pip dependencies, see
    SpikeSortingRecording.dependencies field
RecordingRecompute: Attempt recompute of an analysis file.
"""

from pathlib import Path
from typing import Union

import datajoint as dj
import pynwb
from h5py import File as h5py_File
from hdmf import __version__ as hdmf_version
from hdmf.build import TypeMap
from hdmf.spec import NamespaceCatalog
from pynwb.spec import NWBDatasetSpec, NWBGroupSpec, NWBNamespace
from spikeinterface import __version__ as si_version

from spyglass.common import AnalysisNwbfile
from spyglass.settings import analysis_dir, temp_dir
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_hash import NwbfileHasher, get_file_namespaces

schema = dj.schema("cbroz_temp")  # TODO: spikesorting_v1_usage or _recompute


@schema
class RecordingVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> SpikeSortingRecording
    ---
    core = '' : varchar(32)
    hdmf_common = '' : varchar(32)
    hdmf_experimental = '' : varchar(32)
    ndx_franklab_novela = '' : varchar(32)
    spyglass = '' : varchar(64)
    """

    @property
    def valid_ver_restr(self):
        """Return a restriction of self for the current environment."""
        return self.namespace_dict(pynwb.get_manager().type_map)

    @property
    def this_env(self):
        """Return restricted version of self for the current environment."""
        return self & self.valid_ver_restr

    def namespace_dict(self, type_map: TypeMap):
        """Remap namespace names to hyphenated field names for DJ compatibility."""
        hyph_fields = [f.replace("_", "-") for f in self.heading.names]
        name_cat = type_map.namespace_catalog
        return {
            field.replace("-", "_"): name_cat.get_namespace(field).get(
                "version", None
            )
            for field in name_cat.namespaces
            if field in hyph_fields
        }

    def make(self, key):
        parent = (SpikeSortingRecording & key).fetch1()
        path = AnalysisNwbfile().get_abs_path(parent["analysis_file_name"])

        insert = key.copy()
        insert.update(get_file_namespaces(path))

        with h5py_File(path, "r") as f:
            script = f.get("general/source_script")
            if script is not None:
                script = str(script[()]).split("=")[1].strip().replace("'", "")
            insert["spyglass"] = script

        self.insert1(insert)


@schema
class RecordingRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingVersions
    attempt_id: varchar(32) # name for environment used to attempt recompute
    ---
    dependencies: blob # dict of pip dependencies
    """

    @property
    def pip_deps(self):
        return dict(
            pynwb=pynwb.__version__,
            hdmf=hdmf_version,
            spikeinterface=si_version,
            # TODO: add othes?
        )

    def attempt_all(self, attempt_id):
        inserts = [
            {**key, "attempt_id": attempt_id, "dependencies": self.pip_deps}
            for key in RecordingVersions().this_env.fetch("KEY", as_dict=True)
        ]
        self.insert(inserts, skip_duplicates=True)


@schema
class RecordingRecompute(dj.Computed):
    definition = """
    -> RecordingRecomputeSelection
    ---
    matched: bool
    """

    class Name(dj.Part):
        definition = """
        -> master
        name : varchar(255)
        ---
        missing_from: enum('old', 'new')
        """

    class Hash(dj.Part):
        definition = """
        -> master
        name : varchar(255)
        ---
        old=null: longblob
        new=null: longblob
        """

    key_source = RecordingVersions().this_env * RecordingRecomputeSelection
    old_dir = Path(analysis_dir)
    # see /stelmo/cbroz/temp_rcp/ for existing recomputed
    new_dir = Path(temp_dir) / "spikesort_v1_recompute"
    ignore_files = set()

    def _get_subdir(self, key):
        file = key["analysis_file_name"] if isinstance(key, dict) else key
        parts = file.split("_")
        subdir = "_".join(parts[:-1])
        return subdir + "/" + file

    def _get_paths(self, key):
        old = self.old_dir / self._get_subdir(key)
        new = self.new_dir / self._get_subdir(key)
        return old, new

    def _hash_existing(self):
        for nwb in self.new_dir.rglob("*.nwb"):
            if nwb.with_suffix(".hash").exists():
                continue
            try:
                logger.info(f"Hashing {nwb}")
                _ = self._hash_one(nwb)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning(f"Error: {e.__class__.__name__}: {nwb.name}")
                continue

    def _hash_one(self, path):
        hasher = NwbfileHasher(path, verbose=False)
        with open(path.with_suffix(".hash"), "w") as f:
            f.write(hasher.hash)
        return hasher.hash

    def make(self, key):
        if self & key:
            return

        parent = (
            SpikeSortingRecording
            * RecordingVersions
            * RecordingRecomputeSelection
            & key
        ).fetch1()

        # Ensure dependencies unchanged since selection insert
        parent_deps = parent["dependencies"]
        for dep, ver in RecordingRecomputeSelection().pip_deps.items():
            if parent_deps.get(dep, None) != ver:
                raise ValueError(f"Please run this key with {parent_deps}")

        old, new = self._get_paths(parent)

        if not new.exists():  # if the file has yet to be recomputed
            try:
                new_vals = SpikeSortingRecording()._make_file(
                    parent,
                    recompute_file_name=parent["analysis_file_name"],
                    save_to=new,
                )
            except RuntimeError as e:
                print(f"{e}: {new.name}")
                return
            except ValueError as e:
                e_info = e.args[0]
                if "probe info" in e_info:  # make failed bc missing probe info
                    self.insert1(dict(key, matched=False, diffs=e_info))
                else:
                    print(f"ValueError: {e}: {new.name}")
                return
            except KeyError as err:
                e_info = err.args[0]
                if "MISSING" in e_info:  # make failed bc missing parent file
                    e = e_info.split("MISSING")[0].strip()
                    self.insert1(dict(key, matched=False, diffs=e))
                    self.Name().insert1(
                        dict(
                            key, name=f"Parent missing {e}", missing_from="old"
                        )
                    )
                else:
                    logger.warning(f"KeyError: {err}: {new.name}")
                return
            with open(new.with_suffix(".hash"), "w") as f:
                f.write(new_vals["file_hash"])

        # TODO: how to check the env used to make the file when reading?
        elif new.with_suffix(".hash").exists():
            print(f"\nReading hash {new}")
            with open(new.with_suffix(".hash"), "r") as f:
                new_vals = dict(file_hash=f.read())
        else:
            new_vals = dict(file_hash=self._hash_one(new))

        old_hasher = parent["file_hash"] or NwbfileHasher(
            old, verbose=False, keep_obj_hash=True
        )

        if new_vals["file_hash"] == old_hasher.hash:
            self.insert1(dict(key, match=True))
            new.unlink(missing_ok=True)
            return

        names = []
        hashes = []

        logger.info(f"Compparing mismatched {new}")
        new_hasher = NwbfileHasher(new, verbose=False, keep_obj_hash=True)
        all_objs = set(old_hasher.objs.keys()) | set(new_hasher.objs.keys())

        for obj in all_objs:
            old_obj, old_hash = old_hasher.objs.get(obj, (None, None))
            new_obj, new_hash = new_hasher.objs.get(obj, (None, None))

            if old_hash is None:
                names.append(dict(key, name=obj, missing_from="old"))
            elif new_hash is None:
                names.append(dict(key, name=obj, missing_from="new"))
            elif old_hash != new_hash:
                hashes.append(dict(key, name=obj, old=old_obj, new=new_obj))

        self.insert1(dict(key, matched=False))
        self.Name().insert(names)
        self.Hash().insert(hashes)
