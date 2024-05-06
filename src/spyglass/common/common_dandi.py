import os
import dandi.organize
import dandi.validate
from dandi.validate_types import Severity
import fsspec
import pynwb
import h5py
from uuid import uuid4
from fsspec.implementations.cached import CachingFileSystem

import datajoint as dj

from spyglass.common.common_usage import Export
from spyglass.settings import export_dir

schema = dj.schema("common_dandi")


@schema
class DandiPath(dj.Manual):
    definition = """
    -> Export.File
    ---
    dandiset_id: int
    filename: varchar(255)
    dandi_path: varchar(255)
    """

    def fetch_file_from_dandi(self, key: dict):
        key = (self & key).fetch1("KEY")
        dandiset_id = (self & key).fetch1("dandiset_id")
        dandi_path = (self & key).fetch1("dandi_path")

        # get the s3 url from Dandi
        from dandi.dandiapi import DandiAPIClient

        with DandiAPIClient() as client:
            asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(
                dandi_path
            )
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        # stream the file from s3
        # first, create a virtual filesystem based on the http protocol
        fs = fsspec.filesystem("http")

        # create a cache to save downloaded data to disk (optional)
        fs = CachingFileSystem(
            fs=fs,
            cache_storage=f"{export_dir}/nwb-cache",  # Local folder for the cache
        )

        # Open and return the file
        f = fs.open(s3_url, "rb")
        file = h5py.File(f)
        io = pynwb.NWBHDF5IO(file=file)
        nwbfile = io.read()
        return (io, nwbfile)


def _get_metadata(path):
    # taken from definition within dandi.organize.organize
    # Avoid heavy import by importing within function:
    from dandi.metadata.nwb import get_metadata

    try:
        meta = get_metadata(path)
    except Exception as exc:
        meta = {}
        raise RuntimeError("Failed to get metadata for %s: %s", path, exc)
    meta["path"] = path
    return meta


def translate_name_to_dandi(folder):
    """Uses dandi.organize to translate filenames to dandi paths
    *Note* The name for a given file is dependent on that of all files in the folder

    Parameters
    ----------
    folder : str
        location of files to be translated

    Returns
    -------
    dict
        dictionary of filename to dandi_path translations
    """
    files = [f"{folder}/{f}" for f in os.listdir(folder)]
    metadata = list(map(_get_metadata, files))
    metadata, skip_invalid = dandi.organize.filter_invalid_metadata_rows(
        metadata
    )
    metadata = dandi.organize.create_unique_filenames_from_metadata(
        metadata, required_fields=None
    )
    translations = []
    for file in metadata:
        translation = {
            "filename": file["path"].split("/")[-1],
            "dandi_path": file["dandi_path"],
        }
        translations.append(translation)
    return translations


def validate_dandiset(
    folder, min_severity="ERROR", ignore_external_files=False
):
    """Validate the dandiset directory
    Parameters
    ----------
    folder : str
        location of dandiset to be validated
    min_severity : str
        minimum severity level for errors to be reported
    ignore_external_files : bool
        whether to ignore external file errors. Used if validating
        before the organize step
    """
    validator_result = dandi.validate.validate(folder)
    min_severity = "ERROR"
    min_severity_value = Severity[min_severity].value

    filtered_results = [
        i
        for i in validator_result
        if i.severity is not None and i.severity.value >= min_severity_value
    ]

    if ignore_external_files:
        # ignore external file errors. will be resolved during organize step
        filtered_results = [
            i
            for i in filtered_results
            if not i.message.startswith("Path is not inside")
        ]

    for result in filtered_results:
        print(result.severity, result.message, result.path)

    if filtered_results:
        raise ValueError("Validation failed")


def make_file_obj_id_unique(nwb_path: str):
    """Make the top-level object_id attribute of the file unique

    Parameters
    ----------
    nwb_path : str
        path to the NWB file

    Returns
    -------
    str
        the new object_id
    """
    new_id = str(uuid4())
    with h5py.File(nwb_path, "a") as f:
        f.attrs["object_id"] = new_id
    return new_id
