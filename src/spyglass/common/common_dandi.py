import os
import shutil
from pathlib import Path

import datajoint as dj
import fsspec
import h5py
import pynwb
from fsspec.implementations.cached import CachingFileSystem

from spyglass.utils import logger

try:
    import dandi.download
    import dandi.organize
    import dandi.upload
    import dandi.validate
    from dandi.consts import known_instances
    from dandi.dandiapi import DandiAPIClient
    from dandi.metadata.nwb import get_metadata
    from dandi.organize import OrganizeInvalid
    from dandi.validate_types import Severity

except (ImportError, ModuleNotFoundError) as e:
    (
        dandi.download,
        dandi.organize,
        dandi.upload,
        dandi.validate,
        known_instances,
        DandiAPIClient,
        get_metadata,
        OrganizeInvalid,
        Severity,
    ) = [None] * 9
    logger.warning(e)


from spyglass.common.common_usage import Export
from spyglass.settings import export_dir
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("common_dandi")


@schema
class DandiPath(SpyglassMixin, dj.Manual):
    definition = """
    -> Export.File
    ---
    dandiset_id: varchar(16)
    filename: varchar(255)
    dandi_path: varchar(255)
    dandi_instance = "dandi": varchar(32)
    """

    def fetch_file_from_dandi(self, key: dict):
        dandiset_id, dandi_path, dandi_instance = (self & key).fetch1(
            "dandiset_id", "dandi_path", "dandi_instance"
        )
        dandiset_id = str(dandiset_id)
        # get the s3 url from Dandi
        with DandiAPIClient(
            dandi_instance=known_instances[dandi_instance],
        ) as client:
            asset = client.get_dandiset(dandiset_id).get_asset_by_path(
                dandi_path
            )
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        # stream the file from s3
        # first, create a virtual filesystem based on the http protocol
        fs = fsspec.filesystem("http")

        # create a cache to save downloaded data to disk (optional)
        fsspec_file = CachingFileSystem(
            fs=fs,
            cache_storage=f"{export_dir}/nwb-cache",  # Local folder for cache
        )

        # Open and return the file
        fs_file = fsspec_file.open(s3_url, "rb")
        io = pynwb.NWBHDF5IO(file=h5py.File(fs_file))
        nwbfile = io.read()
        return (io, nwbfile)

    def compile_dandiset(
        self,
        key: dict,
        dandiset_id: str,
        dandi_api_key: str = None,
        dandi_instance: str = "dandi",
    ):
        """Compile a Dandiset from the export.
        Parameters
        ----------
        key : dict
            ExportSelection key
        dandiset_id : str
            Dandiset ID generated by the user on the dadndi server
        dandi_api_key : str, optional
            API key for the dandi server. Optional if the environment variable
            DANDI_API_KEY is set.
        dandi_instance : dandiset's Dandi instance. Defaults to the dev server
        """
        key = (Export & key).fetch1("KEY")
        paper_id = (Export & key).fetch1("paper_id")
        if self & key:
            raise ValueError(
                "Adding new files to an existing dandiset is not permitted. "
                + f"Please rerun after deleting existing entries for {key}"
            )

        # make a temp dir with symbolic links to the export files
        source_files = (Export.File() & key).fetch("file_path")
        paper_dir = f"{export_dir}/{paper_id}"
        os.makedirs(paper_dir, exist_ok=True)
        destination_dir = f"{paper_dir}/dandiset_{paper_id}"
        dandiset_dir = f"{paper_dir}/{dandiset_id}"

        # check if pre-existing directories for dandi export exist.
        # Remove if so to continue
        for dandi_dir in destination_dir, dandiset_dir:
            if os.path.exists(dandi_dir):
                from datajoint.utils import user_choice

                if (
                    user_choice(
                        "Pre-existing dandi export dir exist."
                        + f"Delete existing export folder: {dandi_dir}",
                        default="no",
                    )
                    == "yes"
                ):
                    shutil.rmtree(dandi_dir)
                    continue
                raise RuntimeError(
                    "Directory must be removed prior to dandi export to ensure "
                    + f"dandi-compatability: {dandi_dir}"
                )

        os.makedirs(destination_dir, exist_ok=False)
        for file in source_files:
            if not os.path.exists(
                f"{destination_dir}/{os.path.basename(file)}"
            ):
                os.symlink(file, f"{destination_dir}/{os.path.basename(file)}")

        # validate the dandiset
        validate_dandiset(destination_dir, ignore_external_files=True)

        # given dandiset_id, download the dandiset to the export_dir
        url = (
            f"{known_instances[dandi_instance].gui}"
            + f"/dandiset/{dandiset_id}/draft"
        )
        dandi.download.download(url, output_dir=paper_dir)

        # organize the files in the dandiset directory
        dandi.organize.organize(
            destination_dir, dandiset_dir, invalid=OrganizeInvalid.WARN
        )

        # get the dandi name translations
        translations = translate_name_to_dandi(destination_dir)

        # upload the dandiset to the dandi server
        if dandi_api_key:
            os.environ["DANDI_API_KEY"] = dandi_api_key
        dandi.upload.upload(
            [dandiset_dir],
            dandi_instance=dandi_instance,
        )
        logger.info(f"Dandiset {dandiset_id} uploaded")
        # insert the translations into the dandi table
        translations = [
            {
                **(
                    Export.File() & key & f"file_path LIKE '%{t['filename']}'"
                ).fetch1(),
                **t,
                "dandiset_id": dandiset_id,
                "dandi_instance": dandi_instance,
            }
            for t in translations
        ]
        self.insert(translations, ignore_extra_fields=True)


def _get_metadata(path):
    # taken from definition within dandi.organize.organize
    try:
        meta = get_metadata(path)
    except Exception as exc:
        meta = {}
        raise RuntimeError("Failed to get metadata for %s: %s", path, exc)
    meta["path"] = path
    return meta


def translate_name_to_dandi(folder):
    """Uses dandi.organize to translate filenames to dandi paths

    NOTE: The name for a given file depends on all files in the folder

    Parameters
    ----------
    folder : str
        location of files to be translated

    Returns
    -------
    dict
        dictionary of filename to dandi_path translations
    """
    files = Path(folder).glob("*")
    metadata = list(map(_get_metadata, files))
    metadata, skip_invalid = dandi.organize.filter_invalid_metadata_rows(
        metadata
    )
    metadata = dandi.organize.create_unique_filenames_from_metadata(
        metadata, required_fields=None
    )
    return [
        {"filename": Path(file["path"]).name, "dandi_path": file["dandi_path"]}
        for file in metadata
    ]


def validate_dandiset(
    folder, min_severity="ERROR", ignore_external_files=False
):
    """Validate the dandiset directory

    Parameters
    ----------
    folder : str
        location of dandiset to be validated
    min_severity : str
        minimum severity level for errors to be reported, threshold for failed
        Dandi upload is "ERROR"
    ignore_external_files : bool
        whether to ignore external file errors. Used if validating
        before the organize step
    """
    validator_result = dandi.validate.validate(folder)
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

    if filtered_results:
        raise ValueError(
            "Validation failed\n\t"
            + "\n\t".join(
                [
                    f"{result.severity}: {result.message} in {result.path}"
                    for result in filtered_results
                ]
            )
        )
