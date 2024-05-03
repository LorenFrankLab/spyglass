import os
import dandi.organize
import dandi.validate
from dandi.validate_types import Severity

import datajoint as dj

from .common_usage import Export

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
    """Validate the dandiset directory"""
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
