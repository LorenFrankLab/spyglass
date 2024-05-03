import os
import dandi.organize


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
