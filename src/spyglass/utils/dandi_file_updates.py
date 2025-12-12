from typing import List

import h5py

from spyglass.utils.dj_helper_fn import ExportErrorLog, _resolve_external_table
from spyglass.utils.logging import logger

STR_DTYPE = h5py.special_dtype(vlen=str)


def update_analysis_for_dandi_standard(
    filepath: str,
    age: str = "P4M/P8M",
    resolve_external_table: bool = True,
):
    """Function to resolve common nwb file format errors within the database

    Parameters
    ----------
    filepath : str
        abs path to the file to edit
    age : str, optional
        age to assign animal if missing, by default "P4M/P8M"
    resolve_external_table : bool, optional
        whether to update the external table. Set False if editing file
        outside the database, by default True
    """
    from spyglass.common import LabMember

    LabMember().check_admin_privilege(
        error_message="Admin permissions required to edit existing analysis files"
    )
    file_name = filepath.split("/")[-1]
    # edit the file
    try:
        with h5py.File(filepath, "a") as file:
            # add file_name attribute to general/source_script if missing
            add_source_script_name(
                file=file, script_name="src/spyglass/common/common_nwbfile.py"
            )
            # Adjust to single letter sex identifier
            standardize_sex_identifier(file)

            # replace subject species value "Rat" with "Rattus norvegicus"
            ensure_species_is_latin(file)

            # add subject age dataset "P4M/P8M"
            add_age_if_missing(file, age)

            # format names to "Last, First"
            format_experimenter_names(file)

            # convert any float16 datasets to float32
            convert_float16_to_float32(file)

            # add id column to dynamic tables if missing
            add_id_column_to_dynamic_tables(file)

    except BlockingIOError as e:
        ExportErrorLog().insert1(
            {
                "file": filepath,
                "source": "update_analysis_for_dandi_standard",
            },
            skip_duplicates=True,
        )
        logger.error(f"Could not open {filepath} for editing: {e}")
        return

    # update the datajoint external store table to reflect the changes
    if resolve_external_table:
        location = "raw" if filepath.endswith("_.nwb") else "analysis"
        _resolve_external_table(filepath, file_name, location)


def add_source_script_name(
    file: h5py.File,
    script_name: str,
):
    """Add source script information to general/source_script if missing.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    script_name : str
        The name of the source script.
    """
    if ("general/source_script" in file) and (
        "file_name" not in (grp := file["general/source_script"]).attrs
    ):
        logger.info("Adding file_name attribute to general/source_script")
        grp.attrs["file_name"] = script_name


def standardize_sex_identifier(file: h5py.File):
    """
    Adjust the subject sex identifier to a single letter format.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    """
    sex_value = file["/general/subject/sex"][()].decode("utf-8")

    if sex_value not in ["Female", "Male", "F", "M", "O", "U"]:
        raise ValueError(f"Unexpected value for sex: {sex_value}")
    if len(sex_value) == 1:
        return

    new_sex_value = sex_value[0].upper()
    logger.info(f"Adjusting subject sex: '{sex_value}' -> '{new_sex_value}'")
    file["/general/subject/sex"][()] = new_sex_value


def ensure_species_is_latin(file: h5py.File):
    """
    Ensure the subject species is in Latin binomial form or NCBI taxonomy link.

    For Rat, it updates "Rat" to "Rattus norvegicus".

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    """
    species_value = file["/general/subject/species"][()].decode("utf-8")
    if species_value == "Rat":
        new_species_value = "Rattus norvegicus"
        logger.info(
            f"Adjusting subject species from '{species_value}' to "
            + f"'{new_species_value}'."
        )
        file["/general/subject/species"][()] = new_species_value
        return

    if not (len(species_value.split(" ")) == 2 or "NCBITaxon" in species_value):
        raise ValueError(
            "Dandi upload requires species either be in Latin binomial form"
            + " (e.g., 'Mus musculus' and 'Homo sapiens') or be a NCBI "
            + "taxonomy link (e.g., "
            + "'http://purl.obolibrary.org/obo/NCBITaxon_280675').\n "
            + f"Please update species value of: {species_value}"
        )


def add_age_if_missing(file: h5py.File, age: str = "P4/P8"):
    """
    Add the subject age if not present

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.

    age : str, optional
       age of the subject, by default "P4/P8"
    """
    if "age" in file["/general/subject"]:
        return
    new_age_value = age
    logger.info(f"Adding missing subject age, set to '{new_age_value}'.")
    file["/general/subject"].create_dataset(
        name="age", data=new_age_value, dtype=STR_DTYPE
    )


def dandi_format_names(experimenter: List) -> List:
    """Make names compliant with dandi standard of "Last, First"

    Parameters
    ----------
    experimenter : List
        List of experimenter names

    Returns
    -------
    List
        reformatted list of experimenter names
    """
    for i, name in enumerate(experimenter):
        parts = name.split(" ")
        new_name = " ".join(
            parts[:-1],
        )
        new_name = f"{parts[-1]}, {new_name}"
        experimenter[i] = new_name
    return experimenter


def format_experimenter_names(file: h5py.File):
    """
    Ensure experimenter names are in "Last, First" format.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    """
    experimenter_value = file["/general/experimenter"][:].astype(str)
    new_experimenter_value = dandi_format_names(experimenter_value)
    if experimenter_value != new_experimenter_value:
        new_experimenter_value = new_experimenter_value.astype(STR_DTYPE)
        logger.info(
            f"Adjusting experimenter from {experimenter_value} to "
            + f"{new_experimenter_value}."
        )
        file["/general/experimenter"][:] = new_experimenter_value


def convert_float16_to_float32(file: h5py.File):
    """
    Convert datasets with float16 dtype to float32 dtype.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    """

    float16_datasets = find_float16_datasets(file)
    if not float16_datasets:
        return
    logger.info(
        f"Converting {len(float16_datasets)} float16 datasets to float32"
    )
    for dset_path in float16_datasets:
        convert_dataset_type(file, dset_path, target_dtype="float32")


def add_id_column_to_dynamic_tables(
    file: h5py.File,
    table_path: str,
):
    """
    Add an 'id' column to a dynamic table if missing.

    Parameters
    ----------
    file : h5py.File
        An open HDF5 file object.
    table_path : str
        The HDF5 path to the dynamic table.
    """
    tables_missing_id = find_dynamic_tables_missing_id(file)
    if not tables_missing_id:
        return
    logger.info(
        f"Adding missing id columns to {len(tables_missing_id)} "
        + "dynamic tables"
    )
    for table_path in tables_missing_id:
        add_id_column_to_table(file, table_path)
