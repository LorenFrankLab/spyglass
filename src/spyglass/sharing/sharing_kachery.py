import os

import datajoint as dj
import kachery_cloud as kcl
from datajoint.errors import DataJointError

from spyglass.common.common_lab import Lab  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import config
from spyglass.utils import SpyglassMixin, logger

# define the environment variable name for the kachery zone and the cloud directory
kachery_zone_envar = "KACHERY_ZONE"
kachery_cloud_dir_envar = "KACHERY_CLOUD_DIR"
kachery_resource_url_envar = "KACHERY_RESOURCE_URL"


# set the global default kachery zones
try:
    default_kachery_zone = os.environ[kachery_zone_envar]
    default_kachery_cloud_dir = os.environ[kachery_cloud_dir_envar]
    default_kachery_resource_url = os.environ[kachery_resource_url_envar]
except KeyError:
    default_kachery_zone = None
    default_kachery_cloud_dir = None
    default_kachery_resource_url = None

schema = dj.schema("sharing_kachery")


def kachery_download_file(uri: str, dest: str, kachery_zone_name: str) -> str:
    """Set the kachery resource url and attempt to download."""
    KacheryZone.set_resource_url({"kachery_zone_name": kachery_zone_name})
    return kcl.load_file(uri, dest=dest)


@schema
class KacheryZone(SpyglassMixin, dj.Manual):
    definition = """
    kachery_zone_name: varchar(200) # the name of the kachery zone. Note that this is the same as the name of the kachery resource.
    ---
    description: varchar(200) # description of this zone
    kachery_cloud_dir: varchar(200) # kachery cloud directory on local machine where files are linked
    kachery_proxy: varchar(200) # kachery sharing proxy
    -> Lab
    """

    @staticmethod
    def set_zone(key: dict):
        """Set the kachery zone based on the key to KacheryZone

        Parameters
        ----------
        key : dict
            key defining a single KacheryZone

        """
        try:
            kachery_zone_name, kachery_cloud_dir = (KacheryZone & key).fetch1(
                "kachery_zone_name", "kachery_cloud_dir"
            )
        except DataJointError:
            raise Exception(
                f"{key} does not correspond to a single entry in KacheryZone."
            )
            return None
        # set the new zone and cloud directory
        os.environ[kachery_zone_envar] = kachery_zone_name
        os.environ[kachery_cloud_dir_envar] = kachery_cloud_dir

    @staticmethod
    def reset_zone():
        """Resets the kachery zone environment variable to the default values."""
        if default_kachery_zone is not None:
            os.environ[kachery_zone_envar] = default_kachery_zone
        if default_kachery_cloud_dir is not None:
            os.environ[kachery_cloud_dir_envar] = default_kachery_cloud_dir

    @staticmethod
    def set_resource_url(key: dict):
        """Sets the KACHERY_RESOURCE_URL based on the key corresponding to a
        single Kachery Zone

        Parameters
        ----------
        key : dict
            key to retrieve a single kachery zone
        """
        try:
            kachery_zone_name, kachery_proxy = (KacheryZone & key).fetch1(
                "kachery_zone_name", "kachery_proxy"
            )
        except DataJointError:
            raise Exception(
                f"{key} does not correspond to a single entry in KacheryZone."
            )
        # set the new zone and cloud directory
        os.environ[kachery_zone_envar] = kachery_zone_name
        os.environ[kachery_resource_url_envar] = (
            kachery_proxy + "/r/" + kachery_zone_name
        )

    @staticmethod
    def reset_resource_url():
        """Resets the KACHERY_RESOURCE_URL to the default value."""
        KacheryZone.reset_zone()
        if default_kachery_resource_url is not None:
            os.environ[kachery_resource_url_envar] = (
                default_kachery_resource_url
            )


@schema
class AnalysisNwbfileKacherySelection(SpyglassMixin, dj.Manual):
    definition = """
    -> KacheryZone
    -> AnalysisNwbfile
    """


@schema
class AnalysisNwbfileKachery(SpyglassMixin, dj.Computed):
    definition = """
    -> AnalysisNwbfileKacherySelection
    ---
    analysis_file_uri='': varchar(200)  # the uri of the file
    """

    class LinkedFile(SpyglassMixin, dj.Part):
        definition = """
        -> AnalysisNwbfileKachery
        linked_file_rel_path: varchar(200) # the path for the linked file relative to the SPYGLASS_BASE_DIR environment variable
        ---
        linked_file_uri='': varchar(200) # the uri for the linked file
        """

    def make(self, key):
        """Populate with the uri of the analysis file"""
        analysis_file = key["analysis_file_name"]
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file)

        logger.info(f"Linking {analysis_file} in kachery-cloud...")

        KacheryZone.set_zone(key)  # set the kachery zone

        key["analysis_file_uri"] = kcl.link_file(abs_path)
        kachery_zone = os.environ[kachery_zone_envar]
        kachery_cloud_dir = os.environ[kachery_cloud_dir_envar]

        logger.info(kachery_zone, kachery_cloud_dir)
        logger.info(abs_path)
        logger.info(kcl.load_file(key["analysis_file_uri"]))

        self.insert1(key)

        KacheryZone.reset_zone()  # reset the zone/cloud_dir to the defaults

    @staticmethod
    def download_file(
        analysis_file_name: str, permit_fail: bool = False
    ) -> bool:
        """Download the specified analysis file and associated linked files
        from kachery-cloud if possible

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis file

        Returns
        ----------
        is_success : bool
            True if the file was successfully downloaded, False otherwise
        """
        fetched_list = (
            AnalysisNwbfileKachery & {"analysis_file_name": analysis_file_name}
        ).fetch("analysis_file_uri", "kachery_zone_name")
        downloaded = False
        for uri, kachery_zone_name in zip(fetched_list[0], fetched_list[1]):
            if len(uri) == 0:
                return False
            logger.info("uri:", uri)
            if kachery_download_file(
                uri=uri,
                dest=AnalysisNwbfile.get_abs_path(analysis_file_name),
                kachery_zone_name=kachery_zone_name,
            ):
                downloaded = True
                # now download the linked file(s)
                linked_files = (
                    AnalysisNwbfileKachery.LinkedFile
                    & {"analysis_file_name": analysis_file_name}
                ).fetch(as_dict=True)
                for file in linked_files:
                    uri = file["linked_file_uri"]
                    logger.info(f"attempting to download linked file uri {uri}")
                    linked_file_path = (
                        os.environ["SPYGLASS_BASE_DIR"]
                        + file["linked_file_rel_path"]
                    )
                    if not kachery_download_file(
                        uri=uri,
                        dest=linked_file_path,
                        kachery_zone_name=kachery_zone_name,
                    ):
                        raise Exception(
                            f"Linked file {linked_file_path} cannot be downloaded"
                        )
        if not downloaded and not permit_fail:
            raise Exception(f"{analysis_file_name} cannot be downloaded")

        return downloaded


def share_data_to_kachery(
    restriction={},
    table_list=[],
    zone_name=None,
):
    """Share data to kachery

    Parameters
    ----------
    restriction : dict, optional
        restriction to select what data should be shared from table, by default {}
    table_list : list, optional
        List of tables to share data from, by default []
    zone_name : str, optional
        What kachery zone to share the data to, by default zone in spyglass.settings.config,
        which looks for `KACHERY_ZONE` environmental variable, but defaults to
        'franklab.default'

    Raises
    ------
    ValueError
        Does not allow sharing of all data in table
    """
    if not zone_name:
        zone_name = config["KACHERY_ZONE"]
    kachery_selection_key = {"kachery_zone_name": zone_name}
    if not restriction:
        raise ValueError("Must provide a restriction to the table")
    selection_inserts = []
    for table in table_list:

        # ----------- Copy to master if has AnalysisNwbfile parent -----------
        if custom_analysis := getattr(table, "_custom_analysis_parent", None):
            (table & restriction)._parent_copy_to_common()
        # ----------------------------------------------------------------------

        analysis_file_list = (table & restriction).fetch("analysis_file_name")
        for file in analysis_file_list:  # Add all analysis to shared list
            kachery_selection_key["analysis_file_name"] = file
            selection_inserts.append(kachery_selection_key.copy())

    AnalysisNwbfileKacherySelection.insert(
        selection_inserts, skip_duplicates=True
    )
    AnalysisNwbfileKachery.populate()
