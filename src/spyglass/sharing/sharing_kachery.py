import copy
import os

import datajoint as dj
import kachery_cloud as kcl

from ..common.common_nwbfile import AnalysisNwbfile
from ..common.common_lab import Lab

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
    """set the kachery resource url and attempt to down load the uri into the destination path"""
    KacheryZone.set_resource_url({"kachery_zone_name": kachery_zone_name})
    return kcl.load_file(uri, dest=dest)


@schema
class KacheryZone(dj.Manual):
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
        except:
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
        """Sets the KACHERY_RESOURCE_URL based on the key corresponding to a single Kachery Zone

        Parameters
        ----------
        key : dict
            key to retrieve a single kachery zone
        """
        try:
            kachery_zone_name, kachery_proxy = (KacheryZone & key).fetch1(
                "kachery_zone_name", "kachery_proxy"
            )
        except:
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
        KacheryZone.reset_zone()
        if default_kachery_resource_url is not None:
            os.environ[kachery_resource_url_envar] = default_kachery_resource_url


@schema
class AnalysisNwbfileKacherySelection(dj.Manual):
    definition = """
    -> KacheryZone
    -> AnalysisNwbfile
    """


@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfileKacherySelection
    ---
    analysis_file_uri='': varchar(200)  # the uri of the file
    """

    class LinkedFile(dj.Part):
        definition = """
        -> AnalysisNwbfileKachery
        linked_file_rel_path: varchar(200) # the path for the linked file relative to the SPYGLASS_BASE_DIR environment variable
        ---
        linked_file_uri='': varchar(200) # the uri for the linked file
        """

    def make(self, key):
        # note that we're assuming that the user has initialized a kachery-cloud client with kachery-cloud-init
        # uncomment the line below once we are sharing linked files as well.
        # linked_key = copy.deepcopy(key)
        print(f'Linking {key["analysis_file_name"]} in kachery-cloud...')
        # set the kachery zone
        KacheryZone.set_zone(key)
        key["analysis_file_uri"] = kcl.link_file(
            AnalysisNwbfile().get_abs_path(key["analysis_file_name"])
        )
        print(os.environ[kachery_zone_envar], os.environ[kachery_cloud_dir_envar])
        print(AnalysisNwbfile().get_abs_path(key["analysis_file_name"]))
        print(kcl.load_file(key["analysis_file_uri"]))
        self.insert1(key)

        # we also need to insert any linked files
        # TODO: change this to automatically detect all linked files
        # self.LinkedFile.insert1(key)

        # reset the Kachery zone and cloud_dir to the defaults
        KacheryZone.reset_zone()

    @staticmethod
    def download_file(analysis_file_name: str) -> bool:
        """Download the specified analysis file and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis file

        Returns
        ----------
        is_success : bool
            True if the file was successfully downloaded, False otherwise
        """
        uri, kachery_zone_name = (
            AnalysisNwbfileKachery & {"analysis_file_name": analysis_file_name}
        ).fetch1("analysis_file_uri", "kachery_zone_name")
        if len(uri) == 0:
            return False
        print("uri:", uri)
        if not kachery_download_file(
            uri=uri,
            dest=AnalysisNwbfile.get_abs_path(analysis_file_name),
            kachery_zone_name=kachery_zone_name,
        ):
            raise Exception(f"{analysis_file_name} cannot be downloaded")
        # now download the linked file(s)
        linked_files = (
            AnalysisNwbfileKachery.LinkedFile
            & {"analysis_file_name": analysis_file_name}
        ).fetch(as_dict=True)
        for file in linked_files:
            uri = file["linked_file_uri"]
            print(f"attempting to download linked file uri {uri}")
            linked_file_path = (
                os.environ["SPYGLASS_BASE_DIR"] + file["linked_file_rel_path"]
            )
            if not kachery_download_file(
                uri=uri, dest=linked_file_path, kachery_zone_name=kachery_zone_name
            ):
                raise Exception(f"Linked file {linked_file_path} cannot be downloaded")

        return True
