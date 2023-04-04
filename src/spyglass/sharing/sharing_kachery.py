import copy
import os

import datajoint as dj
import kachery_cloud as kcl

from ..common.common_nwbfile import AnalysisNwbfile, Nwbfile
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
except:
    default_kachery_zone = None
    default_kachery_cloud_dir = None
    default_kachery_resource_url = None

schema = dj.schema("sharing_kachery")


def kachery_download_file(uri: str, dest: str, kachery_zone: str):
    # set the kachery zone and attempt to down load the uri into the destination path
    KacheryZone.set_zone({"kachery_zone": kachery_zone})
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
        """sets the KACHERY_RESOURCE_URL based on the key corresponding to a single Kachery Zone

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
            return None
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
class NwbfileKacherySelection(dj.Manual):
    definition = """
    -> KacheryZone
    -> Nwbfile
    """


@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> NwbfileKacherySelection
    ---
    nwb_file_uri='': varchar(200)  # the uri for underscore NWB file for kachery.
    """

    class LinkedFile(dj.Part):
        definition = """
        -> NwbfileKachery
        linked_file_rel_path: varchar(200) # the relative path to the linked data file (assumes base of SPYGLASS_BASE_DIR)
        ---
        linked_file_uri='': varchar(200) # the uri for the linked file
        """

    def make(self, key):
        # note that we're assuming that the user has initialized a kachery-cloud client with kachery-cloud-init
        linked_key = copy.deepcopy(key)
        print(f'Linking {key["nwb_file_name"]} in kachery-cloud...')
        nwb_abs_path = Nwbfile().get_abs_path(key["nwb_file_name"])
        uri = kcl.link_file(nwb_abs_path)
        key["nwb_file_uri"] = uri
        self.insert1(key)

        # we also need to insert the original NWB file.
        # TODO: change this to automatically detect all linked files
        # For the moment, remove the last character ('_') and add the extension
        linked_file_path = os.path.splitext(nwb_abs_path)[0][:-1] + ".nwb"
        uri = kcl.link_file(linked_file_path)
        linked_key["linked_file_uri"] = uri
        linked_key["linked_file_rel_path"] = str.replace(
            linked_file_path, os.environ["SPYGLASS_BASE_DIR"], ""
        )
        self.LinkedFile.insert1(linked_key)

    @staticmethod
    def download_file(nwb_file_name: str):
        """Download the specified nwb file and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file

        Returns
        ----------
        bool
            True if the file was successfully downloaded, false otherwise
        """
        (nwb_uri, kachery_zone) = (
            NwbfileKachery & {"nwb_file_name": nwb_file_name}
        ).fetch("nwb_file_uri", "kachery_zone")
        if len(nwb_uri) == 0:
            return False
        # check to see if the sha1 is encrypted
        if nwb_uri[0].startswith("sha1-enc://"):
            # decypt the URI
            uri = kcl.decrypt_uri(nwb_uri[0])
        else:
            uri = nwb_uri[0]

        project_id = (KacheryZone & {"kachery_zone": kachery_zone[0]}).fetch1(
            "project_id"
        )
        print(f"attempting to download uri {uri}")

        if not kachery_download_file(
            uri=uri, dest=Nwbfile.get_abs_path(nwb_file_name), project_id=project_id
        ):
            raise Exception(
                f"{Nwbfile.get_abs_path(nwb_file_name)} cannot be downloaded"
            )

        # now download the linked file(s)
        linked_files = (
            NwbfileKachery.LinkedFile & {"nwb_file_name": nwb_file_name}
        ).fetch(as_dict=True)
        for file in linked_files:
            if file["linked_file_uri"].startswith("sha1-enc://"):
                uri = kcl.decrypt_uri(file["linked_file_uri"])
            else:
                uri = file["linked_file_uri"]
            print(f"attempting to download linked file uri {uri}")
            linked_file_path = (
                os.environ["SPYGLASS_BASE_DIR"] + file["linked_file_rel_path"]
            )
            if not kachery_download_file(
                uri=uri, dest=linked_file_path, project_id=project_id
            ):
                raise Exception(f"Linked file {linked_file_path} cannot be downloaded")
                return False

        return True


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
        linked_key = copy.deepcopy(key)
        print(f'Linking {key["analysis_file_name"]} in kachery-cloud...')
        # set the kachery zone
        KacheryZone.set_zone(key)
        uri = kcl.link_file(AnalysisNwbfile().get_abs_path(key["analysis_file_name"]))
        self.insert1(key)

        # we also need to insert any linked files
        # TODO: change this to automatically detect all linked files
        # self.LinkedFile.insert1(key)

        # reset the Kachery zone and cloud_dir to the defaults
        KacheryZone.reset_zone()

    @staticmethod
    def download_file(analysis_file_name: str):
        """Download the specified analysis file and associated linked files from kachery-cloud if possible

        Parameters
        ----------
        analysis_file_name : str
            The name of the analysis file

        Returns
        ----------
        bool
            True if the file was successfully downloaded, false otherwise
        """
        uri, kachery_zone = (
            AnalysisNwbfileKachery & {"analysis_file_name": analysis_file_name}
        ).fetch("analysis_file_uri", "kachery_zone")
        if len(uri) == 0:
            return False

        if not kachery_download_file(
            uri=uri,
            dest=AnalysisNwbfile.get_abs_path(analysis_file_name),
            kachery_zone=kachery_zone,
        ):
            raise Exception(
                f"{AnalysisNwbfile.get_abs_path(analysis_file_name)} cannot be downloaded"
            )
            return False
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
            if not kachery_download_file(uri=uri, dest=linked_file_path):
                raise Exception(f"Linked file {linked_file_path} cannot be downloaded")
                return False

        return True
