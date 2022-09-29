import copy
import os

import datajoint as dj
import kachery_cloud as kcl

from ..common.common_nwbfile import AnalysisNwbfile, Nwbfile

schema = dj.schema("sharing_kachery")


def kachery_download_file(uri: str, dest: str, project_id: str):
    """downloads the specified uri from using kachery cloud.
    First tries to download directly, and if that fails, starts an upload request for the file and then downloads it

    Parameters
    ----------
    uri : str
        the uri of the requested file
    dest : str
        the full path for the downloaded file

    Returns
        str
            The path to the downloaded file or None if the download was unsucessful
    """
    fname = kcl.load_file(uri, dest=dest)
    if fname is None:
        # if we can't load the uri directly, it should be because it is not in the cloud, so we need to start a task to load it
        kcl.request_file_experimental(uri=uri, project_id=project_id)
        if not kcl.load_file(uri, dest=dest):
            return False
    print("File downloaded")
    return True


@schema
class KacherySharingGroup(dj.Manual):
    definition = """
    sharing_group_name: varchar(200) # the name of the group we are sharing with
    ---
    description: varchar(200) # description of this group
    access_group_id = '': varchar(100) # the id for this group on http://cloud.kacheryhub.org/. Leaving this empty implies that the group is public.
    project_id: varchar(100) # the ids of the project for this sharing group
    """


@schema
class NwbfileKacherySelection(dj.Manual):
    definition = """
    -> KacherySharingGroup
    -> Nwbfile
    """


@schema
class NwbfileKachery(dj.Computed):
    definition = """
    -> NwbfileKacherySelection
    ---
    nwb_file_uri='': varchar(200)  # the uri for underscore NWB file for kachery. This may be encrypted (for limited sharing) or not encrypted (for public files)
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
        access_group = (
            KacherySharingGroup & {"sharing_group_name": key["sharing_group_name"]}
        ).fetch1("access_group_id")
        if access_group != "":
            key["nwb_file_uri"] = kcl.encrypt_uri(uri, access_group=access_group)
        else:
            key["nwb_file_uri"] = uri
        self.insert1(key)

        # we also need to insert the original NWB file.
        # TODO: change this to automatically detect all linked files
        # For the moment, remove the last character ('_') and add the extension
        linked_file_path = os.path.splitext(nwb_abs_path)[0][:-1] + ".nwb"
        uri = kcl.link_file(linked_file_path)
        if access_group != "":
            linked_key["linked_file_uri"] = kcl.encrypt_uri(
                uri, access_group=access_group
            )
        else:
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
        nwb_uri, sharing_group_name = (
            NwbfileKachery & {"nwb_file_name": nwb_file_name}
        ).fetch("nwb_file_uri", "sharing_group_name")
        if len(nwb_uri) == 0:
            return False
        # check to see if the sha1 is encrypted
        if nwb_uri[0].startswith("sha1-enc://"):
            # decypt the URI
            uri = kcl.decrypt_uri(nwb_uri[0])
        else:
            uri = nwb_uri[0]

        project_id = (
            KacherySharingGroup & {"sharing_group_name": sharing_group_name[0]}
        ).fetch1("project_id")
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
    -> KacherySharingGroup
    -> AnalysisNwbfile
    """


@schema
class AnalysisNwbfileKachery(dj.Computed):
    definition = """
    -> AnalysisNwbfileKacherySelection
    ---
    analysis_file_uri='': varchar(200)  # the uri of the file; may be encyrpted (limited sharing) or not encrypted (public sharing)
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
        uri = kcl.link_file(AnalysisNwbfile().get_abs_path(key["analysis_file_name"]))
        access_group = (
            KacherySharingGroup & {"sharing_group_name": key["sharing_group_name"]}
        ).fetch1("access_group_id")
        if access_group != "":
            key["analysis_file_uri"] = kcl.encrypt_uri(uri, access_group=access_group)
        else:
            key["analysis_file_uri"] = uri
        self.insert1(key)

        # we also need to insert any linked files
        # TODO: change this to automatically detect all linked files
        # self.LinkedFile.insert1(key)

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
        analysis_uri, sharing_group_name = (
            AnalysisNwbfileKachery & {"analysis_file_name": analysis_file_name}
        ).fetch("analysis_file_uri", "sharing_group_name")
        if len(analysis_uri) == 0:
            return False

        if analysis_uri[0].startswith("sha1-enc://"):
            # decypt the URI
            uri = kcl.decrypt_uri(analysis_uri[0])
        else:
            uri = analysis_uri[0]

        project_id = (
            KacherySharingGroup & {"sharing_group_name": sharing_group_name[0]}
        ).fetch1("project_id")
        print(f"attempting to download uri {uri}")

        if not kachery_download_file(
            uri=uri,
            dest=AnalysisNwbfile.get_abs_path(analysis_file_name),
            project_id=project_id,
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
