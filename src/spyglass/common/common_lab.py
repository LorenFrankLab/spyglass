"""Schema for institution, lab team/name/members. Session-independent."""

import datajoint as dj

from spyglass.utils import SpyglassMixin, logger

from ..utils.nwb_helper_fn import get_nwb_file
from .common_nwbfile import Nwbfile

schema = dj.schema("common_lab")


@schema
class LabMember(SpyglassMixin, dj.Manual):
    definition = """
    lab_member_name: varchar(80)
    ---
    first_name: varchar(200)
    last_name: varchar(200)
    """

    # NOTE that names must be unique here. If there are two neuroscientists
    # named Jack Black that have data in this database, this will create an
    # incorrect linkage. NWB does not yet provide unique IDs for names.

    # NOTE: PR requires table alter.
    class LabMemberInfo(SpyglassMixin, dj.Part):
        definition = """
        # Information about lab member in the context of Frank lab network
        -> LabMember
        ---
        google_user_name         : varchar(200) # For permission to curate
        datajoint_user_name = "" : varchar(200) # For permission to delete
        admin = 0                : bool         # Ignore permission checks
        unique index (datajoint_user_name)
        unique index (google_user_name)
        """

        # NOTE: index present for new instances. pending surgery, not enforced
        # for existing instances.

    _admin = []

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert lab member information from an NWB file.

        Parameters
        ----------
        nwbf: pynwb.NWBFile
            The NWB file with experimenter information.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.
        """
        config = config or dict()
        if isinstance(nwbf, str):
            nwb_file_abspath = Nwbfile.get_abs_path(nwbf, new_file=True)
            nwbf = get_nwb_file(nwb_file_abspath)

        if "LabMember" in config:
            experimenter_list = [
                member_dict["lab_member_name"]
                for member_dict in config["LabMember"]
            ]
        elif nwbf.experimenter is not None:
            experimenter_list = nwbf.experimenter
        else:
            logger.info("No experimenter metadata found.\n")
            return

        for experimenter in experimenter_list:
            cls.insert_from_name(experimenter)

            # each person is by default the member of their own LabTeam
            # (same as their name)

            full_name, first, last = decompose_name(experimenter)
            LabTeam.create_new_team(
                team_name=full_name, team_members=[f"{last}, {first}"]
            )

    @classmethod
    def insert_from_name(cls, full_name):
        """Insert a lab member by name.

        Parameters
        ----------
        full_name : str
            The name to be added.
        """
        _, first, last = decompose_name(full_name)
        cls.insert1(
            dict(
                lab_member_name=f"{first} {last}",
                first_name=first,
                last_name=last,
            ),
            skip_duplicates=True,
        )

    def _load_admin(self):
        """Load admin list."""
        self._admin = list(
            (self.LabMemberInfo & {"admin": True}).fetch("datajoint_user_name")
        ) + ["root"]

    @property
    def admin(self) -> list:
        """Return the list of admin users.

        Note: This is cached. If adding a new admin, run _load_admin
        """
        if not self._admin:
            self._load_admin()
        return self._admin

    @property
    def user_is_admin(self) -> bool:
        """Return True if the user is an admin."""
        return dj.config["database.user"] in self.admin

    def get_djuser_name(self, dj_user) -> str:
        """Return the lab member name associated with a datajoint user name.

        Parameters
        ----------
        dj_user: str
            The datajoint user name.

        Returns
        -------
        str
            The lab member name.
        """
        query = (self.LabMemberInfo & {"datajoint_user_name": dj_user}).fetch(
            "lab_member_name"
        )

        if len(query) != 1:
            remedy = f"delete {len(query)-1}" if len(query) > 1 else "add one"
            raise ValueError(
                f"Could not find exactly 1 datajoint user {dj_user}"
                + " in common.LabMember.LabMemberInfo. "
                + f"Please {remedy}: {query}"
            )

        return query[0]

    def check_admin_privilege(
        self,
        error_message: str = "User does not have database admin privileges",
    ):
        """Check if a user has admin privilege.

        Parameters
        ----------
        error_message: str
            The error message to display if the user is not an admin.
        """
        if not self.user_is_admin:
            raise PermissionError(error_message)


@schema
class LabTeam(SpyglassMixin, dj.Manual):
    definition = """
    team_name: varchar(80)
    ---
    team_description = "": varchar(2000)
    """

    class LabTeamMember(SpyglassMixin, dj.Part):
        definition = """
        -> LabTeam
        -> LabMember
        """

    _shared_teams = {}

    def get_team_members(cls, member, reload=False):
        """Return the set of lab members that share a team with member.

        Parameters
        ----------
        member : str
            The lab member to check.
        reload : bool
            If True, reload the shared teams cache.
        """
        if reload or member not in cls._shared_teams:
            teams = (cls.LabTeamMember & {"lab_member_name": member}).fetch(
                "team_name"
            )
            cls._shared_teams[member] = set(
                (
                    LabTeam.LabTeamMember
                    & [{"team_name": team} for team in teams]
                ).fetch("lab_member_name")
            )
        return cls._shared_teams[member]

    @classmethod
    def create_new_team(
        cls, team_name: str, team_members: list, team_description: str = ""
    ):
        """Create a new team with a list of team members.

        If the lab member does not exist in the database, they will be added.

        Parameters
        ----------
        team_name : str
            The name of the team.
        team_members : str
            The full names of the lab members that are part of the team.
        team_description: str
            The description of the team.
        """
        labteam_dict = {
            "team_name": team_name,
            "team_description": team_description,
        }

        member_list = []
        for team_member in team_members:
            LabMember.insert_from_name(team_member)
            member_dict = {"lab_member_name": decompose_name(team_member)[0]}
            query = (LabMember.LabMemberInfo() & member_dict).fetch(
                "google_user_name"
            )
            if not query:
                logger.info(
                    "To help manage permissions in LabMemberInfo, please add "
                    + f"Google user ID for {team_member}"
                )
            labteammember_dict = {
                "team_name": team_name,
                **member_dict,
            }
            member_list.append(labteammember_dict)
            # clear cache for this member
            _ = cls._shared_teams.pop(team_member, None)

        cls.insert1(labteam_dict, skip_duplicates=True)
        cls.LabTeamMember.insert(member_list, skip_duplicates=True)


@schema
class Institution(SpyglassMixin, dj.Manual):
    definition = """
    institution_name: varchar(80)
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert institution information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with institution information.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        institution_name : string
            The name of the institution found in the NWB or config file,
            or None.
        """
        config = config or dict()
        inst_list = config.get("Institution", [{}])
        if len(inst_list) > 1:
            logger.info(
                "Multiple institution entries not allowed. "
                + "Using the first entry only.\n"
            )
        inst_name = inst_list[0].get("institution_name") or getattr(
            nwbf, "institution", None
        )
        if not inst_name:
            logger.info("No institution metadata found.\n")
            return None

        cls.insert1(dict(institution_name=inst_name), skip_duplicates=True)
        return inst_name


@schema
class Lab(SpyglassMixin, dj.Manual):
    definition = """
    lab_name: varchar(80)
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert lab name information from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The NWB file with lab name information.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        lab_name : string
            The name of the lab found in the NWB or config file, or None.
        """
        config = config or dict()
        lab_list = config.get("Lab", [{}])
        if len(lab_list) > 1:
            logger.info(
                "Multiple lab entries not allowed. Using the first entry only."
            )
        lab_name = lab_list[0].get("lab_name") or getattr(nwbf, "lab", None)
        if not lab_name:
            logger.info("No lab metadata found.\n")
            return None

        cls.insert1(dict(lab_name=lab_name), skip_duplicates=True)
        return lab_name


def decompose_name(full_name: str) -> tuple:
    """Return the full, first and last name parts of a full name.

    Names capitalized. Decomposes either comma- or space-separated.
    If name includes spaces, must use exactly one comma.

    Parameters
    ----------
    full_name: str
        The full name to decompose.

    Returns
    -------
    tuple
        A tuple of the full, first, last name parts.

    Raises
    ------
    ValueError
        When full_name is in an unsupported format
    """

    full_trimmed = "" if not full_name else full_name.strip()

    delim = (
        ", "
        if full_trimmed.count(", ") == 1
        else " " if full_trimmed.count(" ") == 1 else None
    )

    parts = full_trimmed.split(delim)

    if delim is None or len(parts) != 2:  # catch unsupported format
        raise ValueError(
            f"Name has unsupported format for {full_name}. \n\t"
            + "Must use exactly one comma+space (i.e., ', ') or space.\n\t"
            + "And provide exactly two name parts.\n\t"
            + "For example, 'First Last' or 'Last1 Last2, First1 First2' "
            + "if name(s) includes spaces."
        )

    first, last = parts if delim == " " else parts[::-1]
    full = f"{first} {last}"

    return full, first, last
