"""Schema for institution, lab team/name/members. Session-independent."""

from typing import Dict

import datajoint as dj
import pynwb

from spyglass.utils import SpyglassIngestion, SpyglassMixin, logger

from .common_nwbfile import Nwbfile  # noqa: F401

schema = dj.schema("common_lab")


@schema
class LabMember(SpyglassIngestion, dj.Manual):
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
    _expected_duplicates = True

    @property
    def _source_nwb_object_type(self):
        """The NWB object type from which this table can ingest data."""
        return pynwb.NWBFile

    def generate_entries_from_nwb_object(
        self, nwb_obj: pynwb.NWBFile, base_key=None
    ) -> Dict["SpyglassIngestion", list]:
        """Override SpyglassIngestion to make entry for each experimenter."""
        base_key = base_key or dict()
        experimenter_list = nwb_obj.experimenter
        if not experimenter_list:
            logger.info("No experimenter metadata found.\n")
            return dict()

        entries = []
        for experimenter in experimenter_list:
            _, first, last = decompose_name(experimenter)
            entries.append(
                {
                    "lab_member_name": f"{first} {last}",
                    "first_name": first,
                    "last_name": last,
                    **base_key,
                }
            )
        return {self: entries}

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
class LabTeam(SpyglassIngestion, dj.Manual):
    definition = """
    team_name: varchar(80)
    ---
    team_description = "": varchar(2000)
    """

    _expected_duplicates = True

    class LabTeamMember(SpyglassIngestion, dj.Part):
        definition = """
        -> LabTeam
        -> LabMember
        """

    _shared_teams = {}

    @property
    def _source_nwb_object_type(self):
        """The NWB object type from which this table can ingest data."""
        return pynwb.NWBFile

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        base_key = base_key or dict()
        experimenter_list = nwb_obj.experimenter
        if not experimenter_list:
            logger.info("No experimenter metadata found for LabTeam.\n")
            return dict()

        team_entries = []
        team_member_entries = []
        for experimenter in experimenter_list:
            full_name, first, last = decompose_name(experimenter)
            # each person is by default the member of their own LabTeam
            # (same as their name)
            team_i, entry_i = self.create_new_team(
                team_name=full_name,
                team_members=[f"{last}, {first}"],
                team_description="personal team",
                dry_run=True,
            )
            team_entries.append(team_i)
            team_member_entries.extend(entry_i)
        return {self: team_entries, self.LabTeamMember: team_member_entries}

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
        cls,
        team_name: str,
        team_members: list,
        team_description: str = "",
        dry_run: bool = False,
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
        dry_run : bool
            If True, do not insert into the database, just return the
            dictionaries that would be inserted.
        """
        labteam_dict = {
            "team_name": team_name,
            "team_description": team_description,
        }

        member_list = []
        for team_member in team_members:
            if not dry_run:
                LabMember.insert_from_name(team_member)
            member_dict = {"lab_member_name": decompose_name(team_member)[0]}
            query = (LabMember.LabMemberInfo() & member_dict).fetch(
                "google_user_name"
            )
            if not query:
                logger.warning(
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

        if dry_run:
            return labteam_dict, member_list
        cls.insert1(labteam_dict, skip_duplicates=True)
        cls.LabTeamMember.insert(member_list, skip_duplicates=True)


@schema
class Institution(SpyglassIngestion, dj.Manual):
    definition = """
    institution_name: varchar(80)
    """

    _expected_duplicates = True

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"institution_name": "institution"}}

    @property
    def _source_nwb_object_type(self):
        return pynwb.NWBFile

    def _adjust_keys_for_entry(self, keys):
        """Ensure that institution_name is not None."""
        # Prevents attempted insert when nwbfile.institution is None
        return [k for k in keys if k.get("institution_name", None)]


@schema
class Lab(SpyglassIngestion, dj.Manual):
    definition = """
    lab_name: varchar(80)
    """

    _expected_duplicates = True

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"lab_name": "lab"}}

    @property
    def _source_nwb_object_type(self):
        return pynwb.NWBFile

    def insert_from_nwbfile(
        self,
        nwb_file_name: str,
        config: dict = None,
        dry_run=False,
    ):
        """Insert lab name information from an NWB file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file with lab name information.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.
        dry_run : bool
            If True, do not insert into the database, just return the
            dictionaries that would be inserted.

        Returns
        -------
        lab_name : string
            The name of the lab found in the NWB or config file, or None.
        """
        insert_entries = (
            super()
            .insert_from_nwbfile(
                nwb_file_name=nwb_file_name, config=config, dry_run=dry_run
            )
            .get(self)
        )
        if not insert_entries:
            logger.info("No lab metadata found.\n")
            return dict()
        if len(insert_entries) > 1:
            logger.info(
                "Multiple lab entries not allowed. Using the first entry only."
            )
            insert_entries = insert_entries[:1]
        if not dry_run:
            self.insert(insert_entries, skip_duplicates=True)
        return insert_entries


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
