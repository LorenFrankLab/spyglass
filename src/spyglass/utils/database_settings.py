#!/usr/bin/env python
import grp
import os
import sys
import tempfile
from pathlib import Path

import datajoint as dj

GRANT_ALL = "GRANT ALL PRIVILEGES ON "
GRANT_SEL = "GRANT SELECT ON "
CREATE_USR = "CREATE USER IF NOT EXISTS "
TEMP_PASS = " IDENTIFIED BY 'temppass';"
ESC = r"\_%"


class DatabaseSettings:
    def __init__(
        self, user_name=None, host_name=None, target_group=None, debug=False
    ):
        """Class to manage common database settings

        Parameters
        ----------
        user_name : str, optional
            The name of the user to add to the database. Default from dj.config
        host_name : str, optional
            The name of the host to add to the database. Default from dj.config
        target_group : str, optional
            Group to which user belongs. Default is kachery-users
        debug : bool, optional
            Default False. If True, print sql instead of running
        """
        self.shared_modules = [
            f"common{ESC}",
            f"spikesorting{ESC}",
            f"decoding{ESC}",
            f"position{ESC}",
            f"position_linearization{ESC}",
            f"ripple{ESC}",
            f"lfp{ESC}",
        ]
        self.user = user_name or dj.config["database.user"]
        self.host = (
            host_name or dj.config["database.host"] or "lmf-db.cin.ucsf.edu"
        )
        self.target_group = target_group or "kachery-users"
        self.debug = debug

    @property
    def _add_collab_usr_sql(self):
        return [
            # Create the user (if not already created) and set the password
            f"{CREATE_USR}'{self.user}'@'%'{TEMP_PASS}\n",
            # Grant privileges to databases matching the user_name pattern
            f"{GRANT_ALL}`{self.user}{ESC}`.* TO '{self.user}'@'%';\n",
            # Grant SELECT privileges on all databases
            f"{GRANT_SEL}`%`.* TO '{self.user}'@'%';\n",
        ]

    def add_collab_user(self):
        """Add collaborator user with full permissions to shared modules"""
        file = self.write_temp_file(self._add_collab_usr_sql)
        self.run_file(file)

    @property
    def _add_dj_guest_sql(self):
        # Note: changing to temppass for uniformity
        return [
            # Create the user (if not already created) and set the password
            f"{CREATE_USR}'{self.user}'@'%'{TEMP_PASS}\n",
            # Grant privileges
            f"{GRANT_SEL}`%`.* TO '{self.user}'@'%';\n",
        ]

    def add_dj_guest(self):
        """Add guest user with select permissions to shared modules"""
        file = self.write_temp_file(self._add_dj_guest_sql)
        self.run_file(file)

    def _find_group(self):
        # find the kachery-users group
        groups = grp.getgrall()
        group_found = False  # initialize the flag as False
        for group in groups:
            if group.gr_name == self.target_group:
                group_found = (
                    True  # set the flag to True when the group is found
                )
                break

        # Check if the group was found
        if not group_found:
            if self.debug:
                print(f"All groups: {[g.gr_name for g in groups]}")
            sys.exit(
                f"Error: The target group {self.target_group} was not found."
            )

        return group

    def _add_module_sql(self, module_name, group):
        return [
            f"{GRANT_ALL}`{module_name}{ESC}`.* TO `{user}`@'%';\n"
            # get a list of usernames
            for user in group.gr_mem
        ]

    def add_module(self, module_name):
        """Add module to database. Grant permissions to all users in group"""
        print(f"Granting everyone permissions to module {module_name}")
        group = self._find_group()
        file = self.write_temp_file(self._add_module_sql(module_name, group))
        self.run_file(file)

    @property
    def _add_dj_user_sql(self):
        return (
            [
                f"{CREATE_USR}'{self.user}'@'%' "
                + "IDENTIFIED BY 'temppass';\n",
                f"{GRANT_ALL}`{self.user}{ESC}`.* TO '{self.user}'@'%';" + "\n",
            ]
            + [
                f"{GRANT_ALL}`{module}`.* TO '{self.user}'@'%';\n"
                for module in self.shared_modules
            ]
            + [f"{GRANT_SEL}`%`.* TO '{self.user}'@'%';\n"]
        )

    def add_dj_user(self, check_exists=True):
        """Add user to database with permissions to shared modules"""
        if check_exists:
            user_home = Path.home().parent / self.user
            if user_home.exists():
                print("Creating database user ", self.user)
            else:
                sys.exit(
                    f"Error: could not find {self.user} in home dir: {user_home}"
                )

        file = self.write_temp_file(self._add_dj_user_sql)
        self.run_file(file)

    def write_temp_file(self, content: list) -> tempfile.NamedTemporaryFile:
        """Write content to a temporary file and return the file object"""
        file = tempfile.NamedTemporaryFile(mode="w")
        for line in content:
            file.write(line)
        file.flush()

        if self.debug:
            from pprint import pprint  # noqa F401

            pprint(file.name)
            pprint(content)

        return file

    def run_file(self, file):
        """Run commands saved to file in sql"""

        if self.debug:
            return

        os.system(f"mysql -p -h {self.host} < {file.name}")
