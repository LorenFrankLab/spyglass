#!/usr/bin/env python

import os
import sys
import tempfile
from functools import cached_property
from pathlib import Path

import datajoint as dj

from spyglass.utils.logging import logger

SHARED_MODULES = [
    "behavior",
    "common",
    "decoding",
    "lfp",
    "linearization",
    "mua",
    "position",
    "ripple",
    "sharing",
    "spikesorting",
    # EDIT: waveform not used as a spyglass schema prefix
]
GRANT_ALL = "GRANT ALL PRIVILEGES ON "
GRANT_SEL = "GRANT SELECT ON "
GRANT_SHOW = "GRANT SHOW DATABASES ON "
CREATE_USR = "CREATE USER IF NOT EXISTS "
CREATE_ROLE = "CREATE ROLE IF NOT EXISTS "
TEMP_PASS = " IDENTIFIED BY 'temppass';"
ESC = r"\_%"


class DatabaseSettings:
    def __init__(
        self,
        user_name=None,
        host_name=None,
        debug=False,
        target_database=None,
        exec_user=None,
        exec_pass=None,
        test_mode=False,
    ):
        """Class to manage common database settings

        Roles:
        - dj_guest:  select for all prefix
        - dj_collab: select for all prefix, all for user prefix
        - dj_user:   select for all prefix, all for user prefix, all for shared
        - dj_admin:     all for all prefix

        Note: To add dj_user role to all those with common access, run...

        ```python
        query = "SELECT user, host FROM mysql.db WHERE Db LIKE 'common%';"
        users = dj.conn().query(query).fetchall()
        for user in users:
            dj.conn().query(f"GRANT dj_user TO '{user[0][0]}'@'%';")
        ```

        Parameters
        ----------
        user_name : str, optional
            The name of the user to add to the database. Default from dj.config
        host_name : str, optional
            The name of the host to add to the database. Default from dj.config
        debug : bool, optional
            Default False. If True, pprint sql instead of running
        target_database : str, optional
            Default is mysql. Can also be docker container id
        exec_user : str, optional
            User for executing commands. If None, use dj.config
        exec_pass : str, optional
            Password for executing commands. If None, use dj.config
        test_mode : bool, optional
            Default False. If True, prepend sudo to commands for use in CI/CD
            Only true in github actions, not true in local testing.
        """
        self.shared_modules = [f"{m}{ESC}" for m in SHARED_MODULES]
        self.user = user_name or dj.config["database.user"]
        self.host = (
            host_name or dj.config["database.host"] or "lmf-db.cin.ucsf.edu"
        )
        self.debug = debug
        self.target_database = target_database or "mysql"
        self.exec_user = exec_user or dj.config["database.user"]
        self.exec_pass = exec_pass or dj.config["database.password"]
        self.test_mode = test_mode

    @property
    def _create_roles_dict(self):
        return dict(
            guest=[
                f"{CREATE_ROLE}`dj_guest`;\n",
                f"{GRANT_SEL}`%`.* TO `dj_guest`;\n",
            ],
            collab=[
                f"{CREATE_ROLE}`dj_collab`;\n",
                f"{GRANT_SEL}`%`.* TO `dj_collab`;\n",
            ],  # also gets own prefix below
            user=[
                f"{CREATE_ROLE}`dj_user`;\n",
                f"{GRANT_SEL}`%`.* TO `dj_user`;\n",
            ]
            + [
                f"{GRANT_ALL}`{module}`.* TO `dj_user`;\n"
                for module in self.shared_modules
            ],  # also gets own prefix below
            admin=[
                f"{CREATE_ROLE}`dj_admin`;\n",
                f"{GRANT_ALL}`%`.* TO `dj_admin`;\n",
            ],
        )

    @cached_property
    def _create_roles_sql(self):
        return sum(self._create_roles_dict.values(), [])

    def _create_user_sql(self, role):
        """Create user and grant role"""
        return [
            f"{CREATE_USR}'{self.user}'@'%'{TEMP_PASS}\n",  # create user
            f"GRANT {role} TO '{self.user}'@'%';\n",  # grant role
        ]

    @property
    def _user_prefix_sql(self):
        """Grant user all permissions for user prefix"""
        return [
            f"{GRANT_ALL}`{self.user}{ESC}`.* TO '{self.user}'@'%';\n",
        ]

    @property
    def _add_guest_sql(self):
        return self._create_user_sql("dj_guest")

    @property
    def _add_collab_sql(self):
        return self._create_user_sql("dj_collab") + self._user_prefix_sql

    @property
    def _add_user_sql(self):
        return self._create_user_sql("dj_user") + self._user_prefix_sql

    @property
    def _add_admin_sql(self):
        return self._create_user_sql("dj_admin") + self._user_prefix_sql

    def _add_module_sql(self, module_name):
        return [f"{GRANT_ALL}`{module_name}{ESC}`.* TO dj_user;\n"]

    def add_guest(self, *args, **kwargs):
        """Add guest user with select permissions to shared modules"""
        file = self.write_temp_file(self._add_guest_sql)
        self.exec(file)

    def add_collab(self, *args, **kwargs):
        """Add collaborator user with full permissions to shared modules"""
        file = self.write_temp_file(self._add_collab_sql)
        self.exec(file)

    def add_user(self, check_exists=False, *args, **kwargs):
        """Add user to database with permissions to shared modules"""
        if check_exists:
            self.check_user_exists()
        file = self.write_temp_file(self._add_user_sql)
        self.exec(file)

    def add_admin(self, *args, **kwargs):
        """Add admin user with full permissions to all modules"""
        file = self.write_temp_file(self._add_admin_sql)
        self.exec(file)

    def add_module(self, module_name):
        """Add module to database. Grant permissions to all users in group"""
        logger.info(f"Granting everyone permissions to module {module_name}")
        file = self.write_temp_file(self._add_module_sql(module_name))
        self.exec(file)

    def check_user_exists(self):
        """Add user to database with permissions to shared modules"""
        user_home = Path.home().parent / self.user
        if user_home.exists():
            logger.info("Creating database user ", self.user)
        else:
            sys.exit(
                f"Error: couldn't find {self.user} in home dir: {user_home}"
            )

    def add_user_by_role(self, role, check_exists=False):
        """Add a user to the database with the specified role"""
        add_func = {
            "guest": self.add_guest,
            "user": self.add_user,
            "collab": self.add_collab,
            "admin": self.add_admin,
        }
        if role not in add_func:
            raise ValueError(f"Role {role} not recognized")
        add_func[role]()

    def add_roles(self):
        """Add roles to database"""
        file = self.write_temp_file(self._create_roles_sql)
        self.exec(file)

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

    def exec(self, file):
        """Run commands saved to file in sql"""

        if self.debug:
            return

        if self.test_mode:
            prefix = "sudo mysql -h 127.0.0.1 -P 3308 -uroot -ptutorial"
        else:
            prefix = f"mysql -h {self.host} -u {self.exec_user} -p"

        cmd = (
            f"{prefix} < {file.name}"
            if self.target_database == "mysql"
            else f"docker exec -i {self.target_database} mysql -u "
            + f"{self.exec_user} --password={self.exec_pass} < {file.name}"
        )

        os.system(cmd)
