from os import system as os_system
from pathlib import Path
from typing import List

from datajoint import FreeTable
from datajoint import config as dj_config

from spyglass.settings import export_dir
from spyglass.utils import logger


class SQLDumpHelper:
    """Write a series of export files to export_dir/paper_id.

    Includes..
    - .my.cnf file to avoid future password prompt
    - bash script to export data from MySQL database
    - environment.yml file to export conda environment

    Parameters
    ----------
    free_tables : List[FreeTable]
        List of FreeTables to export
    paper_id : str
        Paper ID to use for export file names
    docker_id : str, optional
        Docker container ID to export from. Default None
    spyglass_version : str, optional
        Spyglass version to include in export. Default None
    """

    def __init__(
        self,
        paper_id: str,
        docker_id=None,
        spyglass_version=None,
    ):
        self.paper_id = paper_id
        self.docker_id = docker_id
        self.spyglass_version = spyglass_version

    def _get_credentials(self):
        """Get credentials for database connection."""
        return {
            "user": dj_config["database.user"],
            "password": dj_config["database.password"],
            "host": dj_config["database.host"],
        }

    def _write_sql_cnf(self):
        """Write SQL cnf file to avoid password prompt."""
        cnf_path = Path("~/.my.cnf").expanduser()

        if cnf_path.exists():
            return

        template = "[client]\nuser={user}\npassword={password}\nhost={host}\n"

        with open(str(cnf_path), "w") as file:
            file.write(template.format(**self._get_credentials()))
        cnf_path.chmod(0o600)

    def _bash_escape(self, s):
        """Escape restriction string for bash."""
        s = s.strip()

        replace_map = {
            "WHERE ": "",  # Remove preceding WHERE of dj.where_clause
            "  ": " ",  # Squash double spaces
            "( (": "((",  # Squash double parens
            ") )": ")",
            '"': "'",  # Replace double quotes with single
            "`": "",  # Remove backticks
            " AND ": " \\\n\tAND ",  # Add newline and tab for readability
            " OR ": " \\\n\tOR  ",  # OR extra space to align with AND
            ")AND(": ") \\\n\tAND (",
            ")OR(": ") \\\n\tOR  (",
            "#": "\\#",
        }
        for old, new in replace_map.items():
            s = s.replace(old, new)
        if s.startswith("(((") and s.endswith(")))"):
            s = s[2:-2]  # Remove extra parens for readability
        return s

    def _cmd_prefix(self, docker_id=None):
        """Get prefix for mysqldump command. Includes docker exec if needed."""
        if not docker_id:
            return "mysqldump "
        return (
            f"docker exec -i {docker_id} \\\n\tmysqldump "
            + "-u {user} --password={password} \\\n\t".format(
                **self._get_credentials()
            )
        )

    def write_mysqldump(
        self,
        free_tables: List[FreeTable],
        file_suffix: str = "",
    ):
        """Write mysqlmdump.sh script to export data.

        Parameters
        ----------
        free_tables : List[FreeTable]
            List of FreeTables to export
        file_suffix : str, optional
            Suffix to append to export file names. Default ""
        """
        self._write_sql_cnf()

        paper_dir = (
            Path(export_dir) / self.paper_id
            if not self.docker_id
            else Path(".")
        )
        paper_dir.mkdir(exist_ok=True)

        dump_script = paper_dir / f"_ExportSQL_{self.paper_id}{file_suffix}.sh"
        dump_content = paper_dir / f"_Populate_{self.paper_id}{file_suffix}.sql"

        prefix = self._cmd_prefix(self.docker_id)
        version = (  # Include spyglass version as comment in dump
            "echo '--'\n"
            + f"echo '-- SPYGLASS VERSION: {self.spyglass_version} --'\n"
            + "echo '--'\n\n"
            if self.spyglass_version
            else ""
        )
        create_cmd = (
            "echo 'CREATE DATABASE IF NOT EXISTS {database}; "
            + "USE {database};'\n\n"
        )
        dump_cmd = prefix + '{database} {table} --where="\\\n\t{where}"\n\n'

        tables_by_db = sorted(free_tables, key=lambda x: x.full_table_name)

        with open(dump_script, "w") as file:
            file.write(
                "#!/bin/bash\n\n"
                + f"exec > {dump_content}\n\n"  # Redirect output to sql file
                + f"{version}"  # Include spyglass version as comment
            )

            prev_db = None
            for table in tables_by_db:
                if not (where := table.where_clause()):
                    continue
                where = self._bash_escape(where)
                database, table_name = (
                    table.full_table_name.replace("`", "")
                    .replace("#", "\\#")
                    .split(".")
                )
                if database != prev_db:
                    file.write(create_cmd.format(database=database))
                    prev_db = database
                file.write(
                    dump_cmd.format(
                        database=database, table=table_name, where=where
                    )
                )
        logger.info(f"Export script written to {dump_script}")

        self._export_conda_env()

    def _export_conda_env(self):
        """Export conda environment to paper directory."""
        yml_path = Path(export_dir) / self.paper_id / "environment.yml"
        if yml_path.exists():
            return
        command = f"conda env export > {yml_path}"
        os_system(command)
        logger.info(f"Conda environment exported to {yml_path}")
