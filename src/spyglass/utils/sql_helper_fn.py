import re
from functools import cached_property
from os import system as os_system
from pathlib import Path
from typing import List

import yaml
from datajoint import FreeTable
from datajoint import config as dj_config


class SQLDumpHelper:
    """Write a series of export files to export_dir/paper_id.

    Includes..
    - .my.cnf file to avoid future password prompt
    - bash script to export data from MySQL database
    - environment.yml file to export conda environment

    Parameters
    ----------
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

    @cached_property
    def _export_dir(self):
        """Lazy load export directory."""
        from spyglass.settings import export_dir

        return export_dir

    @cached_property
    def _logger(self):
        """Lazy load logger."""
        from spyglass.utils import logger

        return logger

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

    def _cmd_prefix(self, docker_id=None):
        """Get prefix for mysqldump command. Includes docker exec if needed."""
        default = "mysqldump --hex-blob "
        if not docker_id:
            return default
        return (
            f"docker exec -i {docker_id} \\\n\t{default}"
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
            Path(self._export_dir) / self.paper_id
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
                where = bash_escape_sql(where)
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

        self._remove_encoding(dump_script)
        self._write_version_file()

        self._logger.info(f"Export script written to {dump_script}")

        self._export_conda_env()

    def _remove_encoding(self, dump_script):
        """Remove encoding from dump_content."""
        charset_sed = r"sed -i 's/ DEFAULT CHARSET=[^ ]\w*//g' "
        charset_sed = r"sed -i 's/ DEFAULT COLLATE [^ ]\w*//g' "
        os_system(f"{charset_sed} {dump_script}")

    def _write_version_file(self):
        """Write spyglass version to paper directory."""
        version_file = (
            Path(self._export_dir) / self.paper_id / "spyglass_version"
        )
        if version_file.exists():
            return
        with version_file.open("w") as file:
            file.write(self.spyglass_version)

    def _export_conda_env(self):
        """Export conda environment to paper directory.

        Renames environment name to paper_id.
        """
        yml_path = Path(self._export_dir) / self.paper_id / "environment.yml"
        if yml_path.exists():
            return
        command = f"conda env export > {yml_path}"
        os_system(command)

        # RENAME ENVIRONMENT NAME TO PAPER ID
        with yml_path.open("r") as file:
            yml = yaml.safe_load(file)
        yml["name"] = self.paper_id
        with yml_path.open("w") as file:
            yaml.dump(yml, file)

        self._logger.info(f"Conda environment exported to {yml_path}")


def remove_redundant(s):
    """Remove redundant parentheses from a string.

    '((a=b)OR((c=d)AND((e=f))))' -> '(a=b) OR ((c=d) AND (e=f))'

    Full solve would require content parsing, this removes duplicates.
    https://codegolf.stackexchange.com/questions/250596/remove-redundant-parentheses
    """

    def is_list(x):  # Check if element is a list
        return isinstance(x, list)

    def list_to_str(x):  # Convert list to string
        return "(%s)" % "".join(map(list_to_str, x)) if is_list(x) else x

    def flatten_list(nested):
        ret = [flatten_list(e) if is_list(e) else e for e in nested if e]
        return ret[0] if ret == [[*ret[0]]] else ret  # first if all same

    tokens = repr("\"'" + s)[3:]  # Quote to safely eval the string
    as_list = tokens.translate({40: "',['", 41: "'],'"})  # parens -> square
    flattened = flatten_list(eval(as_list))  # Flatten the nested list
    as_str = list_to_str(flattened)  # back to str

    # space out AND and OR for readability
    return re.sub(r"\b(and|or)\b", r" \1 ", as_str, flags=re.IGNORECASE)


def bash_escape_sql(s, add_newline=True):
    """Escape restriction string for bash.

    Parameters
    ----------
    s : str
        SQL restriction string
    add_newline : bool, optional
        Add newlines for readability around AND & OR. Default True
    """
    s = s.strip()
    if s.startswith("WHERE"):
        s = s[5:].strip()

    # Balance parentheses - because make_condition may unbalance outside parens
    n_open = s.count("(")
    n_close = s.count(")")
    add_open = max(0, n_close - n_open)
    add_close = max(0, n_open - n_close)
    balanced = "(" * add_open + s + ")" * add_close

    s = remove_redundant(balanced)

    replace_map = {
        "  ": " ",  # Squash double spaces
        "( (": "((",  # Squash double parens
        ") )": "))",
        '"': "'",  # Replace double quotes with single
        "`": "",  # Remove backticks
    }

    if add_newline:
        replace_map.update(
            {
                " AND ": " \\\n\tAND ",  # Add newline and tab for readability
                " OR ": " \\\n\tOR  ",  # OR extra space to align with AND
                ")AND(": ") \\\n\tAND (",
                ")OR(": ") \\\n\tOR  (",
                "#": "\\#",
            }
        )
    else:  # Used in ExportMixin
        replace_map.update({"%%%%": "%%"})  # Remove extra percent signs

    for old, new in replace_map.items():
        s = re.sub(re.escape(old), new, s)

    return s
