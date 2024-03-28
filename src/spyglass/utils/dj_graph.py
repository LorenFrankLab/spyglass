"""DataJoint graph traversal and restriction application.

NOTE: read `ft` as FreeTable and `restr` as restriction.
"""

from pathlib import Path
from typing import Dict, List

from datajoint import FreeTable
from datajoint import config as dj_config
from datajoint.condition import make_condition
from datajoint.table import Table

from spyglass.settings import export_dir
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import unique_dicts


class RestrGraph:
    def __init__(
        self,
        seed_table: Table,
        table_name: str = None,
        restriction: str = None,
        leaves: List[Dict[str, str]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Use graph to cascade restrictions up from leaves to all ancestors.

        Parameters
        ----------
        seed_table : Table
            Table to use to establish connection and graph
        table_name : str, optional
            Table name of single leaf, default None
        restriction : str, optional
            Restriction to apply to leaf. default None
        leaves : Dict[str, str], optional
            List of dictionaries with keys table_name and restriction. One
            entry per leaf node. Default None.
        verbose : bool, optional
            Whether to print verbose output. Default False
        """

        self.connection = seed_table.connection
        self.graph = seed_table.connection.dependencies
        self.graph.load()

        self.verbose = verbose
        self.cascaded = False
        self.ancestors = set()
        self.visited = set()
        self.leaves = set()

        if table_name and restriction:
            self.add_leaf(table_name, restriction)
        if leaves:
            self.add_leaves(leaves)

    def __repr__(self):
        l_str = ",\n\t".join(self.leaves) + "\n" if self.leaves else ""
        processed = "Processed" if self.cascaded else "Unprocessed"
        return f"{processed} RestrictionGraph(\n\t{l_str})"

    @property
    def all_ft(self):
        """Get restricted FreeTables from all visited nodes."""
        return [self._get_ft(table, with_restr=True) for table in self.visited]

    @property
    def leaf_ft(self):
        """Get restricted FreeTables from graph leaves."""
        return [self._get_ft(table, with_restr=True) for table in self.leaves]

    def _get_node(self, table):
        """Get node from graph."""
        node = self.graph.nodes.get(table)
        if not node:
            raise ValueError(
                f"Table {table} not found in graph."
                + "\n\tPlease import this table and rerun"
            )
        return node

    def _set_node(self, table, attr="ft", value=None):
        """Set attribute on graph node."""
        _ = self._get_node(table)  # Ensure node exists
        self.graph.nodes[table][attr] = value

    def _get_ft(self, table, with_restr=False):
        """Get FreeTable from graph node. If one doesn't exist, create it."""
        restr = self._get_restr(table) if with_restr else True
        if ft := self._get_node(table).get("ft"):
            return ft & restr
        ft = FreeTable(self.connection, table)
        self._set_node(table, "ft", ft)
        return ft & restr

    def _get_restr(self, table):
        """Get restriction from graph node."""
        table = table if isinstance(table, str) else table.full_table_name
        return self._get_node(table).get("restr")

    def _set_restr(self, table, restriction):
        """Add restriction to graph node. If one exists, merge with new."""
        ft = self._get_ft(table)
        restriction = (  # Convert to condition if list or dict
            make_condition(ft, restriction, set())
            if not isinstance(restriction, str)
            else restriction
        )
        if existing := self._get_restr(table):
            if existing == restriction:
                return
            join = ft & [existing, restriction]
            if len(join) == len(ft & existing):
                return  # restriction is a subset of existing
            restriction = make_condition(
                ft, unique_dicts(join.fetch("KEY", as_dict=True)), set()
            )

        if not isinstance(restriction, str):
            self._log_truncate(
                f"Set Restr {table}: {type(restriction)} {restriction}"
            )
        self._set_node(table, "restr", restriction)

    def _log_truncate(self, log_str, max_len=80):
        """Truncate log lines to max_len and print if verbose."""
        if not self.verbose:
            return
        logger.info(
            log_str[:max_len] + "..." if len(log_str) > max_len else log_str
        )

    def _child_to_parent(
        self,
        child,
        parent,
        restriction,
        attr_map=None,
        primary=True,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Given a child, child's restr, and parent, get parent's restr.

        Parameters
        ----------
        child : str
            child table name
        parent : str
            parent table name
        restriction : str
            restriction to apply to child
        attr_map : dict, optional
            dictionary mapping aliases across parend/child, as pulled from
            DataJoint-assembled graph. Default None. Func will flip this dict
            to convert from child to parent fields.
        primary : bool, optional
            Is parent in child's primary key? Default True. Also derived from
            DataJoint-assembled graph. If True, project only primary key fields
            to avoid secondary key collisions.

        Returns
        -------
        List[Dict[str, str]]
            List of dicts containing primary key fields for restricted parent
            table.
        """

        # Need to flip attr_map to respect parent's fields
        attr_map = (
            {v: k for k, v in attr_map.items() if k != k} if attr_map else {}
        )
        child_ft = self._get_ft(child)
        parent_ft = self._get_ft(parent).proj()
        restr = restriction or self._get_restr(child_ft) or True
        restr_child = child_ft & restr

        if primary:  # Project only primary key fields to avoid collisions
            join = restr_child.proj(**attr_map) * parent_ft
        else:  # Include all fields
            join = restr_child.proj(..., **attr_map) * parent_ft

        ret = unique_dicts(join.fetch(*parent_ft.primary_key, as_dict=True))

        if len(ret) == len(parent_ft):
            self._log_truncate(f"NULL rest {parent}")

        return ret

    def cascade1(self, table, restriction):
        """Cascade a restriction up the graph, recursively on parents.

        Parameters
        ----------
        table : str
            table name
        restriction : str
            restriction to apply
        """
        self._set_restr(table, restriction)
        self.visited.add(table)

        for parent, data in self.graph.parents(table).items():
            if parent in self.visited:
                continue

            if parent.isnumeric():
                parent, data = self.graph.parents(parent).popitem()

            parent_restr = self._child_to_parent(
                child=table,
                parent=parent,
                restriction=restriction,
                **data,
            )

            self.cascade1(parent, parent_restr)  # Parent set on recursion

    def cascade(self) -> None:
        """Cascade all restrictions up the graph."""
        if self.cascaded:
            return
        for table in self.leaves - self.visited:
            restr = self._get_restr(table)
            self._log_truncate(f"Start     {table}: {restr}")
            self.cascade1(table, restr)
        if not self.visited == self.ancestors:
            raise RuntimeError("Cascade: FAIL - incomplete cascade")

        self.cascaded = True

    def add_leaf(self, table_name, restriction, cascade=False) -> None:
        """Add leaf to graph and cascade if requested.

        Parameters
        ----------
        table_name : str
            table name of leaf
        restriction : str
            restriction to apply to leaf
        """
        new_ancestors = set(self._get_ft(table_name).ancestors())
        self.ancestors |= new_ancestors  # Add to total ancestors
        self.visited -= new_ancestors  # Remove from visited to revisit

        self.leaves.add(table_name)
        self._set_restr(table_name, restriction)  # Redundant if cascaded

        if cascade:
            self.cascade1(table_name, restriction)
            self.cascaded = True

    def add_leaves(self, leaves: List[Dict[str, str]], cascade=False) -> None:
        """Add leaves to graph and cascade if requested.

        Parameters
        ----------
        leaves : List[Dict[str, str]]
            list of dictionaries containing table_name and restriction
        cascade : bool, optional
            Whether to cascade the restrictions up the graph. Default False
        """

        if not leaves:
            return
        if not isinstance(leaves, list):
            leaves = [leaves]
        leaves = unique_dicts(leaves)
        for leaf in leaves:
            if not (
                (table_name := leaf.get("table_name"))
                and (restriction := leaf.get("restriction"))
            ):
                raise ValueError(
                    f"Leaf must have table_name and restriction: {leaf}"
                )
            self.add_leaf(table_name, restriction, cascade=False)
        if cascade:
            self.cascade()

    @property
    def as_dict(self) -> List[Dict[str, str]]:
        """Return as a list of dictionaries of table_name: restriction"""
        self.cascade()
        return [
            {"table_name": table, "restriction": self._get_restr(table)}
            for table in self.ancestors
            if self._get_restr(table)
        ]

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

    def _write_mysqldump(
        self, paper_id: str, docker_id=None, spyglass_version=None
    ):
        """Write mysqlmdump.sh script to export data.

        Parameters
        ----------
        paper_id : str
            Paper ID to use for export file names
        docker_id : str, optional
            Docker container ID to export from. Default None
        spyglass_version : str, optional
            Spyglass version to include in export. Default None
        """
        paper_dir = Path(export_dir) / paper_id if not docker_id else Path(".")
        paper_dir.mkdir(exist_ok=True)

        dump_script = paper_dir / f"_ExportSQL_{paper_id}.sh"
        dump_content = paper_dir / f"_Populate_{paper_id}.sql"

        prefix = self._cmd_prefix(docker_id)
        version = (  # Include spyglass version as comment in dump
            "echo '--'\n"
            + f"echo '-- SPYGLASS VERSION: {spyglass_version} --'\n"
            + "echo '--'\n\n"
            if spyglass_version
            else ""
        )
        create_cmd = (
            "echo 'CREATE DATABASE IF NOT EXISTS {database}; "
            + "USE {database};'\n\n"
        )
        dump_cmd = prefix + '{database} {table} --where="\\\n\t{where}"\n\n'

        tables_by_db = sorted(self.all_ft, key=lambda x: x.full_table_name)

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
                database, table_name = table.full_table_name.replace(
                    "`", ""
                ).split(".")
                if database != prev_db:
                    file.write(create_cmd.format(database=database))
                    prev_db = database
                file.write(
                    dump_cmd.format(
                        database=database, table=table_name, where=where
                    )
                )
        logger.info(f"Export script written to {dump_script}")

    def write_export(
        self, paper_id: str, docker_id=None, spyglass_version=None
    ):
        """Write export bash script for all tables in graph.

        Also writes a user-specific .my.cnf file to avoid password prompt.

        Parameters
        ----------
        paper_id : str
            Paper ID to use for export file names
        docker_id : str, optional
            Docker container ID to export from. Default None
        spyglass_version : str, optional
            Spyglass version to include in export. Default None
        """
        self.cascade()
        self._write_sql_cnf()
        self._write_mysqldump(paper_id, docker_id, spyglass_version)

        # TODO: export conda env
