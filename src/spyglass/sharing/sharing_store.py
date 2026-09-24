"""Declare which files are shared through the shared-storage broker.

A selection table names what should be shared, a computed table records what
actually was. Raw and analysis are separate pairs, because `Nwbfile` and
`AnalysisNwbfile` have separate primary keys.

**Declaring a share is a database insert; `populate()` is the transfer** —
unless `sg_config.store_auto_upload` is set. Either way a failed upload leaves
the declaration to retry.

**Nothing here decides who may read anything.** These tables record what the
owner declared; the broker verifies ownership and enforces the result. Hence
`update_visibility` relays to the broker rather than trusting the local row.
"""

from typing import List, Optional

import datajoint as dj

from spyglass.common.common_lab import LabTeam
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("sharing_store")

#: Ordered from most to least restrictive. Any operation that combines several
#: parents' visibility takes the minimum, so that an inherited default can
#: only ever narrow access.
SCOPES = ("private", "group", "public")


def most_restrictive(scopes) -> str:
    """Return the narrowest of several visibility scopes.

    Used where a derived file has more than one parent. Taking the minimum
    means a file assembled from a public and a private source is private:
    combining data must never be a way to widen access to any part of it.

    Parameters
    ----------
    scopes : iterable of str
        Scope names. Unknown names are ignored rather than treated as a
        widening, since an unrecognized scope is not evidence of permission.

    Returns
    -------
    str
        The narrowest scope present, or "private" if none were recognized.

    Examples
    --------
    >>> most_restrictive(["public", "group"])
    'group'
    >>> most_restrictive(["public", "public"])
    'public'
    >>> most_restrictive([])
    'private'
    """
    known = [s for s in scopes if s in SCOPES]

    return min(known, key=SCOPES.index) if known else "private"


class _SharedFile:
    """Behavior common to the raw and analysis sharing tables.

    A mixin rather than a DataJoint base class: the two tables differ in their
    primary key and in the `file_class` the broker files them under, and
    DataJoint does not inherit definitions. Everything that does not depend on
    those two facts lives here.

    Attributes
    ----------
    _file_class : str
        What the broker calls this kind of file, "raw" or "analysis".
    _name_attr : str
        The primary-key attribute naming the file.
    _source_table : dj.Table
        Table holding the file, used to resolve an absolute path.
    """

    _file_class = None
    _name_attr = None
    _source_table = None

    @classmethod
    def file_name(cls, key: dict) -> str:
        """Return the Spyglass file name from a key.

        Parameters
        ----------
        key : dict
            A key containing this table's name attribute.

        Returns
        -------
        str
            The file name.
        """
        return key[cls._name_attr]

    @classmethod
    def abs_path(cls, key: dict) -> str:
        """Return the absolute path of the file a key names.

        Parameters
        ----------
        key : dict
            A key containing this table's name attribute.

        Returns
        -------
        str
            Absolute path on local disk.
        """
        return cls._source_table.get_abs_path(cls.file_name(key))


@schema
class SharedFileSelection(SpyglassMixin, dj.Manual):
    """Raw NWB files declared for sharing, and at what visibility.

    Inserting here declares intent. `SharedFile.populate()` is what uploads.
    """

    definition = """
    # Raw NWB files declared for sharing through the broker
    -> Nwbfile
    ---
    scope = 'public': enum('private', 'group', 'public') # Public by default
    """

    class Team(SpyglassMixin, dj.Part):
        definition = """
        # Teams that may read this file, when scope is 'group'
        -> master
        -> LabTeam
        """

    _file_class = "raw"
    _name_attr = "nwb_file_name"
    _source_table = Nwbfile


@schema
class AnalysisFileSelection(SpyglassMixin, dj.Manual):
    """Analysis NWB files declared for sharing, and at what visibility."""

    definition = """
    # Analysis NWB files declared for sharing through the broker
    -> AnalysisNwbfile
    ---
    scope = 'public': enum('private', 'group', 'public') # Public by default
    inherited = 0: bool # Written by inheritance
    """

    class Team(SpyglassMixin, dj.Part):
        definition = """
        # Teams that may read this file, when scope is 'group'
        -> master
        -> LabTeam
        """

    class Parent(SpyglassMixin, dj.Part):
        definition = """
        # Files this declaration was inherited from
        -> master
        parent_name: varchar(64)
        file_class: enum('raw', 'analysis')
        """

    _file_class = "analysis"
    _name_attr = "analysis_file_name"
    _source_table = AnalysisNwbfile


class _UploadMixin(_SharedFile):
    """The transfer half: hash, register, upload, and record the result.

    `make` is identical for raw and analysis files once `_file_class` and
    `_name_attr` are known, so it lives here rather than twice.
    """

    #: Set on the concrete table to its matching selection table.
    _selection = None

    def _declared_visibility(self, key: dict) -> tuple:
        """Return the scope and teams declared for this file.

        Parameters
        ----------
        key : dict
            Selection key.

        Returns
        -------
        tuple of (str, list of str)
            Scope name and team names.

        Raises
        ------
        ValueError
            If the scope is "group" but no team is named. The broker rejects
            that too; catching it here means the error names the row rather
            than arriving as a 422 from a service the user did not call.
        """
        scope = (self._selection & key).fetch1("scope")
        teams = list((self._selection.Team & key).fetch("team_name"))

        if scope == "group" and not teams:
            raise ValueError(
                f"{self.file_name(key)} is declared 'group' but names no "
                + f"team. Insert into {self._selection.__name__}.Team, or "
                + "set scope to 'private'."
            )

        return scope, teams

    # Tripartite `make`, not a monolithic one, because the work between the
    # fetch and the insert is hashing and then uploading a file that is
    # routinely tens of gigabytes. A single `make` runs entirely inside one
    # database transaction, so the MySQL connection would sit idle for the
    # whole transfer and hit `wait_timeout` — failing the populate *after* the
    # object store had accepted the bytes, leaving a registration on the
    # broker with no row here. Splitting it keeps the transaction to the
    # insert alone.

    def make_fetch(self, key):
        """Read the declared visibility and resolve the file's path.

        Every database query belongs here; nothing below touches the database
        until `make_insert`.

        Parameters
        ----------
        key : dict
            Selection key.

        Returns
        -------
        tuple
            `(scope, teams, path)`, passed on to `make_compute`.
        """
        scope, teams = self._declared_visibility(key)

        return (scope, teams, self.abs_path(key))

    def make_compute(self, key, scope, teams, path):
        """Hash the file and upload it. No database access.

        The digest is taken over the file's raw bytes, because that is what
        the object store verifies on arrival. A file whose bytes are already
        stored — by this user or another — is deduplicated to the existing
        object and only the registration is written.

        Parameters
        ----------
        key : dict
            Selection key.
        scope : str
            Declared visibility.
        teams : list of str
            Declared teams.
        path : str
            Absolute path of the file to upload.

        Returns
        -------
        tuple
            `(file_id, sha256, deduplicated, content_md5)`, passed on to
            `make_insert`.

        Raises
        ------
        RuntimeError
            If no broker is configured.
        """
        from spyglass.sharing.store_client import get_client
        from spyglass.utils.nwb_hash import digest_file

        client = get_client()

        if not client.configured:
            raise RuntimeError(
                "No shared-storage broker is configured. Set `store_url` in "
                + "dj_local_conf.json before populating."
            )

        digests = digest_file(
            path, algorithms=client.upload_digests(), show_progress=True
        )

        result = client.upload(
            path,
            spyglass_name=self.file_name(key),
            file_class=self._file_class,
            scope=scope,
            teams=teams,
            sha256=digests["sha256"],
            content_md5=digests.get("md5"),
        )

        return (
            result["file_id"],
            digests["sha256"],
            bool(result.get("deduplicated")),
            digests.get("md5"),
        )

    def make_insert(self, key, file_id, digest, deduplicated, content_md5):
        """Record what the broker accepted.

        Parameters
        ----------
        key : dict
            Selection key.
        file_id : str
            The broker's id for this registration.
        digest : str
            SHA-256 of the file's bytes.
        deduplicated : bool
            True if the object was already stored by someone.
        content_md5 : str
            MD5 declared to the broker. Recorded so a later audit need not
            re-read every file; see the column comment for what it does and
            does not attest.
        """
        self.insert1(
            {
                **key,
                "file_id": file_id,
                "sha256": digest,
                "deduplicated": deduplicated,
                "content_md5": content_md5,
            }
        )

    def update_visibility(
        self, key: dict, scope: str, teams: Optional[List[str]] = None
    ):
        """Change who may read an already-uploaded file.

        Relayed to the broker, which verifies ownership: a teammate who can
        read a file cannot widen access to it. The local declaration is
        updated only after the broker accepts the change, so the tables never
        claim a visibility the broker did not apply.

        Re-scoping a raw follows into the derivatives that inherited from it;
        see `_resync_inherited`.

        Parameters
        ----------
        key : dict
            Key naming exactly one uploaded file.
        scope : str
            "private", "group", or "public".
        teams : list of str, optional
            `LabTeam` names. Required when `scope` is "group".

        Raises
        ------
        ValueError
            If `scope` is unknown, is "group" with no team named, or names a
            team that does not exist.
        StoreForbidden
            If this identity does not own the file.
        """
        from spyglass.sharing.store_client import get_client

        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; pick one of {SCOPES}.")
        if scope == "group" and not teams:
            raise ValueError(
                "A 'group' scope naming no team grants access to nobody. "
                + "Name a team, or use 'private'."
            )

        teams = list(teams or []) if scope == "group" else []

        # Before the broker call: a name the local FK rejects would other-
        # wise fail after the broker had already applied the change.
        unknown = set(teams) - set(LabTeam.fetch("team_name"))
        if unknown:
            raise ValueError(
                f"No such LabTeam: {', '.join(sorted(unknown))}. "
                + "Create the team before sharing to it."
            )

        file_id = (self & key).fetch1("file_id")
        selection_key = (self._selection & key).fetch1("KEY")

        get_client().set_visibility(file_id, scope=scope, teams=teams)

        # Only now is the declaration true. One transaction, so a failure
        # cannot leave a 'group' row naming no team.
        with self.connection.transaction:
            self._selection.update1({**selection_key, "scope": scope})
            (self._selection.Team & selection_key).delete_quick()
            self._selection.Team.insert(
                [{**selection_key, "team_name": t} for t in teams]
            )

        logger.info(f"{self.file_name(selection_key)} is now {scope}.")

        self._resync_inherited(selection_key)

    def _resync_inherited(self, key: dict) -> None:
        """Re-derive the derivatives that inherited from this raw.

        Inheritance copies a parent's scope at registration, so a raw
        re-scoped afterward leaves them at the old audience. Narrows or
        widens; skips rows the user scoped by hand, which `inherited` marks.

        Never raises — the raw's own change is already applied.

        Parameters
        ----------
        key : dict
            Selection key of the raw file whose visibility just changed.
        """
        if self._file_class != "raw":
            return

        derived = AnalysisFileSelection & {"inherited": 1}
        derived &= AnalysisNwbfile & {"nwb_file_name": key[self._name_attr]}

        for name in derived.fetch("analysis_file_name"):
            self._rederive(name)

    def _rederive(self, analysis_file_name: str) -> None:
        """Recompute one derivative's visibility from its recorded parents.

        Its parents, not the raw alone: one that named `share_parents` is as
        narrow as the narrowest of them.

        Parameters
        ----------
        analysis_file_name : str
            The derived file to re-derive.
        """
        key = {"analysis_file_name": analysis_file_name}
        parents = AnalysisFileSelection.Parent & key

        if not parents:  # declared before parents were recorded
            logger.warning(
                f"{analysis_file_name} inherited its visibility before "
                + "parents were recorded, so it cannot be re-derived. Set it "
                + "with `update_visibility`, or redeclare it."
            )
            return

        by_class = {
            kind: list((parents & {"file_class": kind}).fetch("parent_name"))
            for kind in ("raw", "analysis")
        }
        inherited = inherited_visibility(
            raw_files=by_class["raw"], analysis_files=by_class["analysis"]
        )

        if inherited is None:  # a parent is no longer declared at all
            return

        scope, teams = inherited
        current = declared_visibility(analysis_file_name, "analysis")

        if current is None or (scope, set(teams)) == current:
            return

        try:
            if SharedAnalysisFile & key:  # uploaded, so the Broker must agree
                SharedAnalysisFile().update_visibility(
                    key, scope=scope, teams=teams
                )
            else:  # not uploaded; the next populate carries the new scope
                with self.connection.transaction:
                    AnalysisFileSelection.update1({**key, "scope": scope})
                    (AnalysisFileSelection.Team & key).delete_quick()
                    AnalysisFileSelection.Team.insert(
                        [{**key, "team_name": t} for t in teams]
                    )
                logger.info(f"{analysis_file_name} is now {scope}.")
        except Exception as err:  # noqa: BLE001 - see _resync_inherited
            logger.warning(
                f"Could not re-scope {analysis_file_name} to {scope}: {err}. "
                + "It still carries its parent's previous visibility; an "
                + "owner of it can run `update_visibility` by hand."
            )


@schema
class SharedFile(SpyglassMixin, _UploadMixin, dj.Computed):
    """Raw NWB files that have been uploaded to the shared store."""

    definition = """
    # Raw NWB files present in the shared store
    -> SharedFileSelection
    ---
    file_id: varchar(64)       # the broker's id for this registration
    sha256: char(64)           # digest of the file's bytes
    deduplicated = 0: bool     # the object was already stored by someone
    content_md5 = null: char(32)  # declared, NOT proof the bytes were checked
    """

    _file_class = "raw"
    _name_attr = "nwb_file_name"
    _source_table = Nwbfile
    _selection = SharedFileSelection


@schema
class SharedAnalysisFile(SpyglassMixin, _UploadMixin, dj.Computed):
    """Analysis NWB files that have been uploaded to the shared store."""

    definition = """
    # Analysis NWB files present in the shared store
    -> AnalysisFileSelection
    ---
    file_id: varchar(64)       # the broker's id for this registration
    sha256: char(64)           # digest of the file's bytes
    deduplicated = 0: bool     # the object was already stored by someone
    content_md5 = null: char(32)  # declared, NOT proof the bytes were checked
    """

    _file_class = "analysis"
    _name_attr = "analysis_file_name"
    _source_table = AnalysisNwbfile
    _selection = AnalysisFileSelection


def declared_visibility(
    file_name: str, file_class: str = "analysis"
) -> Optional[tuple]:
    """Return the visibility declared for one file, or None.

    None means "never declared", which is different from "declared private".
    Inheritance depends on the distinction: a derived file whose parents were
    never shared should not be queued at all, whereas one whose parent is
    explicitly private is queued as private.

    Parameters
    ----------
    file_name : str
        Spyglass file name.
    file_class : str, optional
        "raw" or "analysis".

    Returns
    -------
    tuple of (str, set of str), or None
        Scope and team names, or None if the file was never declared.
    """
    selection = (
        SharedFileSelection if file_class == "raw" else AnalysisFileSelection
    )
    key = {selection._name_attr: file_name}
    rows = (selection & key).fetch("scope")

    if not len(rows):
        return None

    return rows[0], set((selection.Team & key).fetch("team_name"))


def inherited_visibility(raw_files=(), analysis_files=()) -> Optional[tuple]:
    """Return the narrowest visibility across a set of parent files.

    The rule is intersection, in both dimensions. The scope is the narrowest
    any parent declared, and the teams are the teams *every* group-scoped
    parent named — because a reader who is on only one parent's team can
    reconstruct nothing the derived file does not already expose to them.

    A `public` parent contributes no team restriction, since it restricts
    nobody. A group scope that intersects to no team at all collapses to
    private rather than to a share that names nobody.

    A parent that was *never declared* is narrower than any scope, so one
    undeclared parent makes the answer None. Skipping it instead would let a
    public raw file pull a derived result into the open alongside an analysis
    file its owner never chose to share.

    Parameters
    ----------
    raw_files : iterable of str, optional
        Parent `Nwbfile` names.
    analysis_files : iterable of str, optional
        Parent `AnalysisNwbfile` names.

    Returns
    -------
    tuple of (str, list of str), or None
        Scope and team names, or None if any parent was never declared shared.
    """
    declared = [
        declared_visibility(name, kind)
        for names, kind in ((raw_files, "raw"), (analysis_files, "analysis"))
        for name in names
    ]

    if not declared or any(found is None for found in declared):
        return None

    scope = most_restrictive(s for s, _ in declared)

    if scope != "group":
        return scope, []

    grouped = [teams for s, teams in declared if s == "group"]
    shared_teams = set.intersection(*grouped) if grouped else set()

    if not shared_teams:
        # A group scope naming no team grants access to nobody, and the
        # broker refuses it. Private says the same thing and is honest.
        return "private", []

    return "group", sorted(shared_teams)


def queue_inherited_share(
    analysis_file_name: str, raw_files=(), analysis_files=()
) -> Optional[dict]:
    """Declare a derived file at its parents' visibility.

    Queuing only. The row is a database insert; `SharedAnalysisFile.populate()`
    is what transfers the bytes. Uploading as a side effect of creation is
    `AnalysisNwbfileBuilder`'s business, under `sg_config.store_auto_upload`,
    and it drives it from the return value below.

    Does nothing when no parent was ever shared. That is the whole safety
    property: a default that queued anything would be a default that widened
    access to a file nobody asked to share.

    Parameters
    ----------
    analysis_file_name : str
        The derived file, already registered in `AnalysisNwbfile`.
    raw_files : iterable of str, optional
        Parent `Nwbfile` names.
    analysis_files : iterable of str, optional
        Parent `AnalysisNwbfile` names.

    Returns
    -------
    dict or None
        The declaration that was written, or None if nothing was queued.
    """
    key = {"analysis_file_name": analysis_file_name}

    if AnalysisFileSelection & key:
        return None  # already declared; inheritance never overrides a choice

    inherited = inherited_visibility(raw_files, analysis_files)

    if inherited is None:
        return None

    scope, teams = inherited

    AnalysisFileSelection.insert1({**key, "scope": scope, "inherited": 1})
    AnalysisFileSelection.Team.insert([{**key, "team_name": t} for t in teams])
    AnalysisFileSelection.Parent.insert(
        [
            {**key, "parent_name": name, "file_class": kind}
            for names, kind in (
                (raw_files, "raw"),
                (analysis_files, "analysis"),
            )
            for name in names
        ]
    )

    logger.info(
        f"Queued {analysis_file_name} for sharing as {scope}"
        + (f" with {', '.join(teams)}" if teams else "")
        + ". Run SharedAnalysisFile.populate() to upload."
    )

    return {**key, "scope": scope, "teams": teams}


def share_file(
    file_name: str,
    scope: str = "public",
    teams: Optional[List[str]] = None,
    file_class: str = "analysis",
    populate: bool = True,
):
    """Declare one file for sharing, and upload it.

    A convenience over the two-step insert, for the common case. The
    equivalent of `share_data_to_kachery`, minus the zone.

    Parameters
    ----------
    file_name : str
        Spyglass file name.
    scope : str, optional
        "private", "group", or "public". Defaults to "public", matching the
        selection tables: declaring a share is an explicit act, and the point
        of it is to be read. Pass "private" or "group" to narrow.
    teams : list of str, optional
        `LabTeam` names. Required when `scope` is "group".
    file_class : str, optional
        "raw" or "analysis". Chooses which pair of tables to use.
    populate : bool, optional
        Transfer immediately. False declares the share and leaves the upload
        for a later `populate()` — useful when queueing many large files.

    Calling this again for the same file replaces the declaration, which is
    how a share declared too widely gets narrowed *before* it is uploaded.
    Once the file is in the store, use `update_visibility` instead: only that
    relays the change to the broker, and only the broker's copy is what any
    reader is actually checked against.

    Returns
    -------
    dict
        The selection key that was written.

    Raises
    ------
    ValueError
        If `scope` or `file_class` is unknown, or a group scope names no team.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; pick one of {SCOPES}.")
    if scope == "group" and not teams:
        raise ValueError(
            "A 'group' scope naming no team grants access to nobody. "
            + "Name a team, or use 'private'."
        )
    if file_class not in ("raw", "analysis"):
        raise ValueError(f"Unknown file_class {file_class!r}.")

    is_raw = file_class == "raw"
    selection = SharedFileSelection if is_raw else AnalysisFileSelection
    shared = SharedFile if is_raw else SharedAnalysisFile

    key = {selection._name_attr: file_name}

    # Not `insert1(..., skip_duplicates=True)`. A second call is how a user
    # *narrows* a share they declared too widely, and skipping the duplicate
    # would leave the old scope and the old teams in place while returning as
    # though it had worked.
    if selection & key:
        selection.update1({**key, "scope": scope})
    else:
        selection.insert1({**key, "scope": scope})

    # Teams are replaced, not added to, and only a group scope keeps any:
    # a row left behind from an earlier group declaration would take effect
    # again the moment the scope was flipped back.
    (selection.Team & key).delete_quick()
    if scope == "group":
        selection.Team.insert([{**key, "team_name": t} for t in teams])

    if populate:
        shared.populate(key)

    return key
