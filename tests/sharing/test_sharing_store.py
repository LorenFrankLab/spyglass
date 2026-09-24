"""Tests for the shared-store sharing schema.

The broker is faked throughout. What these tests are about is the boundary
Spyglass owns: that declaring a share is only a database insert, that a group
scope naming no team is refused rather than quietly meaning private, and that
the local declaration is never updated ahead of the broker accepting it.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def store(common):
    """The sharing_store module, imported after the schema is available."""
    from spyglass.sharing import sharing_store

    return sharing_store


@pytest.fixture
def fake_client():
    """A broker client that records calls instead of making them."""
    calls = []

    def _upload(path, **kwargs):
        calls.append(("upload", path, kwargs))
        return {"file_id": "f1", "deduplicated": False}

    def _set_visibility(file_id, **kwargs):
        calls.append(("visibility", file_id, kwargs))
        return {"file_id": file_id, **kwargs}

    client = SimpleNamespace(
        configured=True,
        logged_in=True,
        calls=calls,
        upload=_upload,
        set_visibility=_set_visibility,
        # What a broker on a store that ignores the sha256 header asks for.
        upload_digests=lambda: ["md5", "sha256"],
    )

    with patch("spyglass.sharing.store_client.get_client", return_value=client):
        yield client


@pytest.fixture
def broker_configured():
    """Point the session at a broker, so the builder queues inherited shares.

    Without this the builder returns before importing `sharing_store` at all,
    which is the behavior an instance with no broker should have.
    """
    from spyglass.settings import sg_config

    prior = sg_config.store_url
    sg_config.store_url = "https://store.example.org"

    yield

    sg_config.store_url = prior


@pytest.fixture
def declared(store, mini_copy_name):
    """A raw file declared private, cleaned up afterward."""
    key = {"nwb_file_name": mini_copy_name}
    store.SharedFileSelection.insert1({**key, "scope": "private"})

    yield key

    (store.SharedFileSelection & key).delete(
        safemode=False, force_permission=True
    )


@pytest.fixture
def build_analysis(store, mini_copy_name, common):
    """Build analysis files, and remove both row and file afterward.

    The file on disk has to go too. Leaving it behind makes it an orphan that
    `AnalysisNwbfile.cleanup` will find, which fails an unrelated test in
    another suite rather than this one.
    """
    made = []

    def _build():
        with common.AnalysisNwbfile().build(mini_copy_name) as builder:
            name = builder.analysis_file_name
        made.append(name)
        return name

    yield _build

    for name in made:
        key = {"analysis_file_name": name}
        path = Path(common.AnalysisNwbfile.get_abs_path(name))
        (store.AnalysisFileSelection & key).delete(
            safemode=False, force_permission=True
        )
        (common.AnalysisNwbfile & key).delete(
            safemode=False, force_permission=True
        )
        path.unlink(missing_ok=True)


# ------------------------------ visibility ------------------------------


def test_most_restrictive_never_widens(store):
    """A derived file is as private as its most private parent."""
    assert store.most_restrictive(["public", "private"]) == "private"
    assert store.most_restrictive(["public", "group"]) == "group"
    assert store.most_restrictive(["public", "public"]) == "public"


def test_unknown_scope_is_not_evidence_of_permission(store):
    """An unrecognized scope falls back to private, not to public."""
    assert store.most_restrictive(["nonsense"]) == "private"
    assert store.most_restrictive([]) == "private"


# ------------------------------ declaring -------------------------------


def test_declaring_a_share_transfers_nothing(store, declared, fake_client):
    """The insert is the declaration; populate is the transfer."""
    assert fake_client.calls == []
    assert len(store.SharedFile & declared) == 0


def test_group_with_no_team_is_refused(store, declared, fake_client):
    """A group scope naming nobody is an error, not a silent private."""
    store.SharedFileSelection.update1({**declared, "scope": "group"})

    with pytest.raises(ValueError, match="names no team"):
        store.SharedFile()._declared_visibility(declared)


def test_share_file_rejects_a_group_with_no_team(store):
    """The convenience helper refuses before writing anything."""
    with pytest.raises(ValueError, match="grants access to nobody"):
        store.share_file("x.nwb", scope="group")


def test_share_file_rejects_an_unknown_scope(store):
    """Scope names are closed; a typo must not become a permission."""
    with pytest.raises(ValueError, match="Unknown scope"):
        store.share_file("x.nwb", scope="everyone")


# ------------------------------- upload ---------------------------------


def test_populate_uploads_and_records(store, declared, fake_client):
    """populate hashes the bytes, uploads, and records the broker's id."""
    store.SharedFile.populate(declared)

    kind, path, kwargs = fake_client.calls[0]
    assert kind == "upload"
    assert kwargs["file_class"] == "raw"
    assert kwargs["scope"] == "private"
    assert len(kwargs["sha256"]) == 64  # a byte digest, not an NwbfileHasher

    row = (store.SharedFile & declared).fetch1()
    assert row["file_id"] == "f1"
    assert row["sha256"] == kwargs["sha256"]


def test_populate_without_a_broker_says_so(store, declared):
    """A missing store_url names itself rather than failing in transport."""
    unconfigured = SimpleNamespace(configured=False)
    table = store.SharedFile()
    fetched = table.make_fetch(declared)

    with patch(
        "spyglass.sharing.store_client.get_client", return_value=unconfigured
    ):
        with pytest.raises(RuntimeError, match="No shared-storage broker"):
            table.make_compute(declared, *fetched)


def test_hash_and_upload_hold_no_transaction(store, declared, fake_client):
    """The slow half of populate must not sit inside a database transaction.

    Hashing and uploading a multi-gigabyte session would otherwise hold the
    MySQL connection idle past `wait_timeout`, failing the populate *after*
    the object store accepted the bytes. The tripartite split is what keeps
    the transaction down to the insert.
    """
    from datajoint.autopopulate import AutoPopulate

    table = store.SharedFile()

    # DataJoint's own `make` dispatches to the three below. Overriding it is
    # what would put the transfer back inside the transaction.
    assert type(table).make is AutoPopulate.make
    for step in ("make_fetch", "make_compute", "make_insert"):
        assert callable(getattr(table, step))

    in_transaction = []

    def _upload(path, **kwargs):
        in_transaction.append(table.connection.in_transaction)
        return {"file_id": "f1", "deduplicated": False}

    fake_client.upload = _upload
    store.SharedFile.populate(declared)

    assert in_transaction == [False]


# ---------------------------- changing scope ----------------------------


def test_update_visibility_relays_to_the_broker(store, declared, fake_client):
    """The broker decides; the local row records what it accepted."""
    store.SharedFile.populate(declared)

    store.SharedFile().update_visibility(declared, scope="public")

    kind, file_id, kwargs = fake_client.calls[-1]
    assert (kind, file_id, kwargs["scope"]) == ("visibility", "f1", "public")
    assert (store.SharedFileSelection & declared).fetch1("scope") == "public"


def test_a_refused_change_leaves_the_declaration_alone(
    store, declared, fake_client
):
    """The tables must never claim a visibility the broker did not apply."""
    from spyglass.sharing.store_client import StoreForbidden

    store.SharedFile.populate(declared)

    def _refuse(file_id, **kwargs):
        raise StoreForbidden("Only the owner may change visibility.")

    fake_client.set_visibility = _refuse

    with pytest.raises(StoreForbidden):
        store.SharedFile().update_visibility(declared, scope="public")

    assert (store.SharedFileSelection & declared).fetch1("scope") == "private"


def test_update_visibility_rejects_a_group_with_no_team(
    store, declared, fake_client
):
    """Refused locally, before a round trip the broker would also refuse."""
    store.SharedFile.populate(declared)

    with pytest.raises(ValueError, match="grants access to nobody"):
        store.SharedFile().update_visibility(declared, scope="group")


# ----------------------------- inheritance ------------------------------


def test_visibility_of_an_undeclared_file_is_none(store, mini_copy_name):
    """Never declared is not the same as declared private."""
    assert store.declared_visibility(mini_copy_name, "raw") is None


def test_inheritance_is_none_when_no_parent_is_shared(store, mini_copy_name):
    """A derived file of unshared parents is not queued at all."""
    assert store.inherited_visibility(raw_files=[mini_copy_name]) is None


def test_inheritance_takes_the_narrowest_scope(store, declared, mini_copy_name):
    """A public parent cannot widen a private one."""
    with patch.object(
        store,
        "declared_visibility",
        side_effect=[("public", set()), ("private", set())],
    ):
        assert store.inherited_visibility(
            raw_files=["a"], analysis_files=["b"]
        ) == ("private", [])


def test_group_teams_intersect(store):
    """A reader must be on every group parent's team, not just one."""
    with patch.object(
        store,
        "declared_visibility",
        side_effect=[("group", {"A", "B"}), ("group", {"B", "C"})],
    ):
        assert store.inherited_visibility(
            raw_files=["a"], analysis_files=["b"]
        ) == ("group", ["B"])


def test_a_public_parent_imposes_no_team(store):
    """Public restricts nobody, so it narrows the scope but not the teams."""
    with patch.object(
        store,
        "declared_visibility",
        side_effect=[("public", set()), ("group", {"A"})],
    ):
        assert store.inherited_visibility(
            raw_files=["a"], analysis_files=["b"]
        ) == ("group", ["A"])


def test_disjoint_teams_collapse_to_private(store):
    """A group share naming nobody is private, not a share the broker refuses."""
    with patch.object(
        store,
        "declared_visibility",
        side_effect=[("group", {"A"}), ("group", {"B"})],
    ):
        assert store.inherited_visibility(
            raw_files=["a"], analysis_files=["b"]
        ) == ("private", [])


def test_queue_inherited_share_is_a_no_op_without_a_shared_parent(
    store, mini_copy_name, build_analysis
):
    """Registering a derived file of unshared parents queues nothing."""
    analysis = build_analysis()

    assert (
        store.queue_inherited_share(analysis, raw_files=[mini_copy_name])
        is None
    )
    assert (
        len(store.AnalysisFileSelection & {"analysis_file_name": analysis}) == 0
    )


def test_the_builder_queues_an_inherited_share(
    store, declared, broker_configured, build_analysis
):
    """Sharing a derived file takes no action beyond sharing its parent.

    The parent is declared private by the `declared` fixture, so registering a
    file built from it should queue that same scope with no user involvement —
    and should queue only, leaving the upload to a later populate.
    """
    key = {"analysis_file_name": build_analysis()}

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "private"
    assert len(store.SharedAnalysisFile & key) == 0  # queued, not uploaded


def test_the_builder_queues_nothing_without_a_shared_parent(
    store, broker_configured, build_analysis
):
    """No default may widen access to a file nobody asked to share."""
    key = {"analysis_file_name": build_analysis()}

    assert len(store.AnalysisFileSelection & key) == 0


def test_inheritance_never_overrides_an_explicit_choice(
    store, declared, mini_copy_name, broker_configured, build_analysis
):
    """A share the user declared by hand is left as they set it."""
    analysis = build_analysis()
    key = {"analysis_file_name": analysis}
    store.AnalysisFileSelection.update1({**key, "scope": "public"})

    assert (
        store.queue_inherited_share(analysis, raw_files=[mini_copy_name])
        is None
    )
    assert (store.AnalysisFileSelection & key).fetch1("scope") == "public"


def test_no_broker_means_the_builder_touches_nothing(mini_copy_name, common):
    """An instance with no broker must not grow sharing tables as a side effect.

    Importing `sharing_store` declares its schema, so the builder checks
    `store_url` before importing anything at all. Most instances are attached
    to no broker and should see no trace of one.
    """
    from spyglass.settings import sg_config

    prior = sg_config.store_url
    sg_config.store_url = ""

    with patch(
        "spyglass.sharing.sharing_store.queue_inherited_share"
    ) as queued:
        try:
            with common.AnalysisNwbfile().build(mini_copy_name) as builder:
                analysis = builder.analysis_file_name
        finally:
            sg_config.store_url = prior

    try:
        queued.assert_not_called()
    finally:
        (common.AnalysisNwbfile & {"analysis_file_name": analysis}).delete(
            safemode=False, force_permission=True
        )


# -------------------------- narrowing an existing --------------------------


def test_share_file_narrows_an_existing_declaration(
    store, mini_copy_name, common
):
    """A second call is how a too-wide share gets locked down.

    `skip_duplicates` would leave the old scope and the old teams in place
    while returning as though the narrowing had worked.
    """
    key = {"nwb_file_name": mini_copy_name}
    existing = common.LabTeam.fetch("team_name")
    team = existing[0] if len(existing) else "My Team"
    if not len(existing):
        common.LabTeam.insert1({"team_name": team}, skip_duplicates=True)

    try:
        store.share_file(
            mini_copy_name,
            scope="group",
            teams=[team],
            file_class="raw",
            populate=False,
        )
        store.share_file(
            mini_copy_name, scope="private", file_class="raw", populate=False
        )

        assert (store.SharedFileSelection & key).fetch1("scope") == "private"
        # A stale team row would take effect again on a flip back to group.
        assert len(store.SharedFileSelection.Team & key) == 0
    finally:
        (store.SharedFileSelection & key).delete(
            safemode=False, force_permission=True
        )


def test_update_visibility_rejects_an_unknown_team(
    store, declared, fake_client
):
    """Checked before the broker call, so the two cannot end up disagreeing."""
    store.SharedFile.populate(declared)
    before = len(fake_client.calls)

    with pytest.raises(ValueError, match="No such LabTeam"):
        store.SharedFile().update_visibility(
            declared, scope="group", teams=["Team That Does Not Exist"]
        )

    assert len(fake_client.calls) == before  # broker never asked
    assert (store.SharedFileSelection & declared).fetch1("scope") == "private"


def test_an_undeclared_parent_blocks_inheritance(store):
    """Never-declared is narrower than private, so it cannot be skipped."""
    with patch.object(
        store, "declared_visibility", side_effect=[("public", set()), None]
    ):
        assert (
            store.inherited_visibility(raw_files=["a"], analysis_files=["b"])
            is None
        )


def test_an_upload_makes_the_name_resolvable_by_hash(
    store, declared, fake_client
):
    """What populate records is what settles an ambiguous name later.

    `SharedFileSelection` is keyed on the file name, so one instance maps a
    name to exactly one digest. `StoreBackend` resolves by that digest rather
    than by the name the broker indexes per owner.
    """
    from spyglass.utils.file_backends import StoreBackend

    store.SharedFile.populate(declared)

    recorded = (store.SharedFile & declared).fetch1("sha256")

    assert StoreBackend()._known_hash(declared["nwb_file_name"]) == recorded


# ----------------------------- auto upload ------------------------------


@pytest.fixture
def auto_upload():
    """Turn on transfer-as-you-declare for one test, then restore it."""
    from spyglass.settings import sg_config

    prior = sg_config.store_auto_upload
    sg_config.store_auto_upload = True

    yield

    sg_config.store_auto_upload = prior


def test_auto_upload_is_off_by_default(
    store, declared, build_analysis, broker_configured, fake_client
):
    """The declaration is still queued, but no bytes leave the host."""
    analysis = build_analysis()

    assert store.AnalysisFileSelection & {"analysis_file_name": analysis}
    assert fake_client.calls == [], "Registered a file and uploaded it"


def test_auto_upload_transfers_on_creation(
    store, declared, build_analysis, broker_configured, auto_upload, fake_client
):
    """With the flag set, registering a derivative also transfers it."""
    analysis = build_analysis()
    key = {"analysis_file_name": analysis}

    kinds = [call[0] for call in fake_client.calls]

    assert kinds == ["upload"], f"Expected one upload, got {kinds}"
    assert (store.SharedAnalysisFile & key).fetch1("file_id") == "f1"


def test_auto_upload_transfers_only_the_new_file(
    store,
    declared,
    build_analysis,
    broker_configured,
    auto_upload,
    fake_client,
    mini_copy_name,
    common,
):
    """A bare populate() would drain every declaration on the instance."""
    with common.AnalysisNwbfile().build(mini_copy_name) as builder:
        bystander = builder.analysis_file_name

    other = {"analysis_file_name": bystander}
    path = Path(common.AnalysisNwbfile.get_abs_path(bystander))

    try:
        fake_client.calls.clear()  # the bystander's own auto upload
        (store.SharedAnalysisFile & other).delete(
            safemode=False, force_permission=True
        )

        build_analysis()

        assert len(fake_client.calls) == 1, "Uploaded more than the new file"
    finally:
        (store.AnalysisFileSelection & other).delete(
            safemode=False, force_permission=True
        )
        (common.AnalysisNwbfile & other).delete(
            safemode=False, force_permission=True
        )
        path.unlink(missing_ok=True)


def test_a_failed_auto_upload_leaves_the_analysis_intact(
    store,
    declared,
    build_analysis,
    broker_configured,
    auto_upload,
    fake_client,
    caplog,
    common,
):
    """A network blip must cost a retry, not hours of compute.

    The analysis file is finished by the time the upload is attempted.
    """

    def _boom(path, **kwargs):
        raise RuntimeError("connection reset")

    fake_client.upload = _boom

    analysis = build_analysis()  # must not raise
    key = {"analysis_file_name": analysis}

    assert store.AnalysisFileSelection & key, "Lost the declaration"
    assert common.AnalysisNwbfile & key, "Lost the analysis file"
    assert not (store.SharedAnalysisFile & key), "Recorded a failed upload"
    assert "connection reset" in caplog.text
    assert "Could not upload" in caplog.text, "Warning blamed the wrong half"


def test_auto_upload_does_nothing_without_a_shared_parent(
    store, build_analysis, broker_configured, auto_upload, fake_client
):
    """Nothing inherited means nothing sent."""
    analysis = build_analysis()

    assert not (store.AnalysisFileSelection & {"analysis_file_name": analysis})
    assert fake_client.calls == []


def test_auto_upload_does_nothing_without_a_broker(
    mini_copy_name, common, auto_upload, fake_client
):
    """The flag does not override the no-broker early return."""
    from spyglass.settings import sg_config

    prior = sg_config.store_url
    sg_config.store_url = ""

    with patch(
        "spyglass.sharing.sharing_store.queue_inherited_share"
    ) as queued:
        try:
            with common.AnalysisNwbfile().build(mini_copy_name) as builder:
                analysis = builder.analysis_file_name
        finally:
            sg_config.store_url = prior

    try:
        queued.assert_not_called()
        assert fake_client.calls == []
    finally:
        (common.AnalysisNwbfile & {"analysis_file_name": analysis}).delete(
            safemode=False, force_permission=True
        )


# --------------------------- default visibility -------------------------


def test_a_share_declared_without_a_scope_is_public(store, mini_copy_name):
    """The schema exists to share data; a default nobody can read does not."""
    key = {"nwb_file_name": mini_copy_name}
    store.SharedFileSelection.insert1(key)

    try:
        assert (store.SharedFileSelection & key).fetch1("scope") == "public"
    finally:
        (store.SharedFileSelection & key).delete(
            safemode=False, force_permission=True
        )


def test_the_public_default_never_reaches_inheritance(store):
    """A derived file takes its parents' scopes, never the column default.

    One file has one scope; this list is one entry per parent.
    """
    parent_scopes = ["private", "public"]  # a private raw, a public analysis

    assert store.most_restrictive(parent_scopes) == "private"
    assert store.most_restrictive([]) == "private", "No parents is not public"

    with patch.object(store, "declared_visibility", return_value=None):
        assert store.inherited_visibility(raw_files=["never_shared"]) is None


# ------------------------- inherited re-sync ----------------------------


@pytest.fixture
def team_names(common):
    """Two LabTeams to scope shares to, removed afterward."""
    names = ["Resync Alpha", "Resync Beta"]

    for name in names:
        common.LabTeam.insert1({"team_name": name}, skip_duplicates=True)

    yield names

    for name in names:
        (common.LabTeam & {"team_name": name}).delete(
            safemode=False, force_permission=True
        )


def test_narrowing_a_raw_narrows_its_derivatives(
    store, declared, build_analysis, broker_configured, fake_client, team_names
):
    """A raw re-scoped after registration must not leave derivatives behind."""
    alpha, _ = team_names
    store.SharedFileSelection.update1({**declared, "scope": "group"})
    store.SharedFileSelection.Team.insert1({**declared, "team_name": alpha})

    analysis = build_analysis()
    key = {"analysis_file_name": analysis}

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "group"

    store.SharedFile.populate(declared)
    store.SharedFile().update_visibility(declared, scope="private")

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "private"
    assert not (store.AnalysisFileSelection.Team & key)


def test_widening_a_raw_widens_its_derivatives(
    store, declared, build_analysis, broker_configured, fake_client
):
    """A raw opened up carries its derivatives with it."""
    analysis = build_analysis()
    key = {"analysis_file_name": analysis}

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "private"

    store.SharedFile.populate(declared)
    store.SharedFile().update_visibility(declared, scope="public")

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "public"


def test_widening_stops_at_another_parents_restriction(
    store,
    declared,
    build_analysis,
    broker_configured,
    fake_client,
    common,
    mini_copy_name,
):
    """Re-derivation reads every recorded parent, not just the raw."""
    # A hand-declared private analysis file for the derivative to also
    # inherit from. `inherited = 0` is what makes it a choice rather than
    # another row the cascade re-derives — were it inherited from this same
    # raw, widening it along with everything else would be correct.
    with common.AnalysisNwbfile().build(mini_copy_name) as builder:
        upstream = builder.analysis_file_name

    up_key = {"analysis_file_name": upstream}
    store.AnalysisFileSelection.update1({**up_key, "scope": "private"})
    store.AnalysisFileSelection.update1({**up_key, "inherited": 0})

    with common.AnalysisNwbfile().build(
        mini_copy_name, share_parents=[upstream]
    ) as builder:
        derived = builder.analysis_file_name

    key = {"analysis_file_name": derived}
    path = Path(common.AnalysisNwbfile.get_abs_path(derived))
    up_path = Path(common.AnalysisNwbfile.get_abs_path(upstream))

    try:
        assert (store.AnalysisFileSelection & key).fetch1("scope") == "private"

        store.SharedFile.populate(declared)
        store.SharedFile().update_visibility(declared, scope="public")

        assert (store.AnalysisFileSelection & key).fetch1(
            "scope"
        ) == "private", "Widened past the analysis parent"
    finally:
        for k, p in ((key, path), (up_key, up_path)):
            (store.AnalysisFileSelection & k).delete(
                safemode=False, force_permission=True
            )
            (common.AnalysisNwbfile & k).delete(
                safemode=False, force_permission=True
            )
            p.unlink(missing_ok=True)


def test_a_hand_declared_derivative_is_left_alone(
    store, declared, build_analysis, broker_configured, fake_client
):
    """A scope the user chose is a choice, not a default to re-derive."""
    analysis = build_analysis()
    key = {"analysis_file_name": analysis}

    # What a hand declaration looks like: inherited stays 0.
    store.AnalysisFileSelection.update1({**key, "inherited": 0})
    store.AnalysisFileSelection.update1({**key, "scope": "public"})

    store.SharedFile.populate(declared)
    store.SharedFile().update_visibility(declared, scope="private")

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "public"


def test_a_refused_narrowing_warns_and_keeps_going(
    store, declared, build_analysis, broker_configured, fake_client, caplog
):
    """A refusal names the file and moves on, rather than failing the call."""
    # The derivative must start wider than where the raw is headed, or there
    # is nothing to narrow and no relay to refuse.
    store.SharedFileSelection.update1({**declared, "scope": "public"})

    analysis = build_analysis()
    key = {"analysis_file_name": analysis}

    assert (store.AnalysisFileSelection & key).fetch1("scope") == "public"

    store.SharedFile.populate(declared)
    store.SharedAnalysisFile.populate(key)

    # The refusal lands on the derivative's relay, not the raw's own.
    refused = RuntimeError("This identity does not own the file.")

    with patch.object(
        store.SharedAnalysisFile, "update_visibility", side_effect=refused
    ):
        store.SharedFile().update_visibility(declared, scope="private")

    assert (store.SharedFileSelection & declared).fetch1("scope") == "private"
    assert "Could not re-scope" in caplog.text
    assert analysis in caplog.text


# ------------------------------ checksums -------------------------------


@pytest.mark.parametrize(
    "algorithms", [("sha256",), ("md5",), ("sha256", "md5")]
)
def test_the_file_is_read_once(tmp_path, algorithms):
    """However many digests are asked for, the read happens once."""
    from spyglass.utils import nwb_hash

    target = tmp_path / "bytes.bin"
    target.write_bytes(b"spyglass" * 4096)

    opens = []
    real_open = Path.open

    def _counting_open(self, *args, **kwargs):
        opens.append(self)
        return real_open(self, *args, **kwargs)

    with patch.object(Path, "open", _counting_open):
        digests = nwb_hash.digest_file(target, algorithms=algorithms)

    assert opens == [target], f"Read the file {len(opens)} times"
    assert set(digests) == set(algorithms)


def test_the_digests_match_hashlib(tmp_path):
    """Both, against hashlib on the same bytes.

    A chunking mistake is invisible to a self-consistent implementation.
    """
    import hashlib

    from spyglass.utils import nwb_hash

    payload = b"".join(bytes([i % 256]) for i in range(70000))
    target = tmp_path / "bytes.bin"
    target.write_bytes(payload)

    # A chunk size that does not divide the payload evenly.
    digests = nwb_hash.digest_file(
        target, algorithms=("sha256", "md5"), chunk_size=4096
    )

    assert digests["sha256"] == hashlib.sha256(payload).hexdigest()
    assert digests["md5"] == hashlib.md5(payload).hexdigest()


def test_sha256_file_still_returns_a_string(tmp_path):
    """The wrapper keeps its documented contract."""
    import hashlib

    from spyglass.utils.nwb_hash import sha256_file

    payload = b"unchanged"
    target = tmp_path / "bytes.bin"
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_upload_sends_the_md5(store, declared, fake_client):
    """The digest reaches the broker, where it is signed into the URL."""
    store.SharedFile.populate(declared)

    _, _, kwargs = fake_client.calls[0]

    assert len(kwargs["content_md5"]) == 32
    assert len(kwargs["sha256"]) == 64


def test_a_deduplicated_upload_still_records_what_it_declared(
    store, declared, fake_client
):
    """Recorded even when no bytes moved, and not to be read as verified."""

    def _dedup(path, **kwargs):
        fake_client.calls.append(("upload", path, kwargs))
        return {"file_id": "f1", "deduplicated": True}

    fake_client.upload = _dedup
    store.SharedFile.populate(declared)

    row = (store.SharedFile & declared).fetch1()

    assert row["deduplicated"] == 1
    assert row["content_md5"] and len(row["content_md5"]) == 32

    comment = store.SharedFile.heading.attributes["content_md5"].comment
    assert (
        "NOT proof" in comment
    ), f"Column comment reassures wrongly: {comment}"


def test_only_the_requested_digests_are_computed(
    store, declared, fake_client, monkeypatch
):
    """A broker whose store verifies sha256 must not cost an extra digest."""
    from spyglass.utils import nwb_hash

    fake_client.upload_digests = lambda: ["sha256"]
    asked = []

    real = nwb_hash.digest_file

    def _record(path, algorithms=("sha256",), **kwargs):
        asked.append(sorted(algorithms))
        return real(path, algorithms=algorithms, **kwargs)

    monkeypatch.setattr(nwb_hash, "digest_file", _record)
    store.SharedFile.populate(declared)

    assert asked == [["sha256"]], f"Computed {asked}"

    _, _, kwargs = fake_client.calls[0]

    assert kwargs["content_md5"] is None
    assert (store.SharedFile & declared).fetch1("content_md5") is None
