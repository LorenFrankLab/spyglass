# Client-side work for the shared-storage broker

What remains inside `spyglass` before a user can read or share files through the
broker. The server side is substantially built; none of the client half exists
yet.

Every claim below was checked against the code on branch `store` at `f744f2be`,
not against a plan.

## Where things actually stand

| Question                                             | Answer                                    |
| ---------------------------------------------------- | ----------------------------------------- |
| Can the client talk to the broker today?             | **No.** No broker-facing code exists.     |
| Can the broker run against a real Spyglass database? | **No.** See T1.                           |
| Is the backend protocol merged?                      | No — PR #1662 is open, changes requested. |

Evidence for "nothing exists": `src/spyglass/sharing/` contains only
`sharing_kachery.py`; the resolution chain in
`src/spyglass/utils/file_backends.py:407` is
`[LocalBackend(), KacheryBackend(), DandiBackend()]`; and `pyproject.toml`
declares no HTTP client.

______________________________________________________________________

## T1. Add `github_user_name` to `LabMember.LabMemberInfo`

**File:** `src/spyglass/common/common_lab.py` (lines 34-38), plus an alter entry
in `CHANGELOG.md`.

`LabMemberInfo` has `google_user_name` and `datajoint_user_name` and no GitHub
column. The broker reflects this table to map a GitHub login to a lab member and
thence to teams, and it checks for the column during application startup — so a
missing column is not a degraded feature, it is a broker that **refuses to
boot**.

This is the one task that blocks the server, so it comes first regardless of how
the rest is sequenced. It is also small: one column, one `alter()`.

Give it a **unique index**, as `google_user_name` and `datajoint_user_name`
already have. One GitHub login must map to at most one lab member: the broker
resolves a login to a member and then to that member's teams, so two rows
claiming the same login would hand one person another's access. GitHub logins
are globally unique, so the index is the only place this can go wrong.

Two further points make this table load-bearing:

- **It must be admin-only writable** on any instance served by a broker. Whoever
    can insert a `github_user_name` row decides who is verified and may upload.
    The same already has to be true of `LabTeam.LabTeamMember`.
- **A user with no row is not an error.** The broker treats them as an
    unaffiliated reader with no teams, who can reach public files only.

## T2. Byte-level SHA-256 hashing

**File:** new helper, near `src/spyglass/utils/nwb_hash.py`.

The broker addresses objects by the SHA-256 of the file's bytes, and the object
store verifies that hash on upload — a mismatch is refused, so the client cannot
approximate this.

`NwbfileHasher` cannot be reused. It is **md5** (`nwb_hash.py:4`), and it hashes
HDF5 *datasets* rather than file bytes, so it answers a different question.
There is no `sha256` anywhere in `src/spyglass`.

Two files with identical contents must produce identical digests, since that is
what makes deduplication work.

## T3. Device-flow login and credential storage

**File:** new `src/spyglass/sharing/store_client.py`; a console script in
`pyproject.toml`.

The broker's own CLI is admin-only — accounts, tiers, audit — and is never
installed alongside the client, so login is entirely the client's job.

The flow: `POST /auth/device` returns a code and a URL; the user types the code
at `github.com/login/device`; the client polls `POST /auth/token` until it stops
returning 428. No browser callback, so it works over SSH and in containers.

Store the returned token at `0600`. It is a broker credential and grants nothing
on GitHub.

## T4. HTTP client for the broker API

**File:** same module as T3; `pyproject.toml` dependencies.

Six endpoints: two `/auth`, `/file/resolve`, `POST /file`, `/file/{id}/content`,
`PATCH /file/{id}/visibility`.

No HTTP library is currently declared. Note that `common_dandi.py` already
imports `fsspec` as an undeclared transitive dependency, so a direct dependency
is overdue regardless.

Upload needs care: `POST /file` returns an upload URL **and headers**. The
headers carry the checksum the store verifies against and are covered by the URL
signature — sending the bytes without them fails the upload rather than skipping
the check.

## T5. `StoreBackend`

**File:** `src/spyglass/utils/file_backends.py`.

Implement `has` / `stream` / `download` and insert into the chain after
`LocalBackend`, so a local copy still wins and DANDI remains the fallback.

`has()` must treat 403 and 404 the same way — both fall through to the next
backend. Be aware this means a permission denial is currently indistinguishable
from a missing file, which silently masks a file the user could actually read.
See Q2.

## T6. Sharing schema

**Files:** new `src/spyglass/sharing/sharing_store.py`;
`src/spyglass/sharing/__init__.py`.

Selection tables parallel to `sharing_kachery`, separate for raw and analysis
files, with a part table listing teams for group visibility. Declaring a share
is a database insert; `populate()` is what transfers.

## T7. Configuration surface

**Files:** `src/spyglass/settings.py`, `dj_local_conf_example.json`,
`scripts/install.py`, `tests/setup/test_config_schema.py`,
`tests/setup/test_install.py`.

The broker URL and related settings. This is the same five-file footprint that
`prefer_download` used in PR #1662, so there is a worked example to follow.

## T8. Visibility inheritance on analysis files

**File:** `src/spyglass/utils/mixins/analysis_builder.py`.

The builder already enforces its lifecycle and auto-registers on exit. At
registration it should also queue a sharing row, inheriting visibility from the
parent — the intersection where there are several, so a default never widens
access.

## T9. Rewrite `03_Data_Sync`

**Files:** `notebooks/03_Data_Sync.ipynb` and
`notebooks/py_scripts/03_Data_Sync.py` — both, per repo convention.

Rewrite in place rather than adding a notebook. `03_Data_Sync` is currently all
Kachery, and two live sharing tutorials during the deprecation window would be
actively confusing. Keep a short Kachery appendix until it is removed.

Cover, in this order:

1. **Linking your GitHub identity.** Until `github_user_name` is set, the broker
    sees an unaffiliated reader with no teams. Non-obvious, and the most likely
    support question.
2. **One-time login.** Emphasize no browser callback is needed. Say where the
    token lives and that it grants nothing on GitHub.
3. **Tiers.** An unverified account reads public files only and cannot upload.
    Without this, a first upload returns 403 with no explanation.
4. **Declaring a share.** The three scopes; that `group` naming no team is
    rejected rather than silently meaning private.
5. **`populate()` is the transfer.** Declaring is a database insert; nothing
    crosses the network until populate. Retry is just re-running it.
6. **Reading someone else's file.** `get_nwb_file` is unchanged. How to confirm
    it streamed, and `prefer_download` for slow links.
7. **Changing visibility.** Owner only — a teammate who can read cannot widen.
8. **Inheritance.** Derived files take the most restrictive parent visibility.
9. **Quota.** What a throttle looks like, and that it is expected.

Admin-facing setup — pointing an instance at a broker — belongs in
`docs/src/ForDevelopers/`, not a notebook. Different audience.

## T10. Update the backend developer page

**File:** `docs/src/ForDevelopers/FileBackends.md`.

Its chain table documents three backends and describes the chain as fixed, so
adding a fourth means editing the prose, not just the list.

______________________________________________________________________

## Open questions for the broker author

These are contract-level and worth settling before T5, because the answers
change what the client has to do.

**Q1. Resolving by name is ambiguous.** Registration is deliberately per-owner:
two people registering the same bytes get separate rows. Nothing enforces
uniqueness of `spyglass_name`, and the resolve endpoint returns the first
matching row. So two owners with `minirec20230622_.nwb` produce a
nondeterministic winner — and a reader can be handed someone else's private row
and refused a file they could actually read. The response carries no owner
field, so the client cannot disambiguate either.

**Q2. No way to read visibility back.** The stated goal is that a user can query
which of their files are shared *and at what visibility*. There is no list
endpoint and the file response carries no visibility field; the visibility
endpoint only writes. Either the Spyglass tables are accepted as the sole
record, with no drift detection, or the API needs a way to read it back.

**Q3. No token expiry or refresh.** The token response is
`access_token`/`tier`/`github_login` — no expiry, no refresh endpoint. The
client can only re-run device flow after a 401. Any plan that says the client
"handles token refresh" is not implementable as written.

**Q4. `file_class` is client-declared and unvalidated.** Any policy that treats
raw and analysis files differently rests on a value the uploader set.

## Suggested order

T1 first — it unblocks the server. Then land PR #1662, settle Q1 and Q2, then
T2-T4 (login and transport), T5 (the backend), T6-T8 (sharing), and T9-T10
(documentation) last, once the behaviour they describe has stopped moving.
