# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: spyglass
#     language: python
#     name: python3
# ---

# # Sync Data
#

# ## Overview
#

# This notebook covers sharing NWB files through a **shared-storage broker**: a
# small service your lab runs that decides who may read a file and hands out
# short-lived download links. Spyglass never holds an object-store credential.
#
# In order:
#
# 1. [Linking your GitHub identity](#1-link-your-github-identity)
# 2. [One-time login](#2-log-in-once)
# 3. [Tiers](#3-tiers-what-your-account-may-do)
# 4. [Declaring a share](#4-declare-a-share)
# 5. [`populate()` is the transfer](#5-populate-is-the-transfer)
# 6. [Reading someone else's file](#6-read-someone-elses-file)
# 7. [Changing visibility](#7-change-visibility)
# 8. [Inheritance](#8-derived-files-inherit-visibility)
# 9. [Quota](#9-quota)
#
# Kachery is being retired. Its instructions are kept in an
# [appendix](#appendix-kachery-deprecated) until it is removed.
#
# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass.
#
# - To set up your Spyglass environment and database, see
#   [the Setup notebook](./00_Setup.ipynb)
# - For additional info on DataJoint syntax, including table definitions and
#   inserts, see
#   [these additional tutorials](https://github.com/datajoint/datajoint-tutorials)
#
# Pointing a whole database instance at a broker is an admin task and is
# covered in `docs/src/ForDevelopers/`, not here.
#

# ## Imports
#

# +
import datajoint as dj
import spyglass.common as sgc
import spyglass.sharing as sgs
from spyglass.settings import sg_config

import warnings

warnings.filterwarnings("ignore")
# -

# Your instance is attached to a broker if `store_url` is set. If it is empty,
# nothing in this notebook applies, and `get_nwb_file` keeps working exactly as
# it does today.
#

sg_config.store_url

# ## 1. Link your GitHub identity
#

# **This is the step most likely to be missed, and it is silent when it is.**
#
# The broker knows you by your GitHub login. It maps that login to a
# `LabMember` through `LabMemberInfo.github_user_name`, and from there to your
# `LabTeam` memberships. Until that column is set, the broker sees an
# unaffiliated reader on no team, who can fetch **public files only** — a
# perfectly valid account that quietly cannot see your lab's data.
#

sgc.LabMember.LabMemberInfo & {"lab_member_name": "Firstname Lastname"}

# Set it with:
#
# ```python
# sgc.LabMember.set_github_user_name("Firstname Lastname", "your-github-login")
# ```
#
# This updates an existing `LabMemberInfo` row; it will not create one. A
# member who has none needs `google_user_name` supplied, and that column is
# uniquely indexed, so there is no placeholder this call could invent without
# taking the one blank slot and breaking the *next* member's link. If you see
# "has no `LabMemberInfo` row", insert the row first — see
# [Insert Data](./02_Insert_Data.ipynb).
#
# On a database attached to a broker, `LabMemberInfo` is **writable by admins
# only**. It has to be: anyone who could edit it could point their own GitHub
# login at a better-connected lab member and be handed that member's files. So
# the call above is expected to fail for an ordinary user, with a
# `PermissionError` naming the exact update to request:
#
# ```text
# PermissionError:
#   `common_lab.LabMember.LabMemberInfo` is admin-only on this instance.
#   Ask an admin to run:
#
#     LabMember.LabMemberInfo.update1(
#         {"lab_member_name": "Firstname Lastname",
#          "github_user_name": "your-github-login"}
#     )
# ```
#
# Send that to your database admin. Nothing else in this notebook will behave
# as documented until it is done.
#

# ## 2. Log in once
#

# Login uses GitHub's **device flow**. There is no browser callback, so it
# works unchanged over SSH and inside a container — you are never waiting on a
# `localhost` redirect that cannot reach you.
#
# From a terminal:
#
# ```bash
# spyglass-store login
# ```
#
# Or from Python:
#
# ```python
# sgs.get_client().login()
# ```
#
# Either prints a URL and a short code. Open the URL, type the code, and the
# client stops polling as soon as you approve.
#

sgs.get_client().logged_in

# **What is stored, and where.** The token lands in
# `~/.spyglass/store_token.json` at mode `0600`, keyed by broker URL so a
# second instance does not silently reuse the first one's credential.
# `SPYGLASS_STORE_TOKEN` overrides the location, which is what a container
# wants.
#
# **It grants nothing on GitHub.** Our OAuth app requests *zero* scopes. The
# GitHub token is used once, by the broker, to learn your login, and is then
# dropped; what you keep is a broker token that can read nothing on GitHub.
#
# There is no expiry and no refresh endpoint. If a call starts failing with
# "the broker did not accept this token", run `spyglass-store login` again.
#

# ## 3. Tiers: what your account may do
#

# Every new login starts in the `unverified` tier until an admin promotes it.
#
# | Tier         | Upload    | Read                        |
# | ------------ | --------- | --------------------------- |
# | `unverified` | no        | public only, throttled      |
# | `verified`   | yes       | per permissions             |
# | `trusted`    | yes, bulk | per permissions, unmetered  |
# | `admin`      | yes       | all                         |
#
# **An unverified account cannot upload.** A first `populate()` will fail with
# `StoreForbidden: This identity may not upload.` That is not a bug and not a
# misconfiguration — ask your ServerHost admin to whitelist you.
#

client = sgs.get_client()
client.tier, client.github_login

# The tier shown here is from your last login and can be stale: the broker
# re-reads it on every call, so a promotion takes effect without logging in
# again.
#

# ## 4. Declare a share
#

# Visibility has three scopes:
#
# | Scope     | Who can read it                         |
# | --------- | --------------------------------------- |
# | `private` | you only                                |
# | `group`   | members of the `LabTeam`s you name      |
# | `public`  | anyone with an account                  |
#
# Declaring a share is a **database insert**. Raw and analysis files live in
# separate tables, because `Nwbfile` and `AnalysisNwbfile` have separate
# primary keys.
#

nwb_copy_filename = "minirec20230622_.nwb"

sgs.SharedFileSelection.insert1(
    {"nwb_file_name": nwb_copy_filename, "scope": "group"},
    skip_duplicates=True,
)
sgs.SharedFileSelection.Team.insert1(
    {"nwb_file_name": nwb_copy_filename, "team_name": "My Team"},
    skip_duplicates=True,
)

# A `group` scope that names no team is **rejected**, not silently treated as
# private. A share visible to nobody is almost always a mistake, and saying so
# is better than quietly doing something other than what was asked.
#
# For the common case there is a one-line helper:
#
# ```python
# sgs.share_file(
#     nwb_copy_filename,
#     scope="group",
#     teams=["My Team"],
#     file_class="raw",
# )
# ```
#
# Calling it again for the same file **replaces** the declaration, teams
# included, which is how you narrow a share declared too widely before it is
# uploaded. Once the file is in the store, use `update_visibility` instead —
# only that relays the change to the broker, and only the broker's copy is
# what a reader is actually checked against.
#

sgs.SharedFileSelection()

# ## 5. `populate()` is the transfer
#

# **Nothing has crossed the network yet.** The rows above are a declaration.
# The upload happens when you populate:
#

sgs.SharedFile.populate()

# That hashes the file's bytes with SHA-256, registers the hash and size with
# the broker, and uploads to a signed URL — unless someone already stored
# identical bytes, in which case only the registration is written and the
# transfer is skipped. Files are content-addressed, so the same bytes are
# stored once no matter how many people share them.
#

sgs.SharedFile()

# **Retry is just re-running it.** Because declaring and transferring are
# separate steps, a failed upload leaves the declaration intact; `populate()`
# picks up the rows that have no matching entry yet. Nothing to clean up, and
# nothing to redeclare.
#
# Analysis files use the parallel pair:
#

sgs.AnalysisFileSelection()

# ```python
# sgs.SharedAnalysisFile.populate()
# ```
#

# ## 6. Read someone else's file
#

# **`get_nwb_file` is unchanged.** There is no separate download step and no
# new function to call. `fetch_nwb`, `fetch1_dataframe`, and everything built
# on them work as they always have:
#

# +
from spyglass.lfp.v1 import LFPV1

(
    LFPV1
    & {
        "nwb_file_name": "Winnie20220713_.nwb",
        "target_interval_list_name": "pos 0 valid times",
    }
).fetch1_dataframe()
# -

# Behind that, Spyglass walks an ordered chain of backends and takes the first
# that has the file: local disk, then the shared store, then Kachery, then
# DANDI. A local copy always wins.
#

from spyglass.utils.file_backends import get_backends

[b.name for b in get_backends()]

# **Confirming it streamed.** A file read over the network was never written
# to disk, and `file_is_remote` reports which happened:
#

# +
from spyglass.utils.nwb_helper_fn import file_is_remote

file_is_remote(sgc.Nwbfile.get_abs_path(nwb_copy_filename))
# -

# Streaming means only the chunks your analysis touches cross the network, and
# reads are cached locally so the same chunk is not fetched twice. On a slow or
# metered link the arithmetic inverts — many small range requests cost more
# than one sequential transfer — so set `prefer_download` and get the whole
# file in one go:
#
# ```python
# sg_config.prefer_download = True  # this session
# ```
#
# Or in `dj_local_conf.json`, for a machine that is always on a slow link:
#
# ```json
# {"custom": {"prefer_download": true}}
# ```
#
# One caveat worth knowing: a file you are *refused* and a file that does not
# exist look identical to the resolution chain — both simply fall through to
# the next backend. If a file you expect to be able to read is not found,
# check step 1 before assuming it was never shared.
#

# ## 7. Change visibility
#

# **Owner only.** A teammate who can read a file cannot widen who else sees it.
# The change is relayed to the broker, which verifies ownership; the local
# declaration is updated only after the broker accepts it, so these tables
# never claim a visibility that was refused.
#

# ```python
# sgs.SharedFile().update_visibility(
#     {"nwb_file_name": nwb_copy_filename},
#     scope="public",
# )
# ```
#
# Note that visibility lives in DataJoint and is enforced by the broker; there
# is no second place to configure it.
#

# ## 8. Derived files inherit visibility
#

# Sharing a downstream result takes **no extra action beyond having shared its
# parent**. When `AnalysisFileBuilder` registers a file, it queues a sharing
# row inheriting the parent's visibility.
#
# Where a file has several parents, inheritance takes the **intersection** —
# the narrowest scope any parent declared, and the teams *every* group-scoped
# parent named. A result built from a public source and a private one is
# private. Combining data is never a way to widen access to any part of it.
#
# Two consequences worth stating:
#
# - A derived file of parents that were never shared is **not queued at all**.
#   No default may widen access to something nobody asked to share.
# - Inheritance never overrides a scope you set by hand.
#
# If a result draws on analysis files beyond its raw parent, name them so its
# visibility cannot exceed theirs:
#
# ```python
# with AnalysisNwbfile().build(
#     nwb_file_name, share_parents=[upstream_analysis_file]
# ) as builder:
#     builder.add_nwb_object(my_data, "results")
# ```
#
# Inheritance queues; it does not upload. Run `populate()` when you are ready.
#

sgs.AnalysisFileSelection()

# ## 9. Quota
#

# Reads are metered per tier, over a rolling window. Exceeding the allowance
# raises `StoreQuotaExceeded` with a `retry_after` giving the broker's own
# estimate of when capacity frees up:
#
# ```python
# from spyglass.sharing import StoreQuotaExceeded
#
# try:
#     nwbf = get_nwb_file(path)
# except StoreQuotaExceeded as err:
#     print(f"Throttled; retry in {err.retry_after}s")
# ```
#
# **This is expected, not a failure.** An unverified account is throttled by
# design. A file already charged in the current window costs nothing more, so
# a long streaming read that re-follows the redirect hundreds of times is
# billed once.
#
# Note that being throttled is the one broker refusal that does *not* fall
# through to the next backend. A missing or forbidden file quietly moves on to
# DANDI; a quota refusal raises, because the file is there and readable and
# waiting is cheaper than recomputing it.
#
# If you are throttled routinely, you want a tier promotion, not a retry loop.
#

# ## Appendix: Kachery (deprecated)
#

# Kachery sharing still works during the deprecation window, and is removed
# once the shared store replaces it. New work should use the tables above.
#
# Kachery is download-only, has no streaming path, and does not support raw
# files. It requires the optional dependency:
#
# ```bash
# pip install spyglass-neuro[kachery-cloud]
# ```
#
# A [Kachery Zone](https://github.com/flatironinstitute/kachery-cloud/blob/main/doc/create_kachery_zone.md)
# is a cloud storage host, set by environment variable or DataJoint config:
#
# ```json
# "custom": {
#    "kachery_zone": "franklab.default",
#    "kachery_dirs": {"cloud": "/your/base/path/.kachery-cloud"}
# }
# ```
#
# To share analysis files to a zone:
#
# ```python
# zone_name = config.get("KACHERY_ZONE")
#
# for file in (
#     sgc.AnalysisNwbfile() & {"nwb_file_name": nwb_copy_filename}
# ).fetch("analysis_file_name"):
#     sgs.AnalysisNwbfileKacherySelection.insert1(
#         {"kachery_zone_name": zone_name, "analysis_file_name": file},
#         skip_duplicates=True,
#     )
#
# sgs.AnalysisNwbfileKachery.populate()
# ```
#
# Or by source table:
#
# ```python
# from spyglass.sharing import share_data_to_kachery
# from spyglass.lfp.v1 import LFPV1
#
# share_data_to_kachery(
#     table_list=[LFPV1],
#     restriction={"nwb_file_name": nwb_copy_filename},
#     zone_name=zone_name,
# )
# ```
#
# Access is managed outside Spyglass, at
# `https://kachery-gateway.figurl.org/admin?zone=your_zone`, under
# Admin/Authorization Settings. That split — permissions in a web console,
# declarations in DataJoint — is one of the reasons Kachery is being replaced.
#

# # Up Next
#

# In the [next notebook](./04_Merge_Tables.ipynb), we'll explore the details of
# a table tier unique to Spyglass, Merge Tables.
#
