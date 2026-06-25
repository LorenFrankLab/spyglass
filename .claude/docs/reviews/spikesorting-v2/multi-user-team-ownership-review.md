# Spike Sorting V2 Multi-User and Team Ownership Review

Date: 2026-06-25

Scope: team ownership, shared-database usage, permission boundaries, destructive
operations across teams, session groups, shared artifact groups, recompute
reclamation, curation provenance, and export namespaces. This is a different
lens from schema evolution, destructive-operation mechanics, downstream
consumer contracts, docs integrity, and dependency/runtime safety.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 models team identity in several useful places: `RecordingSelection` includes
`team_name`, `SessionGroup` names are scoped by owner, session-group members
carry their own `team_name`, and UnitMatch validates that pinned curations match
the member identity. Those are good foundations for shared-lab operation.

The main gap is that `team_name` is mostly a namespace and provenance field, not
an enforced authorization boundary. Insert paths check that the team exists, but
not that the current DataJoint user belongs to that team. Delete permissions
still flow through session experimenters rather than v2 owner teams. Several
admin/global surfaces, including recompute deletion, paper export, sort-group
overwrite, and shared artifact groups, can therefore affect another team's rows
or files in a shared database.

## What Looks Solid

- `RecordingSelection` includes `team_name` in the stable selection identity,
  which prevents two teams from accidentally sharing the same recording
  selection row (`src/spyglass/spikesorting/v2/recording.py:754-762`).
- `SessionGroup` master rows are namespaced by `session_group_owner`, and member
  rows include their own `team_name`, so the schema can represent both
  single-team and mixed-team groups
  (`src/spyglass/spikesorting/v2/session_group.py:51-83`).
- `SessionGroup.create_group()` makes mixed-team use explicit by allowing
  per-member `team_name` overrides and defaulting missing member teams to the
  group owner (`src/spyglass/spikesorting/v2/session_group.py:96-105`,
  `src/spyglass/spikesorting/v2/session_group.py:161-168`).
- UnitMatch validates that each pinned curation belongs to the exact
  `(nwb_file_name, sort_group_id, interval_list_name, team_name)` member
  identity (`src/spyglass/spikesorting/v2/unit_matching.py:1044-1088`).
- `SortGroupV2` overwrite is no longer silent: callers must explicitly opt into
  deletion and confirmation before replacing an existing session's sort groups
  (`src/spyglass/spikesorting/v2/recording.py:138-146`,
  `src/spyglass/spikesorting/v2/recording.py:253-285`).

## Findings

### 1. High: sort-group overwrite is session-global and can cascade through every team's downstream rows

`SortGroupV2` is keyed only by `Session` and `sort_group_id`; it has no owner
team (`src/spyglass/spikesorting/v2/recording.py:149-155`). Downstream
`RecordingSelection` rows are team-scoped, but they depend on those
session-global sort groups (`src/spyglass/spikesorting/v2/recording.py:754-762`).

The overwrite path selects all sort groups for an NWB file and deletes them via
`existing.cautious_delete()` when `delete_existing_entries=True, confirm=True`
(`src/spyglass/spikesorting/v2/recording.py:249-285`). The confirmation guard is
good, but the unit of deletion is still the whole session's sort-group set, not
the caller's team.

Impact: one team rerunning `set_group_by_*` on a shared session can cascade
delete another team's recordings, artifact detections, sortings, and curations
for the same `nwb_file_name` and `sort_group_id`.

Recommended fix: either add owner scoping to sort-group definitions or block
destructive replacement when downstream rows include owner teams outside the
caller-authorized team set. Add a two-team same-session test where Team A cannot
overwrite sort groups that Team B has populated downstream.

### 2. High: delete permission checks session experimenters, not v2 owner teams

V2 tables inherit `CautiousDeleteMixin` through `SpyglassMixin`, and v2 delete
overrides still delegate to that mechanism before their side-effect cleanup
(`src/spyglass/utils/dj_mixin.py:18-24`,
`src/spyglass/spikesorting/v2/sorting.py:2084-2154`,
`src/spyglass/spikesorting/v2/artifact.py:1104-1153`). The mixin maps the
DataJoint user to a lab member and then checks whether that user shares a team
with each `Session.Experimenter`
(`src/spyglass/utils/mixins/cautious_delete.py:90-150`). If it cannot find a
session path, it warns and permits the delete
(`src/spyglass/utils/mixins/cautious_delete.py:110-126`).

That policy predates v2's team-scoped selection identity. It does not inspect
`RecordingSelection.team_name`, `SessionGroup.session_group_owner`, or member
teams.

Impact: a user on the session experimenter's team can delete another team's v2
analysis for that session. Conversely, a team that legitimately owns the v2
analysis may be denied if it is not tied to the session experimenter in the old
permission model.

Recommended fix: add a v2 ownership resolver that derives affected teams from
`RecordingSelection.team_name`, `SessionGroup.session_group_owner`, and member
`team_name` rows, then require the current user to belong to all affected owner
teams or be a database admin. Add cross-team delete tests for `Recording`,
`ArtifactDetection`, `Sorting`, `CurationV2`, `SessionGroup`, and UnitMatch
tables.

### 3. High: `team_name` is treated as a namespace, not an access boundary

Pipeline docs describe `team_name` as the LabTeam owning the sort, but the
preflight check only verifies that the `LabTeam` row exists
(`src/spyglass/spikesorting/v2/_pipeline_run.py:133-135`,
`src/spyglass/spikesorting/v2/_pipeline_preflight.py:387-392`).
`run_v2_pipeline()` passes the caller-supplied team directly into
`RecordingSelection.insert_selection()`
(`src/spyglass/spikesorting/v2/_pipeline_run.py:299-307`).

`CurationV2.insert_curation()` takes a `sorting_key`, not an authorizing team,
curator, or requester (`src/spyglass/spikesorting/v2/curation.py:192-205`).
The inserted curation master stores source and description, then registers into
`SpikeSortingOutput`, but it does not persist who created it or prove that the
current user can curate that sorting
(`src/spyglass/spikesorting/v2/curation.py:750-764`,
`src/spyglass/spikesorting/v2/curation.py:815-842`).

Impact: a user with shared DB write access can accidentally or intentionally
pass another team's `team_name`, reuse that team's selection identity, curate a
known `sorting_id`, and create merge-table outputs over another team's data.

Recommended fix: enforce `database.user -> LabMemberInfo -> LabTeamMember`
membership or admin status at `RecordingSelection.insert_selection()`,
`run_v2_pipeline()`, `CurationV2.insert_curation()`, visualization/export entry
points, and any public helpers that accept v2 IDs. Add allowed/forbidden
Alice/Bob-style tests where the foreign team row exists but the user is not a
member.

### 4. Medium-high: recompute deletion is documented as admin-facing but has no admin or team gate

The storage docs label recompute helpers as an "Admin surface"
(`docs/src/Features/SpikeSortingV2StorageManagement.md:101-106`). The deletion
authorization itself is based on matched recompute rows in the current
`UserEnvironment`, optional stale-env override, and the artifact age gate
(`src/spyglass/spikesorting/v2/recompute.py:481-489`,
`src/spyglass/spikesorting/v2/recompute.py:1019-1063`). It does not check
database admin status, owner team, or session-group membership.

Impact: in a shared database, any writer who can create or match recompute rows
can reclaim/delete recording or analyzer cache files for another team's v2
artifacts once the environment and age gates pass.

Recommended fix: require admin/service-account privilege, or explicit ownership
over the affected artifact, for `dry_run=False`, `force_stale_env=True`,
`remove_matched`, and `update_secondary`. If the intent is trusted-operator-only,
document that contract more plainly and make the API reject non-admin users.
Add non-admin tests around both recording and analyzer recompute deletion.

### 5. Medium: mixed-team session groups are representable but not authorized

`SessionGroup` has both a group owner and per-member team identity
(`src/spyglass/spikesorting/v2/session_group.py:65-83`). `create_group()` accepts
member-level `team_name` overrides and inserts those rows directly
(`src/spyglass/spikesorting/v2/session_group.py:161-168`). Later concat
selection resolves each member's `RecordingSelection` by that member team, so
the member team is operationally meaningful
(`src/spyglass/spikesorting/v2/session_group.py:443-450`).

The code validates shape and scientific consistency, and UnitMatch validates
curation-to-member identity. It does not validate that the current user can use
the group owner team and every member team.

Impact: one team can create concat or matching workflows that point at another
team's recordings or curations. Depending on row availability this may fail
later as "missing Recording", or it may proceed over another team's data without
a policy check.

Recommended fix: require authorization for `session_group_owner` and all member
`team_name` values. Make cross-team collaborations explicit in docs, including
who must belong to which teams. Test unauthorized foreign-member groups fail,
while a user who belongs to both teams passes.

### 6. Medium: `SharedArtifactGroup` is globally named and ownerless

`SharedArtifactGroup` is keyed only by `shared_artifact_group_name` and stores a
secondary `Session` FK (`src/spyglass/spikesorting/v2/artifact.py:211-215`).
Members are arbitrary populated `Recording` rows
(`src/spyglass/spikesorting/v2/artifact.py:217-223`). The insert helper checks
that members exist and are compatible, but not their teams
(`src/spyglass/spikesorting/v2/artifact.py:226-330`). Its docstring explicitly
says names must be unique within the installation
(`src/spyglass/spikesorting/v2/artifact.py:238-242`).

Impact: common names such as `day1_artifacts` collide across teams, and one team
can build shared artifact masks over another team's recordings.

Recommended fix: add an owner/team namespace to `SharedArtifactGroup` and
validate all member recordings against the caller's authorized teams. If schema
changes are deferred, make examples use explicit lab-wide names and document
that this is a global admin namespace. Add tests for same-name groups across two
teams and rejection of foreign recordings.

### 7. Medium: paper export is global by `paper_id` and has no owner

`ExportSelection` has `paper_id`, `analysis_id`, version, and time, with a
unique index on `(paper_id, analysis_id)` but no owner team or user field
(`src/spyglass/common/common_usage.py:100-110`). Starting the same export
selection resumes the existing `export_id` and deletes an existing `Export` row
for that selection (`src/spyglass/common/common_usage.py:140-153`).
`Export.populate_paper()` populates the max export for a `paper_id`
(`src/spyglass/common/common_usage.py:485-535`).

Impact: two teams sharing a database can overwrite or replace each other's
export metadata by reusing a paper id and analysis id, or by populating the same
paper namespace.

Recommended fix: add an `export_owner` or `team_name`, or document that
`paper_id` must be globally unique within an installation. Add a two-team test
that the same `paper_id` either isolates cleanly by owner or fails with an
actionable message.

### 8. Medium-low: LabTeam mutation controls exist but are not applied to team creation

`LabMemberInfo` stores `datajoint_user_name` and an `admin` flag
(`src/spyglass/common/common_lab.py:29-39`), and `LabTeam.check_admin_privilege()`
can enforce admin status (`src/spyglass/common/common_lab.py:142-156`).
`LabTeam.create_new_team()` inserts `LabTeam` and `LabTeamMember` rows without
calling that check (`src/spyglass/common/common_lab.py:227-281`).

Impact: if database grants allow ordinary users to write `common_lab`, those
users can reshape the membership data that future v2 authorization checks would
depend on. This may already be controlled by deployment-level DB grants, but the
code path itself does not express the policy.

Recommended fix: require admin or existing team-owner privilege for
LabTeam/LabTeamMember mutations, or explicitly document that database grants are
the enforcement boundary. Add non-admin mutation denial tests if code-level
checks are introduced.

### 9. Low: docs still blur availability and ownership semantics

The v2 docs describe `team_name` and `session_group_owner` in ownership terms,
but the code currently treats them mainly as namespaces. Separately, the docs
provide a UnitMatch workflow while the status section still says cross-session
unit matching is not yet available.

Impact: users may assume team rows protect access or that UnitMatch is still a
placeholder. Both assumptions are operationally important in a shared lab.

Recommended fix: align docs with the chosen policy. If team identity is only
metadata, say so directly. If it is intended as an access boundary, implement
the checks above and document the required `LabMemberInfo` / `LabTeamMember`
setup. Update UnitMatch status to the current support level and fixture/extra
requirements.

## Suggested Priority

1. Define the policy first: is `team_name` a namespace/provenance tag, or an
   access boundary?
2. If it is an access boundary, add a small shared authorization helper and use
   it in insert, curation, delete, recompute, session-group, shared-artifact, and
   export entry points.
3. Add two-user/two-team integration tests before making broad schema changes.
4. Revisit sort-group and shared-artifact schema only after the policy is
   explicit, because those changes affect migration shape.

