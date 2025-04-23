# Contributing to Spyglass

This document provides an overview of the Spyglass development, and provides
guidance for folks looking to contribute to the project itself. For information
on setting up custom tables, skip to Code Organization.

## Development workflow

New contributors should follow the
[Fork-and-Branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).
See GitHub instructions
[here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

Regular contributors may choose to follow the
[Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
for features that will involve multiple contributors.

The local version is dynamically managed via `hatch`, so be sure to run
`git pull --tags; hatch build` to update the version for any new local install.

## Code organization

- Tables are grouped into schemas by topic (e.g., `common_metrics`)
- Schemas
    - Are defined in a `py` pile.
    - Correspond to MySQL 'databases'.
    - Are organized into modules (e.g., `common`) by folders.
- The _common_ module
    - In principle, contains schema that are shared across all projects.
    - In practice, contains shared tables (e.g., Session) and the first draft of
        schemas that have since been split into their own
        modality-specific\
        modules (e.g., `lfp`)
    - Should not be added to without discussion.
- A pipeline
    - Refers to a set of tables used for processing data of a particular modality
        (e.g., LFP, spike sorting, position tracking).
    - May span multiple schema.
- For analysis that will be only useful to you, create your own schema.

## Misc

- During development, we suggest using a Docker container. See
    [example](../notebooks/00_Setup.ipynb).
- `numpy` style docstrings will be interpreted by API docs. To check for
    compliance, monitor the output when building docs (see `docs/README.md`)

## Making a release

Spyglass follows [Semantic Versioning](https://semver.org/) with versioning of
the form `X.Y.Z` (e.g., `0.4.2`).

1. In `CITATION.cff`, update the `version` key.
2. Make a pull request with changes.
3. After the pull request is merged, pull this merge commit and tag it with
    `git tag {version}`
4. Publish the new release tag. Run `git push origin {version}`. This will
    rebuild docs and push updates to PyPI.
5. Make a new
    [release on GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
