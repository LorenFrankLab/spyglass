# Building Docs

## Adding new pages

`mkdocs.yml` is the site configuration file. To add a new page, edit the `nav`
section of this file. New pages should be either:

1. A Markdown file in `docs/src/`.
2. A Jupyter notebook in `notebooks/`, linked under `docs/src/notebooks/`.

After adding a notebook, run `bash ./docs/link-docs.sh` from the repository root
and commit the new symlink. Edit notebooks in `notebooks/`; the linked
documentation uses the same source.

The remainder of `mkdocs.yml` specifies the site's
[configuration](https://www.mkdocs.org/user-guide/configuration/)

## Deployment

## GitHub

Whenever a new tag is pushed, GitHub actions will run
`.github/workflows/publish-docs.yml`. From the repository, select the Actions
tab, and then the 'Publish Docs' workflow on the left to monitor progress. The
process can also be manually triggered by selecting 'Run workflow' on the right.

To deploy on your own fork without a tag, follow turn on github pages in
settings, following a `documentation` branch, and then push to `test_branch`.
This branch is protected on `LorenFranklin/spyglass`, but not on forks.

## Testing

To test edits to the site, be sure docs dependencies are installed:

```console
cd /your/path/to/spyglass
pip install -e ".[docs]"
```

From the repository root, build the site without deploying or changing the
documentation branch:

```console
mkdocs build -f ./docs/mkdocs.yml
```

Review the build output for broken links and docstring parsing warnings. Use
`mkdocs build --strict -f ./docs/mkdocs.yml` to make warnings fail the build;
existing warnings elsewhere on the site may need separate cleanup.

For a local preview with automatic reloading, run:

```console
mkdocs serve -f ./docs/mkdocs.yml
```

Open `http://localhost:8000/` to inspect navigation, notebook rendering, code
examples, and narrow-screen layouts. Notebooks render without execution; a
successful docs build does not verify their code cells. The v2 notebook
execution checks are described in `tests/README.md`.

The `mike` package used by `docs/build-docs.sh` manages versioned deployments
and creates commits on the local `documentation` branch. Use that script for
release/version preview work (`bash ./docs/build-docs.sh serve`), or with `push`
to publish.

The following items can be commented out in `mkdocs.yml` to reduce build time:

- `mkdocstrings`: Turns code docstrings to API pages.
- `mkdocs-jupyter`: Generates tutorial pages from notebooks.

To end the process in your console, use `ctrl+c`.

If your new submodule is causing a build error (e.g., "Could not collect ..."),
you may need to add `__init__.py` files to the submodule directories.
