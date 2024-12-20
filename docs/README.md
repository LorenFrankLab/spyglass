# Building Docs

## Adding new pages

`mkdocs.yml` is the site configuration file. To add a new page, edit the `nav`
section of this file. New pages should be either:

1. A markdown file in the `docs/` directory.
2. A Jupyter notebook in the `notebooks/` directory.

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
pip install .[docs]
```

Then, run the build script:

```console
bash ./docs/build-docs.sh serve
```

Notably, this will make a copy of notebooks in `docs/src/notebooks`. Changes to
the root notebooks directory may not be reflected when rebuilding.

Use a browser to navigate to `localhost:8000/` to inspect the site. For
auto-reload of markdown files during development, use
`mkdocs serve -f ./docs/mkdosc.yaml`. The `mike` package used in the build
script manages versioning, but does not support dynamic reloading.

The following items can be commented out in `mkdocs.yml` to reduce build time:

- `mkdocstrings`: Turns code docstrings to API pages.
- `mkdocs-jupyter`: Generates tutorial pages from notebooks.

To end the process in your console, use `ctrl+c`.

If your new submodule is causing a build error (e.g., "Could not collect ..."),
you may need to add `__init__.py` files to the submodule directories.
