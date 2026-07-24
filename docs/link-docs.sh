#!/bin/bash
# Regenerate the committed symlinks that surface top-level repo files under
# docs/src/ for the documentation build. Run from the repo root:
#   > bash ./docs/link-docs.sh
#
# These symlinks replace the per-build `cp` commands that used to live in
# build-docs.sh. They are committed to the repo, so this script only needs to
# be re-run when the set of source files changes (e.g. a notebook is added).
set -euo pipefail

# Top-level files surfaced at docs/src/
ln -sfn ../../CHANGELOG.md ./docs/src/CHANGELOG.md
ln -sfn ../../LICENSE ./docs/src/LICENSE.md
ln -sfn ../../notebook-images ./docs/src/notebook-images

# Quickstart under GettingStarted/
ln -sfn ../../../QUICKSTART.md ./docs/src/GettingStarted/QUICKSTART.md

# Notebooks: one symlink per notebook, README.md -> index.md, plus images
mkdir -p ./docs/src/notebooks
for nb in ./notebooks/*.ipynb; do
    base=$(basename "$nb")
    ln -sfn "../../../notebooks/$base" "./docs/src/notebooks/$base"
done
ln -sfn ../../../notebooks/README.md ./docs/src/notebooks/index.md
ln -sfn ../../../notebook-images ./docs/src/notebooks/notebook-images

echo "Doc symlinks refreshed under docs/src/"
