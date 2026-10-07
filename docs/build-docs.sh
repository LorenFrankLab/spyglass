#!/bin/bash
# Run this script from repo root to serve site: > bash ./docs/build-docs.sh serve
# Then, navigate to localhost:8000/ to inspect site, then ctrl+c to exit
# For auto-reload during dev, use `mkdocs serve -f ./docs/mkdocs.yml`

# Top-level repo files (CHANGELOG, LICENSE, QUICKSTART, notebooks, and
# notebook-images) are surfaced under docs/src/ via committed symlinks, so no
# copying is needed here. To refresh those links (e.g. after adding a
# notebook), run ./docs/link-docs.sh

# Verify each notebook has a matching committed symlink under docs/src (does
# not modify anything). Catches a notebook added without running link-docs.sh.
missing_links=""
for nb in ./notebooks/*.ipynb; do
    link="./docs/src/notebooks/$(basename "$nb")"
    if [ ! -L "$link" ]; then
        missing_links="${missing_links}\n  - ${link}"
    fi
done
if [ -n "$missing_links" ]; then
    echo "ERROR: missing notebook symlinks under docs/src/notebooks/:" >&2
    echo -e "$missing_links" >&2
    echo "Run ./docs/link-docs.sh and commit the new symlinks." >&2
    exit 1
fi

# Function for checking major version format: #.#
check_format() {
    local version="$1"
    if [[ $version =~ ^[0-9]+\.[0-9]+$ ]]; then
        return 0
    else
        return 1
    fi
}

# Check if the MAJOR_VERSION not defined or does not meet format criteria
if [ -z "$MAJOR_VERSION" ] || ! check_format "$MAJOR_VERSION"; then
  full_version=$(git describe --tags --abbrev=0)
  export MAJOR_VERSION="${full_version:0:3}"
fi
if ! check_format "$MAJOR_VERSION"; then
  export MAJOR_VERSION="dev" # Fallback to dev if still not valid
fi
echo "$MAJOR_VERSION"

# Get ahead of errors
export JUPYTER_PLATFORM_DIRS=1
jupyter notebook --generate-config -y &> /dev/null
jupyter trust ./docs/src/notebooks/*.ipynb &> /dev/null

# Generate site docs
mike deploy "$MAJOR_VERSION" --config ./docs/mkdocs.yml -b documentation \
  2>&1 | grep -v 'kernel_spec' # Suppress kernel_spec errors

# Label this version as latest, set as default
mike alias "$MAJOR_VERSION" latest -u --config ./docs/mkdocs.yml -b documentation
# mike set-default latest --config ./docs/mkdocs.yml -b documentation

# # Serve site to localhost
if [ "$1" == "serve" ]; then # If first arg is serve, serve docs
  mike serve --config ./docs/mkdocs.yml -b documentation | grep -v 'kernel_spec'
elif [ "$1" == "push" ]; then # if first arg is push
    if [ -z "$2" ]; then # When no second arg, use local git user
        git_user=$(git config user.name)
    else # Otherwise, accept second arg as git user
        git_user="${2}"
    fi # Push mike results to relevant branch
    export url="https://github.com/${git_user}/spyglass.git"
    git push "$url" documentation
else
    echo "Docs built. "
    echo "  Add 'serve' as script arg to serve. "
    echo "  Add 'push' to push to your fork."
    echo "  Use additional arg to dictate push-to fork"
fi
