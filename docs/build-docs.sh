#!/bin/bash
# Run this script from repo root to serve site: > bash ./docs/build-docs.sh serve
# Then, navigate to localhost:8000/ to inspect site, then ctrl+c to exit
# For auto-reload during dev, use `mkdocs serve -f ./docs/mkdosc.yaml`


# Copy top-level repo files for docs display
cp ./CHANGELOG.md ./docs/src/
cp ./LICENSE ./docs/src/LICENSE.md
cp -r ./notebooks/ ./docs/src/
cp -r ./notebook-images ./docs/src/notebooks

# Get major version
FULL_VERSION=$(python -c "from spyglass import __version__; print(__version__)")
export MAJOR_VERSION="${FULL_VERSION%.*}"
echo "$MAJOR_VERSION"

# Generate site docs
mike deploy "$MAJOR_VERSION" --config ./docs/mkdocs.yml -b documentation

# Label this version as latest, set as default
mike alias "$MAJOR_VERSION" latest --config ./docs/mkdocs.yml -b documentation
mike set-default latest --config ./docs/mkdocs.yml -b documentation

# # Serve site to localhost
if [ "$1" == "serve" ]; then # If first arg is serve, serve docs
  mike serve --config ./docs/mkdocs.yml -b documentation
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
