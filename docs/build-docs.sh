#!/bin/bash
# Run this script from repo root to serve site: > bash ./docs/build-docs.sh serve
# Then, navigate to localhost:8000/ to inspect site, then ctrl+c to exit
# For auto-reload during dev, use `mkdocs serve -f ./docs/mkdosc.yaml`


# Copy top-level repo files for docs display
cp ./CHANGELOG.md ./docs/src/
cp ./LICENSE ./docs/src/LICENSE.md
mkdir -p ./docs/src/notebooks
cp ./notebooks/*ipynb ./docs/src/notebooks/
cp ./notebooks/*md ./docs/src/notebooks/
mv ./docs/src/notebooks/README.md ./docs/src/notebooks/index.md
cp -r ./notebook-images ./docs/src/notebooks/
cp -r ./notebook-images ./docs/src/

# Get major version
version_line=$(grep "__version__ =" ./src/spyglass/_version.py)
version_string=$(echo "$version_line" | awk -F"[\"']" '{print $2}')
export MAJOR_VERSION="${version_string:0:3}"
echo "$MAJOR_VERSION"

# Get ahead of errors
export JUPYTER_PLATFORM_DIRS=1
# jupyter notebook --generate-config

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
