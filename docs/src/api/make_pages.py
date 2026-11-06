"""Generate the api pages and navigation."""

from pathlib import Path
import runpy

import mkdocs_gen_files
from mkdocs.utils import log

ignored_stems = ["__init__", "_version"]

added = 0
add_limit = None

nav = mkdocs_gen_files.Nav()
# Read the dependency-free export catalog without importing the runtime tables.
# Griffe cannot infer pipeline.__all__ from its imported, unpacked tuple.
pipeline_exports = runpy.run_path(
    "src/spyglass/spikesorting/v2/_orchestration/exports.py"
)["PIPELINE_FACADE_EXPORTS"]
for path in sorted(Path("src/spyglass/").glob("**/*.py")):
    if (
        path.stem in ignored_stems
        or "cython" in path.stem
        or "checkpoint" in path.stem
    ):
        continue
    rel_path = path.relative_to("src/spyglass")
    if rel_path.parts[:2] == ("spikesorting", "v2") and any(
        part.startswith("_") for part in rel_path.parts[2:]
    ):
        # The v2 API map promises stability only for public import paths.
        continue

    # parts[0] is the src directory, ignore as of mkdocstrings-python 1.9.1
    module_path = ".".join([p for p in path.with_suffix("").parts[1:]])

    with mkdocs_gen_files.open(f"api/{rel_path.with_suffix('')}.md", "w") as f:
        print(f"::: {module_path}", file=f)
        if module_path == "spyglass.spikesorting.v2.pipeline":
            print("    options:\n      members:", file=f)
            for name in pipeline_exports:
                print(f"        - {name}", file=f)
    nav[rel_path.parts] = f"{rel_path.with_suffix('')}.md"

    if add_limit is not None:
        if added < add_limit:
            log.warning(f"Generated {rel_path.with_suffix('')}.md")
            added += 1
        else:
            break


with mkdocs_gen_files.open("api/navigation.md", "w") as nav_file:
    nav_file.write("* [Overview](../api/index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
