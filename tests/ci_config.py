import os
from pathlib import Path

import datajoint as dj

# NOTE this env var is set in the GitHub Action directly
data_dir = Path(os.environ["SPYGLASS_BASE_DIR"])

raw_dir = data_dir / "raw"
analysis_dir = data_dir / "analysis"

dj.config["database.host"] = "localhost"
dj.config["database.user"] = "root"
dj.config["database.password"] = "tutorial"
dj.config["stores"] = {
    "raw": {"protocol": "file", "location": str(raw_dir), "stage": str(raw_dir)},
    "analysis": {
        "protocol": "file",
        "location": str(analysis_dir),
        "stage": str(analysis_dir),
    },
}
dj.config.save_global()
