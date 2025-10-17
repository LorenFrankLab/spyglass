import sys

import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.utils.dj_mixin import SpyglassAnalysis

user_prefix = dj.config.get("custom", {}).get("database.prefix")

if not user_prefix:
    sys.exit(
        "Please set up config['custom']['database.prefix']"
        + f"Found: {user_prefix}"
    )

# NOTE: For multiple custom analysis tables, set the prefix accordingly
schema = dj.schema(f"{user_prefix}_nwbfile")


@schema
class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
    definition = """This definition is managed by SpyglassAnalysis"""
