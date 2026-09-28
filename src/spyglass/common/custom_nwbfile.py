import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile  # noqa F401
from spyglass.utils.dj_mixin import SpyglassAnalysis

db_prefix = dj.config.get("custom", {}).get("database.prefix")
username = dj.config.get("database.user")
user_prefix = db_prefix or username

if not user_prefix:
    raise ValueError(
        "Cannot create custom AnalysisNwbfile table: No prefix configured.\n"
        "Custom tables require a database prefix to isolate your files.\n"
        "This is typically set to your username automatically.\n\n"
        "To fix this:\n"
        "1. Set your database user: dj.config['database.user'] = 'username'\n"
        "   OR\n"
        "2. Set prefix: dj.config['custom']['database.prefix'] = 'myteam'\n\n"
        "Add this to your dj_local_conf.json or set it before importing.\n"
        f"Current config: database.user={username}, "
        f"custom.database.prefix={db_prefix}\n\n"
        "See: docs/Features/AnalysisTables.md#using-a-custom-table"
    )

# NOTE: For multiple custom analysis tables, set the prefix accordingly
schema = dj.schema(f"{user_prefix}_nwbfile")


@schema
class AnalysisNwbfile(SpyglassAnalysis, dj.Manual):
    definition = """This definition is managed by SpyglassAnalysis"""
