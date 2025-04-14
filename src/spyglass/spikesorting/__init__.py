# Initializing table to prevent recompute of cached properties
from spyglass.common.common_user import UserEnvironment
from spyglass.utils import logger

USER_TBL = UserEnvironment()
ENV_ID = USER_TBL.this_env.get("env_id", None)
logger.info(f"Initializing UserEnvironment for spikesorting: {ENV_ID}")

__all__ = [
    "USER_TBL",
]
