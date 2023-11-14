#!/usr/bin/env python
import sys
from warnings import warn

from spyglass.utils.database_settings import DatabaseSettings

if __name__ == "__main__":
    warn(
        "This script is deprecated. "
        + "Use spyglass.utils.database_settings.DatabaseSettings instead."
    )
    DatabaseSettings().add_module(sys.argv[1])
