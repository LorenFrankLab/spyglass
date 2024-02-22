#!/usr/bin/python

import os
import sys


def main(*args):
    database_user, base_dir, filename = args + (None,) * (3 - len(args))

    os.environ["SPYGLASS_BASE_DIR"] = base_dir  # need to set for import to work

    from spyglass.settings import SpyglassConfig  # noqa F401

    config = SpyglassConfig(base_dir=base_dir)
    save_method = (
        "local"
        if filename == "dj_local_conf.json"
        else "global" if filename is None else "custom"
    )

    config.save_dj_config(
        save_method=save_method,
        filename=filename,
        base_dir=base_dir,
        database_user=database_user,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
