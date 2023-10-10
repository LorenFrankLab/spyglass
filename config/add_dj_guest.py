#!/usr/bin/env python
import os
import sys
import tempfile

shared_modules = [
    "common\_%",
    "spikesorting\_%",
    "decoding\_%",
    "position\_%",
    "position_linearization\_%",
    "ripple\_%",
    "lfp\_%",
]


def add_user(user_name):
    # create a temporary file for the command
    file = tempfile.NamedTemporaryFile(mode="w")

    # Create the user (if not already created) and set password
    file.write(
        f"CREATE USER IF NOT EXISTS '{user_name}'@'%' IDENTIFIED BY 'Data_$haring';\n"
    )

    # Grant privileges
    file.write(f"GRANT SELECT ON `%`.* TO '{user_name}'@'%';\n")

    file.flush()

    # run those commands in sql
    os.system(f"mysql -p -h lmf-db.cin.ucsf.edu < {file.name}")


if __name__ == "__main__":
    add_user(sys.argv[1])
