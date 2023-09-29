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
    if os.path.isdir(f"/home/{user_name}"):
        print("Creating database user ", user_name)
    else:
        sys.exit(f"Error: user_name {user_name} does not exist in /home.")

    # create a tempoary file for the command
    file = tempfile.NamedTemporaryFile(mode="w")
    create_user_query = f"CREATE USER IF NOT EXISTS '{user_name}'@'%' IDENTIFIED BY 'temppass';\n"
    grant_privileges_query = (
        f"GRANT ALL PRIVILEGES ON `{user_name}\_%`.* TO '{user_name}'@'%';"
    )

    file.write(create_user_query + "\n")
    file.write(grant_privileges_query + "\n")
    for module in shared_modules:
        file.write(
            f"GRANT ALL PRIVILEGES ON `{module}`.* TO '{user_name}'@'%';\n"
        )
    file.write(f"GRANT SELECT ON `%`.* TO '{user_name}'@'%';\n")
    file.flush()

    # run those commands in sql
    os.system(f"mysql -p -h lmf-db.cin.ucsf.edu < {file.name}")


if __name__ == "__main__":
    add_user(sys.argv[1])
