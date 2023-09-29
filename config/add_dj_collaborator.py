#!/usr/bin/env python
import os
import sys
import tempfile


def add_collab_user(user_name):
    # create a temporary file for the command
    file = tempfile.NamedTemporaryFile(mode="w")

    # Create the user (if not already created) and set the password
    file.write(
        f"CREATE USER IF NOT EXISTS '{user_name}'@'%' IDENTIFIED BY 'temppass';\n"
    )

    # Grant privileges to databases matching the user_name pattern
    file.write(
        f"GRANT ALL PRIVILEGES ON `{user_name}\_%`.* TO '{user_name}'@'%';\n"
    )

    # Grant SELECT privileges on all databases
    file.write(f"GRANT SELECT ON `%`.* TO '{user_name}'@'%';\n")

    file.flush()

    # run those commands in sql
    os.system(f"mysql -p -h lmf-db.cin.ucsf.edu < {file.name}")


if __name__ == "__main__":
    add_collab_user(sys.argv[1])
