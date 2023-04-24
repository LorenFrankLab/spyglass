#!/usr/bin/env python
import grp
import os
import sys
import tempfile

target_group = "kachery-users"


def add_module(module_name):
    print(f"Granting everyone permissions to module {module_name}")

    # create a tempoary file for the command
    file = tempfile.NamedTemporaryFile(mode="w")

    # find the kachery-users group
    groups = grp.getgrall()
    for group in groups:
        if group.gr_name == target_group:
            break

    # get a list of usernames
    for user in group.gr_mem:
        file.write(f"GRANT ALL PRIVILEGES ON `{module_name}\_%`.* TO `{user}`@'%';\n")
    file.flush()

    # run those commands in sql
    os.system(f"mysql -p -h lmf-db.cin.ucsf.edu < {file.name}")


if __name__ == "__main__":
    add_module(sys.argv[1])
