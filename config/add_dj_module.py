#!/usr/bin/env python
import grp
import os
import sys
import tempfile

TARGET_GROUP = "kachery-users"


def add_module(module_name):
    print(f"Granting everyone permissions to module {module_name}")

    # create a tempoary file for the command
    file = tempfile.NamedTemporaryFile(mode="w")

    # find the kachery-users group
    groups = grp.getgrall()
    group_found = False  # initialize the flag as False
    for group in groups:
        if group.gr_name == TARGET_GROUP:
            group_found = True  # set the flag to True when the group is found
            break

    # Check if the group was found
    if not group_found:
        sys.exit(f"Error: The target group {TARGET_GROUP} was not found.")

    # get a list of usernames
    for user in group.gr_mem:
        file.write(
            f"GRANT ALL PRIVILEGES ON `{module_name}\_%`.* TO `{user}`@'%';\n"
        )
    file.flush()

    # run those commands in sql
    os.system(f"mysql -p -h lmf-db.cin.ucsf.edu < {file.name}")


if __name__ == "__main__":
    add_module(sys.argv[1])
