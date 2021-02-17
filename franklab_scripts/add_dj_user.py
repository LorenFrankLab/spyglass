#!/usr/bin/env python


import sys
import tempfile
import os

# get the argument and make sure it's a username
username = sys.argv[1]
if os.path.isdir('/home/' + username):
    print('Creating database user ', username)
else:
    errormessage = 'Error: username ' + username + ' does not exist in /home.'
    exit(errormessage)

# create a tempoary file for the command
file = tempfile.NamedTemporaryFile(mode='w')

file.write("GRANT ALL PRIVILEGES ON `{}\_%`.* TO `{}`@\'%\' IDENTIFIED BY \'temppass\';\n".format(username, username))
file.write("GRANT ALL PRIVILEGES ON `common\_%`.* TO `{}`@'%';\n".format(username))
file.write("GRANT SELECT ON `%`.* TO `{}`@'%';\n".format(username))
file.flush()

# run those commands in sql
os.system('mysql -p -h lmf-db.cin.ucsf.edu < {}'.format(file.name))
# os.system('cat {}'.format(file.name))
