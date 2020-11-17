# Edit and then run this script after activating the nwb_datajoint environment and a mysql login has been created for you


import datajoint as dj

dj.config['database.host'] = "lmf-db.cin.ucsf.edu"

# replace with your user name
dj.config['database.user'] = "user_name_here"

# change your password
dj.set_password()

# replace "password_here" with the password that you used in the dj.set_password() above
dj.config['database.password'] = "password_here"

dj.config.save_global()




