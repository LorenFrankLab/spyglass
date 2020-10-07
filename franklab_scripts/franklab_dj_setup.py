# run this script after activating the nwb_datajoint environment and a mysql login has been created for you


import datajoint as dj

dj.config['database.host'] = "lmf-db.cin.ucsf.edu"
dj.config['database.user'] = "user_name_here"

# change your password
dj.set_password()

dj.config['database.password'] = "password_here"
dj.config.save_global()




