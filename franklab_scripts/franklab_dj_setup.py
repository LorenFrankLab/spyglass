# run this script after activating the nwb_datajoint environment and after someone creates a mysql login for you


import datajoint as dj

dj.config['database.host'] = "lmf-db.cin.ucsf.edu"
dj.config['database.user'] = "user_name_here"
dj.config['database.password'] = "password_here"
dj.config.save_global()



