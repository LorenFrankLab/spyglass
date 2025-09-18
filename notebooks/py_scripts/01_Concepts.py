# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: spy
#     language: python
#     name: python3
# ---

# # Concepts
#

# ## Intro
#

# _Developer Note:_ if you may make a PR in the future, be sure to copy this
# notebook, and use the `gitignore` prefix `temp` to avoid future conflicts.
#
# This is one notebook in a multi-part series on Spyglass. To set up your Spyglass environment and database, see
# [the Setup notebook](./00_Setup.ipynb)
#
# This notebook will introduce foundational concepts that will help in
# understanding how to work with Spyglass pipelines.
#

# ## Other materials
#
# DataJoint is an tool that helps us create Python classes for
# tables that exist on a shared SQL server. Many Spyglass
# imports are DataJoint tables like this.
#
# Any 'introduction to SQL' will give an overview of relational data models as
# a primer on how DataJoint tables within Spyglass will interact with one-another,
# and the ways we can interact with them. A quick primer may help with the
# specifics ahead.
#
# For an overview of DataJoint, including table definitions and inserts, see
# [DataJoint tutorials](https://github.com/datajoint/datajoint-tutorials).

# ## Common Errors
#
# Skip this for now, but refer back if you hit issues.
#
#
# ### Integrity
#
# ```console
# IntegrityError: Cannot add or update a child row: a foreign key constraint fails (`schema`.`_table`, CONSTRAINT `_table_ibfk_1` FOREIGN KEY (`parent_field`) REFERENCES `other_schema`.`parent_name` (`parent_field`) ON DELETE RESTRICT ON UPDATE CASCADE)
# ```
#
# `IntegrityError` during `insert` means that some part of the key you're
# inserting doesn't exist in the parent of the table you're inserting into. You
# can explore which that may be by doing the following...
#
# ```python
# my_key = dict(value=key)  # whatever you're inserting
# MyTable.insert1(my_key)  # error here
# parents = MyTable().find_insert_fail(my_key)
# ```
#
# If any of the printed tables are empty, you know you need to insert into that
# table (or another ancestor up the pipeline) first. This code will not work if
# there are aliases in the table (i.e., `proj` in the definition). In that case,
# you'll need to modify your `parent_key` to reflect the renaming.
#
# The error message itself will tell you which table is the limiting parent. After
# `REFERENCES` in the error message, you'll see the parent table and the column
# that is causing the error.
#
# ### Permission
#
# ```console
# ('Insufficient privileges.', "INSERT command denied to user 'username'@'127.0.0.1' for table '_table_name'", 'INSERT INTO `schema_name`.`table_name`(`field1`,`field2`) VALUES (%s,%s)')
# ```
#
# This is a MySQL error that means that either ...
#
# - You don't have access to the command you're trying to run (e.g., `INSERT`)
# - You don't have access to this command on the schema you're trying to run it on
#
# To see what permissions you have, you can run the following ...
#
# ```python
# dj.conn().query("SHOW GRANTS FOR CURRENT_USER();").fetchall()
# ```
#
# If you think you should have access to the command, you contact your database
# administrator (e.g., Chris in the Frank Lab). Please share the output of the
# above command with them.
#
# ### Type
#
# ```console
# TypeError: example_function() got an unexpected keyword argument 'this_arg'
# ```
#
# This means that you're calling a function with an argument that it doesn't
# expect (e.g., `example_function(this_arg=5)`). You can check the function's
# accepted arguments by running `help(example_function)`.
#
# ```console
# TypeError: 'NoneType' object is not iterable
# ```
#
# This means that some function is trying to do something with an object of an
# unexpected type. For example, if might by running `for item in variable: ...`
# when `variable` is `None`. You can check the type of the variable by going into
# debug mode and running `type(variable)`.
#
# ### KeyError
#
# ```console
# KeyError: 'field_name'
# ```
#
# This means that you're trying to access a key in a dictionary that doesn't
# exist. You can check the keys of the dictionary by running `variable.keys()`. If
# this is in your custom code, you can get a key and supply a default value if it
# doesn't exist by running `variable.get('field_name', default_value)`.
#
# ### DataJoint
#
# ```console
# DataJointError("Attempt to delete part table {part} before deleting from its master {master} first.")
# ```
#
# This means that DataJoint's delete process found a part table with a foreign key
# reference to the data you're trying to delete. You need to find the master table
# listed and delete from that table first.
#

# ## Debug Mode
#
# To fix an error, you may want to enter 'debug mode'. VSCode has a dedicated
# featureful [extension](https://code.visualstudio.com/docs/python/debugging)
# for making use of the UI, but you can choose to use Python's built-in tool.
#
# To enter into debug mode, you can add the following line to your code ...
#
# ```python
# __import__("pdb").set_trace()
# ```
#
# This will set a breakpoint in your code at that line. When you run your code, it
# will pause at that line and you can explore the variables in the current frame.
# Commands in this mode include ...
#
# - `u` and `d` to move up and down the stack
# - `l` to list the code around the current line
# - `q` to quit the debugger
# - `c` to continue running the code
# - `h` for help, which will list all the commands
#
# `ipython` and jupyter notebooks can launch a debugger automatically at the last
# error by running `%debug`.
#
#

# ## Up Next
#

# Next, we'll try [inserting data](./01_Insert_Data.ipynb)
