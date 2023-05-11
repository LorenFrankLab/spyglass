# Object-oriented interface
Instead of interacting directly with datajoint tables, give the user a more object-oriented interface. For example, ingesting an NWB file returns a spyglass `session` object:
```python
import spyglass as sg
nwb_file = 'test.nwb'
session = sg.insert_session(nwb_file)
```
You can access everything about the session (e.g. information in the common table) as _attributes_ of the session object. Calling these attributes prints the datajoint table:
```python
# examples
session.electrode
session.raw
session.interval
```
To do analysis, set parameters by calling _methods_ associated with the sesison object
```python
session.insert_electrode_group(electrode_group_name='tetrode1', electrode_ids=[1,2,3,4])
session.insert_interval(interval_name='epoch1', interval=[10,20])
```
You can also get parameters already saved in the database
```python
print(sg.default_filter_params)
```
and then run the computation
```python
# this runs LFP.populate under the hood
session.compute_lfp(electrode_group_name='tetrode1', interval_name='epoch1', filter_param_name='default')
# this creates a figurl
session.lfp.generate_figurl()
```
Later, the user can reinstantiate the session object from the nwb file.
```python
session = sg.load_session('test.nwb')
```
Advantages
- simpler and more intuitive than interacting with datajoint tables
- awkwardness in the pipelines (e.g. Selection tables) are hidden
Disadvantages
- cannot do queries; for this we would just need to use the datajoint API
- possibly limited in scope