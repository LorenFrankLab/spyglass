# Creating figurl views

### Spike sorting recording view

```python
import nwb_datajoint.common as ndc
import nwb_datajoint.figurl_views as ndf

query = ...

# To replace:
# (ndf.SpikeSortingRecordingView & query).delete()

ndf.SpikeSortingRecordingView.populate([(ndc.SpikeSortingRecording & query).proj()])
```

### Spike sorting view

```python
import nwb_datajoint.common as ndc
import nwb_datajoint.figurl_views as ndf

query = ...

# To replace:
# (ndf.SpikeSortingView & query).delete()

ndf.SpikeSortingView.populate([(ndc.SpikeSorting & query).proj()])
```