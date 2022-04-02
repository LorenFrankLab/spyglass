# Session groups

A session group is a collection of sessions. Each group has a name (primary key) and a description.

```python
from spyglass.common import SessionGroup

# Create a new session group
SessionGroup.add_group('test_group_1', 'Description of test group 1')

# Get the table of session groups
SessionGroup()

# Add a session to the group
SessionGroup.add_session_to_group('RN2_20191110_.nwb', 'test_group_1')

# Remove a session from a group
# SessionGroup.remove_session_from_group('RN2_20191110_.nwb', 'test_group_1')

# Get all sessions in group
SessionGroup.get_group_sessions('test_group_1')

# Update the description of a session group
SessionGroup.update_session_group_description('test_group_1', 'Test description')
```