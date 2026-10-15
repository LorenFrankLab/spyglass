"""Spyglass position tracking package.

Provides tables and utilities for tracking animal position from video:

- ``PositionOutput``: merge table that unifies all position sources
  (DLC V1, Trodes, PoseV2, ImportedPose) behind a single fetch interface.
- ``v2``: modern pose-estimation pipeline built on ndx-pose and DataJoint.
- ``v1``: legacy DLC and Trodes pipelines (maintained for backwards compat).
"""

from spyglass.position.position_merge import PositionOutput
