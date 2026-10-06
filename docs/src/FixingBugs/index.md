# Fixing Bugs

This directory documents **one-time bugfix workflows**: tools written to detect
and retroactively repair data affected by a specific, now-fixed bug.

Unlike the guides under [Features](../Features/index.md), which describe ongoing
capabilities of Spyglass, these pages and their associated tables exist to
support a single repair effort tied to a specific issue. They are not part of
the normal user workflow, and the associated tables/tools may be deprecated once
the repair has been completed across all affected data.

- [Label Repair (#1513)](./LabelRepair.md) - Detects and repairs unit labels
    affected by a bug in `AutomaticCuration.get_labels` (PR #1281).
