# Features

This directory contains a series of explainers on tools that have been added to
Spyglass.

- [Analysis File Tables](./AnalysisTables.md) - Guide to creating and managing
    analysis NWB files.
- [Export](./Export.md) - How to export an analysis.
- [FigURL](./FigURL.md) - How to use FigURL to share figures.
- [Merge Tables](./Merge.md) - Tables for pipeline versioning
- [Mixin](./Mixin.md) - Spyglass-specific functionalities to DataJoint tables,
    including fetching NWB files, long-distance restrictions, and permission
    checks on delete operations.
- [Interval](./Intervals.md) - The `Interval` class for creating, combining, and
    querying time windows. Includes migration from deprecated `interval_list_*`
    functions.
- [Populate](./Populate.md) - Tri-part make pattern for long-running
    computations. Includes migration from deprecated `_use_transaction = False`.
