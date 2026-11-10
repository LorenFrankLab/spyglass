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
- Spike Sorting v2 (`spyglass.spikesorting.v2`): the how-to is the
    [v2 notebooks](../notebooks/10_Spike_SortingV2.ipynb). The
    [migration guide](./SpikeSortingV2_Migration.md) covers moving from v1 and
    the setup to read before running v2. The
    [developer guide](../ForDevelopers/SpikeSortingV2Architecture.md) covers how
    the code is organized.
