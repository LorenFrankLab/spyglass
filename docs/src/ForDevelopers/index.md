# For Developers

This folder covers the process of developing new pipelines and features to be
used with Spyglass.

## Contributing

If you're looking to contribute to the project itself, either by adding a new
pipeline or improving an existing one, please review the article on
[contributing](./Contribute.md).

Any computation that might be useful for more that one project is a good
candidate for contribution. If you're not sure, feel free to
[open an issue](https://github.com/LorenFrankLab/spyglass/issues/new) to
discuss.

## Management

If you're looking to declare and manage your own instance of Spyglass, please
review the article on [database management](./Management.md).

## Custom

This folder also contains a number of articles on understanding pipelines in
order to develop your own.

- [Mixin Architecture](./Classes.md) explains the mixin-based class architecture,
    including design goals, organization, and usage patterns.
- [Code for Reuse](./Reuse.md) discusses good practice for writing readable and
    reusable code in Python.
- [Table Types](./TableTypes.md) explains the different table motifs in Spyglass
    and how to use them.
- [Schema design](./Schema.md) explains the anatomy of a Spyglass schema and
    gives a model for writing your version of each of the types of tables.
- [Pipeline design](./CustomPipelines.md) explains the process of turning an
    existing analysis into a Spyglass pipeline.
- [Using NWB](./UsingNWB.md) explains how to use the NWB format in Spyglass.

If you'd like help in developing a new pipeline, please reach out to the
Spyglass team via our
[discussion board](https://github.com/LorenFrankLab/spyglass/discussions).
