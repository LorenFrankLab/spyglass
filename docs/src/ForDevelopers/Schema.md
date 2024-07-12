# Schema Design

This document gives a detailed overview of how to read a schema script,
including explations of the different components that define a pipeline.

1. Goals of a schema
2. Front matter
    1. Imports
    2. Schema declaration
3. Table syntax
    1. Class inheritance
    2. Explicit table types
    3. Definitions
    4. Methods
4. Conceptual table types

Some of this will be redundant with general Python best practices and DataJoint
documentation, but it is important be able to read a schema, espically if you
plan to write your own.

Later sections will depend on information presented in the article on
[Table Types](./TableTypes.md).

## Goals of a schema

- At its core, DataJoint is just a mapping between Python and SQL.
- SQL is a language for managing relational databases.
- DataJoint is opinionated about how to structure the database, and limits SQL's
    potential options in way that promotes good practices.
- Python stores ...
    - A copy of table definitions, that may be out of sync with the database.
    - Methods for processing data, that may be out of sync with existing data.

Good data provenance requires good version control and documentation to keep
these in sync.

## Example schema

This is the full example schema referenced in subsections below.

<details><summary>Full Schema</summary>

```python
"""Schema example for custom pipelines

Note: `noqa: F401` is a comment that tells linters to ignore the fact that
`Subject` seems unused in the file. If this table is only used in a table
definition string, the linter will not recognize it as being used.
"""

import random  # Package import
from typing import Union  # Individual class import
from uuid import UUID

import datajoint as dj  # Aliased package import
from custom_package.utils import process_df, schema_prefix  # custom functions
from spyglass.common import RawPosition, Subject  # noqa: F401
from spyglass.utils import SpyglassMixin  # Additional Spyglass features

schema = dj.schema(schema_prefix + "_example")  # schema name from string


# Model to demonstrate DataJoint syntax
@schema  # Decorator to define a table in the schema on the server
class ExampleTable(SpyglassMixin, dj.Manual):  # Inherit SpyglassMixin class
    """Table Description"""  # Table docstring, one-line if possible

    definition = """ # Table comment
    primary_key1 : uuid # randomized string
    primary_key2 : int  # integer
    ---
    secondary_key1 : varchar(32) # string of max length 32
    -> Subject # Foreign key reference, inherit primary key of this table
    """


# Model to demonstrate field aliasing with `proj`
@schema
class SubjBlinded(SpyglassMixin, dj.Manual):
    """Blinded subject table."""  # Class docstring for `help()`

    definition = """
    subject_id: uuid # id
    ---
    -> Subject.proj(actual_id='subject_id')
    """

    @property  # Static information, Table.property
    def pk(self):
        """Return the primary key"""  # Function docstring for `help()`
        return self.heading.primary_key

    @staticmethod  # Basic func with no reference to self instance
    def _subj_dict(subj_uuid: UUID):  # Type hint for argument
        """Return the subject dict"""
        return {"subject_id": subj_uuid}

    @classmethod  # Class, not instance. Table.func(), not Table().func()
    def hash(cls, argument: Union[str, dict] = None):  # Default value
        """Example class method"""
        return dj.hash.key_hash(argument)

    def blind_subjects(self, restriction: Union[str, dict]):  # Union is "or"
        """Import all subjects selected by the restriction"""
        insert_keys = [
            {
                **self._subj_dict(self.hash(key)),
                "actual_id": key["subject_id"],
            }
            for key in (Subject & restriction).fetch("KEY")
        ]
        self.insert(insert_keys, skip_duplicates=True)

    def return_subj(self, key: str):
        """Return the entry in subject table"""
        if isinstance(key, dict):  # get rid of extra values
            key = key["subject_id"]
        key = self._subj_dict(key)
        actual_ids = (self & key).fetch("actual_id")
        ret = [{"subject_id": actual_id} for actual_id in actual_ids]
        return ret[0] if len(ret) == 1 else ret


@schema
class MyParams(SpyglassMixin, dj.Lookup):  # Lookup allows for default values
    """Parameter table."""

    definition = """
    param_name: varchar(32)
    ---
    params: blob
    """
    contents = [  # Default values as list of tuples
        ["example1", {"A": 1, "B": 2}],
        ["example2", {"A": 3, "B": 4}],
    ]

    @classmethod
    def insert_default(cls):  # Not req for dj.Lookup, but Spyglass convention
        """Insert default values."""  # skip_duplicates prevents errors
        cls().insert(rows=cls.contents, skip_duplicates=True)


@schema
class MyAnalysisSelection(SpyglassMixin, dj.Manual):
    """Selection table."""  # Pair subjects and params for computation

    definition = """
    -> SubjBlinded
    -> MyParams
    """

    def insert_all(self, param_name="example1"):  # Optional helper function
        """Insert all subjects with given param name"""
        self.insert(
            [
                {**subj_key, "param_name": param_name}
                for subj_key in SubjBlinded.fetch("KEY")
            ],
            skip_duplicates=True,
        )


@schema
class MyAnalysis(SpyglassMixin, dj.Computed):
    """Analysis table."""

    # One or more foreign keys, no manual input
    definition = """
    -> MyAnalysisSelection
    """

    class MyPart(SpyglassMixin, dj.Part):
        """Part table."""

        definition = """
        -> MyAnalysis
        ---
        result: int
        """

    def make(self, key):
        # Prepare for computation
        this_subj = SubjBlinded().return_subj(key["subject_id"])
        param_key = {"param_name": key["param_name"]}
        these_param = (MyParams & param_key).fetch1("params")

        # Perform computation.
        # Ideally, all data is linked with foreign keys, but not enforced
        for pos_obj in RawPosition.PosObject * (Subject & this_subj):
            dataframe = (RawPosition.PosObject & pos_obj).fetch1_dataframe()
            result = process_df(dataframe, **these_param)

        part_inserts = []  # Prepare inserts, to minimize insert calls
        for _ in range(10):
            result += random.randint(0, 100)
            part_inserts.append(dict(key, result=result))

        self.insert1(key)  # Insert into 'master' first, then all parts
        self.MyPart().insert(rows=part_inserts, skip_duplicates=True)
```

</details>

## Front matter

At the beginning of the schema file, you'll find ...

- Script docstring
- Imports
    - Aliased imports
    - Package imports
    - Individual imports
    - Relative imports
- Schema declaration

```python
"""Schema example for custom pipelines

Note: `noqa: F401` is a comment that tells linters to ignore the fact that
`Subject` seems unused in the file. If this table is only used in a table
definition string, the linter will not recognize it as being used.
"""

import random  # Package import
from typing import Union  # Individual class import
from uuid import UUID

import datajoint as dj  # Aliased package import
from custom_package.utils import process_df, schema_prefix  # custom functions
from spyglass.common import RawPosition, Subject  # noqa: F401
from spyglass.utils import SpyglassMixin  # Additional Spyglass features

schema = dj.schema(schema_prefix + "_example")  # schema name from string
```

- The `schema` variable determines the name of the schema in the database.
- Existing schema prefixes (e.g., `common`) should not be added to without
    discussion with the Spyglass team.
- Database admins may be interested in limiting privileges on a per-prefix
    basis. For example, Frank Lab members use ...
- Their respective usernames for solo work
- Project-specific prefixes for shared work.

## Table syntax

Each table is defined as a Python class, with a `definition` attribute that
contains the SQL-like table definition.

### Class inheritance

The parentheses in the class definition indicate that the class inherits from.

This table is ...

- A `SpyglassMixin` class, which provides a number of useful methods specific to
    Spyglass as discussed in the [mixin article](../Features/Mixin.md).
- A DataJoint `Manual` table, which is a table that is manually populated.

```python
@schema  # Decorator to define a table in the schema on the server
class ExampleTable(SpyglassMixin, dj.Manual):  # Inherit SpyglassMixin class
    pass
```

### Table types

- [DataJoint types](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/tiers/):
    - `Manual` tables are manually populated.
    - `Lookup` tables can be populated on declaration, and rarely change.
    - `Computed` tables are populated by a method runs computations on upstream
        entries.
    - `Imported` tables are populated by a method that imports data from another
        source.
    - `Part` tables are used to store data that is conceptually part of another
        table.
- [Spyglass conceptual types](./TableTypes.md):
    - Optional upstream Data tables from a previous pipeline.
    - Parameter tables (often `dj.Lookup`) store parameters for analysis.
    - Selection tables store pairings of parameters and data to be analyzed.
    - Compute tables (often `dj.Computed`) store the results of analysis.
    - Merge tables combine data from multiple pipeline versions.

### Definitions

Each table can have a docstring that describes the table, and must have a
`definition` attribute that contains the SQL-like table definition.

- `#` comments are used to describe the table and its columns.

- `---` separates the primary key columns from the data columns.

- `field : datatype` defines a column using a
    [SQL datatype](https://datajoint.com/docs/core/datajoint-python/0.14/design/tables/attributes/)


- `->` indicates a foreign key reference to another table.

```python
@schema  # Decorator to define a table in the schema on the server
class ExampleTable(SpyglassMixin, dj.Manual):  # Inherit SpyglassMixin class
    """Table Description"""  # Table docstring, one-line if possible

    definition = """ # Table comment
    primary_key1 : uuid # randomized string
    primary_key2 : int  # integer
    ---
    secondary_key1 : varchar(32) # string of max length 32
    -> Subject # Foreign key reference, inherit primary key of this table
    """
```

### Methods

Many Spyglss tables have methods that provide functionality for the pipeline.

Check out our [API documentation](../api/index.md) for a full list of available
methods.

This example models subject blinding to demonstrate ...

- An aliased foreign key in the definition, using `proj` to rename the field.
- A static property that returns the primary key.
- A static method that returns a dictionary of subject information.
- A class method that hashes an argument.
- An instance method that self-inserts subjects based on a restriction.
- An instance method that returns the unblinded subject information.

```python
# Model to demonstrate field aliasing with `proj`
@schema
class SubjBlinded(SpyglassMixin, dj.Manual):
    """Blinded subject table."""  # Class docstring for `help()`

    definition = """
    subject_id: uuid # id
    ---
    -> Subject.proj(actual_id='subject_id')
    """

    @property  # Static information, Table.property
    def pk(self):
        """Return the primary key"""  # Function docstring for `help()`
        return self.heading.primary_key

    @staticmethod  # Basic func with no reference to self instance
    def _subj_dict(subj_uuid: UUID):  # Type hint for argument
        """Return the subject dict"""
        return {"subject_id": subj_uuid}

    @classmethod  # Class, not instance. Table.func(), not Table().func()
    def hash(cls, argument: Union[str, dict] = None):  # Default value
        """Example class method"""
        return dj.hash.key_hash(argument)

    def blind_subjects(self, restriction: Union[str, dict]):  # Union is "or"
        """Import all subjects selected by the restriction"""
        insert_keys = [
            {
                **self._subj_dict(self.hash(key)),
                "actual_id": key["subject_id"],
            }
            for key in (Subject & restriction).fetch("KEY")
        ]
        self.insert(insert_keys, skip_duplicates=True)

    def return_subj(self, key: str):
        """Return the entry in subject table"""
        if isinstance(key, dict):  # get rid of extra values
            key = key["subject_id"]
        key = self._subj_dict(key)
        actual_ids = (self & key).fetch("actual_id")
        ret = [{"subject_id": actual_id} for actual_id in actual_ids]
        return ret[0] if len(ret) == 1 else ret
```

### Example Table Types

#### Params Table

This stores the set of values that may be used in an analysis. For analyses that
are unlikely to change, consider specifying all parameters in the table's
secondary keys. For analyses that may have different parameters, of when
depending on outside packages, consider a `blob` datatype that can store a
python dictionary.

```python
@schema
class MyParams(SpyglassMixin, dj.Lookup):  # Lookup allows for default values
    """Parameter table."""

    definition = """
    param_name: varchar(32)
    ---
    params: blob
    """
    contents = [  # Default values as list of tuples
        ["example1", {"A": 1, "B": 2}],
        ["example2", {"A": 3, "B": 4}],
    ]

    @classmethod
    def insert_default(cls):  # Not req for dj.Lookup, but Spyglass convention
        """Insert default values."""  # skip_duplicates prevents errors
        cls().insert(rows=cls.contents, skip_duplicates=True)
```

#### Selection Table

This is the staging area to pair sessions with parameter sets. Depending on what
is inserted, you might pair the same subject with different parameter sets, or
different subjects with the same parameter set.

```python
@schema
class MyAnalysisSelection(SpyglassMixin, dj.Manual):
    """Selection table."""  # Pair subjects and params for computation

    definition = """
    -> SubjBlinded
    -> MyParams
    """

    def insert_all(self, param_name="example1"):  # Optional helper function
        """Insert all subjects with given param name"""
        self.insert(
            [
                {**subj_key, "param_name": param_name}
                for subj_key in SubjBlinded.fetch("KEY")
            ],
            skip_duplicates=True,
        )
```

#### Compute Table

This is how processing steps are paired with data entry. By running
`MyAnalysis().populate()`, the `make` method is called for each foreign key
pairing in the selection table. The `make` method should end in one or one
inserts into the compute table.

```python
@schema
class MyAnalysis(SpyglassMixin, dj.Computed):
    """Analysis table."""

    # One or more foreign keys, no manual input
    definition = """
    -> MyAnalysisSelection
    """

    class MyPart(SpyglassMixin, dj.Part):
        """Part table."""

        definition = """
        -> MyAnalysis
        ---
        result: int
        """

    def make(self, key):
        # Prepare for computation
        this_subj = SubjBlinded().return_subj(key["subject_id"])
        param_key = {"param_name": key["param_name"]}
        these_param = (MyParams & param_key).fetch1("params")

        # Perform computation.
        # Ideally, all data is linked with foreign keys, but not enforced
        for pos_obj in RawPosition.PosObject * (Subject & this_subj):
            dataframe = (RawPosition.PosObject & pos_obj).fetch1_dataframe()
            result = process_df(dataframe, **these_param)

        part_inserts = []  # Prepare inserts, to minimize insert calls
        for _ in range(10):
            result += random.randint(0, 100)
            part_inserts.append(dict(key, result=result))

        self.insert1(key)  # Insert into 'master' first, then all parts
        self.MyPart().insert(rows=part_inserts, skip_duplicates=True)
```

To see how tables of a given schema relate to one another, use a
[schema diagram](https://datajoint.com/docs/core/datajoint-python/0.14/design/diagrams/)
