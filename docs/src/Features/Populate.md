# Populate and Long-Running Computations

## Why

DataJoint wraps each `populate` call in a database transaction. This protects
data integrity, but it also holds a table lock for the entire duration of the
computation. For analyses that take seconds, this is fine. For analyses that
take minutes or hours (spike sorting, LFP filtering, decoding), the open
transaction blocks other users from modifying related tables — even for
unrelated sessions.

The old workaround was to set `_use_transaction = False` on the table class.
This bypassed the transaction wrapper entirely, which removed the lock but also
removed the data-integrity guarantees. It is now deprecated.

## What: tri-part make

The replacement is the **tri-part make** pattern: split the monolithic `make`
method into three methods with explicit responsibilities.

| Method                   | Responsibility                   | DB access  |
| ------------------------ | -------------------------------- | ---------- |
| `make_fetch(key)`        | Read inputs from upstream tables | Read only  |
| `make_compute(key, ...)` | Run the computation              | None       |
| `make_insert(key, ...)`  | Write results to the database    | Write only |

Spyglass's `populate` calls these three methods in sequence. `make_fetch` and
`make_compute` run outside the transaction; `make_insert` runs inside one. The
long computation no longer holds a lock, but the database write is still atomic.

## How

```python
import datajoint as dj
from spyglass.utils import SpyglassMixin

schema = dj.schema("my_schema")


@schema
class MyHeavyTable(SpyglassMixin, dj.Computed):
    definition = """
    -> UpstreamTable
    ---
    result: float
    """

    _parallel_make = True  # enables tri-part populate

    def make_fetch(self, key):
        """Read inputs. No database writes allowed here."""
        data = (UpstreamTable & key).fetch1("raw_data")
        params = (ParameterTable & key).fetch1("params")
        return [data, params]

    def make_compute(self, key, data, params):
        """Run the computation. No database access allowed here."""
        result = heavy_analysis(data, params)  # can take minutes
        return [{"result": result}]

    def make_insert(self, key, insert_dict):
        """Write results. Runs inside a transaction."""
        self.insert1(dict(key, **insert_dict))
```

**Rules:**

1. `make_fetch` must only read — no inserts, updates, or deletes.
2. `make_fetch` must be deterministic: the same key always returns the same
    data.
3. `make_compute` must not access the database at all.
4. `make_insert` is the only method that writes to the database.
5. Each method returns a list; the next method receives those values as
    positional arguments after `key`.

For a detailed walkthrough with a real Spyglass table, see
[Custom Pipelines — Make Method](../ForDevelopers/CustomPipelines.md#make-method).

## Migration from `_use_transaction = False`

If your table currently sets `_use_transaction = False`, Spyglass emits a
deprecation warning on every `populate` call and falls back to the old
no-transaction behavior. Migrate by removing the attribute and splitting your
`make` into three methods.

#### Before

```python
@schema
class MyHeavyTable(SpyglassMixin, dj.Computed):
    definition = """
    -> UpstreamTable
    ---
    result: float
    """

    _use_transaction = False  # deprecated — remove this

    def make(self, key):
        # step 1: fetch
        data = (UpstreamTable & key).fetch1("raw_data")
        params = (ParameterTable & key).fetch1("params")

        # step 2: compute (long-running, holds no lock under old pattern)
        result = heavy_analysis(data, params)

        # step 3: insert
        self.insert1(dict(key, result=result))
```

#### After

```python
@schema
class MyHeavyTable(SpyglassMixin, dj.Computed):
    definition = """
    -> UpstreamTable
    ---
    result: float
    """

    _parallel_make = True  # replaces _use_transaction = False

    def make_fetch(self, key):
        data = (UpstreamTable & key).fetch1("raw_data")
        params = (ParameterTable & key).fetch1("params")
        return [data, params]

    def make_compute(self, key, data, params):
        result = heavy_analysis(data, params)
        return [{"result": result}]

    def make_insert(self, key, insert_dict):
        self.insert1(dict(key, **insert_dict))
```

**NOTE:** The `no_transaction_make` deprecation warning is triggered by the
`_use_transaction = False` class attribute — not by calling `populate` with
`use_transaction=False` as a keyword argument. The keyword argument is still a
supported way to override transaction behavior at call time.
