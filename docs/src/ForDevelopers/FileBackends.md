# File Backends

This article explains how Spyglass resolves an NWB file that is not present on
local disk, and how to add a new source of files.

## Resolution chain

`get_nwb_file` walks an ordered list of backends, asking each whether it has the
file before asking it to open one. The first backend that has the file wins.

| Order | Backend          | Behavior                         |
| ----- | ---------------- | -------------------------------- |
| 1     | `LocalBackend`   | Reads from disk                  |
| 2     | `KacheryBackend` | Downloads, then reads locally    |
| 3     | `DandiBackend`   | Streams over HTTP range requests |

If no backend has the file and the calling table has a `_make_file` method, the
file is recomputed. Otherwise `get_nwb_file` raises `FileNotFoundError`.

The chain is fixed in code, in `spyglass.utils.file_backends`. It is
deliberately not user-configurable: local disk must be tried first, and putting
a network source ahead of it would only ever be a mistake.

```python
from spyglass.utils.file_backends import get_backends

[b.name for b in get_backends()]  # ['local', 'kachery', 'Dandi']
```

`get_backends` returns a copy, so callers cannot reorder the chain in place.

## The `FileBackend` protocol

A backend is any object with a `name`, a `supports_streaming` flag, and three
methods. `FileBackend` is both the structural type and the base class:
inheriting from it supplies the default `open` and requires `has`, while a class
that implements the same members without inheriting still satisfies `isinstance`
checks.

```python
from spyglass.utils.file_backends import FileBackend


class MyBackend(FileBackend):
    name = "my_store"
    supports_streaming = False

    def has(self, nwb_file_path: str) -> bool:
        """Return True if this backend can supply the file."""

    def download(self, nwb_file_path: str, dest: str | None = None) -> bool:
        """Fetch the file to local disk. Return True on success."""
```

### Required: `has`

Marked `@abstractmethod`, so a subclass that omits it raises `TypeError` at
instantiation. Keep it cheap. It runs on every unresolved file, and its job is
only to decide whether a transfer is worth attempting.

Return `False` rather than raising when the backend is unavailable. For example,
`KacheryBackend.has` checks whether `kachery_cloud` is importable before it
queries the database, so a missing optional dependency skips the backend instead
of breaking file access.

### Streaming or downloading: `open` and `download`

Backends differ in what they can do, so only one of these is needed:

- **Download-only** backends implement `download` and inherit the default
    `open`, which downloads the file and then reads the local copy. If
    `download` returns `False`, the default `open` raises `FileNotFoundError`,
    and `get_nwb_file` moves on to the next backend.
- **Streaming** backends override `open` to return an `(io, nwbfile)` pair read
    over the network, and leave `download` alone. The inherited `download`
    raises `NotImplementedError` naming the backend.

Set `supports_streaming` to match. It is declared rather than inferred, so
callers can reason about a backend without invoking it.

A backend that reports a file in `has` but then fails to supply it does not halt
resolution: `get_nwb_file` catches `FileNotFoundError` and continues down the
chain.

## How the shipped backends map on

**`LocalBackend`** checks `os.path.exists` and reads the file directly. It
neither streams nor downloads, so it overrides `open`.

**`KacheryBackend`** is download-only; kachery-cloud has no streaming path. Its
`has` is a restriction on `AnalysisNwbfileKachery`, which is cheaper than the
older behavior of attempting a transfer to find out. Kachery is deprecated and
will be removed once the shared store replaces it.

**`DandiBackend`** is streaming-only. It resolves a file under either the
analysis or raw naming scheme, then reads it through `fsspec` with a local
cache, so only the chunks an analysis touches cross the network.

## Detecting a streamed file

`file_is_remote(path)` reports whether an open file is backed by HTTP rather
than disk. Use it to skip operations that assume a local file, such as DataJoint
filepath checksums.

```python
from spyglass.utils.nwb_helper_fn import file_is_remote
```

!!! note

    This was previously `file_from_dandi`. That name still works but is deprecated:
    the check was never DANDI-specific, since it detects any HTTP-backed filesystem.

## Adding a backend

1. Implement the protocol as above.
2. Add an instance to the chain in `spyglass.utils.file_backends`, ordered by
    cost: cheap and local before slow and remote.
3. Add tests that do not require network access. The existing suite in
    `tests/utils/test_file_backends.py` uses fakes to assert chain order and
    fall-through behavior without touching a real store.

Registering a backend at runtime is not supported. Backends are declared in the
chain, which keeps resolution order predictable and reviewable.
