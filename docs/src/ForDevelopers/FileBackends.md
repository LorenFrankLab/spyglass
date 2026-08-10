# File Backends

This article explains how Spyglass resolves an NWB file that is not present on
local disk, and how to add a new source of files.

## Resolution chain

`get_nwb_file` walks an ordered list of backends, asking each whether it has the
file before asking it to open one. The first backend that has the file wins.

| Order | Backend          | Behavior                                     |
| ----- | ---------------- | -------------------------------------------- |
| 1     | `LocalBackend`   | Reads from disk                              |
| 2     | `KacheryBackend` | Downloads, then reads locally                |
| 3     | `DandiBackend`   | Streams, or downloads on request (see below) |

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

## Preferring download over streaming

Streaming is the default for backends that support it, and is usually the right
choice: only the chunks an analysis touches cross the network. On a slow or
metered connection the arithmetic can invert, because many small range requests
cost more than one sequential transfer.

Set `prefer_download` to make every stream-capable backend fetch the whole file
and read the local copy instead. In `dj_local_conf.json`:

```json
{
  "custom": {
    "prefer_download": true
  }
}
```

Or for the current session only, which is the usual case — the same user is fast
on the lab network and slow from a laptop:

```python
from spyglass.settings import sg_config

sg_config.prefer_download = True
```

`sg_config.save_dj_config()` persists it like any other custom key. There is no
environment variable for it: the two forms above cover both the durable and the
one-off case.

Two things it deliberately does not do. It never reorders the chain, so a file
already on disk is still read from disk. And it never causes a failure: a
backend that can only stream logs a debug message and streams anyway, since
serving the file matters more than honoring a performance preference.

Read it live as `sg_config.prefer_download` rather than importing a module-level
name — the session setter above would not be visible in a value captured at
import.

!!! note

    Streaming already caches. `DandiBackend` reads through an `fsspec`
    `CachingFileSystem` backed by `{export_dir}/nwb-cache`, so re-reading the same
    chunks does not re-cross the network. If a user reports slowness, check whether
    they are paying for first reads or for a cold cache before reaching for this
    setting.

## The `FileBackend` protocol

A backend is any object with a `name`, a `supports_streaming` flag, and the
methods below. `FileBackend` is both the structural type and the base class:
inheriting from it supplies the concrete `open` and requires `has`, while a
class that implements the same members without inheriting still satisfies
`isinstance` checks.

```python
from spyglass.utils.file_backends import FileBackend


class MyBackend(FileBackend):
    name = "my_store"
    supports_streaming = True

    def has(self, nwb_file_path: str) -> bool:
        """Return True if this backend can supply the file."""

    def download(self, nwb_file_path: str, dest: str = None) -> bool:
        """Fetch the file to local disk. Return True on success."""

    def stream(self, nwb_file_path: str):
        """Return an (io, nwbfile) pair read over the network."""
```

### Required: `has`

Marked `@abstractmethod`, so a subclass that omits it raises `TypeError` at
instantiation. Keep it cheap. It runs on every unresolved file, and its job is
only to decide whether a transfer is worth attempting.

Return `False` rather than raising when the backend is unavailable. For example,
`KacheryBackend.has` checks whether `kachery_cloud` is importable before it
queries the database, so a missing optional dependency skips the backend instead
of breaking file access.

### Streaming or downloading: `stream` and `download`

Implement at least one. `open` is concrete and picks between them, so a backend
that can do both needs no dispatch logic of its own:

- **Download-only** backends implement `download`. If it returns `False`, `open`
    raises `FileNotFoundError` and `get_nwb_file` moves on to the next backend.
- **Streaming-only** backends implement `stream`, returning an `(io, nwbfile)`
    pair read over the network.
- **Both** is the most useful shape, because it lets the user choose. `open`
    streams unless `prefer_download` is set.

Whichever you leave unimplemented raises `NotImplementedError` naming the
backend, so a misconfigured chain fails legibly.

Set `supports_streaming` to match your `stream` implementation. It is declared
rather than inferred, so callers can reason about a backend without invoking it.
It describes what the backend *can* do; what `open` actually does on a given
call is that capability narrowed by the user's preference.

Overriding `open` is reserved for backends where the stream/download split does
not apply — `LocalBackend` does it, because reading a file already on disk is
neither.

A backend that reports a file in `has` but then fails to supply it does not halt
resolution: `get_nwb_file` catches `FileNotFoundError` and continues down the
chain.

If your `download` writes to the destination path directly, write to a temporary
path and rename on success. A partial file left at the expected path is worse
than no file, because `LocalBackend` will hand it to the next caller as a valid
local copy.

## How the shipped backends map on

**`LocalBackend`** checks `os.path.exists` and reads the file directly. It
neither streams nor downloads, so it is the one backend that overrides `open`.

**`KacheryBackend`** is download-only; kachery-cloud has no streaming path. Its
`has` is a restriction on `AnalysisNwbfileKachery`, which is cheaper than the
older behavior of attempting a transfer to find out. Kachery is deprecated and
will be removed once the shared store replaces it.

**`DandiBackend`** implements both. It resolves a file under either the analysis
or raw naming scheme, then either reads it through `fsspec` with a local cache —
so only the chunks an analysis touches cross the network — or fetches the whole
file to the local path Spyglass expects, so the next call resolves locally.

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
