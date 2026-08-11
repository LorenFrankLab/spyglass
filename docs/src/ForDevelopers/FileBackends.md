# File Backends

This article explains how Spyglass resolves an NWB file that is not present on
local disk, and how to add a new source of files.

## Resolution chain

`get_nwb_file` walks an ordered list of backends, asking each whether it has the
file before asking it to open one. The first backend that has the file wins.

| Order | Backend          | Behavior                                 |
| ----- | ---------------- | ---------------------------------------- |
| 1     | `LocalBackend`   | Reads from disk                          |
| 2     | `KacheryBackend` | Downloads, then reads locally            |
| 3     | `DandiBackend`   | Streams by default, downloads on request |

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

## Streaming and download

A backend that reads over the network can do so two ways, and which one it uses
is a user setting rather than a fixed property of the backend.

Streaming is the default: only the chunks an analysis touches cross the network.
On a slow or metered connection the arithmetic inverts, because many small range
requests cost more than one sequential transfer. Setting `prefer_download` makes
every stream-capable backend fetch the whole file and read the local copy
instead.

In `dj_local_conf.json`, for a machine that is always on a slow link:

```json
{
  "custom": {
    "prefer_download": true
  }
}
```

Or for one session, where the same user is fast on the lab network and slow from
a laptop:

```python
from spyglass.settings import sg_config

sg_config.prefer_download = True
```

`sg_config.save_dj_config()` persists it like any other custom key. There is no
environment variable; the two forms above cover the durable and the one-off
case. Read the value live as `sg_config.prefer_download` — a module-level name
captured at import would not see the session setter.

The setting changes how a file is fetched, not which backend supplies it. Chain
order is unaffected, so a file already on disk is still read from disk. A
backend that can only stream logs a debug message and streams anyway: serving
the file matters more than honoring a performance preference.

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

    def download(self, nwb_file_path: str, dest: str | None = None) -> bool:
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

### Transferring the file: `stream` and `download`

Implement at least one. `open` is concrete and picks between them, so a backend
that can do both needs no dispatch logic of its own:

- **Download-only** backends implement `download`. If it returns `False`, `open`
    raises `BackendUnavailable` and `get_nwb_file` moves on to the next backend.
- **Streaming-only** backends implement `stream`, returning an `(io, nwbfile)`
    pair read over the network.
- **Both** is the most useful shape, because it lets the user choose.

Whichever you leave unimplemented raises `NotImplementedError` naming the
backend, so a misconfigured chain fails legibly.

If your `download` writes to the destination path directly, write to a temporary
path and rename on success. A partial file left at the expected path is worse
than no file, because `LocalBackend` will hand it to the next caller as a valid
local copy.

Overriding `open` is reserved for backends where the split does not apply —
`LocalBackend` does it, because reading a file already on disk is neither.

### Choosing between them: `will_stream`

`open` calls `will_stream(nwb_file_path)` to pick its path, and the resolver
calls it to record how the file was read, which is what `file_is_remote` reports
later. The inherited implementation combines `supports_streaming` with the
user's `prefer_download` setting; returning per-file answers is fine if your
backend streams some files and downloads others.

Set `supports_streaming` to match your `stream` implementation. It is declared
rather than inferred, so callers can reason about a backend without invoking it.
It describes what the backend *can* do, while `will_stream` answers what a given
call will actually do.

**If you override `open`, override `will_stream` to match.** They are the same
decision, and a disagreement means a streamed file gets treated as local.

### Reporting a miss

Prefer a return value. `has` answers "is this worth attempting?" before any
transfer, and `download` returns `False` for "I tried and got nothing" — `open`
converts that into the exception the resolver expects. Between them these cover
most cases.

`stream` is the exception, because it owes its caller an `(io, nwbfile)` pair
and has no value with which to say no:

```python
from spyglass.utils.file_backends import BackendUnavailable

raise BackendUnavailable(f"'{self.name}' has no asset for {name}")
```

`get_nwb_file` catches **only** `BackendUnavailable` and moves to the next
backend. Everything else propagates. The distinction is load-bearing: a corrupt
file, a dropped connection, or a DataJoint error is not a miss, and treating it
as one would hide the real fault and silently recompute a file that already
exists.

`BackendUnavailable` subclasses `FileNotFoundError`, so callers catching the
broader type catch it too; only resolution is narrow.

## The shipped backends

**`LocalBackend`** checks `os.path.exists` and reads the file directly. It
neither streams nor downloads, so it is the one backend that overrides `open`.

**`KacheryBackend`** is download-only; kachery-cloud has no streaming path. Its
`has` is a restriction on `AnalysisNwbfileKachery`. Kachery is deprecated and
will be removed once the shared store replaces it.

**`DandiBackend`** implements both. A single lookup resolves the file under
either the analysis or raw naming scheme and backs `has` as well as both
transfer methods, so the archive is queried once per question. Streaming reads
through `fsspec` with a local cache; downloading fetches the whole file to the
local path Spyglass expects, so the next call resolves locally.

## Detecting a streamed file

`file_is_remote(path)` reports whether an open file was read over the network.
Use it to skip operations that assume a local file, such as DataJoint filepath
checksums.

```python
from spyglass.utils.nwb_helper_fn import file_is_remote
```

The answer comes from the backend that opened the file: the resolver asks
`will_stream` and records it, so any streaming backend is recognized — HTTP, S3,
ROS3, or something not yet written.

It reports what happened, not what the backend can do. A stream-capable backend
that downloaded because the user set `prefer_download` produced a real local
file, so `file_is_remote` returns `False` and checksums proceed normally. Paths
that are not open return `False`.

!!! note

    This was previously `file_from_dandi`. That name still works but is deprecated:
    the check is not DANDI-specific, and detects any streaming backend.

## Adding a backend

1. Implement the protocol as above.
2. Add an instance to the chain in `spyglass.utils.file_backends`, ordered by
    cost: cheap and local before slow and remote.
3. Add tests that do not require network access. The existing suite in
    `tests/utils/test_file_backends.py` uses fakes to assert chain order and
    fall-through behavior without touching a real store.

Registering a backend at runtime is not supported. Backends are declared in the
chain, which keeps resolution order predictable and reviewable.
