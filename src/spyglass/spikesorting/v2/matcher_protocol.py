"""Plugin interface + registry for cross-session unit matchers.

A *matcher* takes per-session, wrapper-prepared waveform bundles and returns
pairwise cross-session unit matches. The interface is deliberately narrow so
external backends (UnitMatch first, others later) can be added without touching
the DataJoint tables:

- :class:`SessionMatcherInput` -- one wrapper-prepared bundle per session. The
  matcher reads only these directories; it never sees a ``SortingAnalyzer``
  object, a recording, or a Spyglass table key.
- :class:`MatchPair` -- one cross-session match, keyed by
  ``(sorting_id, curation_id, unit_id)`` on each side.
- :class:`MatcherProtocol` -- the ``match(session_inputs, params)`` contract.
- :func:`register_matcher` / :func:`get_matcher` -- the name -> backend +
  per-matcher params-schema registry that ``MatcherParameters`` validates
  against at insert time.

This module is pure Python with no DataJoint dependency, so it is importable
and testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spyglass.spikesorting.v2.exceptions import UnknownMatcherError


@dataclass(frozen=True)
class SessionMatcherInput:
    """One per-session bundle the wrapper prepares for the matcher.

    Attributes
    ----------
    session_key : dict
        ``{"sorting_id": UUID, "curation_id": int}`` identifying the curated
        sorting this bundle was extracted from. The matcher echoes these keys
        back on every :class:`MatchPair`; it does not use them to read data.
    waveform_dir : pathlib.Path
        Directory holding the per-unit waveform arrays in the backend's
        expected layout (for UnitMatch: ``RawWaveforms/Unit{id}_RawSpikes.npy``
        of shape ``(spike_width, n_channels, 2)`` plus a ``cluster_group.tsv``).
    channel_positions_path : pathlib.Path
        ``.npy`` file of shape ``(n_channels, 2)`` with the probe geometry.
    recording_date : Any
        A canonical UTC ISO 8601 string derived from
        ``Session.session_start_time`` (used for chronological drift ordering;
        UTC-normalized so plain string comparison is chronological); may be
        ``None`` when a backend does not need it.
    """

    session_key: dict
    waveform_dir: Path
    channel_positions_path: Path
    recording_date: Any = None


@dataclass(frozen=True)
class MatchPair:
    """One cross-session unit match, keyed by curated-unit identity per side.

    ``drift_estimate_um`` and ``fdr_estimate`` have no per-pair source in
    UnitMatch (drift is applied internally per session-pair; FDR is a
    session-level diagnostic), so they default to ``0.0`` / ``None`` and a
    backend only sets them if it genuinely produces per-pair values.
    """

    session_a_sorting_id: str
    session_a_curation_id: int
    unit_a_id: int
    session_b_sorting_id: str
    session_b_curation_id: int
    unit_b_id: int
    match_probability: float
    drift_estimate_um: float = 0.0
    fdr_estimate: float | None = None


@runtime_checkable
class MatcherProtocol(Protocol):
    """Structural interface every cross-session matcher backend implements.

    A backend is any object with a ``name`` string and a ``match`` method; it
    need not subclass anything. ``match`` consumes wrapper-prepared bundles and
    returns the cross-session matches, returning ``[]`` for the degenerate
    single-session case (one input) rather than raising.
    """

    name: str

    def match(
        self,
        session_inputs: list[SessionMatcherInput],
        params: dict,
    ) -> list[MatchPair]: ...


#: name -> backend instance
_MATCHER_REGISTRY: dict[str, MatcherProtocol] = {}
#: name -> per-matcher Pydantic params schema (validates ``MatcherParameters``)
_SCHEMA_REGISTRY: dict[str, type] = {}


def register_matcher(
    matcher: MatcherProtocol, schema: type, *, replace: bool = False
) -> None:
    """Register a matcher backend and its params schema under ``matcher.name``.

    Parameters
    ----------
    matcher : MatcherProtocol
        A backend with a ``name`` attribute and a ``match`` method.
    schema : type
        The Pydantic model validating that matcher's ``MatcherParameters``
        ``params`` blob.
    replace : bool, optional
        Explicit maintenance override. Registering a name already held by a
        DIFFERENT backend class raises ``ValueError`` unless ``replace=True`` --
        ``MatcherParameters`` stores only the matcher name, so silently pointing
        a name at different code would make existing rows dispatch elsewhere.
        Re-registering the SAME backend class is always idempotent and needs no
        flag (this is how :func:`register_default_matchers` self-heals), so the
        built-in registration does NOT use ``replace`` -- it is reserved for a
        deliberate swap to genuinely different code.

    Raises
    ------
    TypeError
        If ``matcher`` does not satisfy :class:`MatcherProtocol`.
    ValueError
        If ``matcher.name`` is already registered to a different backend class
        and ``replace`` is ``False``.
    """
    if not isinstance(matcher, MatcherProtocol) or not callable(
        getattr(matcher, "match", None)
    ):
        raise TypeError(
            f"{matcher!r} does not satisfy MatcherProtocol (needs a `name` "
            "attribute and a callable `match(session_inputs, params)` method)."
        )
    existing = _MATCHER_REGISTRY.get(matcher.name)
    if (
        existing is not None
        and type(existing) is not type(matcher)
        and (not replace)
    ):
        raise ValueError(
            f"A matcher named {matcher.name!r} is already registered to a "
            f"different backend ({type(existing).__module__}."
            f"{type(existing).__qualname__}). MatcherParameters rows store only "
            "the matcher name, so re-pointing a name at different code would "
            "silently re-route existing rows. Pick a distinct name, or pass "
            "replace=True to override deliberately."
        )
    _MATCHER_REGISTRY[matcher.name] = matcher
    _SCHEMA_REGISTRY[matcher.name] = schema


def register_default_matchers() -> None:
    """Register the built-in matcher backends (idempotent, self-healing).

    The backends register themselves as an import side effect, so the registry
    is empty until a backend module is imported. This makes that bootstrap
    explicit -- callers (and the lookups below) can ensure the built-ins are
    present without depending on import order -- and re-registers them even if
    the registry was cleared (e.g. by a test fixture).
    """
    # Function-level import avoids an import cycle (the backend imports this
    # module) and keeps the optional UnitMatchPy import lazy (the backend only
    # imports UnitMatchPy when its match()/extract path actually runs).
    from spyglass.spikesorting.v2 import _unitmatch_backend

    _unitmatch_backend.register()


def _registered_matchers() -> frozenset[str]:
    """Return the set of registered matcher names (built-ins ensured)."""
    register_default_matchers()
    return frozenset(_MATCHER_REGISTRY)


def _raise_unknown(name: str) -> None:
    raise UnknownMatcherError(
        f"Unknown matcher {name!r}. Registered matchers: "
        f"{sorted(_registered_matchers())}. To add a new matcher, implement "
        "MatcherProtocol and register it via register_matcher() before "
        "inserting parameters."
    )


def get_matcher(name: str) -> MatcherProtocol:
    """Return the registered backend for ``name`` or raise UnknownMatcherError."""
    register_default_matchers()
    if name not in _MATCHER_REGISTRY:
        _raise_unknown(name)
    return _MATCHER_REGISTRY[name]


def _get_matcher_schema(name: str) -> type:
    """Return the params schema for ``name`` or raise UnknownMatcherError."""
    register_default_matchers()
    if name not in _SCHEMA_REGISTRY:
        _raise_unknown(name)
    return _SCHEMA_REGISTRY[name]
