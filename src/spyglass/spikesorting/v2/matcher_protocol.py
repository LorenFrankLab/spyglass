"""Plugin interface + registry for cross-session unit matchers.

A *matcher* takes wrapper-prepared bundles, one per matching input,
and returns pairwise cross-session unit matches. The interface is deliberately
narrow so backends (UnitMatch ships built in) can be added without touching
the DataJoint tables:

- :class:`SessionMatcherInput` -- one wrapper-prepared bundle per matching
  input (a sort of a single recording or of a same-day concatenation). The
  matcher reads only these directories; it never sees a ``SortingAnalyzer``
  object, a recording, or a Spyglass table key.
- :class:`MatchPair` -- one cross-session match, keyed by
  ``(sorting_id, curation_id, unit_id)`` on each side.
- :class:`MatcherProtocol` -- the ``match(session_inputs, params)`` contract.
- :class:`MatcherGeometryValidator` -- optional backend geometry preflight.
- :class:`MatcherInputPreparer` -- the independent preparation contract, from
  resolved :class:`MatcherInputSource` to :class:`PreparedMatcherInput`.
- :func:`register_matcher` / :func:`get_matcher` -- the name -> backend +
  input preparer + per-matcher params-schema registry that ``MatcherParameters`` validates
  against at insert time.

This module is pure Python with no DataJoint dependency, so it is importable
and testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spyglass.spikesorting.v2.exceptions import UnknownMatcherError

WAVEFORM_BUNDLE_LAYOUT = "split_half_waveforms"
WAVEFORM_BUNDLE_VERSION = 1


def _resolve_bundle_path(path, legacy_path, name: str) -> Path | None:
    """Resolve a canonical path and its constructor compatibility alias."""
    if (
        path is not None
        and legacy_path is not None
        and Path(path) != Path(legacy_path)
    ):
        raise ValueError(f"Conflicting {name} and legacy alias paths")
    resolved = path if path is not None else legacy_path
    return Path(resolved) if resolved is not None else None


@dataclass(frozen=True, init=False)
class SessionMatcherInput:
    """One bundle the wrapper prepares for the matcher per matching input.

    A matching input is one curated sort, of a single recording or of a
    same-day concatenation; its bundle holds that sort's units once each.
    Bundles are passed in ``input_index`` (chronological) order.

    Attributes
    ----------
    curation_key : dict
        ``{"sorting_id": UUID, "curation_id": int}`` identifying the curated
        sorting this bundle was extracted from. The matcher echoes these keys
        back on every :class:`MatchPair`; it does not use them to read data.
    bundle_dir : pathlib.Path
        Directory holding the prepared files in the backend's expected layout.
    layout, layout_version : str, int
        Format identifier and positive version, interpreted by the backend.
        The shared layout is ``split_half_waveforms`` version 1: per-unit
        ``RawWaveforms/Unit{id}_RawSpikes.npy`` arrays of shape
        ``(spike_width, n_channels, 2)`` plus ``cluster_group.tsv``.
    geometry_path : pathlib.Path or None
        Optional geometry file. In the shared waveform layout it is a ``.npy``
        file of shape ``(n_channels, 2)``. Other layouts define their own
        metadata, or omit geometry when it is not used.
    recording_date : Any
        The input's frozen start time
        (``UnitMatchSelection.Input.input_start_time``: the earliest
        ``Session.session_start_time`` among its constituent recordings) as a
        canonical UTC ISO 8601 string, so plain string comparison is
        chronological; may be ``None`` when a backend does not need it.

    ``waveform_dir`` and ``channel_positions_path`` remain constructor and
    read-only attribute aliases for ``bundle_dir`` and ``geometry_path``.
    The original four positional arguments retain their meaning. Legacy
    calls default to the shared waveform layout/version; new preparers should
    declare the layout they actually write. A backend must check the layouts
    and versions it supports before consuming files.
    """

    curation_key: dict
    bundle_dir: Path
    layout: str
    layout_version: int
    geometry_path: Path | None
    recording_date: Any = None

    def __init__(
        self,
        curation_key: dict,
        waveform_dir: Path | None = None,
        channel_positions_path: Path | None = None,
        recording_date: Any = None,
        *,
        bundle_dir: Path | None = None,
        layout: str = WAVEFORM_BUNDLE_LAYOUT,
        layout_version: int = WAVEFORM_BUNDLE_VERSION,
        geometry_path: Path | None = None,
    ):
        directory = _resolve_bundle_path(bundle_dir, waveform_dir, "bundle_dir")
        if directory is None:
            raise TypeError(
                "SessionMatcherInput requires bundle_dir (or waveform_dir)"
            )
        geometry = _resolve_bundle_path(
            geometry_path, channel_positions_path, "geometry_path"
        )
        if not isinstance(layout, str) or not layout.strip():
            raise ValueError("Matcher bundle layout must be a nonempty string")
        if (
            not isinstance(layout_version, Integral)
            or isinstance(layout_version, bool)
            or layout_version < 1
        ):
            raise ValueError(
                "Matcher bundle layout_version must be a positive integer"
            )
        for name, value in (
            ("curation_key", curation_key),
            ("bundle_dir", directory),
            ("layout", layout),
            ("layout_version", int(layout_version)),
            ("geometry_path", geometry),
            ("recording_date", recording_date),
        ):
            object.__setattr__(self, name, value)

    @property
    def waveform_dir(self) -> Path:
        """Compatibility alias for bundle_dir."""
        return self.bundle_dir

    @property
    def channel_positions_path(self) -> Path | None:
        """Compatibility alias for geometry_path."""
        return self.geometry_path


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


@dataclass(frozen=True)
class MatcherInputSource:
    """Resolved, artifact-masked recording and curated units for preparation.

    Preparers receive file-backed SI objects, not tables or database access.
    ``statistics_spans`` are the frozen intervals inside which waveform
    windows may be sampled. The identity and date must survive preparation.
    """

    curation_key: dict
    recording: Any
    sorting: Any
    recording_date: Any
    statistics_spans: Any


@dataclass(frozen=True)
class PreparedMatcherInput:
    """A backend-ready bundle and units excluded from that bundle."""

    session_input: SessionMatcherInput
    excluded_unit_ids: tuple[int, ...] = ()
    exclusion_reason: str = ""


@runtime_checkable
class MatcherInputPreparer(Protocol):
    """Prepare one input in the layout required by a registered matcher.

    An optional ``preparer_version() -> str | None`` records its implementation
    version. Both preparers and backends may implement
    :class:`MatcherProvenanceProvider` to describe result-determining assets.
    """

    def prepare(
        self,
        source: MatcherInputSource,
        directory: Path,
        params: dict,
        job_kwargs: dict,
    ) -> PreparedMatcherInput: ...


@runtime_checkable
class MatcherProvenanceProvider(Protocol):
    """Optional model/feature-definition fingerprints for producer provenance.

    Return named fingerprints as nonempty strings. This metadata does not
    change run identity: result-determining fingerprints must also be fields
    in the producer's immutable parameter schema, and the producer must verify
    the loaded assets against them. The hook must not run inference or prepare
    files; it is also called for a single-input run that skips those operations.
    """

    def provenance_fingerprints(self, params: dict) -> dict[str, str]: ...


@runtime_checkable
class MatcherGeometryValidator(Protocol):
    """Optional geometry requirements owned by a matcher backend.

    Selection calls this before inserting a new multi-input selection. Arrays
    are the effective channel positions of the curated recordings, in input
    order; labels identify inputs for errors. The validated matcher parameters
    allow a backend to choose its own geometry policy. Raise ``ValueError``
    for unsupported geometry. No table or recording objects are passed.

    Backends without this hook impose no selection-time geometry requirement.
    Inference must also validate the actual prepared geometry it consumes;
    input preparation may transform it after this preflight.
    """

    def validate_geometry(
        self, named_positions: list[tuple[Any, Any]], params: dict
    ) -> None: ...


@runtime_checkable
class MatcherProtocol(Protocol):
    """Structural interface every cross-session matcher backend implements.

    A backend is any object with a ``name`` string and a ``match`` method; it
    need not subclass anything. ``match`` consumes wrapper-prepared bundles and
    returns the cross-session matches, returning ``[]`` for the degenerate
    single-session case (one input) rather than raising.
    An optional ``backend_version() -> str | None`` records its library version;
    :class:`MatcherProvenanceProvider` can record model/feature fingerprints.
    """

    name: str

    def match(
        self,
        session_inputs: list[SessionMatcherInput],
        params: dict,
    ) -> list[MatchPair]:
        """Match units across the prepared per-input bundles.

        Parameters
        ----------
        session_inputs : list[SessionMatcherInput]
            One wrapper-prepared bundle per matching input, in ``input_index``
            (chronological) order. Each bundle's ``curation_key`` is the
            identity the matcher echoes back on the pairs it returns.
        params : dict
            The matcher's ``MatcherParameters.params`` blob, validated at
            insert time against the params schema registered with this
            backend (see :func:`register_matcher`).

        Returns
        -------
        list[MatchPair]
            One :class:`MatchPair` per matched unit pair, each side identified
            by the ``(sorting_id, curation_id)`` of an input's
            ``curation_key`` plus a unit id from that input's bundle. The two
            sides must come from different inputs, every key must be one of
            the ``curation_key`` values passed in, and an unordered pair must
            appear at most once (not also in reversed orientation). Order is
            not significant. Curation and unit IDs must be Python or NumPy
            integers; booleans, floats and strings are rejected. Units
            excluded by preparation cannot appear in pairs.
            The caller orients and orders the pairs, and
            ``UnitMatch.make`` raises ``ValueError`` on a pair that breaks
            these rules. ``[]`` when fewer than two inputs are given or
            nothing matches.
        """
        ...


#: name -> backend instance
_MATCHER_REGISTRY: dict[str, MatcherProtocol] = {}
#: name -> per-matcher Pydantic params schema (validates ``MatcherParameters``)
_SCHEMA_REGISTRY: dict[str, type] = {}
_PREPARER_REGISTRY: dict[str, MatcherInputPreparer] = {}
#: Names owned by shipped backends. Registering one of these loads the
#: built-in first so the replacement rule applies regardless of import order.
_BUILTIN_MATCHER_NAMES: frozenset[str] = frozenset({"unitmatch"})
_bootstrapping = False


def register_matcher(
    matcher: MatcherProtocol,
    schema: type,
    *,
    replace: bool = False,
    input_preparer: MatcherInputPreparer | None = None,
) -> None:
    """Register a matcher backend and its params schema under ``matcher.name``.

    Parameters
    ----------
    matcher : MatcherProtocol
        A backend with a ``name`` attribute and a ``match`` method.
        May also implement :class:`MatcherGeometryValidator`; otherwise
        selection does not read or compare its inputs' channel positions.
    schema : type
        The Pydantic model validating that matcher's ``MatcherParameters``
        ``params`` blob.
    input_preparer : MatcherInputPreparer, optional
        Prepare the backend's input layout. Defaults to the shared dense
        split-half waveform layout, extracted without UnitMatchPy. Supply a
        preparer for another layout; the matcher still consumes only bundles.
    replace : bool, optional
        Explicit maintenance override. Registering a name already held by a
        DIFFERENT backend class raises ``ValueError`` unless ``replace=True`` --
        ``MatcherParameters`` stores only the matcher name, so silently pointing
        a name at different code would make existing rows dispatch elsewhere.
        Re-registering the SAME backend class with the same preparer class is
        idempotent and needs no flag. Omitting the preparer retains the one
        already registered. A different preparer class also needs ``replace``.
        This is how :func:`register_default_matchers` self-heals, so the
        built-in registration does NOT use ``replace`` -- it is reserved for a
        deliberate swap to genuinely different code.

    Raises
    ------
    TypeError
        If ``matcher`` does not satisfy :class:`MatcherProtocol`.
    ValueError
        If ``matcher.name`` is already registered to a different backend or
        preparer class and ``replace`` is ``False``.
    """
    if not isinstance(matcher, MatcherProtocol) or not callable(
        getattr(matcher, "match", None)
    ):
        raise TypeError(
            f"{matcher!r} does not satisfy MatcherProtocol (needs a `name` "
            "attribute and a callable `match(session_inputs, params)` method)."
        )
    geometry_validator = getattr(matcher, "validate_geometry", None)
    if geometry_validator is not None and not callable(geometry_validator):
        raise TypeError(
            "validate_geometry must be callable when supplied by a matcher backend."
        )
    preparer_supplied = input_preparer is not None
    if input_preparer is None:
        from spyglass.spikesorting.v2._waveform_bundles import (
            WaveformInputPreparer,
        )

        input_preparer = WaveformInputPreparer()
    if not isinstance(input_preparer, MatcherInputPreparer) or not callable(
        getattr(input_preparer, "prepare", None)
    ):
        raise TypeError(
            "input_preparer must implement prepare(source, directory, params, job_kwargs)"
        )
    if (
        matcher.name in _BUILTIN_MATCHER_NAMES
        and matcher.name not in _MATCHER_REGISTRY
    ):
        # A built-in name claimed before its backend was imported would
        # otherwise bypass the replacement rule below (startup order must
        # not decide whether persisted MatcherParameters rows re-route).
        register_default_matchers()
    existing = _MATCHER_REGISTRY.get(matcher.name)
    existing_preparer = _PREPARER_REGISTRY.get(matcher.name)
    if (
        not preparer_supplied
        and existing is not None
        and type(existing) is type(matcher)
        and existing_preparer is not None
    ):
        input_preparer = existing_preparer
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
    if (
        existing is not None
        and existing_preparer is not None
        and type(existing_preparer) is not type(input_preparer)
        and not replace
    ):
        raise ValueError(
            f"Matcher {matcher.name!r} already has a different input preparer. "
            "Use a distinct matcher name, or replace=True to override deliberately."
        )
    _MATCHER_REGISTRY[matcher.name] = matcher
    _SCHEMA_REGISTRY[matcher.name] = schema
    _PREPARER_REGISTRY[matcher.name] = input_preparer


def is_registered(name: str) -> bool:
    """Return whether ``name`` is currently registered (no bootstrap)."""
    return name in _MATCHER_REGISTRY


def register_default_matchers() -> None:
    """Install any built-in matcher backend that is MISSING (fill-only).

    The backends register themselves as an import side effect, so the registry
    is empty until a backend module is imported. This makes that bootstrap
    explicit -- callers (and the lookups below) can ensure the built-ins are
    present without depending on import order -- and re-installs a built-in
    if the registry was cleared (e.g. by a test fixture). A name that is
    already registered is left alone, so lookups never swap the installed
    object and an explicit ``register_matcher(..., replace=True)`` of a
    built-in name survives later lookups.
    """
    global _bootstrapping
    if _bootstrapping:  # re-entered from the built-in's own register_matcher
        return
    _bootstrapping = True
    try:
        # Function-level import avoids an import cycle (the backend imports
        # this module) and keeps the optional UnitMatchPy import lazy (the
        # backend only imports UnitMatchPy when its match() path
        # actually runs).
        from spyglass.spikesorting.v2 import _unitmatch_backend

        _unitmatch_backend.register()
    finally:
        _bootstrapping = False


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


def get_input_preparer(name: str) -> MatcherInputPreparer:
    """Return the preparer registered with the named matcher."""
    get_matcher(name)
    return _PREPARER_REGISTRY[name]


def get_geometry_validator(name: str) -> MatcherGeometryValidator | None:
    """Return the backend's optional geometry preflight, without adding policy."""
    matcher = get_matcher(name)
    if isinstance(matcher, MatcherGeometryValidator) and callable(
        getattr(matcher, "validate_geometry", None)
    ):
        return matcher
    return None


def _get_matcher_schema(name: str) -> type:
    """Return the params schema for ``name`` or raise UnknownMatcherError."""
    register_default_matchers()
    if name not in _SCHEMA_REGISTRY:
        _raise_unknown(name)
    return _SCHEMA_REGISTRY[name]
