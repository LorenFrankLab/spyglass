"""Plugin interface + registry for cross-session unit matchers.

A *matcher* takes prepared bundles, one per matching input, and returns
pairwise cross-session unit matches. Backends (UnitMatch is built in) plug in
without touching the DataJoint tables:

- :class:`SessionMatcherInput`: one prepared bundle per matching input (a
  sort of one recording or of a same-day concatenation). The matcher reads
  only these directories, never analyzers, recordings, or table keys.
- :class:`MatchPair`: one cross-session match, keyed by
  ``(sorting_id, curation_id, unit_id)`` on each side.
- :class:`MatcherProtocol`: the ``match(session_inputs, params)`` contract.
- :class:`MatcherGeometryValidator`: optional backend geometry preflight.
- :class:`MatcherInputPreparer`: preparation from
  :class:`MatcherInputSource` to :class:`PreparedMatcherInput`.
- :func:`register_matcher` / :func:`get_matcher`: the registry of name to
  backend, input preparer, and params schema that ``MatcherParameters``
  validates against.

No DataJoint dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spyglass.spikesorting.v2.exceptions import UnknownMatcherError

WAVEFORM_BUNDLE_LAYOUT = "split_half_waveforms"
WAVEFORM_BUNDLE_VERSION = 1


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

    The shared waveform layout/version are the defaults. Preparers using
    another layout declare the format they write; a backend checks supported
    layouts and versions before consuming files. Paths are normalized to
    ``Path`` objects when the bundle is created.
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
        bundle_dir: Path,
        *,
        layout: str = WAVEFORM_BUNDLE_LAYOUT,
        layout_version: int = WAVEFORM_BUNDLE_VERSION,
        geometry_path: Path | None = None,
        recording_date: Any = None,
    ):
        if bundle_dir is None:
            raise TypeError("SessionMatcherInput requires bundle_dir")
        directory = Path(bundle_dir)
        geometry = Path(geometry_path) if geometry_path is not None else None
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


@dataclass(frozen=True)
class MatchPair:
    """One cross-session unit match, keyed by curated-unit identity per side.

    ``drift_estimate_um`` and ``fdr_estimate`` are ``None`` unless a backend
    produces per-pair values; UnitMatch does not.
    """

    session_a_sorting_id: str
    session_a_curation_id: int
    unit_a_id: int
    session_b_sorting_id: str
    session_b_curation_id: int
    unit_b_id: int
    match_probability: float
    drift_estimate_um: float | None = None
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
        Allow re-pointing a registered name at a different backend or
        preparer class. Without it that raises ``ValueError``, because
        ``MatcherParameters`` stores only the name and existing rows would
        dispatch to different code. Re-registering the same classes is
        idempotent; omitting the preparer keeps the registered one.

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
        from spyglass.spikesorting.v2._matching.waveforms import (
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
        # Load the built-in first so import order cannot bypass the
        # replacement rule below.
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
    """Install any built-in matcher backend that is missing.

    Backends register on import, so lookups call this to avoid depending on
    import order. A name already registered is left alone, so a
    ``replace=True`` override of a built-in survives later lookups.
    """
    global _bootstrapping
    if _bootstrapping:  # re-entered from the built-in's own register_matcher
        return
    _bootstrapping = True
    try:
        # Lazy import: avoids a cycle and defers optional UnitMatchPy.
        from spyglass.spikesorting.v2._matching import (
            unitmatch_backend as _unitmatch_backend,
        )

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
