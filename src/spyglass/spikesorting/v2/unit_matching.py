"""Cross-session unit tracking for chronic recordings -- not available yet.

UnitMatch is a v2-introduced feature with no v1 equivalent; it is
planned for a future v2 release. See the spike-sorting-v2 documentation
in the project repository for the roadmap.
"""


def __getattr__(name):
    # Dunder names (``__path__``, ``__all__``, ``__spec__``, ``__file__``,
    # ...) are probed defensively by the import machinery, pickle, and
    # inspection tools. Always raise ``AttributeError`` for those so the
    # probes get the answer they expect; the custom message is reserved
    # for real public-API names.
    if name.startswith("__"):
        raise AttributeError(name)
    # ``ImportError`` (not ``AttributeError``) for public names so the
    # message survives the ``from m import X`` flattening path -- CPython
    # collapses only ``AttributeError`` into the generic "cannot import
    # name" ``ImportError``. No v1 fallback exists -- this is a
    # v2-introduced feature.
    raise ImportError(
        f"spyglass.spikesorting.v2.unit_matching.{name!r} is not "
        "available yet; cross-session unit matching is planned for a "
        "future v2 release. See the module docstring for details."
    )
