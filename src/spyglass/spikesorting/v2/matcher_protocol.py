"""Plugin interface for cross-session unit matchers -- not available yet.

This is the plugin interface for future cross-session matchers; it has
no v1 equivalent and lands together with ``unit_matching`` in a future
v2 release. See the spike-sorting-v2 documentation in the project
repository for the roadmap.
"""


def __getattr__(name):
    """Reject public-name access; matcher plugin interface is not available."""
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
        f"spyglass.spikesorting.v2.matcher_protocol.{name!r} is not "
        "available yet; the cross-session matcher plugin interface is "
        "planned for a future v2 release. See the module docstring for "
        "details."
    )
