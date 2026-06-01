"""Spike-sorting quality-metric curation -- not yet ported to v2.

The v1 equivalent (``spyglass.spikesorting.v1.metric_curation``) ships
the ``MetricCuration``, ``MetricCurationParameters``,
``WaveformParameters``, and ``MetricParameters`` tables. v2's port is on
the roadmap; use the v1 chain in the interim::

    from spyglass.spikesorting.v1 import (
        MetricCuration,
        MetricCurationParameters,
        WaveformParameters,
        MetricParameters,
    )

For roadmap details, see the spike-sorting-v2 documentation in the
project repository.
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
    # name" ``ImportError``.
    raise ImportError(
        f"spyglass.spikesorting.v2.metric_curation.{name!r} is not yet "
        "implemented in v2. Use the v1 fallback "
        "`from spyglass.spikesorting.v1.metric_curation import "
        f"{name}` (where available), or wait for the v2 port. See the "
        "module docstring for details."
    )
