"""Helpers for building simulated ground-truth recordings for v2 validation.

This package converts MEArec ground-truth simulations into NWB files that are
structurally identical to ``trodes_to_nwb`` output, so Spyglass ingests them
through the normal session-insertion path. It is validation tooling, not part of
the runtime spike-sorting pipeline.
"""
