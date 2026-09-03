"""Unit tests for least-loaded CUDA device selection (issue #1676).

A bare ``device="cuda"`` resolves to ``cuda:0`` in PyTorch, which on a
shared multi-GPU host may be the one saturated device.
``resolve_cuda_device`` picks the best GPU instead, preferring one that is
genuinely unused over one that merely has the most free bytes.

CI has no GPU, so ``cuda_memory_info`` -- the single function that touches
``torch`` -- is patched. That is a real seam, not mock theatre: it exists
precisely to isolate "what the hardware currently reports" from the
selection policy, and the policy is what these tests exercise. The
``torch``-touching body itself is ``# pragma: no cover``.
"""

from unittest.mock import patch

import pytest

GIB = 2**30
MODULE = "spyglass.position.v2.utils.device"


def mem(*free_total_gib):
    """Build a ``{index: (free, total)}`` map from (free, total) GiB pairs."""
    return {
        i: (int(free * GIB), int(total * GIB))
        for i, (free, total) in enumerate(free_total_gib)
    }


@pytest.fixture
def fns():
    """Import under test (deferred: keeps DB imports out of collection)."""
    from spyglass.position.v2.utils.device import (
        _select_device,
        resolve_cuda_device,
    )

    return _select_device, resolve_cuda_device


class TestSelectDevice:
    """The pure selection rule, independent of any hardware query."""

    def test_picks_index_with_most_free(self, fns):
        select, _ = fns
        assert select(mem((1, 80), (70, 80), (8, 80))) == 1

    def test_ties_resolve_to_lowest_index(self, fns):
        """Deterministic on an idle machine, where every GPU is equal."""
        select, _ = fns
        assert select(mem((79, 80), (79, 80), (79, 80))) == 0

    def test_idle_small_card_beats_busy_large_card(self, fns):
        """The reason free bytes alone is the wrong rule.

        cuda:0 has more free memory in absolute terms, but it is sharing a
        card with someone else's 20 GiB job; cuda:1 is untouched. Prefer
        the idle card -- no contention, and no co-tenant that might grow
        into the memory this run was counting on.
        """
        select, _ = fns
        assert select(mem((60, 80), (39, 40))) == 1

    def test_falls_back_to_most_free_when_all_busy(self, fns):
        """With no idle GPU, absolute free memory is the best signal left."""
        select, _ = fns
        assert select(mem((10, 80), (30, 80), (20, 80))) == 1

    def test_empty_raises(self, fns):
        select, _ = fns
        with pytest.raises(ValueError, match="No CUDA devices"):
            select({})


class TestResolveCudaDevice:
    """End-to-end resolution policy against a faked hardware report."""

    @pytest.mark.parametrize("device", [None, "", "cpu"])
    def test_non_cuda_is_passthrough_without_querying(self, fns, device):
        """Non-CUDA requests never query the GPU at all."""
        _, resolve = fns
        with patch(f"{MODULE}.cuda_memory_info") as mock_query:
            assert resolve(device) == device
        mock_query.assert_not_called()

    def test_bare_cuda_resolves_to_emptiest(self, fns):
        """The reported bug: 'cuda' must not blindly mean cuda:0."""
        _, resolve = fns
        info = {
            0: (int(0.5 * GIB), 80 * GIB),
            1: (70 * GIB, 80 * GIB),
            9: (77 * GIB, 80 * GIB),
        }
        with patch(f"{MODULE}.cuda_memory_info", return_value=info):
            assert resolve("cuda") == "cuda:9"

    def test_explicit_index_is_honored(self, fns):
        """An explicitly named device is not second-guessed."""
        _, resolve = fns
        info = {0: (77 * GIB, 80 * GIB), 3: (40 * GIB, 80 * GIB)}
        with patch(f"{MODULE}.cuda_memory_info", return_value=info):
            assert resolve("cuda:3") == "cuda:3"

    def test_explicit_but_full_warns_and_proceeds(self, fns):
        """Explicit choice wins, but the likely OOM is called out."""
        _, resolve = fns
        info = {0: (int(0.5 * GIB), 80 * GIB), 9: (77 * GIB, 80 * GIB)}
        with (
            patch(f"{MODULE}.cuda_memory_info", return_value=info),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            assert resolve("cuda:0") == "cuda:0"
        warning = mock_logger.warning.call_args[0][0]
        assert "cuda:9" in warning, "should point at the emptier device"

    def test_all_devices_full_raises_with_breakdown(self, fns):
        """Fail immediately rather than OOM minutes into inference."""
        _, resolve = fns
        info = {
            0: (int(0.5 * GIB), 80 * GIB),
            1: (int(0.25 * GIB), 80 * GIB),
        }
        with patch(f"{MODULE}.cuda_memory_info", return_value=info):
            with pytest.raises(RuntimeError, match="enough free memory"):
                resolve("cuda")

    def test_threshold_can_be_disabled(self, fns):
        """min_free_gib=None accepts whatever is available."""
        _, resolve = fns
        info = {
            0: (int(0.5 * GIB), 80 * GIB),
            1: (int(0.25 * GIB), 80 * GIB),
        }
        with patch(f"{MODULE}.cuda_memory_info", return_value=info):
            assert resolve("cuda", min_free_gib=None) == "cuda:0"
