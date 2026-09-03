"""CUDA device selection for V2 pose inference.

Motivation (issue #1676): a bare ``device="cuda"`` is not a request for
"a GPU" -- PyTorch resolves it to ``cuda:0`` specifically. On a shared
multi-GPU host that is an arbitrary and frequently wrong choice: the
reported failure had ``cuda:0`` at 0.5 GiB free (saturated by other
users) while nine sibling A100s sat at ~77 GiB free each. Inference then
died with a CUDA OOM part-way through a long run.

That failure is invisible to
:func:`~spyglass.position.v2.utils.nwb_io.check_gpu_available`, which
only answers "is any GPU visible" (``torch.cuda.is_available()``, True in
that case). This module answers the follow-up question -- *which* GPU.

Lives under ``position/v2/utils/`` rather than the tool-agnostic
``position/utils/`` for the same reason ``check_gpu_available`` does: DLC
3.x/PyTorch is the only V2 inference backend, and ``torch.cuda`` device
strings mean nothing to V1's TensorFlow-based pipeline (which takes an
integer ``gputouse`` instead).

Selection is deliberately *not* a stored parameter. Which GPU is free
changes between runs, so freezing an index into ``PoseEstimParams`` would
reintroduce the same bug with a different number; the choice is made at
dispatch and logged.
"""

from typing import Mapping, Tuple, Union

from spyglass.utils.logging import logger

BYTES_PER_GIB = 2**30

# A floor for "this device is obviously too full to try", not an estimate
# of what inference actually needs (that depends on batch size, image
# resolution, and backbone). Its only job is to convert a run that would
# OOM minutes in into an immediate, explanatory failure.
DEFAULT_MIN_FREE_GIB = 2.0

# Fraction of total memory below which a device counts as unused. Not
# zero: an idle GPU still carries a few hundred MiB of display server and
# other processes' CUDA contexts (~1.6% of an 80 GiB card in the issue
# #1676 report), which should not disqualify it.
DEFAULT_IDLE_FRAC = 0.05


def _select_device(
    mem_by_index: Mapping[int, Tuple[int, int]],
    idle_frac: float = DEFAULT_IDLE_FRAC,
) -> int:
    """Return the index of the best device to run on.

    Prefers a genuinely *unused* GPU over one that merely has the most
    free bytes. On a mixed-capacity host the two differ: a busy 80 GiB
    card sharing 60 GiB with someone else's job beats an idle 40 GiB card
    on free memory alone, yet the idle card is the better neighbor and
    the more predictable performer (no contention for SMs or PCIe
    bandwidth, no risk of the co-tenant job growing into the memory this
    run was counting on).

    Pure helper, split out from the ``torch`` query so the selection rule
    itself is testable on a plain mapping without a GPU present.

    Parameters
    ----------
    mem_by_index : Mapping[int, tuple of (int, int)]
        Per CUDA device index, a ``(free_bytes, total_bytes)`` pair.
    idle_frac : float, optional
        Fraction of total memory a device may have in use and still count
        as unused, by default :data:`DEFAULT_IDLE_FRAC`.

    Returns
    -------
    int
        Index of the chosen device. Ties resolve to the lowest index, so
        selection is deterministic on an idle machine.

    Raises
    ------
    ValueError
        If ``mem_by_index`` is empty.
    """
    if not mem_by_index:
        raise ValueError("No CUDA devices to select from.")

    def used_frac(i: int) -> float:
        free, total = mem_by_index[i]
        return (total - free) / total if total else 1.0

    candidates = [i for i in mem_by_index if used_frac(i) <= idle_frac]
    if not candidates:  # every GPU is in use -- fall back to most free
        candidates = list(mem_by_index)

    # sorted() first so ties break to the lowest index, not dict order.
    return max(sorted(candidates), key=lambda i: mem_by_index[i][0])


def cuda_memory_info() -> dict:  # pragma: no cover - requires a GPU
    """Return ``(free_bytes, total_bytes)`` per visible CUDA device.

    Indices are PyTorch's own, i.e. already relative to
    ``CUDA_VISIBLE_DEVICES``, so the returned keys can be formatted
    straight into a ``"cuda:N"`` string without remapping.

    Returns
    -------
    dict
        Mapping of device index to ``(free_bytes, total_bytes)``.
    """
    import torch  # pragma: no cover

    return {  # pragma: no cover
        i: torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())
    }


def _describe(mem_by_index: Mapping[int, Tuple[int, int]]) -> str:
    """Render per-device free memory for an error or log message."""
    return ", ".join(
        f"cuda:{i}={mem_by_index[i][0] / BYTES_PER_GIB:.1f}"
        for i in sorted(mem_by_index)
    )


def resolve_cuda_device(
    device: Union[str, None],
    min_free_gib: Union[float, None] = DEFAULT_MIN_FREE_GIB,
) -> Union[str, None]:
    """Resolve a bare ``"cuda"`` request to the best available device.

    Parameters
    ----------
    device : str or None
        Requested device, e.g. ``"cuda"``, ``"cuda:3"``, ``"cpu"``, or
        ``None``. Anything that is not a CUDA request is returned
        unchanged without importing ``torch``.
    min_free_gib : float or None, optional
        Refuse to dispatch if the chosen device has less than this much
        free memory, by default :data:`DEFAULT_MIN_FREE_GIB`. Pass
        ``None`` to skip the check and accept whatever is available.

    Returns
    -------
    str or None
        ``"cuda:N"`` when *device* was a bare ``"cuda"``; otherwise
        *device* unchanged.

    Raises
    ------
    RuntimeError
        If the best available device has less than *min_free_gib* free.

    Notes
    -----
    An explicit ``"cuda:N"`` is honored, not second-guessed -- the caller
    named that device on purpose. It still gets a warning when it looks
    too full to succeed, since silently proceeding to an OOM is the
    behavior this module exists to eliminate.
    """
    if not device or "cuda" not in str(device).lower():
        return device

    mem_by_index = cuda_memory_info()
    if not mem_by_index:  # pragma: no cover - requires a GPU-less host
        return device  # pragma: no cover

    explicit = str(device).lower().startswith("cuda:")
    if explicit:
        index = int(str(device).split(":", 1)[1])
        entry = mem_by_index.get(index)
        free = entry[0] if entry else None
    else:
        index = _select_device(mem_by_index)
        free, total = mem_by_index[index]
        device = f"cuda:{index}"
        logger.info(
            f"Resolved device='cuda' to '{device}' "
            f"({free / BYTES_PER_GIB:.1f} of "
            f"{total / BYTES_PER_GIB:.1f} GiB free), the best of "
            f"{len(mem_by_index)} visible device(s). "
            f"Free by device (GiB): {_describe(mem_by_index)}."
        )

    if free is None:  # pragma: no cover - torch would reject it first
        return device  # pragma: no cover

    if min_free_gib is not None and free < min_free_gib * BYTES_PER_GIB:
        best = _select_device(mem_by_index)
        detail = (
            f"cuda:{index} has only {free / BYTES_PER_GIB:.1f} GiB free "
            f"(need at least {min_free_gib:.1f} GiB)."
        )
        if explicit:
            logger.warning(
                f"{detail} It was requested explicitly, so it will be used "
                f"anyway -- but cuda:{best} currently has "
                f"{mem_by_index[best][0] / BYTES_PER_GIB:.1f} GiB free. "
                "Expect a CUDA out-of-memory error. Pass device='cuda' to "
                "let Spyglass pick the least-loaded GPU instead."
            )
            return device
        raise RuntimeError(
            f"No CUDA device has enough free memory. {detail} Free memory "
            f"by device (GiB): {_describe(mem_by_index)}. Wait for another "
            "job to finish, or use a CPU-configured PoseEstimParams entry "
            "(device='cpu')."
        )

    return device
