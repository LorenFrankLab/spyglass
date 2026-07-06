"""Self-contained FIR filter design and out-of-core FIR filtering.

Vendored from ghostipy (Apache-2.0, https://github.com/kemerelab/ghostipy),
covering only the pieces spyglass uses in ``common_filter.py``:

- ``estimate_taps``  -> FIR tap-count estimate
- ``firdesign``      -> Type I FIR design with spline transition bands (L2)
- ``group_delay``    -> integer group delay of a Type I FIR filter
- ``filter_data_fir``-> overlap-save FFT filtering that streams into a
  (possibly on-disk) preallocated output array, with decimation and
  per-dimension index restrictions.

All public call signatures (parameter names, order, and defaults) are unchanged
from upstream ghostipy -- only type annotations have been added -- and numeric
output matches it to floating-point round-off.

Intentional divergences from upstream:

- FFT backend: ``pyfftw`` replaced with ``scipy.fft`` (``workers=`` for
  multithreading), using the real transform (``rfft``/``irfft``) for real input
  and the full complex transform otherwise. This removes the M1-Mac /
  conda-forge install friction documented in the spyglass setup and is the only
  change that touches the numeric path (round-off only).
- Block selection at boundaries: when the requested (exclusive) output stop
  lands exactly on an overlap-save block boundary, the trailing empty block is
  no longer processed. Upstream still read and FFT'd it only to write zero
  samples; skipping it removes a wasted transform and a read one block past the
  needed data (which could fail on a strict lazy/on-disk signal). Output is
  identical.
- Fail-loud / fail-closed hardening that affects only invalid inputs or genuine
  errors (never the valid path spyglass exercises): the M-1 overlap read no
  longer swallows exceptions and silently zero-fills; ``input_index_bounds`` /
  ``output_index_bounds`` treat the stop as exclusive and validate by range
  rather than probing the array; ``estimate_taps`` rejects non-positive
  ``fs``/``tw``/``d1``/``d2`` and deviations so loose the tap estimate would be
  < 1; ``firdesign``/``_firspline`` require an integer ``numtaps`` >= 1 and at
  least two ordered ``band_edges``; the spline power ``p`` must be > 0; ``ds``
  must be an integer >= 1; ``nfft`` must be an integer >= the kernel length;
  ``input_dim_restrictions`` entries must be 1-D integer index arrays restricting
  at most one non-filtered axis; ``output_offset`` must be an integer >= 0 that
  fits within ``outarray``; complex input no longer raises ``UnboundLocalError``.

Design details (spline-transition FIR) follow Burrus et al., 1992.
"""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import cpu_count

import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.signal

__all__ = [
    "estimate_taps",
    "group_delay",
    "firdesign",
    "filter_data_fir",
]


def group_delay(b: np.ndarray) -> int:
    """Group delay of a linear-phase (Type I) FIR filter.

    Parameters
    ----------
    b : numpy.ndarray, shape (N,)
        The filter coefficients. ``N`` must be odd.

    Returns
    -------
    int
        The group delay in samples, ``(N - 1) // 2``.

    Raises
    ------
    ValueError
        If ``b`` has an even number of coefficients, for which the group delay
        is not an integer.
    """
    L = len(b)
    if not L & 1:
        raise ValueError(
            f"There are {L} filter coefficients (an even number), so the group "
            "delay cannot be converted to an integer value"
        )
    return (L - 1) // 2


def estimate_taps(
    fs: float, tw: float, *, d1: float | None = None, d2: float | None = None
) -> int:
    """Estimate the number of taps for a Type I FIR filter.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    tw : float
        Transition bandwidth in Hz.
    d1 : float, optional
        Passband deviation. Default is 0.1% (1e-3).
    d2 : float, optional
        Minimum stopband attenuation. Default is 120 dB (1e-6).

    Returns
    -------
    int
        Number of taps (always odd).

    Raises
    ------
    ValueError
        If ``fs``, ``tw``, ``d1``, or ``d2`` is non-finite or non-positive, or
        if the deviations are so loose that the estimated tap count is < 1
        (i.e. ``10 * d1 * d2 >= 1``).

    References
    ----------
    https://dsp.stackexchange.com/questions/31066
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"'fs' must be finite and positive but got {fs}")
    if not np.isfinite(tw) or tw <= 0:
        raise ValueError(
            f"'tw' (transition bandwidth) must be finite and positive but got {tw}"
        )

    if d1 is None:
        d1 = 1e-3
    if d2 is None:
        d2 = 1e-6
    if not np.isfinite(d1) or not np.isfinite(d2) or d1 <= 0 or d2 <= 0:
        raise ValueError(
            "passband/stopband deviations must be finite and positive but got "
            f"d1={d1}, d2={d2}"
        )

    numtaps = int(np.ceil(2 / 3 * np.log10(1 / (10 * d1 * d2)) * fs / tw))
    if numtaps < 1:
        raise ValueError(
            f"computed numtaps={numtaps} < 1; the deviations d1={d1}, d2={d2} are "
            "too loose (10 * d1 * d2 must be < 1)"
        )
    if not numtaps & 1:
        numtaps += 1
    return numtaps


def _firspline(
    numtaps: int,
    f1: float,
    f2: float,
    *,
    fs: float | None = None,
    p: float | None = None,
) -> np.ndarray:
    """Design a Type I low-pass filter with a spline transition band.

    Parameters
    ----------
    numtaps : int
        Number of coefficients (must be a positive odd integer).
    f1 : float
        Frequency where the amplitude response is 1, in Hz.
    f2 : float
        Frequency where the amplitude response is 0, in Hz.
    fs : float, optional
        Sampling rate in Hz. Default is 2 Hz.
    p : float, optional
        Spline power. Default follows Burrus et al., 1992.

    Returns
    -------
    numpy.ndarray, shape (numtaps,)
        The filter coefficients.

    Raises
    ------
    ValueError
        If ``numtaps`` is not a positive odd integer, if ``fs`` is non-finite or
        non-positive, if ``f1``/``f2`` exceed the Nyquist frequency or are not
        strictly increasing, or if ``p`` is non-finite or non-positive.
    """
    if not isinstance(numtaps, (int, np.integer)):
        raise ValueError(f"numtaps must be an integer but got {numtaps}")
    if numtaps < 1:
        raise ValueError(
            f"numtaps must be a positive odd value but got {numtaps}"
        )
    if not numtaps & 1:
        raise ValueError(f"numtaps must be odd but got {numtaps}")

    if fs is None:
        fs = 2
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"'fs' must be finite and positive but got {fs}")
    nyq = fs / 2

    if f1 > nyq or f2 > nyq:
        raise ValueError(
            f"Got critical frequencies {f1} and {f2} but they must both be less "
            f"than the Nyquist frequency of {nyq}"
        )
    if f1 >= f2:
        raise ValueError(f"f1 must be <{f2} but got {f1}")

    if p is None:
        p = 0.312 * numtaps * (f2 - f1) / nyq
    # Must be strictly positive: p appears in the denominator below, and p == 0
    # silently collapses the spline transition to a rectangular truncation
    # (nan ** 0 == 1) rather than raising.
    if not np.isfinite(p) or p <= 0:
        raise ValueError(f"p must be positive but got {p}")

    # Convert to normalized frequency in radians
    wo = (f1 + f2) / 2 * (1 / nyq * np.pi)
    dw = (f2 - f1) / 2 * (1 / nyq * np.pi)

    nvec = np.arange(1, (numtaps - 1) // 2 + 1)

    # Optimal L2 solution to the ideal lowpass filter
    h = np.sin(wo * nvec) / (np.pi * nvec)

    x = dw * nvec / p
    spline = (np.sin(x) / x) ** p
    h *= spline  # connect transition band with spline

    b = np.hstack((np.flip(h), wo / np.pi, h))  # make linear phase
    return b


def firdesign(
    numtaps: int,
    band_edges: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    fs: float = 1,
    p: float | None = None,
) -> np.ndarray:
    """Design an arbitrary Type I FIR filter with spline transition bands.

    Optimized for an L2 error norm.

    Parameters
    ----------
    numtaps : int
        Number of filter coefficients (must be a positive odd integer).
    band_edges : array_like, shape (2 * n_bands,)
        Critical frequencies of the filter in Hz, an even-length,
        strictly increasing sequence. Do not include 0 or the Nyquist
        frequency.
    desired : array_like, shape (2 * n_bands,)
        Magnitude response at each band edge; each value must be 0 or 1. The
        values must alternate between transition bands (the two edges of a
        transition band differ) and flat bands (the two edges match).
    fs : float, optional
        Sampling rate in Hz. Default is 1 Hz.
    p : float, optional
        Power for the spline transition-band functions. Default follows
        Burrus et al., 1992.

    Returns
    -------
    numpy.ndarray, shape (numtaps,)
        The filter coefficients.

    Raises
    ------
    ValueError
        If ``numtaps`` is not a positive odd integer; if ``fs`` is non-finite
        or non-positive; if ``band_edges`` is empty, has an odd length, differs
        in length from ``desired``, has a non-positive first edge, has a last
        edge >= the Nyquist frequency, or is not strictly increasing; or if
        ``desired`` contains values other than 0/1 or does not follow the
        required transition/flat-band alternation.
    """
    band_edges = np.array(band_edges)
    desired = np.array(desired)

    if not isinstance(numtaps, (int, np.integer)):
        raise ValueError(f"Got {numtaps} for 'numtaps' but must be an integer")
    numtaps = int(numtaps)
    if numtaps < 1:
        raise ValueError(
            f"Got {numtaps} for 'numtaps' but must be a positive odd value"
        )
    if not numtaps & 1:
        raise ValueError(
            f"Got {numtaps} for 'numtaps' but must be an odd value"
        )
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"'fs' must be finite and positive but got {fs}")
    if len(band_edges) == 0:
        raise ValueError("Must have at least two band edges")
    if len(band_edges) % 2 != 0:
        raise ValueError("Must have even number of band edges")
    if len(band_edges) != len(desired):
        raise ValueError("must have equal number of band edges and values")
    if not np.isin(desired, (0, 1)).all():
        raise ValueError("All values must be either 0 or 1")
    if not band_edges[0] > 0:
        raise ValueError("First band edge must be greater than 0")
    if not band_edges[-1] < fs / 2:
        raise ValueError(f"Last band edge must be less than {fs / 2}")
    if not np.all(band_edges[:-1] < band_edges[1:]):
        raise ValueError(
            "'band_edges' must be a monotonically increasing sequence"
        )

    for pair_ind, (v1, v2) in enumerate(zip(desired, desired[1:])):
        edge1, edge2 = band_edges[pair_ind], band_edges[pair_ind + 1]
        if pair_ind % 2 == 0:
            if v1 == v2:
                raise ValueError(
                    f"Got {v1} for band edge {edge1} Hz and {v2} for band edge "
                    f"{edge2} Hz but must be different values"
                )
        else:
            if v1 != v2:
                raise ValueError(
                    f"Got {v1} for band edge {edge1} Hz and {v2} for band edge "
                    f"{edge2} Hz but must be the same values"
                )

    critical_points = band_edges.reshape((-1, 2))
    # low pass prototypes
    prototypes = np.zeros((len(critical_points), numtaps))
    for ind, (f1, f2) in enumerate(critical_points):
        prototypes[ind] = _firspline(numtaps, f1, f2, fs=fs, p=p)

    # center impulse (identity filter), used to invert a lowpass into a highpass
    impulse = scipy.signal.unit_impulse(numtaps, "mid")

    if prototypes.shape[0] == 1:  # single band
        b = prototypes[0]
        if desired[-1] == 1:  # high pass
            b = impulse - prototypes[0]
    else:  # multi-band
        b = np.zeros(numtaps)

        # Magnitude at 0 and Nyquist is the same
        if desired[0] == desired[-1]:
            for ii in range(0, prototypes.shape[0], 2):
                bl = prototypes[ii]
                bh = prototypes[ii + 1]
                b += bh - bl

            # high pass at 0 and Nyquist, so invert
            if desired[-1] == 1:
                b = impulse - b
        else:
            if desired[0] == 0:
                tmp = prototypes[-1]
                b_special = impulse - tmp
                prototypes = prototypes[:-1]
            else:
                b_special = prototypes[0]
                prototypes = prototypes[1:]

            for ii in range(0, prototypes.shape[0] - 1, 2):
                bl = prototypes[ii]
                bh = prototypes[ii + 1]
                b += bh - bl

            b += b_special

    return b


def _osconvolve(
    signal: np.ndarray,
    kernel: npt.ArrayLike,
    *,
    mode: str = "full",
    nfft: int | None = None,
    threads: int = cpu_count(),
    axis: int = -1,
    outarray: np.ndarray | None = None,
    input_index_bounds: Sequence[int] | None = None,
    output_index_bounds: Sequence[int] | None = None,
    describe_dims: bool = False,
    ds: int | None = None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None = None,
    output_offset: int = 0,
    verbose: bool = False,
) -> tuple[tuple[int, ...], str] | np.ndarray:
    """Overlap-save FFT convolution, written to minimize memory usage.

    Streams the convolution block-by-block into ``outarray`` (which may be an
    on-disk array), supporting decimation (``ds``) and restricting which
    indices of the non-filtered axes are used (``input_dim_restrictions``).
    ``mode`` selects ``'full'``, ``'same'``, or ``'valid'`` convolution.

    Ported from ghostipy's ``osconvolve``; the numeric output is identical to
    upstream to floating-point round-off. See the module header for the
    intentional divergences (scipy.fft backend, the exact-boundary block fold,
    and the fail-loud input/read validation), and :func:`filter_data_fir` for
    the meaning of the shared out-of-core parameters.

    Returns
    -------
    tuple of (tuple of int, str), or numpy.ndarray
        If ``describe_dims`` is True, the ``(shape, dtype)`` the output would
        have; otherwise the filled ``outarray``.
    """
    # The kernel is small and always in memory, so normalize it to an array for
    # convenience. The signal is deliberately NOT converted: it may be an
    # on-disk / lazy array, and np.asarray would force it fully into memory.
    kernel = np.asarray(kernel)

    real_output = np.isrealobj(signal) and np.isrealobj(kernel)

    if threads < 1:
        raise ValueError("Must have at least one thread to do the FFT...")
    if kernel.ndim != 1:
        raise ValueError("Kernel must be 1D")
    if kernel.shape[0] < 1:
        raise ValueError("Kernel must have at least one coefficient")

    ###############################################################
    # Determine convolution lengths
    if input_index_bounds is not None:
        if len(input_index_bounds) != 2:
            raise ValueError(
                f"Got {len(input_index_bounds)} elements in input_index_bounds "
                "but must have 2 elements"
            )
        if not input_index_bounds[0] < input_index_bounds[1]:
            raise ValueError(
                f"'input_index_bounds' {input_index_bounds} is not strictly "
                "increasing"
            )
        # The stop bound is exclusive (N is computed as stop - start below), so
        # it may legitimately equal the axis length. Upstream ghostipy probed
        # signal[stop] here, which wrongly rejected a full-length [0, N] range;
        # this range check keeps the exclusive-stop semantics consistent.
        if (
            input_index_bounds[0] < 0
            or input_index_bounds[1] > signal.shape[axis]
        ):
            raise IndexError(
                f"input_index_bounds {list(input_index_bounds)} out of range "
                f"for axis {axis} of length {signal.shape[axis]} "
                "(stop is exclusive, so it may equal the axis length)"
            )
        N = input_index_bounds[1] - input_index_bounds[0]
    else:
        N = signal.shape[axis]

    M = kernel.shape[0]
    tot_length = N + M - 1

    if mode == "full":
        outsize = tot_length
    elif mode == "same":
        outsize = N
    elif mode == "valid":
        if N < M:
            raise ValueError(
                "Cannot do a 'valid' convolution because the input is shorter "
                "than the kernel"
            )
        outsize = N - M + 1
    else:
        raise ValueError(f"Got invalid value {mode} for 'mode'")

    if output_index_bounds is not None:
        if mode != "full":
            raise NotImplementedError(
                "'output_index bounds' currently implemented only for "
                "mode == 'full'"
            )
        if len(output_index_bounds) != 2:
            raise ValueError(
                f"Got {len(output_index_bounds)} elements in "
                "'output_index_bounds' but must have 2 elements"
            )
        # start is an inclusive index into the length-tot_length convolution
        # output; stop is exclusive and so may equal tot_length.
        if output_index_bounds[0] < 0 or output_index_bounds[1] > tot_length:
            raise ValueError(
                f"'output_index_bounds' {list(output_index_bounds)} out of range "
                f"for a full convolution of length {tot_length}; expected "
                f"0 <= start < stop <= {tot_length} (stop is exclusive)"
            )
        if not output_index_bounds[0] < output_index_bounds[1]:
            raise ValueError(
                f"'output_index_bounds' {output_index_bounds} is not strictly "
                "increasing"
            )
        first_ind = output_index_bounds[0]
        last_ind = output_index_bounds[1]
    else:
        first_ind = (tot_length - outsize) // 2
        last_ind = first_ind + outsize

    downsample = False
    if ds is not None:
        # ds == 1 is a valid (no-op) decimation; reject non-integer / 0 / negative
        # up front (consistent with nfft/output_offset) so they fail loudly here
        # instead of as a later ZeroDivisionError.
        if not isinstance(ds, (int, np.integer)) or ds < 1:
            raise ValueError(
                f"'ds' factor must be an integer >= 1 but got {ds!r}"
            )
        ds = int(ds)
        block_offset = 0
        downsample = True

    ###############################################################
    # handle output params
    dtype = "<f8" if real_output else "<c16"

    # set the default expected_shapes...
    expected_shape = list(signal.shape)
    expected_shape[axis] = last_ind - first_ind

    # ... and override the appropriate part if input_dim_restrictions were passed
    # in. The restricted (dim, selection, length) triples are recorded once here
    # and reused for the block-buffer sizing and signal slices below, instead of
    # re-scanning every dimension each time.
    restricted_dims = []
    if input_dim_restrictions is not None:
        if len(input_dim_restrictions) != signal.ndim:
            raise ValueError(
                f"Expected {signal.ndim} elements in 'input_dim_restrictions' "
                f"but got {len(input_dim_restrictions)}"
            )
        if input_dim_restrictions[axis] is not None:
            raise ValueError(
                f"input_dim_restrictions[{axis}] must be set to None"
            )
        # Each restriction must be a 1-D, in-range, strictly increasing integer
        # index array (as spyglass passes for electrode selection, and as h5py
        # requires for fancy indexing). Slices/masks are rejected because shape
        # planning below relies on len(), and restricting more than one
        # non-filtered axis at once would trigger NumPy paired advanced indexing
        # (a Cartesian selection is not what the block reads assume), so both are
        # unsupported and fail loudly rather than producing a wrong shape.
        for dim in range(signal.ndim):
            sel = input_dim_restrictions[dim]
            if dim == axis or sel is None:
                continue
            sel_arr = np.asarray(sel)
            if sel_arr.ndim != 1 or not np.issubdtype(
                sel_arr.dtype, np.integer
            ):
                raise ValueError(
                    f"input_dim_restrictions[{dim}] must be None or a 1-D array "
                    "of integer indices (slices/masks are not supported)"
                )
            if sel_arr.size > 0:
                if np.any(sel_arr < 0) or np.any(sel_arr >= signal.shape[dim]):
                    raise IndexError(
                        f"input_dim_restrictions[{dim}] contains indices out of "
                        f"range for axis length {signal.shape[dim]}"
                    )
                if not np.all(sel_arr[:-1] < sel_arr[1:]):
                    raise ValueError(
                        f"input_dim_restrictions[{dim}] must be strictly "
                        "increasing with no duplicate indices"
                    )
            restricted_dims.append((dim, sel, sel_arr.shape[0]))
            expected_shape[dim] = sel_arr.shape[0]
        if len(restricted_dims) > 1:
            raise ValueError(
                "input_dim_restrictions may restrict at most one non-filtered "
                f"axis, but {len(restricted_dims)} were given"
            )

    # continue modifying expected_shape if downsampling
    if downsample:
        n, mod = divmod(last_ind - first_ind, ds)
        if mod != 0:
            n += 1
        expected_shape[axis] = n

    expected_shape = tuple(expected_shape)
    # Validate nfft in the sizing pass too, so describe_dims rejects what the
    # real filtering call would reject (keep sibling entry points consistent).
    if nfft is not None:
        if not isinstance(nfft, (int, np.integer)):
            raise ValueError(f"'nfft' must be an integer but got {nfft!r}")
        if nfft < M:
            raise ValueError(f"'nfft' must be at least the kernel size of {M}")

    if describe_dims:
        if verbose:
            print(
                f"Output array should have shape {expected_shape} and "
                f"dtype {dtype}"
            )
        return expected_shape, dtype

    if outarray is None:
        outarray = np.zeros(expected_shape, dtype=dtype)
        if verbose:
            print(
                f"Allocated array of shape {outarray.shape} with dtype {dtype}"
            )
    else:
        if verbose:
            print(
                "Checking output array shape is disabled, make sure portion of "
                f"output array has shape {expected_shape}"
            )
        if not real_output and np.isrealobj(outarray):
            raise TypeError(
                "Output array is real but expected one of a complex dtype"
            )

    # Validate the write window fits: a non-integer offset fails later with an
    # opaque "slice indices must be integers", a negative offset silently
    # misplaces the write via NumPy negative indexing, and an over-large one
    # fails deep in the loop with an opaque broadcast error -- catch all here.
    if not isinstance(output_offset, (int, np.integer)):
        raise ValueError(
            f"'output_offset' must be an integer but got {output_offset!r}"
        )
    if output_offset < 0:
        raise ValueError(
            f"'output_offset' must be >= 0 but got {output_offset}"
        )
    if output_offset + expected_shape[axis] > outarray.shape[axis]:
        raise ValueError(
            f"output_offset ({output_offset}) plus {expected_shape[axis]} output "
            f"samples exceeds outarray length {outarray.shape[axis]} on axis {axis}"
        )

    ######################################################################
    if nfft is None:  # Choose good default fft_length
        nfft = 65536
        while nfft < 10 * M:
            nfft *= 4

    L = nfft - (M - 1)

    #############################################################################
    # Signal block buffer (reused each iteration for the fill logic)
    x_dims = list(signal.shape)
    x_dims[axis] = nfft
    for dim, _, length in restricted_dims:
        x_dims[dim] = length
    # Real input (the common case) uses a real FFT: half the buffer memory and
    # ~2x faster than a full complex FFT. Complex input keeps the full transform.
    buf_dtype = np.float64 if real_output else np.complex128
    x = np.zeros(tuple(x_dims), dtype=buf_dtype)

    ##############################################################################
    # FFT of the kernel, computed once and reused for every block.
    y_dims = [1] * signal.ndim
    y_dims[axis] = nfft
    y = np.zeros(tuple(y_dims), dtype=buf_dtype)
    # heterogeneous index-tuple builders (hold ints and slices/arrays)
    y_slices: list = [0] * signal.ndim
    y_slices[axis] = np.s_[0:M]
    y[tuple(y_slices)] = kernel  # remainder stays zero (y is np.zeros)
    if real_output:
        yf = scipy.fft.rfft(y, n=nfft, axis=axis, workers=threads)
    else:
        yf = scipy.fft.fft(y, axis=axis, workers=threads)

    ####################################################################
    start_offset = 0
    if input_index_bounds is not None:
        start_offset = input_index_bounds[0]

    signal_slices_1: list = [np.s_[:]] * signal.ndim
    signal_slices_2: list = [np.s_[:]] * signal.ndim
    for dim, sel, _ in restricted_dims:
        signal_slices_1[dim] = sel
        signal_slices_2[dim] = sel

    x_slices_1 = [np.s_[:]] * x.ndim
    x_slices_2 = [np.s_[:]] * x.ndim

    outarray_slices = [np.s_[:]] * outarray.ndim
    conv_slices = [np.s_[:]] * x.ndim

    outarray_marker = output_offset

    # note that first_ind and last_ind are used to index into the output of the
    # convolution. They are not used to determine where in this function's
    # output array the convolution results are written. outarray_marker does that.
    first_block_to_check, first_offset = divmod(first_ind, L)
    last_block_to_check, last_offset = divmod(last_ind, L)
    # last_ind is an exclusive stop. When it lands exactly on a block boundary
    # (last_offset == 0) the naive divmod points at the *next* block, which then
    # contributes zero samples but is still read -- a wasted FFT, and a read one
    # block past the needed region that can fail on a strict lazy/on-disk input.
    # Fold it into the previous block instead (output is identical: that block's
    # tail slice [..:M-1+L] == [..:nfft] is the same valid region either way).
    if last_offset == 0:
        last_block_to_check -= 1
        last_offset = L

    tot_samples = 0
    for ii in range(first_block_to_check, last_block_to_check + 1):
        start = start_offset + ii * L
        stop = start + L

        # initialize entire block to 0, then fill with appropriate input data
        x[:] = 0
        ind1 = start - (M - 1)
        # fill M - 1 overlap portion of the block. It may extend past the start
        # of the data; if so, place the valid part at the END of the segment.
        if ind1 < 0:
            ind1 = 0
            signal_slices_1[axis] = np.s_[ind1:start]
            signal_chunk = signal[tuple(signal_slices_1)]
            length = signal_chunk.shape[axis]
            diff = M - 1 - length
            x_slices_1[axis] = np.s_[diff : M - 1]
            x[tuple(x_slices_1)] = signal_chunk
        else:
            # A slice read clips at the array bounds rather than raising, so a
            # genuine failure here (e.g. an h5py I/O error on an on-disk signal)
            # must propagate -- upstream swallowed it and silently zero-filled
            # the overlap, corrupting the result without any error.
            signal_slices_1[axis] = np.s_[ind1:start]
            signal_chunk = signal[tuple(signal_slices_1)]
            length = signal_chunk.shape[axis]
            x_slices_1[axis] = np.s_[:length]
            x[tuple(x_slices_1)] = signal_chunk

        # fill L segment of the block
        signal_slices_2[axis] = np.s_[start:stop]
        signal_chunk = signal[tuple(signal_slices_2)]
        length = signal_chunk.shape[axis]
        x_slices_2[axis] = np.s_[M - 1 : M - 1 + length]
        x[tuple(x_slices_2)] = signal_chunk

        # circular convolution of this block via the convolution theorem
        if real_output:
            conv = scipy.fft.irfft(
                scipy.fft.rfft(x, n=nfft, axis=axis, workers=threads) * yf,
                n=nfft,
                axis=axis,
                workers=threads,
            )
        else:
            conv = scipy.fft.ifft(
                scipy.fft.fft(x, axis=axis, workers=threads) * yf,
                axis=axis,
                workers=threads,
            )

        # Every block writes conv[M-1+lo : M-1+hi : step]. The low offset is the
        # first-block head, the decimation carry for later blocks, or 0; the high
        # offset is the last-block tail or the full block length L (note
        # M-1+L == nfft). This single form is equivalent to the four
        # first/last/middle x ds/non-ds cases in upstream's osconvolve.
        if ii == first_block_to_check:
            lo = first_offset
        elif downsample:
            lo = block_offset
        else:
            lo = 0
        hi = last_offset if ii == last_block_to_check else L
        step = ds if downsample else 1
        conv_slices[axis] = np.s_[M - 1 + lo : M - 1 + hi : step]
        if downsample:
            n_samples = conv[tuple(conv_slices)].shape[axis]
            if ii != last_block_to_check:
                # phase of the next block's first retained sample
                block_offset = lo + n_samples * ds - L
        else:
            n_samples = hi - lo
        outarray_slices[axis] = np.s_[
            outarray_marker : outarray_marker + n_samples
        ]

        if real_output:
            outarray[tuple(outarray_slices)] = conv[tuple(conv_slices)].real
        else:
            outarray[tuple(outarray_slices)] = conv[tuple(conv_slices)]
        outarray_marker += n_samples
        tot_samples += n_samples

        if verbose:
            print(f"Computed block {ii} of {last_block_to_check}")

    if not expected_shape[axis] == tot_samples:
        raise ValueError(
            f"Expected to write {expected_shape[axis]} samples for axis {axis} "
            f"but actually wrote {tot_samples}"
        )

    return outarray


def filter_data_fir(
    data: np.ndarray,
    b: npt.ArrayLike,
    *,
    nfft: int | None = None,
    threads: int = cpu_count(),
    axis: int = -1,
    outarray: np.ndarray | None = None,
    input_index_bounds: Sequence[int] | None = None,
    output_index_bounds: Sequence[int] | None = None,
    describe_dims: bool = False,
    ds: int | None = None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None = None,
    output_offset: int = 0,
    verbose: bool = False,
) -> tuple[tuple[int, ...], str] | np.ndarray:
    """Apply an FIR filter to data via overlap-save FFT convolution.

    Uses ``mode='full'``; combined with ``output_index_bounds`` set to
    ``[group_delay, group_delay + N]`` this yields the zero-phase,
    delay-compensated output that spyglass relies on.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be filtered, shape ``(..., n_time, ...)`` with the filtered
        axis given by ``axis``. Must expose ``.ndim``/``.shape`` and support
        slice + integer-array indexing (a NumPy array or an h5py ``Dataset``).
        It is NOT converted to an array, so an on-disk/lazy signal stays on
        disk. Real input yields a real result; complex input a complex result.
    b : array_like, shape (M,)
        Filter coefficients (1-D). Converted to a NumPy array internally.
    nfft : int, optional
        FFT length along the filtered axis; must be an integer >= ``M``.
        Default chosen automatically.
    threads : int, optional
        Number of FFT worker threads (>= 1). Default is the CPU count.
    axis : int, optional
        Axis along which to filter. Default is -1.
    outarray : numpy.ndarray, optional
        Preallocated output (may be on disk). Default allocates in memory. See
        Notes for the dtype contract.
    input_index_bounds : sequence of 2 int, optional
        ``[start, stop)`` indices of the input along ``axis`` (stop exclusive).
    output_index_bounds : sequence of 2 int, optional
        ``[start, stop)`` indices of the full-convolution output to keep (stop
        exclusive).
    describe_dims : bool, optional
        If True, return ``(shape, dtype)`` without filtering. Default False.
    ds : int, optional
        Integer decimation factor (>= 1). Default None (no decimation).
    input_dim_restrictions : sequence, optional
        One entry per dimension of ``data``. The entry for ``axis`` must be
        None; at most one other entry may be set, and it must be a 1-D,
        in-range, strictly increasing array of unique integer indices selecting
        which elements of that (non-filtered) axis to keep -- e.g. a subset of
        electrodes. Slices/masks and restricting more than one axis are not
        supported (they raise).
    output_offset : int, optional
        Offset (>= 0) into ``outarray`` along ``axis`` at which to start
        writing. Default 0.
    verbose : bool, optional
        Print per-block progress. Default False.

    Returns
    -------
    tuple of (tuple of int, str), or numpy.ndarray
        If ``describe_dims`` is True, the ``(shape, dtype)`` the output would
        have. Otherwise the filtered (and optionally decimated) data: when an
        ``outarray`` was supplied, the SAME object is returned (written in
        place); otherwise a newly allocated array is returned.

    Raises
    ------
    ValueError
        On invalid arguments -- e.g. ``threads < 1``, a non-1-D kernel, an
        out-of-range or non-integer ``nfft``/``ds``/``output_offset``, bounds
        that are reversed or out of range, or unsupported
        ``input_dim_restrictions``.
    IndexError
        If ``input_index_bounds`` or a restriction array is out of range.
    TypeError
        If the result is complex but a real-dtype ``outarray`` was supplied.

    Notes
    -----
    Output dtype: real input yields ``float64`` (``'<f8'``), complex input
    yields ``complex128`` (``'<c16'``). If you supply your own ``outarray``, its
    dtype is used as-is and the result is cast into it -- assigning the float
    result into an integer array truncates silently, so match the dtype from
    ``describe_dims`` (a lower-precision float such as ``float32`` is fine).

    Out-of-core streaming protocol (how spyglass filters data larger than RAM):
    call once per interval with ``describe_dims=True`` to get each interval's
    output length, preallocate a single (possibly on-disk) array sized to their
    sum, then call again per interval with that array as ``outarray`` and the
    running cumulative length as ``output_offset``.

    The input is assumed finite: a NaN/inf in any block spreads across that
    whole block's output via the FFT.
    """
    return _osconvolve(
        data,
        b,
        mode="full",
        nfft=nfft,
        threads=threads,
        axis=axis,
        outarray=outarray,
        input_index_bounds=input_index_bounds,
        output_index_bounds=output_index_bounds,
        describe_dims=describe_dims,
        ds=ds,
        input_dim_restrictions=input_dim_restrictions,
        output_offset=output_offset,
        verbose=verbose,
    )
