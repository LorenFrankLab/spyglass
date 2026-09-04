"""Self-contained FIR filter design and out-of-core FIR filtering.

Vendored from ghostipy (Apache-2.0, https://github.com/kemerelab/ghostipy),
covering the FIR subset the spyglass LFP pipeline needs. A copy of the Apache
License 2.0 ships with spyglass at
``spyglass/common/licenses/ghostipy-Apache-2.0.txt``; the modifications made
here are listed under "Intentional divergences from upstream" below.

Vendored from upstream (upstream name -> name here):

- ``estimate_taps``    -> ``estimate_taps``, FIR tap-count estimate
- ``firdesign``        -> ``firdesign``, Type I design with spline transition
  bands (L2), and its ``_firspline`` low-pass prototype helper
- ``group_delay``      -> ``group_delay``, integer group delay of a Type I FIR
- ``filter_data_fir``  -> ``filter_data_fir``, the ``mode='full'`` entry point
- ``osconvolve``       -> ``_osconvolve``, the overlap-save engine beneath it,
  which streams into a (possibly on-disk) preallocated output array with
  decimation and per-dimension index restrictions. Its argument validation and
  output-shape planning were split out into ``_plan_osconvolve``, which has no
  upstream counterpart.

``common_filter.py`` calls ``estimate_taps``, ``firdesign`` and
``filter_data_fir``; ``group_delay`` is vendored and exported for completeness
but spyglass uses its own ``FirFilterParameters.calc_filter_delay`` instead.

Call signatures keep upstream ghostipy's parameter order and defaults, with
clearer parameter names (e.g. ``fs`` -> ``sampling_freq``, ``tw`` ->
``transition_width``, ``p`` -> ``spline_power``, ``b`` -> ``filter_coeffs``)
and added type annotations; numeric output matches upstream to floating-point
round-off (measured: ``firdesign`` coefficients bitwise identical, filtered
output within 4.4e-16 on spyglass's LFP calls).

Intentional divergences from upstream:

- FFT backend: ``pyfftw`` replaced with ``scipy.fft`` (``workers=`` for
  multithreading), using the real transform (``rfft``/``irfft``) for real input
  and the full complex transform otherwise. This removed the M1-Mac /
  conda-forge install friction the spyglass setup notes used to document, and is
  the only change that touches the numeric path for valid input (round-off
  only).
- Block selection at boundaries: when the requested (exclusive) output stop
  lands exactly on an overlap-save block boundary, the trailing empty block is
  no longer processed. Upstream still read and FFT'd it only to write zero
  samples; skipping it removes a wasted transform and a read one block past the
  needed data (which could fail on a strict lazy/on-disk signal). Output is
  identical.
- Overlap placement bug fix: upstream positioned a block's leading M-1 overlap
  samples from the LENGTH of the (clipped) read, which is only correct when the
  read reaches the block start. For a signal shorter than M-1 filtered with an
  ``nfft`` tight enough to need several blocks, the read is clipped at both
  ends and upstream shifted those samples, returning a wrong convolution
  (e.g. ``[2, 4, 4, 4, 10]`` instead of ``[2, 4, 6, 8, 10]``). The position is
  now derived from where the read actually starts. Unreachable at spyglass's
  default ``nfft`` (>= 10x the kernel), so LFP output is unaffected.
- Fail-loud / fail-closed hardening that affects only invalid inputs or genuine
  errors (never the valid path spyglass exercises): the M-1 overlap read no
  longer swallows exceptions and silently zero-fills; ``input_index_bounds`` /
  ``output_index_bounds`` treat the stop as exclusive and validate by range
  rather than probing the array; ``estimate_taps`` rejects non-positive
  ``sampling_freq``, ``transition_width``, ``passband_deviation``, and
  ``stopband_deviation``, and deviations so loose the tap estimate would be < 1;
  ``firdesign``/``_firspline`` require an integer ``numtaps`` >= 1 and at
  least two ordered ``band_edges``; the spline power ``spline_power`` must be
  > 0; ``decimation_factor`` must be an integer >= 1; ``nfft`` must be an
  integer >= the kernel length; ``input_dim_restrictions`` entries must be 1-D
  integer index arrays restricting at most one non-filtered axis;
  ``output_offset`` must be an integer >= 0 that fits within ``outarray``;
  complex input no longer raises ``UnboundLocalError``.
- Provably empty block reads are skipped rather than issued: the leading overlap
  read when a block starts at sample 0 (nothing precedes it), and the main read
  when a trailing block starts at or past the end of the data. Upstream issued
  both unconditionally and hid the fallout in the blanket ``except`` above. h5py
  rejects an empty slice combined with a fancy index of 16 or more elements
  ("Dataspaces don't have hyperslab selections"), which is an ordinary LFP
  configuration. The block buffer is already zeroed, so skipping is also the
  correct fill.
- ``input_dim_restrictions`` may select in any order, including duplicates, and
  rows are returned in the order requested. An unsorted selection is READ as
  sorted unique indices -- all h5py accepts -- then gathered back. Upstream
  passed the array straight through, which worked only for a sorted selection on
  an on-disk signal.
- ``outarray``'s real/complex check reads its ``dtype`` instead of a slice of
  its contents, so no data is read from a (possibly on-disk) output array.
- ``verbose`` is gone from both entry points; progress goes to a module-level
  ``logging`` logger instead of ``print``.

Design details (spline-transition FIR) follow Burrus et al., 1992.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Protocol, TypeVar, overload

import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.signal

logger = logging.getLogger(__name__)


class _ReadableArray(Protocol):
    """Signal source: an in-memory numpy array OR an on-disk h5py Dataset.

    The signal is deliberately never passed through ``np.asarray`` (that would
    pull an on-disk dataset fully into memory), so it is typed by the surface the
    block loop actually uses rather than as ``np.ndarray``. ``dtype`` is required
    as well as read: ``np.isrealobj`` falls back to materialising the whole array
    for an object that lacks it.
    """

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def __getitem__(self, key, /) -> np.ndarray: ...


class _WritableArray(_ReadableArray, Protocol):
    """Output sink, likewise possibly on disk (spyglass writes into NWB)."""

    def __setitem__(self, key, value, /) -> None: ...


# filter_data_fir writes into a supplied outarray IN PLACE and returns that same
# object, so the return type follows the argument: an h5py Dataset in gives an
# h5py Dataset back, not a numpy array. Only the allocate-for-me case returns an
# ndarray.
_OutArrayT = TypeVar("_OutArrayT", bound=_WritableArray)


__all__ = [
    "estimate_taps",
    "group_delay",
    "firdesign",
    "describe_output",
    "filter_data_fir",
]


def _assert_finite_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is finite and strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"'{name}' must be finite and positive but got {value}"
        )


def group_delay(filter_coeffs: np.ndarray) -> int:
    """Group delay of a linear-phase (Type I) FIR filter.

    Parameters
    ----------
    filter_coeffs : numpy.ndarray, shape (N,)
        The filter coefficients. ``N`` must be odd.

    Returns
    -------
    int
        The group delay in samples, ``(N - 1) // 2``.

    Raises
    ------
    ValueError
        If ``filter_coeffs`` has an even number of coefficients, for which the
        group delay is not an integer.
    """
    numtaps = len(filter_coeffs)
    if not numtaps & 1:
        raise ValueError(
            f"There are {numtaps} filter coefficients (an even number), so the "
            "group delay cannot be converted to an integer value"
        )
    return (numtaps - 1) // 2


def estimate_taps(
    sampling_freq: float,
    transition_width: float,
    *,
    passband_deviation: float = 1e-3,
    stopband_deviation: float = 1e-6,
) -> int:
    """Estimate the number of taps for a Type I FIR filter.

    Parameters
    ----------
    sampling_freq : float
        Sampling rate in Hz.
    transition_width : float
        Transition bandwidth in Hz.
    passband_deviation : float, optional
        Passband deviation. Default is 0.1% (1e-3).
    stopband_deviation : float, optional
        Minimum stopband attenuation. Default is 120 dB (1e-6).

    Returns
    -------
    int
        Number of taps (always odd).

    Raises
    ------
    ValueError
        If ``sampling_freq``, ``transition_width``, ``passband_deviation``, or
        ``stopband_deviation`` is non-finite or non-positive, or if the
        deviations are so loose that the estimated tap count is < 1 (i.e.
        ``10 * passband_deviation * stopband_deviation >= 1``).

    References
    ----------
    https://dsp.stackexchange.com/questions/31066
    """
    _assert_finite_positive(sampling_freq, "sampling_freq")
    _assert_finite_positive(transition_width, "transition_width")
    if (
        not np.isfinite(passband_deviation)
        or not np.isfinite(stopband_deviation)
        or passband_deviation <= 0
        or stopband_deviation <= 0
    ):
        raise ValueError(
            "passband/stopband deviations must be finite and positive but got "
            f"passband_deviation={passband_deviation}, "
            f"stopband_deviation={stopband_deviation}"
        )

    deviation_product = 10 * passband_deviation * stopband_deviation
    numtaps = int(
        np.ceil(
            2
            / 3
            * np.log10(1 / deviation_product)
            * sampling_freq
            / transition_width
        )
    )
    if numtaps < 1:
        raise ValueError(
            f"computed numtaps={numtaps} < 1; the deviations "
            f"passband_deviation={passband_deviation}, "
            f"stopband_deviation={stopband_deviation} are too loose "
            "(10 * passband_deviation * stopband_deviation must be < 1)"
        )
    if not numtaps & 1:
        numtaps += 1
    return numtaps


def _firspline(
    numtaps: int,
    pass_freq: float,
    stop_freq: float,
    *,
    sampling_freq: float = 2,
    spline_power: float | None = None,
) -> np.ndarray:
    """Design a Type I low-pass filter with a spline transition band.

    Parameters
    ----------
    numtaps : int
        Number of coefficients (must be a positive odd integer).
    pass_freq : float
        Frequency where the amplitude response is 1, in Hz.
    stop_freq : float
        Frequency where the amplitude response is 0, in Hz.
    sampling_freq : float, optional
        Sampling rate in Hz. Default is 2 Hz.
    spline_power : float, optional
        Spline power. Default follows Burrus et al., 1992.

    Returns
    -------
    numpy.ndarray, shape (numtaps,)
        The filter coefficients.

    Raises
    ------
    ValueError
        If ``numtaps`` is not a positive odd integer, if ``sampling_freq`` is
        non-finite or non-positive, if ``pass_freq``/``stop_freq`` exceed the
        Nyquist frequency or are not strictly increasing, or if
        ``spline_power`` is non-finite or non-positive.
    """
    if not isinstance(numtaps, (int, np.integer)):
        raise ValueError(f"numtaps must be an integer but got {numtaps}")
    if numtaps < 1:
        raise ValueError(
            f"numtaps must be a positive odd value but got {numtaps}"
        )
    if not numtaps & 1:
        raise ValueError(f"numtaps must be odd but got {numtaps}")

    _assert_finite_positive(sampling_freq, "sampling_freq")
    nyquist = sampling_freq / 2

    if pass_freq > nyquist or stop_freq > nyquist:
        raise ValueError(
            f"Got critical frequencies {pass_freq} and {stop_freq} but they "
            f"must both be less than the Nyquist frequency of {nyquist}"
        )
    if pass_freq >= stop_freq:
        raise ValueError(f"pass_freq must be <{stop_freq} but got {pass_freq}")

    if spline_power is None:
        spline_power = 0.312 * numtaps * (stop_freq - pass_freq) / nyquist
    # Must be strictly positive: spline_power appears in the denominator below,
    # and spline_power == 0 silently collapses the spline transition to a
    # rectangular truncation (nan ** 0 == 1) rather than raising.
    if not np.isfinite(spline_power) or spline_power <= 0:
        raise ValueError(
            f"spline_power must be positive but got {spline_power}"
        )

    # Convert to normalized frequency in radians
    center_rad = (pass_freq + stop_freq) / 2 * (1 / nyquist * np.pi)
    half_width_rad = (stop_freq - pass_freq) / 2 * (1 / nyquist * np.pi)

    tap_indices = np.arange(1, (numtaps - 1) // 2 + 1)

    # Optimal L2 solution to the ideal lowpass filter
    half_taps = np.sin(center_rad * tap_indices) / (np.pi * tap_indices)

    spline_arg = half_width_rad * tap_indices / spline_power
    spline = (np.sin(spline_arg) / spline_arg) ** spline_power
    half_taps *= spline  # connect transition band with spline

    # make linear phase
    coeffs = np.hstack((np.flip(half_taps), center_rad / np.pi, half_taps))
    return coeffs


def firdesign(
    numtaps: int,
    band_edges: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    sampling_freq: float = 1,
    spline_power: float | None = None,
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
    sampling_freq : float, optional
        Sampling rate in Hz. Default is 1 Hz.
    spline_power : float, optional
        Power for the spline transition-band functions. Default follows
        Burrus et al., 1992.

    Returns
    -------
    numpy.ndarray, shape (numtaps,)
        The filter coefficients.

    Raises
    ------
    ValueError
        If ``numtaps`` is not a positive odd integer; if ``sampling_freq`` is
        non-finite or non-positive; if ``band_edges`` is empty, has an odd
        length, differs in length from ``desired``, has a non-positive first
        edge, has a last
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
    _assert_finite_positive(sampling_freq, "sampling_freq")
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
    if not band_edges[-1] < sampling_freq / 2:
        raise ValueError(
            f"Last band edge must be less than {sampling_freq / 2}"
        )
    if not np.all(band_edges[:-1] < band_edges[1:]):
        raise ValueError(
            "'band_edges' must be a monotonically increasing sequence"
        )

    for edge_index, (desired_left, desired_right) in enumerate(
        zip(desired, desired[1:])
    ):
        edge_left = band_edges[edge_index]
        edge_right = band_edges[edge_index + 1]
        if edge_index % 2 == 0:
            if desired_left == desired_right:
                raise ValueError(
                    f"Got {desired_left} for band edge {edge_left} Hz and "
                    f"{desired_right} for band edge {edge_right} Hz but must be "
                    "different values"
                )
        else:
            if desired_left != desired_right:
                raise ValueError(
                    f"Got {desired_left} for band edge {edge_left} Hz and "
                    f"{desired_right} for band edge {edge_right} Hz but must be "
                    "the same values"
                )

    critical_points = band_edges.reshape((-1, 2))
    # low pass prototypes
    prototypes = np.zeros((len(critical_points), numtaps))
    for ind, (pass_freq, stop_freq) in enumerate(critical_points):
        prototypes[ind] = _firspline(
            numtaps,
            pass_freq,
            stop_freq,
            sampling_freq=sampling_freq,
            spline_power=spline_power,
        )

    # center impulse (identity filter), used to invert a lowpass into a highpass
    impulse = scipy.signal.unit_impulse(numtaps, "mid")

    if prototypes.shape[0] == 1:  # single band
        coeffs = prototypes[0]
        if desired[-1] == 1:  # high pass
            coeffs = impulse - prototypes[0]
    else:  # multi-band
        coeffs = np.zeros(numtaps)

        # Magnitude at 0 and Nyquist is the same
        if desired[0] == desired[-1]:
            for ii in range(0, prototypes.shape[0], 2):
                lowpass_low = prototypes[ii]
                lowpass_high = prototypes[ii + 1]
                coeffs += lowpass_high - lowpass_low

            # high pass at 0 and Nyquist, so invert
            if desired[-1] == 1:
                coeffs = impulse - coeffs
        else:
            if desired[0] == 0:
                special_band = impulse - prototypes[-1]
                prototypes = prototypes[:-1]
            else:
                special_band = prototypes[0]
                prototypes = prototypes[1:]

            for ii in range(0, prototypes.shape[0] - 1, 2):
                lowpass_low = prototypes[ii]
                lowpass_high = prototypes[ii + 1]
                coeffs += lowpass_high - lowpass_low

            coeffs += special_band

    return coeffs


@dataclass
class _OsPlan:
    """Validated arguments and output plan for one overlap-save convolution.

    Produced by :func:`_plan_osconvolve` and consumed by :func:`_osconvolve`, it
    carries everything the block loop (and :func:`describe_output`) needs, so
    no argument is re-validated once execution starts.
    """

    real_output: bool
    kernel_len: int
    first_ind: int
    last_ind: int
    decimation_factor: int | None
    # (dim, read_sel, gather, out_length) per restricted non-filtered axis:
    # read the signal at ``read_sel`` (sorted unique, which h5py requires) and,
    # when ``gather`` is not None, take ``gather`` along ``dim`` to restore the
    # order the caller asked for. ``out_length`` is the requested length.
    restricted_dims: list[tuple[int, np.ndarray, np.ndarray | None, int]]
    expected_shape: tuple[int, ...]
    dtype: str


def _plan_osconvolve(
    signal: _ReadableArray,
    kernel: np.ndarray,
    *,
    mode: str,
    nfft: int | None,
    threads: int,
    axis: int,
    input_index_bounds: Sequence[int] | None,
    output_index_bounds: Sequence[int] | None,
    decimation_factor: int | None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None,
) -> _OsPlan:
    """Validate arguments and plan the output shape for :func:`_osconvolve`.

    Pure validation and shape planning: raises on any invalid argument it
    receives and returns the :class:`_OsPlan` the block loop (and
    :func:`describe_output`) needs. ``outarray`` and ``output_offset`` are
    validated by :func:`_osconvolve`, which is where the array itself is
    available.
    No FFT runs and no output array is allocated here; ``kernel`` must already
    be a NumPy array.
    """
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
        # The stop bound is exclusive (n_input is computed as stop - start
        # below), so it may legitimately equal the axis length. Upstream
        # ghostipy probed signal[stop] here, which wrongly rejected a
        # full-length [0, n_input] range; this range check keeps the
        # exclusive-stop semantics consistent.
        if (
            input_index_bounds[0] < 0
            or input_index_bounds[1] > signal.shape[axis]
        ):
            raise IndexError(
                f"input_index_bounds {list(input_index_bounds)} out of range "
                f"for axis {axis} of length {signal.shape[axis]} "
                "(stop is exclusive, so it may equal the axis length)"
            )
        n_input = input_index_bounds[1] - input_index_bounds[0]
    else:
        n_input = signal.shape[axis]

    kernel_len = kernel.shape[0]
    tot_length = n_input + kernel_len - 1

    if mode == "full":
        outsize = tot_length
    elif mode == "same":
        outsize = n_input
    elif mode == "valid":
        if n_input < kernel_len:
            raise ValueError(
                "Cannot do a 'valid' convolution because the input is shorter "
                "than the kernel"
            )
        outsize = n_input - kernel_len + 1
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

    if decimation_factor is not None:
        # decimation_factor == 1 is a valid (no-op) decimation; reject
        # non-integer / 0 / negative
        # up front (consistent with nfft/output_offset) so they fail loudly here
        # instead of as a later ZeroDivisionError.
        if (
            not isinstance(decimation_factor, (int, np.integer))
            or decimation_factor < 1
        ):
            raise ValueError(
                "'decimation_factor' must be an integer >= 1 but got "
                f"{decimation_factor!r}"
            )
        decimation_factor = int(decimation_factor)

    ###############################################################
    # handle output params
    dtype = "<f8" if real_output else "<c16"

    # set the default expected_shapes...
    expected_shape = list(signal.shape)
    expected_shape[axis] = last_ind - first_ind

    # ... and override the appropriate part if input_dim_restrictions were passed
    # in. The restriction is recorded once here (see _OsPlan.restricted_dims for
    # the tuple layout) and reused by the _osconvolve block loop for buffer
    # sizing and signal slices, instead of re-scanning every dimension each
    # time.
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
        # Each restriction must be a 1-D, in-range integer index array (as
        # spyglass passes for electrode selection). Slices/masks are rejected
        # because shape planning below relies on len(). Restricting more than
        # one non-filtered axis is unsupported because the shape planning here
        # sets expected_shape[dim] per restricted dim -- i.e. it assumes a
        # Cartesian selection -- while NumPy would instead PAIR the index arrays
        # and broadcast them together (a (3, 2, 10) array indexed with
        # [[0, 1], [0], 0:5] gives (2, 5), not the (2, 1, 5) assumed here). Both
        # fail loudly rather than producing a wrong shape.
        #
        # Any order is accepted, including duplicates, and rows come back in the
        # order requested (spyglass's public FirFilterParameters.filter_data
        # documents no ordering requirement for its `electrodes` argument, and
        # the electrode IDs it sorts can still map to unsorted ElectricalSeries
        # columns). NumPy would fancy-index an in-memory array in any order, but
        # h5py cannot, so an unsorted or repeated selection is READ as sorted
        # unique indices and gathered back into the requested order after each
        # block read -- which works for every signal type.
        for dim in range(signal.ndim):
            sel = input_dim_restrictions[dim]
            if dim == axis or sel is None:
                continue
            sel_arr = np.asarray(sel)
            if sel_arr.ndim == 1 and sel_arr.size == 0:
                # An empty Python list becomes float64, which the integer-dtype
                # check below would reject with a message about dtypes. Selecting
                # no indices is a caller mistake worth naming directly -- e.g. an
                # LFPElectrodeGroup with no electrodes.
                raise ValueError(
                    f"input_dim_restrictions[{dim}] selects no indices; there "
                    "is nothing to filter along that axis"
                )
            if sel_arr.ndim != 1 or not np.issubdtype(
                sel_arr.dtype, np.integer
            ):
                raise ValueError(
                    f"input_dim_restrictions[{dim}] must be None or a 1-D array "
                    "of integer indices (slices/masks are not supported)"
                )
            if sel_arr.size > 0:
                bad = sel_arr[(sel_arr < 0) | (sel_arr >= signal.shape[dim])]
                if bad.size:
                    raise IndexError(
                        f"input_dim_restrictions[{dim}] contains indices out of "
                        f"range for axis length {signal.shape[dim]}: "
                        f"{np.unique(bad).tolist()}"
                    )
            read_sel, gather = sel_arr, None
            if sel_arr.size > 0 and not np.all(sel_arr[:-1] < sel_arr[1:]):
                # Read in sorted-unique order (what h5py accepts) and record the
                # gather that puts the rows back into the requested order.
                read_sel = np.unique(sel_arr)
                gather = np.searchsorted(read_sel, sel_arr)
            restricted_dims.append((dim, read_sel, gather, sel_arr.shape[0]))
            expected_shape[dim] = sel_arr.shape[0]
        if len(restricted_dims) > 1:
            raise ValueError(
                "input_dim_restrictions may restrict at most one non-filtered "
                f"axis, but {len(restricted_dims)} were given"
            )

    # continue modifying expected_shape if decimating
    if decimation_factor is not None:
        n, mod = divmod(last_ind - first_ind, decimation_factor)
        if mod != 0:
            n += 1
        expected_shape[axis] = n

    expected_shape = tuple(expected_shape)
    # nfft is validated here, in the planning pass, so describe_output rejects
    # exactly what the real filtering call would reject. _osconvolve only fills
    # in the default when None.
    if nfft is not None:
        if not isinstance(nfft, (int, np.integer)):
            raise ValueError(f"'nfft' must be an integer but got {nfft!r}")
        if nfft < kernel_len:
            raise ValueError(
                f"'nfft' must be at least the kernel size of {kernel_len}"
            )

    return _OsPlan(
        real_output=real_output,
        kernel_len=kernel_len,
        first_ind=first_ind,
        last_ind=last_ind,
        decimation_factor=decimation_factor,
        restricted_dims=restricted_dims,
        expected_shape=expected_shape,
        dtype=dtype,
    )


def _osconvolve(
    signal: _ReadableArray,
    kernel: npt.ArrayLike,
    *,
    mode: str = "full",
    nfft: int | None = None,
    threads: int = cpu_count(),
    axis: int = -1,
    outarray: _WritableArray | None = None,
    input_index_bounds: Sequence[int] | None = None,
    output_index_bounds: Sequence[int] | None = None,
    decimation_factor: int | None = None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None = None,
    output_offset: int = 0,
) -> _WritableArray:
    """Overlap-save FFT convolution, written to minimize memory usage.

    Streams the convolution block-by-block into ``outarray`` (which may be an
    on-disk array), supporting decimation (``decimation_factor``) and
    restricting which indices of one non-filtered axis are used
    (``input_dim_restrictions``).
    ``mode`` selects ``'full'``, ``'same'``, or ``'valid'`` convolution.

    Ported from ghostipy's ``osconvolve``; for the inputs spyglass uses, the
    numeric output matches upstream to floating-point round-off. It differs
    deliberately where upstream was wrong -- see the overlap-placement fix in the
    module header, which changes the result for a signal shorter than the filter
    under a tight ``nfft``. Argument validation and output-shape
    planning live in :func:`_plan_osconvolve`; this function allocates the output
    (with the ``output_offset``/``outarray`` fit check that needs the array) and
    runs the block loop. It is the general convolution engine (any
    ``mode``), of which :func:`filter_data_fir` is the ``mode='full'`` wrapper
    that spyglass calls. See the module header for the full list of intentional
    divergences from upstream, and :func:`filter_data_fir` for the meaning of the
    shared out-of-core parameters.

    Returns
    -------
    numpy.ndarray or the ``outarray`` type
        The filled ``outarray`` itself, or a newly allocated numpy array if none
        was given.
    """
    # The kernel is small and always in memory, so normalize it to an array for
    # convenience. The signal is deliberately NOT converted: it may be an
    # on-disk / lazy array, and np.asarray would force it fully into memory.
    kernel = np.asarray(kernel)

    plan = _plan_osconvolve(
        signal,
        kernel,
        mode=mode,
        nfft=nfft,
        threads=threads,
        axis=axis,
        input_index_bounds=input_index_bounds,
        output_index_bounds=output_index_bounds,
        decimation_factor=decimation_factor,
        input_dim_restrictions=input_dim_restrictions,
    )
    real_output = plan.real_output
    kernel_len = plan.kernel_len
    first_ind = plan.first_ind
    last_ind = plan.last_ind
    decimation_factor = plan.decimation_factor
    restricted_dims = plan.restricted_dims
    expected_shape = plan.expected_shape
    dtype = plan.dtype
    block_offset = 0

    if outarray is None:
        outarray = np.zeros(expected_shape, dtype=dtype)
        logger.debug(
            "Allocated array of shape %s with dtype %s", outarray.shape, dtype
        )
    else:
        logger.debug(
            "Output array shape not checked; ensure its written portion has "
            "shape %s",
            expected_shape,
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
        while nfft < 10 * kernel_len:
            nfft *= 4

    block_step = nfft - (kernel_len - 1)

    #############################################################################
    # Signal block buffer (reused each iteration for the fill logic)
    block_shape = list(signal.shape)
    block_shape[axis] = nfft
    for dim, _, _, length in restricted_dims:
        block_shape[dim] = length
    # Real input (the common case) uses a real FFT: half the buffer memory and
    # ~2x faster than a full complex FFT. Complex input keeps the full transform.
    buf_dtype = np.float64 if real_output else np.complex128
    block_buffer = np.zeros(tuple(block_shape), dtype=buf_dtype)

    ##############################################################################
    # FFT of the kernel, computed once and reused for every block.
    kernel_shape = [1] * signal.ndim
    kernel_shape[axis] = nfft
    kernel_buffer = np.zeros(tuple(kernel_shape), dtype=buf_dtype)
    # heterogeneous index-tuple builders (hold ints and slices/arrays)
    kernel_slices: list = [0] * signal.ndim
    kernel_slices[axis] = np.s_[0:kernel_len]
    # remainder stays zero (kernel_buffer was allocated with np.zeros)
    kernel_buffer[tuple(kernel_slices)] = kernel
    if real_output:
        kernel_fft = scipy.fft.rfft(
            kernel_buffer, n=nfft, axis=axis, workers=threads
        )
    else:
        kernel_fft = scipy.fft.fft(kernel_buffer, axis=axis, workers=threads)

    ####################################################################
    start_offset = 0
    if input_index_bounds is not None:
        start_offset = input_index_bounds[0]

    signal_slices_1: list = [np.s_[:]] * signal.ndim
    signal_slices_2: list = [np.s_[:]] * signal.ndim
    # At most one non-filtered axis is restricted, so one gather covers it.
    gather_dim = gather_index = None
    for dim, read_sel, gather, _ in restricted_dims:
        signal_slices_1[dim] = read_sel
        signal_slices_2[dim] = read_sel
        if gather is not None:
            gather_dim, gather_index = dim, gather

    def read_signal(slices: list) -> np.ndarray:
        """Read one block chunk, restoring the caller's requested row order."""
        chunk = signal[tuple(slices)]
        if gather_index is None:
            return chunk
        return np.take(chunk, gather_index, axis=gather_dim)

    block_slices_1 = [np.s_[:]] * block_buffer.ndim
    block_slices_2 = [np.s_[:]] * block_buffer.ndim

    outarray_slices = [np.s_[:]] * outarray.ndim
    conv_slices = [np.s_[:]] * block_buffer.ndim

    write_pos = output_offset

    # note that first_ind and last_ind are used to index into the output of the
    # convolution. They are not used to determine where in this function's
    # output array the convolution results are written. write_pos does that.
    first_block_to_check, first_offset = divmod(first_ind, block_step)
    last_block_to_check, last_offset = divmod(last_ind, block_step)
    # last_ind is an exclusive stop. When it lands exactly on a block boundary
    # (last_offset == 0) the naive divmod points at the *next* block, which then
    # contributes zero samples but is still read -- a wasted FFT, and a read one
    # block past the needed region that can fail on a strict lazy/on-disk input.
    # Fold it into the previous block instead (output is identical: that block's
    # tail slice [..:kernel_len-1+block_step] == [..:nfft] is the same valid
    # region either way).
    if last_offset == 0:
        last_block_to_check -= 1
        last_offset = block_step

    samples_written = 0
    for ii in range(first_block_to_check, last_block_to_check + 1):
        start = start_offset + ii * block_step
        stop = start + block_step

        # initialize entire block to 0, then fill with appropriate input data
        block_buffer[:] = 0

        # Fill the kernel_len - 1 overlap portion of the block. Block position p
        # holds absolute sample overlap_start + p, so the chunk goes wherever the
        # (possibly clipped) read actually starts. Deriving that position from
        # the read start handles both clips: a window reaching past the start of
        # the data, and -- for a short signal under a tight nfft -- one whose
        # read also stops early at the end of the data. Placing it from the chunk
        # LENGTH instead is only correct when the read reaches `start`, and
        # silently shifted the overlap when the signal ended first.
        #
        # A slice read clips at the array bounds rather than raising, so a
        # genuine failure here (e.g. an h5py I/O error on an on-disk signal)
        # must propagate -- upstream swallowed it and silently zero-filled the
        # overlap, corrupting the result without any error. What must NOT be
        # issued is a read whose slice is provably empty: h5py rejects an empty
        # slice combined with a fancy index of 16 or more elements, which
        # upstream's blanket except hid. Both reads below are therefore guarded
        # on having something to fetch -- the overlap read when the block starts
        # at sample 0 (nothing precedes it), and the main read when a tail block
        # starts at or past the end of the data. Skipping either leaves the
        # already-zeroed buffer, which is the correct fill.
        signal_len = signal.shape[axis]
        overlap_start = start - (kernel_len - 1)
        read_start = max(overlap_start, 0)
        if read_start < start:
            block_pos = read_start - overlap_start
            signal_slices_1[axis] = np.s_[read_start:start]
            signal_chunk = read_signal(signal_slices_1)
            length = signal_chunk.shape[axis]
            block_slices_1[axis] = np.s_[block_pos : block_pos + length]
            block_buffer[tuple(block_slices_1)] = signal_chunk

        # fill block_step segment of the block
        if start < signal_len:
            signal_slices_2[axis] = np.s_[start:stop]
            signal_chunk = read_signal(signal_slices_2)
            length = signal_chunk.shape[axis]
            block_slices_2[axis] = np.s_[
                kernel_len - 1 : kernel_len - 1 + length
            ]
            block_buffer[tuple(block_slices_2)] = signal_chunk

        # circular convolution of this block via the convolution theorem
        if real_output:
            conv_block = scipy.fft.irfft(
                scipy.fft.rfft(block_buffer, n=nfft, axis=axis, workers=threads)
                * kernel_fft,
                n=nfft,
                axis=axis,
                workers=threads,
            )
        else:
            conv_block = scipy.fft.ifft(
                scipy.fft.fft(block_buffer, axis=axis, workers=threads)
                * kernel_fft,
                axis=axis,
                workers=threads,
            )

        # Every block writes
        # conv_block[kernel_len-1+low_offset : kernel_len-1+high_offset : step].
        # The low offset is the first-block head, the decimation carry for later
        # blocks, or 0; the high offset is the last-block tail or the full block
        # length block_step (note kernel_len-1+block_step == nfft). This single
        # form is equivalent to upstream osconvolve's separate first/last/middle
        # and decimating/non-decimating cases.
        if ii == first_block_to_check:
            low_offset = first_offset
        elif decimation_factor is not None:
            low_offset = block_offset
        else:
            low_offset = 0
        high_offset = last_offset if ii == last_block_to_check else block_step
        step = decimation_factor if decimation_factor is not None else 1
        conv_slices[axis] = np.s_[
            kernel_len - 1 + low_offset : kernel_len - 1 + high_offset : step
        ]
        if decimation_factor is not None:
            n_samples = conv_block[tuple(conv_slices)].shape[axis]
            if ii != last_block_to_check:
                # phase of the next block's first retained sample
                block_offset = (
                    low_offset + n_samples * decimation_factor - block_step
                )
        else:
            n_samples = high_offset - low_offset
        outarray_slices[axis] = np.s_[write_pos : write_pos + n_samples]

        if real_output:
            outarray[tuple(outarray_slices)] = conv_block[
                tuple(conv_slices)
            ].real
        else:
            outarray[tuple(outarray_slices)] = conv_block[tuple(conv_slices)]
        write_pos += n_samples
        samples_written += n_samples

        logger.debug("Computed block %d of %d", ii, last_block_to_check)

    if expected_shape[axis] != samples_written:
        raise ValueError(
            f"Expected to write {expected_shape[axis]} samples for axis {axis} "
            f"but actually wrote {samples_written}"
        )

    return outarray


def describe_output(
    data: _ReadableArray,
    filter_coeffs: npt.ArrayLike,
    *,
    nfft: int | None = None,
    axis: int = -1,
    input_index_bounds: Sequence[int] | None = None,
    output_index_bounds: Sequence[int] | None = None,
    decimation_factor: int | None = None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None = None,
) -> tuple[tuple[int, ...], str]:
    """Shape and dtype :func:`filter_data_fir` would produce, without filtering.

    Sizes the preallocated (possibly on-disk) array for the out-of-core
    streaming protocol described in :func:`filter_data_fir`. No data is read and
    no FFT runs -- this only validates the arguments and plans the output.

    Every argument that affects the answer is accepted and validated exactly as
    :func:`filter_data_fir` validates it, so whatever this accepts, the matching
    filtering call accepts too. The arguments that cannot affect the answer
    (``threads``, ``outarray``, ``output_offset``) are deliberately absent
    rather than accepted and ignored.

    Parameters
    ----------
    See :func:`filter_data_fir` for the shared parameters.

    Returns
    -------
    shape : tuple of int
        Shape the output would have, including the effect of
        ``decimation_factor`` and ``input_dim_restrictions``.
    dtype : str
        ``'<f8'`` for real input, ``'<c16'`` for complex input.

    Raises
    ------
    ValueError, IndexError
        On the same invalid arguments :func:`filter_data_fir` rejects.
    """
    plan = _plan_osconvolve(
        data,
        np.asarray(filter_coeffs),
        mode="full",
        nfft=nfft,
        threads=1,  # no FFT runs here, so this cannot affect the answer
        axis=axis,
        input_index_bounds=input_index_bounds,
        output_index_bounds=output_index_bounds,
        decimation_factor=decimation_factor,
        input_dim_restrictions=input_dim_restrictions,
    )
    logger.debug(
        "Output array should have shape %s and dtype %s",
        plan.expected_shape,
        plan.dtype,
    )
    return plan.expected_shape, plan.dtype


@overload
def filter_data_fir(
    data: _ReadableArray,
    filter_coeffs: npt.ArrayLike,
    *,
    outarray: None = None,
    **kwargs,
) -> np.ndarray: ...


@overload
def filter_data_fir(
    data: _ReadableArray,
    filter_coeffs: npt.ArrayLike,
    *,
    outarray: _OutArrayT,
    **kwargs,
) -> _OutArrayT: ...


def filter_data_fir(
    data: _ReadableArray,
    filter_coeffs: npt.ArrayLike,
    *,
    nfft: int | None = None,
    threads: int = cpu_count(),
    axis: int = -1,
    outarray: _WritableArray | None = None,
    input_index_bounds: Sequence[int] | None = None,
    output_index_bounds: Sequence[int] | None = None,
    decimation_factor: int | None = None,
    input_dim_restrictions: Sequence[npt.ArrayLike | None] | None = None,
    output_offset: int = 0,
) -> _WritableArray:
    """Apply an FIR filter to data via overlap-save FFT convolution.

    This is the public entry point spyglass uses: a thin ``mode='full'`` wrapper
    over the general :func:`_osconvolve` engine, exposing only the parameters
    spyglass needs. Combined with ``output_index_bounds`` set to
    ``[group_delay, group_delay + N]`` the full-mode convolution yields the
    zero-phase, delay-compensated output that spyglass relies on.

    Parameters
    ----------
    data : numpy.ndarray or h5py.Dataset
        The data to be filtered, shape ``(..., n_time, ...)`` with the filtered
        axis given by ``axis``. Must expose ``.ndim``/``.shape``/``.dtype`` and
        support slice + integer-array indexing. It is NOT converted to an array,
        so an on-disk/lazy signal stays on disk. Real input yields a real
        result; complex input a complex result.
    filter_coeffs : array_like, shape (M,)
        Filter coefficients (1-D). Converted to a NumPy array internally.
    nfft : int, optional
        FFT length along the filtered axis; must be an integer >= ``M``.
        Default chosen automatically.
    threads : int, optional
        Number of FFT worker threads (>= 1). Default is the CPU count.
    axis : int, optional
        Axis along which to filter. Default is -1.
    outarray : numpy.ndarray or h5py.Dataset, optional
        Preallocated output (may be on disk; spyglass writes into an NWB
        dataset). Default allocates in memory. See Notes for the dtype
        contract.
    input_index_bounds : sequence of 2 int, optional
        ``[start, stop)`` indices along ``axis`` defining WHICH OUTPUT the
        convolution is computed for (stop exclusive). This is not a promise that
        only that input range is read: the filter still draws its support from
        the neighbouring samples of the full array, so with ``[start, stop)`` the
        result at the window edges depends on data outside the window. To filter
        a window in isolation instead, slice the array first and pass bounds
        relative to the slice -- the two give different edge samples.
    output_index_bounds : sequence of 2 int, optional
        ``[start, stop)`` indices of the full-convolution output to keep (stop
        exclusive).
    decimation_factor : int, optional
        Integer decimation factor (>= 1). Default None (no decimation).
    input_dim_restrictions : sequence, optional
        One entry per dimension of ``data``. The entry for ``axis`` must be
        None; at most one other entry may be set, and it must be a 1-D, in-range
        array of integer indices selecting which elements of that (non-filtered)
        axis to keep -- e.g. a subset of electrodes. Any order is accepted,
        including duplicates, and rows are returned in the order given (an
        unsorted selection is read in sorted order, which is all h5py accepts,
        then gathered back). Slices/masks and restricting more than one axis are
        not supported (they raise).
    output_offset : int, optional
        Offset (>= 0) into ``outarray`` along ``axis`` at which to start
        writing. Default 0.

    Returns
    -------
    numpy.ndarray or the ``outarray`` type
        The filtered (and optionally decimated) data. When an ``outarray`` was
        supplied, the SAME object is returned, written in place -- so an h5py
        ``Dataset`` in gives that ``Dataset`` back, not a numpy array.
        Otherwise a newly allocated numpy array is returned. Use
        :func:`describe_output` to size that array without filtering.

    Raises
    ------
    ValueError
        On invalid arguments -- e.g. ``threads < 1``, a non-1-D kernel, a
        non-integer or out-of-range
        ``nfft``/``decimation_factor``/``output_offset``, bounds that are not
        strictly increasing, ``output_index_bounds`` out of range, or unsupported
        ``input_dim_restrictions``.
    IndexError
        If ``input_index_bounds`` or a restriction array is out of range. Note
        that out-of-range bounds raise ``IndexError`` for the INPUT and
        ``ValueError`` for the OUTPUT.
    TypeError
        If the result is complex but a real-dtype ``outarray`` was supplied.

    Notes
    -----
    Output dtype: real input yields ``float64`` (``'<f8'``), complex input
    yields ``complex128`` (``'<c16'``). If you supply your own ``outarray``, its
    dtype is used as-is and the result is cast into it -- assigning the float
    result into an integer array truncates silently, so match the dtype from
    :func:`describe_output` (a lower-precision float such as ``float32`` is
    fine).

    Out-of-core streaming protocol (how spyglass filters data larger than RAM):
    call :func:`describe_output` once per interval to get each interval's output
    length, preallocate a single (possibly on-disk) array sized to their sum,
    then call this function per interval with that array as ``outarray`` and the
    running cumulative length as ``output_offset``.

    The input is assumed finite: a NaN/inf in any block spreads across that
    whole block's output via the FFT.
    """
    return _osconvolve(
        data,
        filter_coeffs,
        mode="full",
        nfft=nfft,
        threads=threads,
        axis=axis,
        outarray=outarray,
        input_index_bounds=input_index_bounds,
        output_index_bounds=output_index_bounds,
        decimation_factor=decimation_factor,
        input_dim_restrictions=input_dim_restrictions,
        output_offset=output_offset,
    )
