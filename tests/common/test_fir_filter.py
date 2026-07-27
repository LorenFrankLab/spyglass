"""Tests for the vendored scipy.fft FIR filtering module (``_fir_filter``).

Correctness is checked against an independent ``numpy.convolve`` reference
rather than against upstream ghostipy, so the suite has no dependency on
ghostipy. They lock in exactly the behavior spyglass relies on in
``common_filter.py`` (FIR design, out-of-core streaming into a preallocated
array, integer decimation, electrode selection, delay compensation) plus the
edge cases the vendored copy hardened relative to upstream.

Run with:  pytest tests/common/test_fir_filter.py

The tests themselves are pure numerics and need no database, but the import
below reaches ``spyglass.common.__init__``, which activates DataJoint schemas --
so collection still requires the test MySQL container like the rest of the
suite.
"""

import numpy as np
import pytest
from scipy.signal import freqz

from spyglass.common import _fir_filter as fir

FS = 30000.0  # spyglass raw ephys rate


# --------------------------------------------------------------------------- #
# Independent reference
# --------------------------------------------------------------------------- #
def reference_filter(
    data,
    b,
    *,
    axis,
    input_index_bounds=None,
    output_index_bounds=None,
    decimation_factor=None,
    electrodes=None,
):
    """Ground-truth FIR filter for 2D (time x electrode) data via np.convolve.

    Mirrors ``fir.filter_data_fir`` (mode='full'). ``input_index_bounds`` limits
    which OUTPUT samples are produced but the filter still draws its support from
    the real neighboring samples of the full lane (overlap-save, not per-window
    zero-padding), so the convolution output index is offset by ``frm``:
    ``out_lane = conv(full_lane, b)[frm + k1 : frm + k2 : decimation_factor]``.
    """
    assert data.ndim == 2
    time_axis = axis
    # normalize to (electrode, time)
    d = np.moveaxis(data, time_axis, 1)
    if electrodes is not None:
        d = d[electrodes, :]
    n_time = d.shape[1]
    frm, to = (0, n_time) if input_index_bounds is None else input_index_bounds
    tot_sub = (to - frm) + len(b) - 1
    k1, k2 = (
        (0, tot_sub) if output_index_bounds is None else output_index_bounds
    )
    step = decimation_factor or 1
    lanes = [
        np.convolve(d[i], b, mode="full")[frm + k1 : frm + k2 : step]
        for i in range(d.shape[0])
    ]
    out = np.stack(lanes, axis=0)  # (electrode, out_time)
    return np.moveaxis(out, 1, time_axis)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lfp_lowpass():
    """0-400 Hz lowpass spyglass uses (transition 400->425), spline_power=2."""
    numtaps = fir.estimate_taps(FS, 25)
    return fir.firdesign(
        numtaps, [400, 425], [1, 0], sampling_freq=FS, spline_power=2
    )


@pytest.fixture
def rng():
    # Function-scoped so each test draws from a fresh, identically-seeded
    # generator -> deterministic AND independent of test execution order.
    return np.random.default_rng(20240101)


# --------------------------------------------------------------------------- #
# FIR design
# --------------------------------------------------------------------------- #
class TestFirDesign:
    def test_estimate_taps_is_odd(self):
        for tw in (5, 25, 100, 250):
            assert fir.estimate_taps(FS, tw) % 2 == 1

    def test_estimate_taps_scales_inversely_with_transition_width(self):
        # narrower transition band -> more taps
        assert fir.estimate_taps(FS, 10) > fir.estimate_taps(FS, 100)

    def test_group_delay(self):
        assert fir.group_delay(np.zeros(101)) == 50
        with pytest.raises(ValueError, match="group delay"):
            fir.group_delay(np.zeros(100))  # even length has no integer delay

    def test_firdesign_multiband_unequal_endpoints(self):
        # Exercises the desired[0] != desired[-1] multiband branch, which the
        # other design tests (equal endpoints) never reach.
        numtaps = fir.estimate_taps(FS, 5)
        b = fir.firdesign(
            numtaps,
            [1000, 2000, 3000, 4000, 5000, 6000],
            [0, 1, 1, 0, 0, 1],
            sampling_freq=FS,
            spline_power=2,
        )
        assert len(b) == numtaps
        # Exact: taps are built as hstack((flip(half), center, half)).
        np.testing.assert_array_equal(b, b[::-1])  # still Type I symmetric

    def test_lowpass_is_linear_phase(self, lfp_lowpass):
        b = lfp_lowpass
        assert len(b) % 2 == 1
        # Exact, not merely close: assert_allclose would still admit rtol=1e-7.
        np.testing.assert_array_equal(b, b[::-1])  # Type I symmetry

    @pytest.mark.parametrize(
        "band_edges, desired, tw, passband_hz, stopband_hz",
        [
            ([400, 425], [1, 0], 25, [100], [2000]),  # lowpass
            ([300, 325], [0, 1], 25, [3000], [50]),  # highpass
            ([8, 10, 14, 16], [0, 1, 1, 0], 2, [12], [4, 40]),  # bandpass
            ([55, 58, 62, 65], [1, 0, 0, 1], 3, [20, 120], [60]),  # bandstop
        ],
    )
    def test_firdesign_symmetry_and_band_shaping(
        self, band_edges, desired, tw, passband_hz, stopband_hz
    ):
        # Every filter type must be Type I (symmetric) AND actually shape its
        # bands -- symmetry alone would pass for a filter that passes the wrong
        # band, and spyglass designs bandpass filters, not just the lowpass.
        numtaps = fir.estimate_taps(FS, tw)
        b = fir.firdesign(
            numtaps, band_edges, desired, sampling_freq=FS, spline_power=2
        )
        assert len(b) == numtaps
        np.testing.assert_array_equal(b, b[::-1])  # Type I linear phase (exact)
        w, h = freqz(b, worN=16384, fs=FS)
        mag = np.abs(h)
        for f in passband_hz:
            assert mag[np.argmin(np.abs(w - f))] > 0.99  # passband ~= 1
        for f in stopband_hz:
            assert mag[np.argmin(np.abs(w - f))] < 1e-2  # stopband rejected

    def test_firdesign_is_deterministic(self):
        numtaps = fir.estimate_taps(FS, 25)
        b1 = fir.firdesign(
            numtaps, [400, 425], [1, 0], sampling_freq=FS, spline_power=2
        )
        b2 = fir.firdesign(
            numtaps, [400, 425], [1, 0], sampling_freq=FS, spline_power=2
        )
        np.testing.assert_array_equal(b1, b2)

    @pytest.mark.parametrize(
        "numtaps, band_edges, desired, match",
        [
            (100, [400, 425], [1, 0], "odd value"),  # even numtaps
            (101, [400], [1], "even number of band edges"),  # odd # edges
            (
                101,
                [400, 425],
                [1],
                "equal number of band edges",
            ),  # len mismatch
            (101, [400, 425], [1, 2], "either 0 or 1"),  # desired not 0/1
            (
                101,
                [400, FS],
                [1, 0],
                "Last band edge must be less",
            ),  # >= nyquist
            (
                101,
                [425, 400],
                [1, 0],
                "monotonically increasing",
            ),  # non-monotonic
            (
                101,
                [0, 425],
                [1, 0],
                "First band edge must be greater",
            ),  # edge==0
            (
                101,
                [1, 2, 3, 4],
                [0, 0, 1, 0],
                "must be different",
            ),  # bad pairing
            (-1, [400, 425], [1, 0], "positive odd"),  # negative taps
            (101, [], [], "at least two band edges"),  # empty edges
        ],
    )
    def test_firdesign_validation(self, numtaps, band_edges, desired, match):
        # match= ensures each case fails for the INTENDED reason, so a future
        # reorder of the validation checks can't let one pass on the wrong branch.
        with pytest.raises(ValueError, match=match):
            fir.firdesign(
                numtaps, band_edges, desired, sampling_freq=FS, spline_power=2
            )

    @pytest.mark.parametrize("bad_tw", [0, -5])
    def test_estimate_taps_rejects_nonpositive_tw(self, bad_tw):
        # tw <= 0 previously gave a negative tap count / opaque OverflowError.
        with pytest.raises(ValueError, match="transition_width"):
            fir.estimate_taps(FS, bad_tw)

    def test_estimate_taps_rejects_nonpositive_fs(self):
        with pytest.raises(ValueError, match="sampling_freq"):
            fir.estimate_taps(0, 25)

    @pytest.mark.parametrize(
        "fs, tw, d1, d2",
        [
            (np.nan, 25, None, None),
            (np.inf, 25, None, None),
            (FS, np.nan, None, None),
            (FS, np.inf, None, None),
            (FS, 25, np.nan, 1e-6),
            (FS, 25, np.inf, 1e-6),
            (FS, 25, 1e-3, np.nan),
            (FS, 25, 1e-3, np.inf),
        ],
    )
    def test_estimate_taps_rejects_nonfinite_values(self, fs, tw, d1, d2):
        with pytest.raises(ValueError, match="finite"):
            fir.estimate_taps(
                fs, tw, passband_deviation=d1, stopband_deviation=d2
            )

    @pytest.mark.parametrize("d1, d2", [(0, 1e-6), (1e-3, 0), (-1e-3, 1e-6)])
    def test_estimate_taps_rejects_nonpositive_deviations(self, d1, d2):
        # d1/d2 <= 0 previously escaped as ZeroDivisionError / NaN-conversion.
        with pytest.raises(ValueError, match="deviations"):
            fir.estimate_taps(
                FS, 25, passband_deviation=d1, stopband_deviation=d2
            )

    def test_estimate_taps_rejects_too_loose_deviations(self):
        # 10 * d1 * d2 >= 1 -> log10 <= 0 -> a nonsensical numtaps < 1.
        with pytest.raises(ValueError, match="too loose"):
            fir.estimate_taps(
                FS, 25, passband_deviation=0.5, stopband_deviation=0.5
            )

    @pytest.mark.parametrize("bad_p", [0, -1])
    def test_firdesign_rejects_nonpositive_p(self, bad_p):
        # p <= 0 previously "succeeded", silently collapsing the spline
        # transition to a rectangular truncation (nan ** 0 == 1).
        with pytest.raises(ValueError, match="spline_power must be positive"):
            fir.firdesign(
                101, [400, 425], [1, 0], sampling_freq=FS, spline_power=bad_p
            )


# --------------------------------------------------------------------------- #
# filter_data_fir correctness vs np.convolve
# --------------------------------------------------------------------------- #
class TestFilterCorrectness:
    N = 40_000

    @pytest.mark.parametrize("time_axis", [0, 1])
    @pytest.mark.parametrize("decimation_factor", [None, 4, 15])
    def test_delay_compensated_filter(
        self, lfp_lowpass, rng, time_axis, decimation_factor
    ):
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        n_elec = 5
        shape = (self.N, n_elec) if time_axis == 0 else (n_elec, self.N)
        data = rng.standard_normal(shape)
        oib = [delay, delay + self.N]

        out = fir.filter_data_fir(
            data,
            b,
            axis=time_axis,
            output_index_bounds=oib,
            decimation_factor=decimation_factor,
        )
        ref = reference_filter(
            data,
            b,
            axis=time_axis,
            output_index_bounds=oib,
            decimation_factor=decimation_factor,
        )
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("time_axis", [0, 1])
    @pytest.mark.parametrize(
        "n_elec, N, electrodes, frm, to, decimation_factor",
        [
            (6, 40_000, [0, 2, 5], 1000, 30000, 5),  # single FFT block
            (4, 200_000, [0, 3], 1500, 190_000, 15),  # spans several blocks
        ],
        ids=["single_block", "multiblock"],
    )
    def test_restriction_input_bounds_decimation(
        self,
        lfp_lowpass,
        rng,
        time_axis,
        n_elec,
        N,
        electrodes,
        frm,
        to,
        decimation_factor,
    ):
        # Electrode restriction + input bounds + decimation. The multiblock case
        # (N far exceeds the ~59k FFT block stride) exercises the
        # first/middle/last-block branches and the decimation block_offset carry
        # that the single-block case never reaches -- the out-of-core machinery
        # the module exists for and that the pyfftw -> scipy.fft swap touched.
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        electrodes = np.array(electrodes)
        n = to - frm
        shape = (N, n_elec) if time_axis == 0 else (n_elec, N)
        data = rng.standard_normal(shape)
        idr = [None, None]
        idr[1 - time_axis] = np.s_[electrodes]

        out = fir.filter_data_fir(
            data,
            b,
            axis=time_axis,
            input_index_bounds=[frm, to],
            output_index_bounds=[delay, delay + n],
            decimation_factor=decimation_factor,
            input_dim_restrictions=idr,
        )
        ref = reference_filter(
            data,
            b,
            axis=time_axis,
            input_index_bounds=[frm, to],
            output_index_bounds=[delay, delay + n],
            decimation_factor=decimation_factor,
            electrodes=electrodes,
        )
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    def test_interval_shorter_than_filter(self, lfp_lowpass, rng):
        # A valid_times interval can be shorter than the filter (N < M); the
        # single-block overlap-save still matches the direct convolution.
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        short_N = 200  # << len(b) == 6401
        data = rng.standard_normal((3, short_N))
        oib = [delay, delay + short_N]
        out = fir.filter_data_fir(data, b, axis=1, output_index_bounds=oib)
        ref = reference_filter(data, b, axis=1, output_index_bounds=oib)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    def test_single_electrode_restriction(self, lfp_lowpass, rng):
        # A length-1 restriction (one electrode) must keep the axis, giving a
        # (1, n_time) result rather than collapsing the dimension.
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        N = 20_000
        data = rng.standard_normal((4, N))
        electrodes = np.array([2])
        out = fir.filter_data_fir(
            data,
            b,
            axis=1,
            output_index_bounds=[delay, delay + N],
            input_dim_restrictions=[np.s_[electrodes], None],
        )
        ref = reference_filter(
            data,
            b,
            axis=1,
            output_index_bounds=[delay, delay + N],
            electrodes=electrodes,
        )
        assert out.shape == (1, N) == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Out-of-core streaming (the spyglass access pattern)
# --------------------------------------------------------------------------- #
class TestStreaming:
    """Mirror FirFilterParameters.filter_data: several valid_times intervals
    filtered+decimated into ONE preallocated array via describe_output/offset.
    """

    @pytest.mark.parametrize("time_axis", [0, 1])
    @pytest.mark.parametrize(
        "N, valid_times",
        [
            # single-block intervals (each tot_length < FFT block stride)
            (60_000, [(500, 20000), (20000, 45000), (45000, 59000)]),
            # multi-block intervals, with lengths deliberately NOT divisible by
            # decimation_factor=10 so the per-interval decimation phase reset
            # is exercised too
            (250_000, [(500, 88007), (88007, 175003), (175003, 249000)]),
        ],
    )
    def test_multi_interval_stream_into_preallocated(
        self, lfp_lowpass, rng, time_axis, N, valid_times
    ):
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        decimation_factor = 10
        n_elec = 4
        shape = (N, n_elec) if time_axis == 0 else (n_elec, N)
        data = rng.standard_normal(shape)
        electrodes = [0, 1, 3]

        elec_axis = 1 - time_axis
        idr = [None, None]
        idr[elec_axis] = np.s_[electrodes]

        # size each interval's output, accumulate offsets (as spyglass does)
        out_shape = [0, 0]
        out_shape[elec_axis] = len(electrodes)
        offsets = [0]
        for frm, to in valid_times:
            s, _ = fir.describe_output(
                data,
                b,
                axis=time_axis,
                input_index_bounds=[frm, to],
                output_index_bounds=[delay, delay + to - frm],
                decimation_factor=decimation_factor,
                input_dim_restrictions=idr,
            )
            out_shape[time_axis] += s[time_axis]
            offsets.append(offsets[-1] + s[time_axis])

        outarray = np.empty(tuple(out_shape), dtype=data.dtype)
        for i, (frm, to) in enumerate(valid_times):
            ret = fir.filter_data_fir(
                data,
                b,
                axis=time_axis,
                input_index_bounds=[frm, to],
                output_index_bounds=[delay, delay + to - frm],
                decimation_factor=decimation_factor,
                input_dim_restrictions=idr,
                outarray=outarray,
                output_offset=offsets[i],
            )
            assert ret is outarray  # writes into the caller's (on-disk) array

        # reference: convolve each selected electrode's full lane ONCE, then
        # slice per interval (calling reference_filter per interval would redo
        # the full-lane convolution for every interval).
        lanes = np.moveaxis(data, time_axis, 1)[electrodes]  # (electrode, time)
        full = [np.convolve(lane, b, "full") for lane in lanes]
        pieces = [
            np.moveaxis(
                np.stack(
                    [
                        fc[frm + delay : to + delay : decimation_factor]
                        for fc in full
                    ]
                ),
                1,
                time_axis,
            )
            for frm, to in valid_times
        ]
        ref = np.concatenate(pieces, axis=time_axis)
        assert outarray.shape == ref.shape
        np.testing.assert_allclose(outarray, ref, atol=1e-9, rtol=1e-9)

    def test_h5py_on_disk_signal_and_output(self, lfp_lowpass, rng, tmp_path):
        # The module's reason to exist: filter an on-disk h5py signal into an
        # on-disk h5py output. Exercises real h5py read/write semantics (fancy
        # index must be increasing-order, slice assignment) that the in-memory
        # ndarray tests do not.
        h5py = pytest.importorskip("h5py")
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        n_elec, N, decimation_factor = 5, 40_000, 8
        arr = rng.standard_normal((n_elec, N))
        electrodes = np.array([0, 2, 4])  # h5py requires increasing order
        idr = [np.s_[electrodes], None]
        oib = [delay, delay + N]

        path = str(tmp_path / "sig.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=arr)
        with h5py.File(path, "r+") as f:
            sig = f["data"]
            shape, dtype = fir.describe_output(
                sig,
                b,
                axis=1,
                output_index_bounds=oib,
                decimation_factor=decimation_factor,
                input_dim_restrictions=idr,
            )
            out = f.create_dataset("out", shape=shape, dtype=dtype)
            fir.filter_data_fir(
                sig,
                b,
                axis=1,
                output_index_bounds=oib,
                decimation_factor=decimation_factor,
                input_dim_restrictions=idr,
                outarray=out,
            )
            result = out[...]

        ref = reference_filter(
            arr,
            b,
            axis=1,
            output_index_bounds=oib,
            decimation_factor=decimation_factor,
            electrodes=electrodes,
        )
        assert result.shape == ref.shape
        np.testing.assert_allclose(result, ref, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------- #
# describe_output contract
# --------------------------------------------------------------------------- #
class TestDescribeOutput:
    @pytest.mark.parametrize("time_axis", [0, 1])
    @pytest.mark.parametrize("decimation_factor", [None, 7])
    def test_describe_output_matches_actual_output(
        self, lfp_lowpass, rng, time_axis, decimation_factor
    ):
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        N = 25_000
        shape = (N, 3) if time_axis == 0 else (3, N)
        data = rng.standard_normal(shape)
        oib = [delay, delay + N]

        shape_pred, dtype_pred = fir.describe_output(
            data,
            b,
            axis=time_axis,
            output_index_bounds=oib,
            decimation_factor=decimation_factor,
        )
        out = fir.filter_data_fir(
            data,
            b,
            axis=time_axis,
            output_index_bounds=oib,
            decimation_factor=decimation_factor,
        )
        assert tuple(shape_pred) == out.shape
        assert np.dtype(dtype_pred) == out.dtype == np.float64


# --------------------------------------------------------------------------- #
# _osconvolve modes (reachable only via the private function; spyglass uses full)
# --------------------------------------------------------------------------- #
class TestDescribeOutputParity:
    """describe_output must accept exactly what filter_data_fir accepts."""

    N = 5000

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"nfft": 4.0}, "nfft"),
            ({"decimation_factor": 0}, "decimation_factor"),
            ({"input_index_bounds": [10, 5]}, "increasing"),
            ({"output_index_bounds": [5, 1]}, "increasing"),
            # one entry per dimension; the signal here is 1-D
            ({"input_dim_restrictions": [None, None]}, "Expected 1"),
        ],
    )
    def test_rejects_what_filtering_rejects(
        self, lfp_lowpass, rng, kwargs, match
    ):
        # The sizing and filtering passes share _plan_osconvolve, so anything
        # one rejects the other must reject. Before the split these were one
        # function behind a describe_dims flag, and the flag returned early --
        # so a bad output_offset was validated on one path and silently ignored
        # on the other. Pin the parity that replaced it.
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        with pytest.raises((ValueError, IndexError), match=match):
            fir.describe_output(x, b, axis=0, **kwargs)
        with pytest.raises((ValueError, IndexError), match=match):
            fir.filter_data_fir(x, b, axis=0, **kwargs)

    def test_signature_omits_arguments_that_cannot_affect_the_answer(self):
        # threads/outarray/output_offset do not change the shape or dtype, so
        # describe_output does not take them. Passing one is a caller error and
        # must say so, rather than being accepted and ignored as the old
        # describe_dims=True path did.
        import inspect

        params = inspect.signature(fir.describe_output).parameters
        for name in ("threads", "outarray", "output_offset", "describe_dims"):
            assert name not in params
        with pytest.raises(TypeError):
            fir.describe_output(
                np.zeros(100), np.ones(11) / 11, axis=0, output_offset=-99
            )


class TestConvolveModes:
    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_mode_matches_numpy(self, lfp_lowpass, rng, mode):
        b = lfp_lowpass
        x = rng.standard_normal(
            20_000
        )  # N > M so 'same'/'valid' are well-defined
        out = fir._osconvolve(x, b, mode=mode)
        ref = np.convolve(x, b, mode=mode)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Edge cases hardened relative to upstream, plus validation
# --------------------------------------------------------------------------- #
class TestEdgeCasesAndValidation:
    N = 30_000

    def test_full_length_input_bounds_accepted(self, lfp_lowpass, rng):
        # upstream rejected input_index_bounds=[0, N]; here it must equal no-bounds
        b = lfp_lowpass
        data = rng.standard_normal((3, self.N))
        with_bounds = fir.filter_data_fir(
            data, b, axis=1, input_index_bounds=[0, self.N]
        )
        no_bounds = fir.filter_data_fir(data, b, axis=1)
        np.testing.assert_array_equal(with_bounds, no_bounds)

    def test_full_output_bounds_accepted(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        tot = self.N + len(b) - 1
        out = fir.filter_data_fir(x, b, axis=0, output_index_bounds=[0, tot])
        np.testing.assert_allclose(
            out, np.convolve(x, b, "full"), atol=1e-9, rtol=1e-9
        )

    def test_complex_input_runs_and_matches(self, lfp_lowpass, rng):
        # upstream raised UnboundLocalError for complex input; port handles it
        b = lfp_lowpass
        x = rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N)
        out = fir._osconvolve(x, b, mode="full")
        assert np.iscomplexobj(out)
        np.testing.assert_allclose(
            out, np.convolve(x, b, "full"), atol=1e-9, rtol=1e-9
        )

    def test_complex_input_multiblock_decimation_restriction(
        self, lfp_lowpass, rng
    ):
        # The complex fallback path shares the block/decimation/restriction
        # machinery with the real (rfft) path, but the suite otherwise only
        # exercises it single-block/1-D. Cover the full combination.
        b = lfp_lowpass
        delay = (len(b) - 1) // 2
        big_N = 200_000  # spans several FFT blocks
        data = rng.standard_normal((4, big_N)) + 1j * rng.standard_normal(
            (4, big_N)
        )
        electrodes = np.array([0, 3])
        out = fir.filter_data_fir(
            data,
            b,
            axis=1,
            output_index_bounds=[delay, delay + big_N],
            decimation_factor=15,
            input_dim_restrictions=[np.s_[electrodes], None],
        )
        full = [
            np.convolve(data[e], b, "full")[delay : delay + big_N : 15]
            for e in electrodes
        ]
        ref = np.stack(full, axis=0)
        assert out.shape == ref.shape and np.iscomplexobj(out)
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("bad_factor", [0, -3, 4.0])
    def test_decimation_below_one_or_noninteger_fails_closed(
        self, lfp_lowpass, rng, bad_factor
    ):
        # decimation_factor must be an integer >= 1 (consistent with nfft and
        # output_offset).
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        with pytest.raises(ValueError, match="decimation_factor"):
            fir.filter_data_fir(x, b, axis=0, decimation_factor=bad_factor)

    def test_decimation_one_is_noop(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        np.testing.assert_array_equal(
            fir.filter_data_fir(x, b, axis=0, decimation_factor=1),
            fir.filter_data_fir(x, b, axis=0),
        )

    def test_input_bounds_out_of_range_raise(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        with pytest.raises(IndexError, match="out of range"):
            fir.filter_data_fir(
                x, b, axis=0, input_index_bounds=[0, self.N + 1]
            )
        with pytest.raises(IndexError, match="out of range"):
            fir.filter_data_fir(x, b, axis=0, input_index_bounds=[-1, self.N])

    def test_input_bounds_not_increasing_raise(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        with pytest.raises(ValueError, match="not strictly increasing"):
            fir.filter_data_fir(x, b, axis=0, input_index_bounds=[100, 100])

    def test_output_bounds_out_of_range_raise(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        tot = self.N + len(b) - 1
        with pytest.raises(ValueError, match="out of range"):
            fir.filter_data_fir(x, b, axis=0, output_index_bounds=[0, tot + 1])

    def test_bad_mode_and_kernel_shape_raise(self, lfp_lowpass, rng):
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        with pytest.raises(ValueError, match="invalid value"):
            fir._osconvolve(x, b, mode="bogus")
        with pytest.raises(ValueError, match="Kernel must be 1D"):
            fir._osconvolve(x, np.ones((3, 3)))  # kernel must be 1D
        with pytest.raises(ValueError, match="at least one coefficient"):
            fir._osconvolve(x, np.array([]))
        with pytest.raises(ValueError, match="at least one thread"):
            fir.filter_data_fir(x, b, axis=0, threads=0)  # need >= 1 thread

    def test_negative_output_offset_rejected(self, lfp_lowpass, rng):
        # A negative offset silently misplaces the write via NumPy negative
        # indexing; it must fail loudly instead.
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        out = np.empty(self.N + len(b) - 1, dtype=np.float64)
        with pytest.raises(ValueError, match="output_offset"):
            fir.filter_data_fir(x, b, axis=0, outarray=out, output_offset=-10)

    def test_output_offset_past_end_rejected(self, lfp_lowpass, rng):
        # An offset that pushes the write past the end of a supplied outarray
        # must raise clearly rather than fail with an opaque broadcast error.
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        delay = (len(b) - 1) // 2
        out = np.empty(self.N, dtype=np.float64)
        with pytest.raises(ValueError, match="exceeds outarray"):
            fir.filter_data_fir(
                x,
                b,
                axis=0,
                output_index_bounds=[delay, delay + self.N],
                outarray=out,
                output_offset=100,
            )

    def test_noninteger_output_offset_rejected(self, lfp_lowpass, rng):
        # A float offset (within range) previously slipped past validation and
        # failed later with "slice indices must be integers".
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        out = np.empty(self.N + len(b) - 1, dtype=np.float64)
        with pytest.raises(ValueError, match="output_offset.*integer"):
            fir.filter_data_fir(x, b, axis=0, outarray=out, output_offset=1.0)

    def test_noninteger_nfft_rejected(self, lfp_lowpass, rng):
        # A non-integer nfft must be rejected in BOTH the sizing and filtering
        # passes -- previously the sizing pass accepted it and the real call
        # died with a raw TypeError. describe_output shares _plan_osconvolve
        # with filter_data_fir precisely so the two cannot drift.
        b = lfp_lowpass
        x = rng.standard_normal(self.N)
        big_nfft = float(2 * len(b))  # > M, but a float
        with pytest.raises(ValueError, match="nfft.*integer"):
            fir.describe_output(x, b, axis=0, nfft=big_nfft)
        with pytest.raises(ValueError, match="nfft.*integer"):
            fir.filter_data_fir(x, b, axis=0, nfft=big_nfft)

    def test_complex_output_into_real_outarray_rejected(self):
        # A complex result must not be silently written into a real outarray;
        # the dtype guard rejects it (checks the outarray's dtype, no data read).
        x = np.arange(60.0) + 1j * np.arange(60.0)  # complex -> complex output
        b = np.array([1.0, 2.0, 1.0])
        real_out = np.empty(60 + len(b) - 1, dtype=np.float64)
        with pytest.raises(TypeError, match="complex"):
            fir._osconvolve(x, b, mode="full", outarray=real_out)

    def test_integer_outarray_truncates_toward_zero(self, rng):
        # Production writes into an int16 ElectricalSeries: filter_data_nwb
        # allocates the output with the RAW dtype, so the float64 filter result
        # is cast into it. Every other numeric test here is float64-in/out, so
        # pin the cast semantics: numpy assignment truncates toward zero, it
        # does NOT round. Anyone "fixing" this to np.rint would shift every
        # stored LFP sample by up to half an ADC count.
        x = (rng.standard_normal(self.N) * 100).astype(np.int16)
        b = np.ones(51) / 51
        float_out = fir.filter_data_fir(x, b, axis=0)
        int_out = np.empty(self.N + len(b) - 1, dtype=np.int16)
        fir.filter_data_fir(x, b, axis=0, outarray=int_out)
        np.testing.assert_array_equal(int_out, float_out.astype(np.int16))
        # Compared against np.rint the two differ, which is the point.
        assert not np.array_equal(int_out, np.rint(float_out).astype(np.int16))

    def test_integer_outarray_amplifies_roundoff_to_one_count(self, rng):
        # Truncation is discontinuous at every integer, so the ~1e-15 difference
        # between this FFT convolution and a direct np.convolve lands on either
        # side of a boundary for a small fraction of samples -- turning
        # round-off into a full ADC count. The float results agree to 1e-9; the
        # truncated ones do not agree exactly. Documented so a future reader
        # does not mistake this for a filtering bug.
        x = (rng.standard_normal(self.N) * 100).astype(np.int16)
        b = np.ones(51) / 51
        float_out = fir.filter_data_fir(x, b, axis=0)
        direct = np.convolve(x.astype(np.float64), b, "full")
        np.testing.assert_allclose(float_out, direct, atol=1e-9, rtol=1e-9)
        diff = np.abs(
            float_out.astype(np.int16).astype(np.int32)
            - direct.astype(np.int16).astype(np.int32)
        )
        # Pin both bounds: it really does happen (so the hazard stays
        # documented if the FFT backend changes), and it never exceeds one
        # count (so a genuine filtering error could not hide behind it).
        assert diff.max() == 1
        assert 0 < np.count_nonzero(diff) < 0.05 * diff.size

    def test_kernel_accepts_python_list(self, rng):
        # b is documented array-like and converted internally, so a plain list
        # must work (spyglass passes a numpy array, but the contract is broader).
        x = rng.standard_normal(2000)
        out = fir.filter_data_fir(x, [1.0, 2.0, 1.0], axis=0)
        np.testing.assert_allclose(
            out, np.convolve(x, [1.0, 2.0, 1.0], "full"), atol=1e-9, rtol=1e-9
        )

    def test_slice_restriction_rejected(self, lfp_lowpass, rng):
        # input_dim_restrictions entries must be integer-index arrays, not slices
        b = lfp_lowpass
        data = rng.standard_normal((6, self.N))
        with pytest.raises(ValueError, match="1-D array of integer indices"):
            fir.filter_data_fir(
                data, b, axis=1, input_dim_restrictions=[np.s_[0:2], None]
            )

    @pytest.mark.parametrize("electrodes", [np.array([2, 0]), np.array([1, 1])])
    def test_in_memory_restriction_allows_any_order(
        self, lfp_lowpass, rng, electrodes
    ):
        # NumPy fancy-indexes an in-memory array in any order, and the public
        # FirFilterParameters.filter_data documents no ordering requirement for
        # its `electrodes` argument, so out-of-order / repeated selections must
        # keep working -- and must come back in the REQUESTED order.
        b = lfp_lowpass
        data = rng.standard_normal((6, self.N))
        out = fir.filter_data_fir(
            data, b, axis=1, input_dim_restrictions=[electrodes, None]
        )
        ref = reference_filter(data, b, axis=1, electrodes=electrodes)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("electrodes", [np.array([2, 0]), np.array([1, 1])])
    def test_lazy_restriction_allows_any_order(
        self, lfp_lowpass, rng, electrodes, tmp_path
    ):
        # h5py cannot fancy-index out of order or with duplicates, so an
        # unsorted selection is read as sorted unique indices and gathered back.
        # filter_data_nwb maps sorted electrode IDs onto ElectricalSeries
        # columns, which can be unsorted, and it filters straight from h5py --
        # so on-disk signals have to accept the same orders in-memory ones do.
        h5py = pytest.importorskip("h5py")
        b = lfp_lowpass
        data = rng.standard_normal((6, self.N))
        path = str(tmp_path / "sig.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
        with h5py.File(path, "r") as f:
            shape, _ = fir.describe_output(
                f["data"],
                b,
                axis=1,
                input_dim_restrictions=[electrodes, None],
            )
            out = fir.filter_data_fir(
                f["data"],
                b,
                axis=1,
                input_dim_restrictions=[electrodes, None],
            )
        ref = reference_filter(data, b, axis=1, electrodes=electrodes)
        assert shape == ref.shape
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    def test_restriction_indices_out_of_range_raise(self, lfp_lowpass, rng):
        b = lfp_lowpass
        data = rng.standard_normal((6, self.N))
        with pytest.raises(IndexError, match="out of range"):
            fir.filter_data_fir(
                data,
                b,
                axis=1,
                input_dim_restrictions=[np.array([0, 6]), None],
            )

    def test_multiple_restricted_axes_rejected(self, lfp_lowpass, rng):
        # Only one non-filtered axis may be restricted (NumPy paired advanced
        # indexing would otherwise not do the intended Cartesian selection).
        b = lfp_lowpass
        data = rng.standard_normal((3, 2, self.N))
        idr = [np.array([0, 1]), np.array([0]), None]  # two restricted axes
        with pytest.raises(ValueError, match="at most one non-filtered axis"):
            fir.filter_data_fir(data, b, axis=2, input_dim_restrictions=idr)

    def test_boundary_output_no_read_past_data_end(self):
        # When the (exclusive) output stop lands exactly on an overlap-save block
        # boundary, the module must NOT process an empty trailing block that
        # reads one block past the needed data -- wasteful, and unsafe for a
        # strict lazy/on-disk signal. nfft=8 -> block stride 6; a length-10 full
        # convolution has stop=12=2*stride, i.e. exactly on a boundary.
        reads = []

        class _RecordReads(np.ndarray):
            def __getitem__(self, key):
                k = key[0] if isinstance(key, tuple) else key
                if isinstance(k, slice) and k.start is not None:
                    reads.append(int(k.start))
                return super().__getitem__(key)

        x = np.arange(10.0).view(_RecordReads)
        b = np.array([1.0, 2.0, 1.0])
        out = fir._osconvolve(x, b, mode="full", nfft=8)
        assert not [
            s for s in reads if s >= len(x)
        ]  # no read at/beyond data end
        np.testing.assert_allclose(
            out, np.convolve(np.arange(10.0), b, "full"), atol=1e-9
        )

    def test_overlap_read_error_propagates_not_zero_filled(self):
        # An error reading the M-1 overlap segment of a block (e.g. a real h5py
        # I/O failure on an on-disk signal) must propagate, NOT be swallowed and
        # silently zero-filled -- which would corrupt the filtered output with no
        # error. This forces multiple blocks (nfft=8, L=6) so a middle-block
        # overlap read at signal[4:6] occurs, and makes that read fail.
        class _RaiseOnOverlapRead(np.ndarray):
            def __getitem__(self, key):
                k = key[0] if isinstance(key, tuple) else key
                if isinstance(k, slice) and k.start == 4 and k.stop == 6:
                    raise RuntimeError("overlap read failed")
                return super().__getitem__(key)

        x = np.arange(12.0).view(_RaiseOnOverlapRead)
        b = np.array([1.0, 2.0, 1.0])
        with pytest.raises(RuntimeError, match="overlap read failed"):
            fir._osconvolve(x, b, mode="full", nfft=8)

    def test_h5py_interval_starting_at_sample_zero(self, rng, tmp_path):
        # An interval starting at sample 0 has nothing before it, so the first
        # block's M-1 overlap window is empty. h5py rejects an empty slice
        # combined with a fancy index of >= 16 elements ("Dataspaces don't have
        # hyperslab selections"), so that read must not be issued at all -- the
        # block buffer is already zeroed, which is the correct fill. Upstream
        # hid this behind a blanket except; failing loudly on genuine read
        # errors means the empty case has to be skipped explicitly.
        h5py = pytest.importorskip("h5py")
        n_time, n_elec = 4000, 32  # >= 16 electrodes triggers the h5py path
        arr = rng.standard_normal((n_time, n_elec))
        b = np.ones(101) / 101
        delay = (len(b) - 1) // 2
        path = str(tmp_path / "es.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=arr)
        electrodes = np.arange(n_elec)
        with h5py.File(path, "r") as f:
            out = fir.filter_data_fir(
                f["data"],
                b,
                axis=0,
                input_index_bounds=[0, 3000],
                output_index_bounds=[delay, delay + 3000],
                input_dim_restrictions=[None, electrodes],
            )
        ref = reference_filter(
            arr,
            b,
            axis=0,
            input_index_bounds=[0, 3000],
            output_index_bounds=[delay, delay + 3000],
            electrodes=electrodes,
        )
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    def test_h5py_tail_block_starting_past_data_end(self, rng, tmp_path):
        # The main block read signal[start:stop] is empty whenever a tail block
        # begins at or past the end of the data -- which happens when the output
        # stop lands just past a block boundary. Same h5py failure as the empty
        # overlap read (>= 16 electrodes), so it needs the same guard. Sized so
        # the last block starts exactly at the end of the data: with nfft=512
        # and a 101-tap kernel the block stride is 412, and n=412 puts the
        # second block's read at signal[412:824] on a 412-sample signal.
        h5py = pytest.importorskip("h5py")
        n_time, n_elec = 412, 32  # >= 16 electrodes triggers the h5py path
        b = np.ones(101) / 101
        delay = (len(b) - 1) // 2
        arr = rng.standard_normal((n_time, n_elec))
        path = str(tmp_path / "tail.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=arr)
        electrodes = np.arange(n_elec)
        with h5py.File(path, "r") as f:
            out = fir.filter_data_fir(
                f["data"],
                b,
                axis=0,
                nfft=512,
                input_index_bounds=[0, n_time],
                output_index_bounds=[delay, delay + n_time],
                input_dim_restrictions=[None, electrodes],
            )
        ref = reference_filter(
            arr,
            b,
            axis=0,
            input_index_bounds=[0, n_time],
            output_index_bounds=[delay, delay + n_time],
            electrodes=electrodes,
        )
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize(
        "signal, kernel, nfft",
        [
            # A signal shorter than the M-1 overlap combined with an nfft tight
            # enough to need several blocks: every block after the first reads an
            # overlap window that starts before the data AND ends after it, so
            # the read is clipped at BOTH ends.
            ([2.0], [1.0, 2.0, 3.0, 4.0, 5.0], 5),
            ([2.0, -1.0], [1.0, 2.0, 3.0, 4.0, 5.0], 6),
            ([1.0, 2.0, 3.0], [1.0, -1.0, 2.0, 0.5, 3.0, 1.0, -2.0], 8),
        ],
    )
    def test_tight_nfft_short_signal_matches_numpy(self, signal, kernel, nfft):
        # nfft >= kernel length is the documented contract, so every accepted
        # nfft must give the true convolution. Placing the clipped overlap from
        # its LENGTH (rather than from where the read actually starts) used to
        # shift these blocks and return e.g. [2, 4, 4, 4, 10] for
        # [2, 4, 6, 8, 10].
        x, b = np.asarray(signal), np.asarray(kernel)
        out = fir.filter_data_fir(x, b, axis=0, nfft=nfft, threads=1)
        np.testing.assert_allclose(
            out, np.convolve(x, b, "full"), atol=1e-9, rtol=1e-9
        )

    def test_tight_nfft_oracle_sweep(self):
        # Randomized sweep over the whole accepted (signal length, kernel
        # length, nfft) region, against the np.convolve oracle. Fixed seed ->
        # reproducible.
        sweep_rng = np.random.default_rng(20240102)
        for _ in range(400):
            n = int(sweep_rng.integers(1, 12))
            numtaps = int(sweep_rng.integers(1, 5)) * 2 + 1
            nfft = int(sweep_rng.integers(numtaps, 3 * numtaps + 2))
            x = sweep_rng.standard_normal(n)
            b = sweep_rng.standard_normal(numtaps)
            out = fir.filter_data_fir(x, b, axis=0, nfft=nfft, threads=1)
            np.testing.assert_allclose(
                out,
                np.convolve(x, b, "full"),
                atol=1e-9,
                rtol=1e-9,
                err_msg=f"n={n} numtaps={numtaps} nfft={nfft}",
            )
