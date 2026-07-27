# Third-party licenses

Licenses for third-party code vendored into Spyglass. Spyglass itself is
distributed under the MIT license (see `LICENSE` at the repository root); the
files listed here cover code that carries its own terms.

## ghostipy-Apache-2.0.txt

Covers `spyglass/common/_fir_filter.py`, which is derived from
[ghostipy](https://github.com/kemerelab/ghostipy) (Kemere Lab, Rice University;
author Joshua Chu), licensed under the Apache License, Version 2.0. The file is
a verbatim copy of ghostipy's `LICENSE`, included to satisfy Apache-2.0 §4(a).

Only the FIR design and out-of-core overlap-save filtering that Spyglass uses
were vendored (`estimate_taps`, `firdesign`, `group_delay`, and `osconvolve` as
`filter_data_fir`). The code has been modified: the `pyfftw` FFT backend was
replaced with `scipy.fft`, parameters were renamed and type-annotated, argument
validation was tightened, and an overlap-placement bug was fixed. Those changes
are enumerated under "Intentional divergences from upstream" in the module
docstring of `_fir_filter.py`, per Apache-2.0 §4(b).
