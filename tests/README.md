# PyTests

[![codecov](https://codecov.io/gh/LorenFrankLab/spyglass/graph/badge.svg?token=QEJIIFN2S8)](https://codecov.io/gh/LorenFrankLab/spyglass)

## Environment

To facilitate headless testing of various Qt-based tools as well as Tensorflow,
`pyproject.toml` includes some environment variables associated with the
display. These are...

- `QT_QPA_PLATFORM`: Set to `offscreen` to prevent the need for a display.
- `TF_ENABLE_ONEDNN_OPTS`: Set to `1` to enable Tensorflow optimizations.
- `TF_CPP_MIN_LOG_LEVEL`: Set to `2` to suppress Tensorflow warnings.

<!-- - `DISPLAY`: Set to `:0` to prevent the need for a display. -->

## Options

This directory is contains files for testing the code. Simply by running
`pytest` from the root directory, all tests will be run with default parameters
specified in `pyproject.toml`. Notable optional parameters include...

- Coverage items. The coverage report indicates what percentage of the code was
    included in tests.

    - `--cov=spyglass`: Which package should be described in the coverage report
    - `--cov-report term-missing`: Include lines of items missing in coverage

- Verbosity.

    - `-v`: List individual tests, report pass/fail
    - `--quiet-spy`: Default False. When True, print and other logging statements
        from Spyglass are silenced.

- Data and database.

    - `--base_dir`: Default `./tests/test_data/`. Where to store downloaded and
        created files.
    - `--no-teardown`: Default False. When True, docker database tables are
        preserved on exit. Set to false to inspect output items after testing.
    - `--no-docker`: Default False, launch Docker container from python. When
        True, no server is started and tests attempt to connect to existing
        container. For github actions, `--no-docker` is set to configure the
        container class as null.
    - `--no-dlc`: Default False. When True, skip data downloads for and tests of
        features that require DeepLabCut.

- Incremental running.

    - `-s`: No capture. By including `from IPython import embed; embed()` in a
        test, and using this flag, you can open an IPython environment from within
        a test
    - `-v`: Verbose. List individual tests, report pass/fail.
    - `--sw`: Stepwise. Continue from previously failed test when starting again.
    - `--pdb`: Enter debug mode if a test fails.
    - `tests/test_file.py -k test_name`: To run just a set of tests, specify the
        file name at the end of the command. To run a single test, further specify
        `-k` with the test name.

When customizing parameters, comment out the `addopts` line in `pyproject.toml`.

```console
pytest -m current --quiet-spy --no-teardown tests/test_file.py -k test_name
```
