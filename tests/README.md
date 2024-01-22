# PyTests

This directory is contains files for testing the code. Simply by running
`pytest` from the root directory, all tests will be run with default parameters
specified in `pyproject.toml`. Notable optional parameters include...

- Coverage items. The coverage report indicates what percentage of the code was
    included in tests.

    - `--cov=spyglatss`: Which package should be described in the coverage report
    - `--cov-report term-missing`: Include lines of items missing in coverage

- Verbosity.

    - `-v`: List individual tests, report pass/fail
    - `--quiet-spy`: Default False. When True, print and other logging statements
        from Spyglass are silenced.

- Data and database.

    - `--no-server`: Default False, launch Docker container from python. When
        True, no server is started and tests attempt to connect to existing
        container.
    - `--no-teardown`: Default False. When True, docker database tables are
        preserved on exit. Set to false to inspect output items after testing.
    - `--my-datadir ./rel-path/`: Default `./tests/test_data/`. Where to store
        created files.

- Incremental running.

    - `-m`: Run tests with the
        [given marker](https://docs.pytest.org/en/6.2.x/usage.html#specifying-tests-selecting-tests)
        (e.g., `pytest -m current`).
    - `--sw`: Stepwise. Continue from previously failed test when starting again.
    - `-s`: No capture. By including `from IPython import embed; embed()` in a
        test, and using this flag, you can open an IPython environment from within
        a test
    - `--pdb`: Enter debug mode if a test fails.
    - `tests/test_file.py -k test_name`: To run just a set of tests, specify the
        file name at the end of the command. To run a single test, further specify
        `-k` with the test name.

When customizing parameters, comment out the `addopts` line in `pyproject.toml`.

```console
pytest -m current --quiet-spy --no-teardown tests/test_file.py -k test_name
```
