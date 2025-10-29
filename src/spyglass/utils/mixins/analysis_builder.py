"""AnalysisFileBuilder: Safe analysis file creation with lifecycle enforcement.

This module provides AnalysisFileBuilder, a context manager that enforces
the CREATE → POPULATE → REGISTER lifecycle for analysis NWB files, preventing
common user errors like forgetting registration or modifying registered files.
"""

from typing import Optional, Tuple

import pynwb

from spyglass.utils.logging import logger


class AnalysisFileBuilder:
    """Context manager for safe analysis file creation.

    Enforces CREATE → POPULATE → REGISTER lifecycle with automatic state
    tracking and fail-fast error messages.

    The builder prevents common errors:
    - Forgetting to call add() (auto-registration on __exit__)
    - Calling helpers after registration (state checks with clear errors)
    - Opening files in write mode after registration (prevented)
    - Losing track of file state (explicit state machine)
    - Creating orphaned files on exception (logged for cleanup)

    Parameters
    ----------
    analysis_table : AnalysisMixin instance
        The table to create/register file in (master or custom)
    nwb_file_name : str
        Parent NWB file name

    Attributes
    ----------
    analysis_file_name : str
        Name of created analysis file (set after __enter__)
    nwb_file_name : str
        Parent NWB file name

    Examples
    --------
    Basic usage with helper methods:
    >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
    ...     builder.add_nwb_object(my_data, "results")
    ...     file = builder.analysis_file_name
    # File automatically registered on exit!

    Multiple helper calls:
    >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
    ...     builder.add_nwb_object(position_data, "position")
    ...     builder.add_nwb_object(velocity_data, "velocity")
    ...     file = builder.analysis_file_name

    Direct NWB I/O for complex operations:
    >>> with AnalysisNwbfile().build(nwb_file_name) as builder:
    ...     with builder.open_for_write() as io:
    ...         nwbf = io.read()
    ...         nwbf.add_unit(spike_times=times, id=unit_id)
    ...         io.write(nwbf)
    ...     file = builder.analysis_file_name

    See Also
    --------
    AnalysisMixin.build : Factory method that creates this builder
    """

    def __init__(self, analysis_table, nwb_file_name: str):
        """Initialize builder with table and parent file name.

        Parameters
        ----------
        analysis_table : AnalysisMixin instance
            The table to create/register file in
        nwb_file_name : str
            Parent NWB file name
        """
        self._table = analysis_table
        self.nwb_file_name = nwb_file_name
        self.analysis_file_name = None
        self._state = "INIT"
        self._exception_occurred = False

    def __enter__(self):
        """Create analysis file (CREATE phase).

        Returns
        -------
        self : AnalysisFileBuilder
            Builder instance with analysis_file_name set

        Raises
        ------
        Exception
            Any exception from underlying file creation
        """
        self.analysis_file_name = self._table.create(self.nwb_file_name)
        self._state = "CREATED"
        logger.debug(
            f"Created analysis file: {self.analysis_file_name} "
            f"(parent: {self.nwb_file_name})"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Register file automatically if no exception (REGISTER phase).

        Parameters
        ----------
        exc_type : type or None
            Exception type if exception occurred
        exc_val : Exception or None
            Exception instance if exception occurred
        exc_tb : traceback or None
            Exception traceback if exception occurred

        Returns
        -------
        False
            Never suppresses exceptions
        """
        self._exception_occurred = exc_type is not None

        if self._exception_occurred:
            # Log failed file for cleanup
            logger.warning(
                f"Analysis file '{self.analysis_file_name}' created but not "
                f"registered due to exception: {exc_type.__name__}. "
                f"File will be detected and cleaned up by "
                f"AnalysisNwbfile.cleanup()"
            )
            return False  # Don't suppress exception

        # Always auto-register on successful exit
        try:
            self.register()
        except Exception as e:
            logger.error(
                f"Failed to register analysis file "
                f"'{self.analysis_file_name}': {e}"
            )
            raise  # Re-raise registration error

        return False  # Never suppress exceptions

    def _ensure_created(self, method_name: str):
        """Ensure file is in CREATED state (not INIT or REGISTERED).

        This helper validates that population methods (add_nwb_object, add_units,
        open_for_write) are called at the correct time in the lifecycle.

        Parameters
        ----------
        method_name : str
            Name of method being called (for error message)

        Raises
        ------
        ValueError
            If called before __enter__ (INIT state) or after registration (REGISTERED state)
        """
        if self._state == "INIT":
            raise ValueError(
                f"Cannot call {method_name}() before entering context manager. "
                f"Use: 'with table.build(...) as builder: "
                f"builder.{method_name}(...)'"
            )
        elif self._state == "REGISTERED":
            raise ValueError(
                f"Cannot call {method_name}() in state: REGISTERED. "
                f"This method must be called AFTER create() but BEFORE "
                f"register()/add(). The file '{self.analysis_file_name}' is "
                f"already registered and its checksum is locked. "
                f"To modify the file, you must create a new one using "
                f"AnalysisNwbfile.copy()."
            )
        # CREATED state is valid, no error

    def register(self):
        """Register the analysis file in the database.

        This is called automatically by __exit__. Can also be called manually
        if needed (idempotent - does nothing if already registered).

        Raises
        ------
        ValueError
            If file is not yet created
        """
        if self._state == "REGISTERED":
            return  # Already registered, do nothing (idempotent)

        if self._state != "CREATED":
            raise ValueError(
                f"Cannot register file in state: {self._state}. "
                f"File must be created first (use context manager)."
            )

        self._table.add(self.nwb_file_name, self.analysis_file_name)
        self._state = "REGISTERED"
        logger.debug(
            f"Registered analysis file: {self.analysis_file_name} "
            f"(parent: {self.nwb_file_name})"
        )

    def add_nwb_object(
        self, nwb_object, table_name: str = "pandas_table"
    ) -> str:
        """Add NWB object to analysis file (POPULATE phase).

        Parameters
        ----------
        nwb_object : pynwb object, DataFrame, or ndarray
            Object to add to file scratch space
        table_name : str, default="pandas_table"
            Name for object in scratch space

        Returns
        -------
        object_id : str
            NWB object ID for retrieval

        Raises
        ------
        ValueError
            If called before create or after registration
        """
        self._ensure_created("add_nwb_object")
        return self._table.add_nwb_object(
            self.analysis_file_name, nwb_object, table_name
        )

    def add_units(
        self,
        units: dict,
        units_valid_times: dict,
        units_sort_interval: dict,
        metrics: Optional[dict] = None,
    ) -> Tuple[str, str]:
        """Add spike sorting units to analysis file (POPULATE phase).

        Parameters
        ----------
        units : dict
            Unit ID → spike times mapping
        units_valid_times : dict
            Unit ID → valid time intervals mapping
        units_sort_interval : dict
            Unit ID → sorting intervals mapping
        metrics : dict, optional
            Unit ID → metrics mapping

        Returns
        -------
        units_object_id : str
            Object ID for units table
        waveforms_object_id : str
            Object ID for waveforms

        Raises
        ------
        ValueError
            If called before create or after registration
        """
        self._ensure_created("add_units")
        return self._table.add_units(
            self.analysis_file_name,
            units,
            units_valid_times,
            units_sort_interval,
            metrics,
        )

    def add_units_metrics(self, metrics: dict) -> str:
        """Add unit metrics to analysis file (POPULATE phase).

        Parameters
        ----------
        metrics : dict
            Metrics dictionary

        Returns
        -------
        object_id : str
            NWB object ID

        Raises
        ------
        ValueError
            If called before create or after registration
        """
        self._ensure_created("add_units_metrics")
        return self._table.add_units_metrics(self.analysis_file_name, metrics)

    def add_units_waveforms(self, waveform_extractor) -> str:
        """Add unit waveforms to analysis file (POPULATE phase).

        Parameters
        ----------
        waveform_extractor : spikeinterface.WaveformExtractor
            Waveform extractor object

        Returns
        -------
        object_id : str
            NWB object ID

        Raises
        ------
        ValueError
            If called before create or after registration
        """
        self._ensure_created("add_units_waveforms")
        return self._table.add_units_waveforms(
            self.analysis_file_name, waveform_extractor
        )

    def open_for_write(self):
        """Open analysis file for direct NWB I/O (POPULATE phase).

        Returns context manager for NWBHDF5IO in append mode.
        Only works if file is CREATED but not yet REGISTERED.

        Returns
        -------
        io : pynwb.NWBHDF5IO
            Context manager for file I/O

        Raises
        ------
        ValueError
            If called before create or after registration

        Examples
        --------
        >>> with builder.open_for_write() as io:
        ...     nwbf = io.read()
        ...     nwbf.add_unit(spike_times=times, id=unit_id)
        ...     io.write(nwbf)
        """
        self._ensure_created("open_for_write")
        path = self.get_path()
        return pynwb.NWBHDF5IO(path=path, mode="a", load_namespaces=True)

    def get_path(self) -> str:
        """Get absolute filesystem path to analysis file.

        Returns
        -------
        path : str
            Absolute path to analysis file

        Raises
        ------
        ValueError
            If file not yet created
        """
        if self._state == "INIT":
            raise ValueError(
                "File not yet created. Call this method inside the "
                "'with' block after entering the context manager."
            )
        return self._table.get_abs_path(self.analysis_file_name)
