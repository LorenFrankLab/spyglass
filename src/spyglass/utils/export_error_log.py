"""Log table for errors encountered during file edits."""

import datajoint as dj

from spyglass.utils.logging import logger

schema = dj.schema("common_export_error_log")


@schema
class ExportErrorLog(dj.Manual):
    definition = """
    file: varchar(255)  # file being processed
    source: varchar(255)  # source of the error (e.g., table name or function)
    ---
    """

    @staticmethod
    def _logger_warning(key):
        logger.warning(
            f"Logging export error for file: {key.get('file', 'unknown')}"
            + f" from source: {key.get('source', 'unknown')}"
        )

    def insert1(self, key, **kwargs):
        """Insert a new entry into the ExportErrorLog table.

        Parameters
        ----------
        key : dict
            Dictionary containing the primary key fields for the table.
        **kwargs : dict
            Additional keyword arguments for non-primary key fields.
        """
        self._logger_warning(key)
        super().insert1(key, **kwargs)
