from functools import cached_property
from os import environ as os_environ

import datajoint as dj
import requests
from packaging.version import parse as version_parse

from spyglass import __version__ as sg_verion
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import str_to_bool

schema = dj.schema("common_version")


@schema
class SpyglassVersions(dj.Manual):
    """Track available Spyglass versions.

    This table is populated with the current Spyglass version from PyPI. It is
    used to check if the current version is outdated based on letting the user
    be behind by a certain number of versions. Default is 1. If a developer is
    interested in adjusting this value, please submit a GitHub issue.

    An alternative approach would track minor vs patch versions.
    """

    definition = """
    version: varchar(16) # Spyglass version
    ---
    release_date: date   # Release date
    """

    def fetch_from_pypi(self) -> None:
        """Populate SpyglassVersion table with the current version."""
        response = requests.get("https://pypi.org/pypi/spyglass-neuro/json")

        if not response.ok:
            logger.error(f"Failed to fetch info from PyPI: {response.reason}")
            return

        versions = response.json()["releases"]
        inserts = [
            {"version": version, "release_date": releases[0]["upload_time"]}
            for version, releases in versions.items()
            if releases and "a" not in version and "b" not in version
        ]  # Skip alpha and beta versions

        self.insert(inserts, skip_duplicates=True)

    @property
    def env_var(self) -> bool:
        """Get SPYGLASS_UPDATED environment variable."""
        return str_to_bool(os_environ.get("SPYGLASS_UPDATED", "false"))

    @cached_property
    def is_up_to_date(self) -> bool:
        """Check if the current Spyglass version is up to date.

        Returns
        -------
        bool
            True if the current Spyglass version is up to date, False otherwise.
        """
        if self.env_var:
            return True

        return self.check_updated(allowed_versions_behind=1)

    def check_updated(self, allowed_versions_behind=1) -> bool:
        """Check if the current Spyglass version is updated.

        Parameters
        ----------
        allowed_versions_behind: int, optional
            Number of versions behind to consider up to date. Default is 1.

        Returns
        -------
        bool
            True if the current Spyglass version is up to date.

        Raises
        ------
        RuntimeError
            If the current Spyglass version is outdated.
        """
        if self.env_var:
            return True

        if not len(self):  # Assume populated means updated via weekly cron
            self.fetch_from_pypi()

        earliest_permit = self.fetch(
            "version",
            order_by="release_date DESC",
            limit=allowed_versions_behind + 1,
        )[-1]

        if version_parse(sg_verion) >= version_parse(earliest_permit):
            logger.info(
                f"Spyglass version {sg_verion} is up to date. "
                + f"Latest version is {earliest_permit}."
            )
            os_environ["SPYGLASS_UPDATED"] = "true"
            return True

        raise RuntimeError(
            f"Please upgrade Spyglass version.\n\tHave: {sg_verion}\n\t"
            + f"\n\tNeed: {earliest_permit} or later"
        )
