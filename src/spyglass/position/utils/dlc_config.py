"""Read, edit, and write a DeepLabCut project ``config.yaml``.

One home for DLC config manipulation so callers never hand-edit yaml. The
editing methods are pure (no I/O); :meth:`DlcConfig.read` / :meth:`DlcConfig.write`
wrap ``dlc_io.read_yaml`` / ``dlc_io.save_yaml`` (which defer to DeepLabCut's
``read_config`` / ``write_config``, preserving the DLC config structure).

Used by :class:`~spyglass.position.utils.tool_strategies.DLCStrategy` to manage
a training project's config; keeps ad-hoc yaml out of ``train.py``.
"""

from pathlib import Path
from typing import Iterable


class DlcConfig:
    """A DeepLabCut ``config.yaml`` and the edits made to it.

    Parameters
    ----------
    project_dir : str or pathlib.Path
        Directory containing the project ``config.yaml``.
    config : dict
        The parsed config contents.
    """

    def __init__(self, project_dir, config: dict):
        self.project_dir = Path(project_dir)
        self.config = config

    # ── I/O ──────────────────────────────────────────────────────────────
    @classmethod
    def read(cls, project_dir) -> "DlcConfig":
        """Read the config from *project_dir* (defers to DLC's ``read_config``)."""
        from spyglass.position.utils.dlc_io import read_yaml

        _, cfg = read_yaml(Path(project_dir))
        return cls(project_dir, cfg)

    def write(self) -> str:
        """Write ``config.yaml`` back to ``project_dir``; return its path."""
        from spyglass.position.utils.dlc_io import save_yaml

        return save_yaml(str(self.project_dir), self.config, filename="config")

    # ── Reads ────────────────────────────────────────────────────────────
    @property
    def video_sets(self) -> dict:
        """The ``video_sets`` mapping (path → crop entry)."""
        return self.config.get("video_sets", {})

    def video_names(self) -> set:
        """Basenames of the videos referenced in ``video_sets``."""
        return {Path(k).name for k in self.video_sets}

    # ── Edits (chainable, in place) ──────────────────────────────────────
    def set(self, key, value) -> "DlcConfig":
        """Set an arbitrary top-level config *key* to *value*."""
        self.config[key] = value
        return self

    def set_bodyparts(self, bodyparts: Iterable[str]) -> "DlcConfig":
        """Replace the ``bodyparts`` list."""
        self.config["bodyparts"] = list(bodyparts)
        return self

    def keep_videos(self, names: Iterable[str]) -> "DlcConfig":
        """Keep only ``video_sets`` whose basename is in *names*.

        Drops stale entries (e.g. a raw ``.h264`` left behind after conversion)
        so ``video_sets`` references exactly the intended videos.
        """
        names = set(names)
        self.config["video_sets"] = {
            k: v for k, v in self.video_sets.items() if Path(k).name in names
        }
        return self
