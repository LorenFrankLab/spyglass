import contextlib
import csv
import inspect
import sys
from pathlib import Path

try:
    from deeplabcut import evaluate_network
    from deeplabcut.utils.auxiliaryfunctions import get_evaluation_folder
except ImportError:  # pragma: no cover
    evaluate_network, get_evaluation_folder = None, None  # pragma: no cover

from spyglass.position.utils import get_most_recent_file


@contextlib.contextmanager
def suppress_print_from_package(package: str = "deeplabcut"):
    """Suppress stdout/stderr writes that originate from *package*.

    Replaces sys.stdout and sys.stderr with a proxy that walks the call stack
    on every write; output whose innermost package-level frame matches
    ``package`` is dropped, everything else passes through unchanged.

    More reliable than patching builtins.print because it also catches tqdm
    progress bars and any code that calls sys.stdout.write() directly.
    """

    class _PackageFilter:
        """Proxy stream: suppress writes from *package*, pass others through."""

        def __init__(self, stream: object) -> None:
            self._stream = stream

        def write(self, text: str) -> int:
            for frame_info in inspect.stack():
                # Real FrameInfo objects store the frame in .frame;
                # test mocks may expose f_globals directly on the object.
                fg = getattr(frame_info, "f_globals", None)
                if fg is None:
                    raw = getattr(frame_info, "frame", None)
                    fg = getattr(raw, "f_globals", {}) if raw else {}
                if fg.get("__name__", "").startswith(package):
                    return len(text)  # drop — came from target package
            return self._stream.write(text)

        def flush(self) -> None:
            return self._stream.flush()

        def __getattr__(self, name: str):
            return getattr(self._stream, name)

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _PackageFilter(old_stdout)
    sys.stderr = _PackageFilter(old_stderr)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_dlc_model_eval(
    yml_path: str,
    model_prefix: str,
    shuffle: int,
    trainingsetindex: int,
    dlc_config: str,
):
    project_path = Path(yml_path).parent
    trainFraction = dlc_config["TrainingFraction"][trainingsetindex]

    with suppress_print_from_package():
        evaluate_network(
            yml_path,
            Shuffles=[shuffle],  # this needs to be a list
            trainingsetindex=trainingsetindex,
            comparisonbodyparts="all",
        )

    eval_folder = get_evaluation_folder(
        trainFraction=trainFraction,
        shuffle=shuffle,
        cfg=dlc_config,
        modelprefix=model_prefix,
    )
    eval_path = project_path / eval_folder
    if not eval_path.exists():
        raise FileNotFoundError(  # pragma: no cover
            f"Couldn't find eval folder: {eval_path}"
        )

    with open(get_most_recent_file(eval_path, ext=".csv"), newline="") as f:
        results = list(csv.DictReader(f, delimiter=","))[0]

    return results
