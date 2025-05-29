import builtins
import contextlib
import csv
from pathlib import Path

try:
    from deeplabcut import evaluate_network
    from deeplabcut.utils.auxiliaryfunctions import get_evaluation_folder
except ImportError:  # pragma: no cover
    evaluate_network, get_evaluation_folder = None, None  # pragma: no cover

from spyglass.position.utils import get_most_recent_file


@contextlib.contextmanager
def suppress_print_from_package(package: str = "deeplabcut"):
    original_print = builtins.print

    def dummy_print(*args, **kwargs):
        stack = [
            frame.f_globals.get("__name__")
            for frame in inspect.stack()
            if hasattr(frame, "f_globals")
        ]
        if any(name and name.startswith(package) for name in stack):
            return  # Suppress if the call comes from the target package
        return original_print(*args, **kwargs)

    import inspect

    builtins.print = dummy_print
    try:
        yield
    finally:
        builtins.print = original_print


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
