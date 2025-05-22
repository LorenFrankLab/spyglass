import csv
from pathlib import Path

from deeplabcut import evaluate_network
from deeplabcut.utils.auxiliaryfunctions import get_evaluation_folder

from spyglass.position.utils import get_most_recent_file


def get_dlc_model_eval(
    yml_path: str,
    project_path: Path,
    model_prefix: str,
    shuffle: int,
    train_fraction: float,
    dlc_config: str,
):
    evaluate_network(
        yml_path,
        Shuffles=[shuffle],  # this needs to be a list
        trainingsetindex=train_fraction,
        comparisonbodyparts="all",
    )

    eval_folder = get_evaluation_folder(
        trainFraction=train_fraction,
        shuffle=shuffle,
        cfg=dlc_config,
        modelprefix=model_prefix,
    )
    eval_path = Path(project_path) / eval_folder
    if not eval_path.exists():
        raise FileNotFoundError(f"Couldn't find eval folder: {eval_path}")

    with open(get_most_recent_file(eval_path, ext=".csv"), newline="") as f:
        results = list(csv.DictReader(f, delimiter=","))[0]

    return results
